"""Local-edit timing and level evaluation.

The evaluator deliberately has no model-specific dependency. Beat positions
and tempi are produced by ``librosa.beat.beat_track``; the small deterministic
NumPy onset detector is reserved for cross-stem alignment. A report contains
the three requested raw measurements only: beat error, cross-stem onset
alignment error, and level deviation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import librosa

from .audio import PCM16_MAX, read_wav
from .models import CandidateResult, EditPlan, EditRegion, MusicProject, Stem


class EvaluationError(ValueError):
    """Raised when an evaluation input cannot be analysed."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Stable DSP and reporting parameters.

    The defaults match the ACE demo's 44.1 kHz PCM16 audio.  The evaluator
    still reads the actual WAV sample rate and derives frame sizes from it.
    """

    reference_context_sec: float = 8.0
    frame_length: int = 2048
    hop_length: int = 512
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    onset_min_interval_ms: float = 30.0
    beat_match_min_ms: float = 40.0
    beat_match_max_ms: float = 120.0
    alignment_max_lag_ms: float = 250.0
    rms_window_ms: float = 400.0
    rms_hop_ms: float = 100.0
    silence_dbfs: float = -60.0
    volume_good_threshold_db: float = 3.0


@dataclass
class _Audio:
    sample_rate: int
    data: np.ndarray


@dataclass
class _Envelope:
    values: np.ndarray
    times: np.ndarray
    sample_rate: int
    hop_length: int


@dataclass
class _Peaks:
    times: np.ndarray
    strengths: np.ndarray


@dataclass
class _FeatureCache:
    audios: dict[str, _Audio] = field(default_factory=dict)
    envelopes: dict[str, _Envelope] = field(default_factory=dict)
    peaks: dict[str, _Peaks] = field(default_factory=dict)


def _as_stem_key(stem: Stem | str) -> str:
    return stem.value if isinstance(stem, Stem) else str(stem)


def _load_audio(path: Path) -> _Audio:
    info, samples = read_wav(path)
    if info.sample_width != 2:
        raise EvaluationError(f"only 16-bit PCM WAV is supported: {path}")
    raw = np.asarray(samples, dtype=np.float32)
    if info.channels < 1:
        raise EvaluationError(f"invalid channel count in {path}")
    raw = raw.reshape(-1, info.channels)
    mono = np.mean(raw, axis=1, dtype=np.float32) / float(PCM16_MAX)
    mono -= np.mean(mono, dtype=np.float32)
    return _Audio(info.sample_rate, mono)


def _validate_audio_set(paths: dict[str, Path]) -> int:
    if set(paths) != {stem.value for stem in Stem}:
        missing = sorted({stem.value for stem in Stem} - set(paths))
        extra = sorted(set(paths) - {stem.value for stem in Stem})
        raise EvaluationError(f"exactly four stems are required; missing={missing}, extra={extra}")
    reference: tuple[int, int] | None = None
    for stem in Stem:
        info, _ = read_wav(paths[stem.value])
        current = (info.sample_rate, info.frames)
        if reference is None:
            reference = current
        elif current != reference:
            raise EvaluationError(f"stem format or duration mismatch: {paths[stem.value]}")
    assert reference is not None
    return reference[0]


def _intervals(region: EditRegion, duration: float, context: float) -> tuple[list[tuple[float, float]], float, float]:
    pre_start = max(0.0, region.start_sec - context)
    post_end = min(duration, region.end_sec + context)
    pre = (pre_start, region.start_sec) if region.start_sec > pre_start else None
    post = (region.end_sec, post_end) if post_end > region.end_sec else None
    values = [item for item in (pre, post) if item is not None and item[1] - item[0] >= 0.1]
    return values, region.start_sec - pre_start, post_end - region.end_sec


def _slice(data: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    lo = max(0, int(round(start * sample_rate)))
    hi = min(len(data), int(round(end * sample_rate)))
    return data[lo:hi]


def _envelope(audio: _Audio, config: EvaluationConfig) -> _Envelope:
    n_fft = max(256, int(config.frame_length))
    hop = max(1, int(config.hop_length))
    signal = audio.data
    if signal.size == 0:
        return _Envelope(np.zeros(0, dtype=np.float32), np.zeros(0), audio.sample_rate, hop)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    frame_count = 1 + max(0, (signal.size - n_fft) // hop)
    starts = np.arange(frame_count, dtype=np.int64) * hop
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(frame_count, n_fft),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    ).copy()
    window = np.hanning(n_fft).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32)
    magnitude /= max(1.0, float(n_fft))
    flux = np.maximum(0.0, magnitude[1:] - magnitude[:-1]).sum(axis=1)
    envelope = np.concatenate(([0.0], flux)).astype(np.float32)
    # Log compression makes the detector less sensitive to loud mastering.
    envelope = np.log1p(envelope * 100.0)
    scale = float(np.percentile(envelope, 95)) if envelope.size else 0.0
    if scale > 1e-8:
        envelope /= scale
    times = (starts + n_fft / 2.0) / float(audio.sample_rate)
    return _Envelope(envelope, times, audio.sample_rate, hop)


def _detect_peaks(envelope: _Envelope, config: EvaluationConfig) -> _Peaks:
    values = envelope.values
    if values.size < 3:
        return _Peaks(np.zeros(0), np.zeros(0))
    # A local adaptive threshold avoids treating a sustained loud chord as a
    # sequence of onsets.  The global floor keeps quiet passages usable.
    radius = max(1, round(0.18 * envelope.sample_rate / envelope.hop_length))
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / float(2 * radius + 1)
    local_mean = np.convolve(values, kernel, mode="same")
    noise = float(np.median(np.abs(values - np.median(values)))) * 1.4826
    threshold = np.maximum(local_mean + max(0.05, 0.55 * noise), float(np.percentile(values, 65)))
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] > threshold[1:-1])
    ) + 1
    if candidates.size == 0:
        return _Peaks(np.zeros(0), np.zeros(0))
    min_distance = max(1, round(config.onset_min_interval_ms / 1000.0 * envelope.sample_rate / envelope.hop_length))
    # Non-maximum suppression gives deterministic one-to-one onset candidates.
    chosen: list[int] = []
    for index in candidates[np.argsort(values[candidates])[::-1]]:
        if all(abs(int(index) - previous) >= min_distance for previous in chosen):
            chosen.append(int(index))
    chosen.sort()
    indices = np.asarray(chosen, dtype=np.int64)
    strengths = values[indices].astype(np.float64)
    return _Peaks(envelope.times[indices].astype(np.float64), strengths)


def _tempo_scalar(value: object) -> float | None:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0:
        return None
    return float(values[0])


def _normalize_tempo(tempo: float, config: EvaluationConfig) -> float:
    """Resolve common half/double-tempo estimates into the configured range."""

    while tempo < config.min_bpm and tempo * 2.0 <= config.max_bpm:
        tempo *= 2.0
    while tempo > config.max_bpm and tempo / 2.0 >= config.min_bpm:
        tempo /= 2.0
    return tempo


def _track_beats(
    audio: _Audio,
    start: float,
    end: float,
    config: EvaluationConfig,
    bpm: float | None = None,
) -> tuple[float | None, np.ndarray]:
    """Run librosa's beat tracker on one continuous fragment.

    Returned beat times use the full-song time axis even though the tracker is
    intentionally run only on the requested fragment.
    """

    signal = _slice(audio.data, audio.sample_rate, start, end)
    if signal.size < max(config.frame_length, config.hop_length * 4):
        return None, np.zeros(0, dtype=np.float64)
    onset_envelope = librosa.onset.onset_strength(
        y=signal,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
        n_fft=config.frame_length,
        aggregate=np.median,
    )
    if onset_envelope.size < 4 or float(np.max(onset_envelope)) <= 1e-8:
        return None, np.zeros(0, dtype=np.float64)
    tempo_value, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
        start_bpm=120.0,
        trim=False,
        bpm=bpm,
        units="frames",
        sparse=True,
    )
    tempo = _tempo_scalar(tempo_value)
    if tempo is not None:
        tempo = _normalize_tempo(tempo, config)
    beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    ).astype(np.float64)
    beat_times += start
    beat_times = beat_times[(beat_times >= start) & (beat_times < end)]
    return tempo, beat_times


def _resolve_tempo_octave(
    raw_tempo: float,
    reference_tempo: float,
    config: EvaluationConfig,
) -> tuple[float, float]:
    """Correct a clear half/double-tempo tracker ambiguity.

    A correction is accepted only when an octave-related candidate is within
    10 percent of the independently tracked reference tempo. The raw value is
    still reported so the adjustment remains auditable.
    """

    candidates = [
        (raw_tempo * factor, factor)
        for factor in (0.5, 1.0, 2.0)
        if config.min_bpm <= raw_tempo * factor <= config.max_bpm
    ]
    adjusted, factor = min(candidates, key=lambda item: abs(item[0] - reference_tempo))
    relative_error = abs(adjusted - reference_tempo) / reference_tempo
    if factor != 1.0 and relative_error <= 0.10:
        return adjusted, factor
    return raw_tempo, 1.0


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = float(np.sum(sorted_weights)) / 2.0
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")])


def _reference_grid(context_beats: np.ndarray, period: float, start: float, end: float) -> np.ndarray:
    """Fit a regular grid to beat-tracker outputs from unedited context."""

    if context_beats.size == 0 or period <= 0 or end <= start:
        return np.zeros(0, dtype=np.float64)
    # Each tracked beat is a candidate phase. Select the phase that minimizes
    # circular timing error over all tracked context beats.
    best_phase = float(context_beats[0])
    best_error = float("inf")
    for candidate in context_beats:
        residuals = np.abs((context_beats - candidate + period / 2.0) % period - period / 2.0)
        error = float(np.median(residuals))
        if error < best_error:
            best_error = error
            best_phase = float(candidate)
    first = best_phase + math.ceil((start - best_phase) / period) * period
    if first >= end:
        return np.zeros(0, dtype=np.float64)
    count = 1 + int(math.floor((end - np.finfo(float).eps - first) / period))
    return first + np.arange(count, dtype=np.float64) * period


def _match_times(expected: np.ndarray, observed: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if expected.size == 0 or observed.size == 0:
        return np.zeros(0), np.zeros(0), np.arange(expected.size, dtype=np.int64)
    remaining = set(range(observed.size))
    matched_expected: list[float] = []
    matched_observed: list[float] = []
    missing: list[int] = []
    for index, value in enumerate(expected):
        if not remaining:
            missing.extend(range(index, expected.size))
            break
        candidates = np.asarray(sorted(remaining), dtype=np.int64)
        nearest = int(candidates[np.argmin(np.abs(observed[candidates] - value))])
        if abs(float(observed[nearest] - value)) <= tolerance:
            matched_expected.append(float(value))
            matched_observed.append(float(observed[nearest]))
            remaining.remove(nearest)
        else:
            missing.append(index)
    return np.asarray(matched_expected), np.asarray(matched_observed), np.asarray(missing, dtype=np.int64)


def _beat_metric(
    original_mix: _Audio,
    edited_mix: _Audio,
    region: EditRegion,
    config: EvaluationConfig,
    cache: _FeatureCache,
) -> dict[str, object]:
    duration = len(original_mix.data) / original_mix.sample_rate
    contexts, pre_sec, post_sec = _intervals(region, duration, config.reference_context_sec)
    context_tempi: list[float] = []
    context_weights: list[float] = []
    context_beat_parts: list[np.ndarray] = []
    for start, end in contexts:
        tempo, beats = _track_beats(original_mix, start, end, config)
        if beats.size:
            context_beat_parts.append(beats)
        if tempo is not None and beats.size >= 2:
            context_tempi.append(tempo)
            context_weights.append(float(beats.size))
    context_beats = np.concatenate(context_beat_parts) if context_beat_parts else np.zeros(0, dtype=np.float64)
    if context_tempi:
        bpm_ref = _weighted_median(context_tempi, context_weights)
    elif context_beats.size >= 2:
        bpm_ref = 60.0 / float(np.median(np.diff(context_beats)))
        bpm_ref = _normalize_tempo(bpm_ref, config)
    else:
        return {
            "status": "insufficient_signal",
            "backend": "librosa.beat.beat_track",
            "reason": "beat tracker found too few beats in unedited context",
            "reference_context_sec": {"pre": round(pre_sec, 4), "post": round(post_sec, 4)},
        }
    period = 60.0 / bpm_ref
    expected = _reference_grid(context_beats, period, region.start_sec, region.end_sec)
    bpm_edit_raw, observed = _track_beats(edited_mix, region.start_sec, region.end_sec, config)
    bpm_edit_tracked = bpm_edit_raw
    tempo_octave_factor = 1.0
    if bpm_edit_raw is not None:
        bpm_edit_tracked, tempo_octave_factor = _resolve_tempo_octave(bpm_edit_raw, bpm_ref, config)
        if tempo_octave_factor != 1.0:
            _, observed = _track_beats(
                edited_mix,
                region.start_sec,
                region.end_sec,
                config,
                bpm=bpm_edit_tracked,
            )
    if expected.size == 0:
        return {
            "status": "insufficient_signal",
            "backend": "librosa.beat.beat_track",
            "reason": "unable to extrapolate a reference beat grid into the edited region",
            "bpm_ref": round(bpm_ref, 4),
            "reference_context_sec": {"pre": round(pre_sec, 4), "post": round(post_sec, 4)},
        }
    tolerance = min(config.beat_match_max_ms / 1000.0, max(config.beat_match_min_ms / 1000.0, period * 0.25))
    matched_expected, matched_observed, missing = _match_times(expected, observed, tolerance)
    errors = matched_observed - matched_expected
    beat_count = max(1, expected.size)
    if errors.size:
        mae = float(np.mean(np.abs(errors)) * 1000.0)
        rmse = float(np.sqrt(np.mean(errors**2)) * 1000.0)
        bias = float(np.mean(errors) * 1000.0)
        p95 = float(np.percentile(np.abs(errors), 95) * 1000.0)
    else:
        mae = rmse = bias = p95 = None
    bpm_edit = bpm_edit_tracked
    if bpm_edit is None and observed.size >= 2:
        bpm_edit = _normalize_tempo(60.0 / float(np.median(np.diff(observed))), config)
    tempo_abs_error = None if bpm_edit is None else abs(bpm_edit - bpm_ref)
    tempo_rel_error = None if tempo_abs_error is None else tempo_abs_error / bpm_ref * 100.0
    return {
        "status": "ok" if matched_expected.size and bpm_edit is not None else "low_confidence",
        "backend": "librosa.beat.beat_track",
        "bpm_ref": round(bpm_ref, 4),
        "bpm_edit_raw": None if bpm_edit_raw is None else round(bpm_edit_raw, 4),
        "bpm_edit": None if bpm_edit is None else round(bpm_edit, 4),
        "tempo_octave_factor": tempo_octave_factor,
        "tempo_abs_error_bpm": None if tempo_abs_error is None else round(tempo_abs_error, 4),
        "tempo_rel_error_pct": None if tempo_rel_error is None else round(tempo_rel_error, 4),
        "beat_mae_ms": None if mae is None else round(mae, 4),
        "beat_rmse_ms": None if rmse is None else round(rmse, 4),
        "beat_p95_ms": None if p95 is None else round(p95, 4),
        "beat_bias_ms": None if bias is None else round(bias, 4),
        "miss_rate": round(float(missing.size / beat_count), 4),
        "extra_rate": round(float(max(0, observed.size - matched_expected.size) / max(1, observed.size)), 4),
        "reference_period_ms": round(period * 1000.0, 4),
        "reference_context_sec": {"pre": round(pre_sec, 4), "post": round(post_sec, 4)},
        "reference_beat_count": int(expected.size),
        "edited_beat_count": int(observed.size),
        "matched_beat_count": int(matched_expected.size),
        "context_tracked_beat_count": int(context_beats.size),
        "match_tolerance_ms": round(tolerance * 1000.0, 4),
    }


def _onset_metric_for_pair(
    original_target: _Peaks,
    edited_target: _Peaks,
    reference: _Peaks,
    region: EditRegion,
    duration: float,
    config: EvaluationConfig,
) -> dict[str, object]:
    contexts, _, _ = _intervals(region, duration, config.reference_context_sec)
    target_context = np.concatenate([original_target.times[(original_target.times >= s) & (original_target.times < e)] for s, e in contexts]) if contexts else np.zeros(0)
    reference_context = np.concatenate([reference.times[(reference.times >= s) & (reference.times < e)] for s, e in contexts]) if contexts else np.zeros(0)
    max_lag = config.alignment_max_lag_ms / 1000.0
    # Estimate the original inter-track lag from nearest onset pairs.  The
    # median is robust to one track having extra ornaments or ghost notes.
    baseline_diffs: list[float] = []
    for value in target_context:
        if reference_context.size:
            nearest = float(reference_context[np.argmin(np.abs(reference_context - value))])
            if abs(value - nearest) <= max_lag:
                baseline_diffs.append(float(value - nearest))
    baseline_lag = float(np.median(baseline_diffs)) if len(baseline_diffs) >= 2 else 0.0
    edited_values = edited_target.times[(edited_target.times >= region.start_sec) & (edited_target.times < region.end_sec)]
    reference_values = reference.times[(reference.times >= region.start_sec) & (reference.times < region.end_sec)]
    tolerance = min(config.beat_match_max_ms / 1000.0, max(config.beat_match_min_ms / 1000.0, 0.08))
    residuals: list[float] = []
    used: set[int] = set()
    for value in edited_values:
        if not reference_values.size:
            continue
        candidates = np.asarray([i for i in range(reference_values.size) if i not in used], dtype=np.int64)
        if candidates.size == 0:
            break
        nearest = int(candidates[np.argmin(np.abs(reference_values[candidates] - (value - baseline_lag)))])
        diff = float(value - reference_values[nearest] - baseline_lag)
        if abs(diff) <= tolerance:
            residuals.append(diff)
            used.add(nearest)
    baseline_mae = float(np.mean(np.abs(np.asarray(baseline_diffs) - baseline_lag)) * 1000.0) if baseline_diffs else None
    if not residuals:
        return {
            "status": "insufficient_signal",
            "reason": "fewer than one matched onset in the edited region",
            "original_context_mae_ms": None if baseline_mae is None else round(baseline_mae, 4),
            "baseline_lag_ms": round(baseline_lag * 1000.0, 4),
            "edited_onset_count": int(edited_values.size),
            "reference_onset_count": int(reference_values.size),
        }
    values = np.asarray(residuals, dtype=np.float64)
    mae = float(np.mean(np.abs(values)) * 1000.0)
    return {
        "status": "ok",
        "alignment_mae_ms": round(mae, 4),
        "alignment_rmse_ms": round(float(np.sqrt(np.mean(values**2)) * 1000.0), 4),
        "alignment_p95_ms": round(float(np.percentile(np.abs(values), 95) * 1000.0), 4),
        "alignment_bias_ms": round(float(np.mean(values) * 1000.0), 4),
        "original_context_mae_ms": None if baseline_mae is None else round(baseline_mae, 4),
        "alignment_degradation_ms": None if baseline_mae is None else round(mae - baseline_mae, 4),
        "baseline_lag_ms": round(baseline_lag * 1000.0, 4),
        "edited_onset_count": int(edited_values.size),
        "reference_onset_count": int(reference_values.size),
        "matched_onset_count": int(values.size),
        "onset_recall": round(float(values.size / max(1, edited_values.size)), 4),
        "unmatched_rate": round(float(1.0 - values.size / max(1, edited_values.size)), 4),
    }


def _alignment_metric(
    original_stems: dict[str, _Audio],
    edited_stems: dict[str, _Audio],
    target_stems: set[str],
    region: EditRegion,
    config: EvaluationConfig,
    cache: _FeatureCache,
) -> dict[str, object]:
    references = [stem.value for stem in Stem if stem.value not in target_stems]
    if not references:
        return {"status": "insufficient_reference_tracks", "reason": "all four stems were edited", "per_target": {}}
    duration = len(next(iter(original_stems.values())).data) / next(iter(original_stems.values())).sample_rate
    per_target: dict[str, object] = {}
    for target in sorted(target_stems):
        target_original_key = f"original:{target}"
        target_edited_key = f"edited:{target}"
        target_original = cache.peaks.setdefault(target_original_key, _detect_peaks(cache.envelopes.setdefault(target_original_key, _envelope(original_stems[target], config)), config))
        target_edited = cache.peaks.setdefault(target_edited_key, _detect_peaks(cache.envelopes.setdefault(target_edited_key, _envelope(edited_stems[target], config)), config))
        per_reference: dict[str, object] = {}
        for reference in references:
            ref_key = f"original:{reference}"
            ref_peaks = cache.peaks.setdefault(ref_key, _detect_peaks(cache.envelopes.setdefault(ref_key, _envelope(original_stems[reference], config)), config))
            per_reference[reference] = _onset_metric_for_pair(target_original, target_edited, ref_peaks, region, duration, config)
        valid = [item for item in per_reference.values() if isinstance(item, dict) and item.get("alignment_mae_ms") is not None]
        valid_degradation = [float(item["alignment_degradation_ms"]) for item in valid if item.get("alignment_degradation_ms") is not None]
        aggregate_mae = float(np.mean([float(item["alignment_mae_ms"]) for item in valid])) if valid else None
        aggregate_degradation = float(np.mean(valid_degradation)) if valid_degradation else None
        per_target[target] = {
            "per_reference": per_reference,
            "aggregate_alignment_mae_ms": None if aggregate_mae is None else round(aggregate_mae, 4),
            "aggregate_alignment_degradation_ms": None if aggregate_degradation is None else round(aggregate_degradation, 4),
        }
    return {"status": "ok", "per_target": per_target, "reference_stems": references}


def _rms_db(values: np.ndarray) -> float:
    if values.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(values.astype(np.float64)))))
    return -120.0 if rms <= 1e-8 else 20.0 * math.log10(rms)


def _rms_frames(audio: _Audio, intervals: Iterable[tuple[float, float]], config: EvaluationConfig) -> np.ndarray:
    window = max(1, round(config.rms_window_ms / 1000.0 * audio.sample_rate))
    hop = max(1, round(config.rms_hop_ms / 1000.0 * audio.sample_rate))
    output: list[float] = []
    for start, end in intervals:
        values = _slice(audio.data, audio.sample_rate, start, end)
        if values.size == 0:
            continue
        if values.size < window:
            output.append(_rms_db(values))
            continue
        starts = range(0, values.size - window + 1, hop)
        output.extend(_rms_db(values[pos : pos + window]) for pos in starts)
    return np.asarray(output, dtype=np.float64)


def _volume_metric(
    original_stems: dict[str, _Audio],
    edited_stems: dict[str, _Audio],
    target_stems: set[str],
    region: EditRegion,
    config: EvaluationConfig,
) -> dict[str, object]:
    duration = len(next(iter(original_stems.values())).data) / next(iter(original_stems.values())).sample_rate
    contexts, pre_sec, post_sec = _intervals(region, duration, config.reference_context_sec)
    per_stem: dict[str, object] = {}
    for target in sorted(target_stems):
        reference_levels = _rms_frames(original_stems[target], contexts, config)
        edited_levels = _rms_frames(edited_stems[target], [(region.start_sec, region.end_sec)], config)
        valid_reference = reference_levels[reference_levels > config.silence_dbfs]
        valid_edited = edited_levels[edited_levels > config.silence_dbfs]
        if valid_reference.size == 0 or valid_edited.size == 0:
            per_stem[target] = {
                "status": "insufficient_signal",
                "reason": "edited or unedited context is below the silence threshold",
                "volume_deviation_db": None,
                "absolute_volume_deviation_db": None,
                "good_threshold_db": config.volume_good_threshold_db,
            }
            continue
        reference_db = float(np.median(valid_reference))
        edited_db = float(np.median(valid_edited))
        deviation = edited_db - reference_db
        per_stem[target] = {
            "status": "ok",
            "reference_rms_dbfs": round(reference_db, 4),
            "edited_rms_dbfs": round(edited_db, 4),
            "volume_deviation_db": round(deviation, 4),
            "absolute_volume_deviation_db": round(abs(deviation), 4),
            "within_good_threshold": bool(abs(deviation) <= config.volume_good_threshold_db),
            "good_threshold_db": config.volume_good_threshold_db,
            "reference_frame_count": int(valid_reference.size),
            "edited_frame_count": int(valid_edited.size),
        }
    return {
        "status": "ok" if any(item.get("status") == "ok" for item in per_stem.values() if isinstance(item, dict)) else "insufficient_signal",
        "per_stem": per_stem,
        "reference_context_sec": {"pre": round(pre_sec, 4), "post": round(post_sec, 4)},
        "good_threshold_db": config.volume_good_threshold_db,
    }


def _load_stem_audio(paths: dict[str, Path]) -> dict[str, _Audio]:
    _validate_audio_set(paths)
    return {stem: _load_audio(path) for stem, path in paths.items()}


def evaluate_stems(
    original_stems: dict[Stem | str, Path],
    edited_stems: dict[Stem | str, Path],
    region: EditRegion,
    target_stems: Sequence[Stem | str],
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Evaluate one edited candidate and return a JSON-serialisable report."""

    cfg = config or EvaluationConfig()
    original_paths = {_as_stem_key(key): Path(value).resolve() for key, value in original_stems.items()}
    edited_paths = {_as_stem_key(key): Path(value).resolve() for key, value in edited_stems.items()}
    sample_rate = _validate_audio_set(original_paths)
    edited_rate = _validate_audio_set(edited_paths)
    if edited_rate != sample_rate:
        raise EvaluationError("original and edited sample rates differ")
    original = _load_stem_audio(original_paths)
    edited = _load_stem_audio(edited_paths)
    original_stereo_length = {len(value.data) for value in original.values()}
    edited_stereo_length = {len(value.data) for value in edited.values()}
    if len(original_stereo_length) != 1 or len(edited_stereo_length) != 1 or original_stereo_length != edited_stereo_length:
        raise EvaluationError("original and edited stem durations differ")
    target_keys = {_as_stem_key(item) for item in target_stems}
    valid_stems = {stem.value for stem in Stem}
    unknown = target_keys - valid_stems
    if unknown:
        raise EvaluationError(f"unknown target stems: {sorted(unknown)}")
    if not target_keys:
        raise EvaluationError("at least one target stem is required")
    cache = _FeatureCache()
    original_mix = _Audio(sample_rate, sum((audio.data for audio in original.values()), np.zeros(len(next(iter(original.values())).data), dtype=np.float32)))
    edited_mix = _Audio(sample_rate, sum((audio.data for audio in edited.values()), np.zeros(len(next(iter(edited.values())).data), dtype=np.float32)))
    beat = _beat_metric(original_mix, edited_mix, region, cfg, cache)
    alignment = _alignment_metric(original, edited, target_keys, region, cfg, cache)
    volume = _volume_metric(original, edited, target_keys, region, cfg)
    warnings: list[str] = []
    if beat.get("status") != "ok":
        warnings.append(f"beat:{beat.get('status')}")
    if alignment.get("status") != "ok":
        warnings.append(f"alignment:{alignment.get('status')}")
    if volume.get("status") != "ok":
        warnings.append(f"volume:{volume.get('status')}")
    return {
        "schema_version": "evaluation.v1",
        "status": "ok" if not warnings else "low_confidence",
        "region": {"start_sec": region.start_sec, "end_sec": region.end_sec},
        "target_stems": sorted(target_keys),
        "metrics": {"beat": beat, "alignment": alignment, "volume": volume},
        "diagnostics": {
            "warnings": warnings,
            "sample_rate": sample_rate,
            "channels": 1,
            "source_format": "16-bit PCM WAV, mono analysis downmix",
            "extractor": {
                "beat_backend": "librosa.beat.beat_track",
                "librosa_version": librosa.__version__,
                "onset_backend": "numpy_dsp",
                "frame_length": cfg.frame_length,
                "hop_length": cfg.hop_length,
            },
        },
    }


def evaluate_candidate(
    project: MusicProject,
    plan: EditPlan,
    candidate: CandidateResult,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Evaluate an existing ACE demo candidate against its source project."""

    original = {stem.value: project.stem_path(stem) for stem in Stem}
    edited = {stem.value: Path(candidate.edited_stems[stem]) for stem in Stem}
    report = evaluate_stems(original, edited, plan.region, [target.stem for target in plan.targets], config)
    report["project_id"] = project.project_id
    report["plan_id"] = plan.plan_id
    report["candidate"] = {
        "seed": candidate.seed,
        "backend": candidate.backend,
        "edited_mix": candidate.edited_mix,
    }
    return report


def save_evaluation(report: dict[str, object], destination: Path) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
