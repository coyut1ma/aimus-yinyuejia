from typing import Dict, Any, Tuple
import numpy as np
import librosa
import mir_eval
from .utils import _safe_float
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 22050
BEAT_TRACK_UNITS = "time"
BEAT_TRACK_TRIM = True

HALF_DOUBLE_RATIO = 2.0

# =========================================================
# INTERNAL HELPERS
# =========================================================


def _as_1d_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def _normalize_beats(beats) -> np.ndarray:
    beats = _as_1d_float_array(beats)
    if beats.size == 0:
        return beats
    beats = np.unique(beats)
    return beats


def _folded_bpm_delta(bpm_ref: float, bpm_est: float) -> float:
    bpm_ref = _safe_float(bpm_ref, default=np.nan)
    bpm_est = _safe_float(bpm_est, default=np.nan)

    if not np.isfinite(bpm_ref) or not np.isfinite(bpm_est) or bpm_ref <= 0 or bpm_est <= 0:
        return np.nan

    candidates = [
        bpm_est,
        bpm_est * HALF_DOUBLE_RATIO,
        bpm_est / HALF_DOUBLE_RATIO,
    ]
    return float(min(abs(bpm_ref - c) for c in candidates if c > 0))


# =========================================================
# BEAT / TEMPO EXTRACTION
# =========================================================

def tempo_and_beats_librosa(audio_path: str, sr: int = SAMPLE_RATE) -> Tuple[float, np.ndarray]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y = np.nan_to_num(y, nan=0.0)

    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        units=BEAT_TRACK_UNITS,
        trim=BEAT_TRACK_TRIM,
    )

    if hasattr(tempo, "__len__") and len(np.atleast_1d(tempo)) > 0:
        tempo_val = float(np.atleast_1d(tempo)[0])
    else:
        tempo_val = float(tempo)

    beats = _normalize_beats(beats)
    return tempo_val, beats


# =========================================================
# MIR_EVAL BEAT METRICS
# =========================================================

def beat_fmeasure_mireval(ref_beats, est_beats) -> Dict[str, float]:
    ref_beats = _normalize_beats(ref_beats)
    est_beats = _normalize_beats(est_beats)

    if ref_beats.size == 0 or est_beats.size == 0:
        return {"F-measure": 0.0, "Information gain": 0.0}

    try:
        scores = mir_eval.beat.evaluate(ref_beats, est_beats)
        return {
            "F-measure": float(scores["F-measure"]),
            "Information gain": float(scores["Information gain"]),
        }
    except Exception as e:
        print(f"mir_eval error: {e}")
        return {"F-measure": 0.0, "Information gain": 0.0}


# =========================================================
# MAIN API
# =========================================================

def rhythm_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    tempo_ref_bpm, beats_ref = tempo_and_beats_librosa(audio_ref)
    tempo_est_bpm, beats_est = tempo_and_beats_librosa(audio_est)

    # Folded Beat Difference (robust to half/double tempo)
    out["delta_bpm_folded"] = _safe_float(
        _folded_bpm_delta(tempo_ref_bpm, tempo_est_bpm),
        0.0,
    )

    # Beat F-measure and Information Gain (mir_eval)
    out["beat_mir_eval"] = beat_fmeasure_mireval(beats_ref, beats_est)

    return out


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute rhythm metrics over matching original/edited wav pairs."
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        required=True,
        help="Directory of original (reference) wav files.",
    )
    parser.add_argument(
        "--edit-dir",
        type=Path,
        required=True,
        help="Directory of edited (estimated) wav files.",
    )
    args = parser.parse_args()

    orig_dir = args.orig_dir
    edit_dir = args.edit_dir

    pairs = sorted(
        (f, edit_dir / f.name)
        for f in orig_dir.glob("*.wav")
        if (edit_dir / f.name).exists()
    )

    if not pairs:
        print("No matching pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} pairs", flush=True)

    flat_keys: list[str] = []
    accumulator: dict[str, list[float]] = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = rhythm_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        for k, v in scores.items():
            if k == "beat_mir_eval":
                for sub_k, sub_v in v.items():
                    key = f"beat_mir_eval.{sub_k}"
                    accumulator[key].append(float(sub_v))
                    if key not in flat_keys:
                        flat_keys.append(key)
            elif isinstance(v, (int, float)):
                accumulator[k].append(float(v))
                if k not in flat_keys:
                    flat_keys.append(k)
        print("ok")

    print("\n--- Results ---")
    print(f"{'Metric':<45}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 70)
    for k in flat_keys:
        vals = accumulator[k]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"{k:<45}  {mean:8.4f}  {std:8.4f}  {len(vals):4d}")
