from typing import Dict, Any, Tuple, List, Optional
from contextlib import contextmanager
import warnings
import numpy as np
import mir_eval
import os
import tempfile
import scipy
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Intervals shorter than this are dropped as numerical noise.
MIN_INTERVAL_DURATION = 1e-10

BOUNDARY_WINDOW_SEC = 0.5
SHIFT_SEARCH_RANGE_SEC = 2.0
SHIFT_STEP_SEC = 0.02

BUG_EXCEPTIONS = (TypeError, AttributeError, NameError, IndexError, KeyError,
                  ImportError)


# ---------- Basic Tools ----------
def _bounds_to_intervals(bounds) -> np.ndarray:
    """1D boundaries -> (n,2) intervals; (n,2) input is filtered and returned as float."""
    b = np.asarray(bounds, float)
    if b.ndim == 1:
        # Remove duplicates, sort, remove non-finite values
        b = b[np.isfinite(b)]
        b = np.unique(b)
        if b.size < 2:
            return np.empty((0, 2), dtype=float)
        return np.column_stack([b[:-1], b[1:]])

    if b.size == 0 or b.ndim != 2 or b.shape[1] != 2:
        return np.empty((0, 2), dtype=float)

    keep = np.isfinite(b).all(axis=1) & ((b[:, 1] - b[:, 0]) > MIN_INTERVAL_DURATION)
    return b[keep]


def _shift_intervals(intv: np.ndarray, d: float) -> np.ndarray:
    """Shift intervals globally; maintain shape."""
    intv = np.asarray(intv, float)
    if intv.size == 0:
        return intv
    return intv + np.array([d, d], dtype=float)


def _crop_overlap_for_detection(ref_int: np.ndarray, est_int: np.ndarray):
    """Crop both to their common window [t0, t1] (t0 >= 0) and rebase so t0 -> 0.

    Rebasing is required by ``mir_eval.segment.validate_structure``, which rejects
    intervals that do not start at 0, and it also guarantees ref/est span the same
    duration (``pairwise``/``ari`` compare frame matrices element-wise).

    Return (ref_c, est_c); if there is no overlap, return (None, None).
    """
    ref_int = np.asarray(ref_int, float)
    est_int = np.asarray(est_int, float)
    if ref_int.size == 0 or est_int.size == 0:
        return None, None

    # Common time window: start point takes max of both and 0; end point takes min of both
    t0 = max(0.0, float(ref_int[0, 0]), float(est_int[0, 0]))
    t1 = min(float(ref_int[-1, 1]), float(est_int[-1, 1]))
    if not (t1 > t0):
        return None, None

    def _clip_valid(intv):
        if intv.size == 0:
            return None
        S = np.maximum(intv[:, 0], t0)
        E = np.minimum(intv[:, 1], t1)
        keep = (E - S) > MIN_INTERVAL_DURATION
        if not np.any(keep):
            return None
        clipped = np.column_stack([S[keep], E[keep]])
        return clipped - np.array([t0, t0], dtype=float)

    ref_c = _clip_valid(ref_int)
    est_c = _clip_valid(est_int)
    if ref_c is None or est_c is None:
        return None, None
    return ref_c, est_c


# ---------- Search for Best Shift ----------
def best_boundary_shift(
    ref_bounds,
    est_bounds,
    window: float = BOUNDARY_WINDOW_SEC,
    search_range: float = SHIFT_SEARCH_RANGE_SEC,
    step: float = SHIFT_STEP_SEC,
):
    """Search for global shift delta in [-search_range, +search_range] maximizing boundary F.

    Because ``detection`` accepts any boundary within ``window`` seconds, the optimum
    is a plateau of tied deltas rather than a single point. We return the median of
    the plateau: its left edge (what an argmax returns) is off by up to ``window``
    seconds, which would misalign the frame labels fed to pairwise/ari.
    """
    ref_int = _bounds_to_intervals(ref_bounds)
    est_int = _bounds_to_intervals(est_bounds)

    if ref_int.size == 0 or est_int.size == 0:
        return {"F": 0.0, "delta": 0.0}

    deltas, f_scores = [], []
    n_failed, first_error = 0, None
    for d in np.arange(-search_range, search_range + 1e-12, step):
        ref_c, est_c = _crop_overlap_for_detection(ref_int, _shift_intervals(est_int, d))
        if ref_c is None or est_c is None:
            continue
        try:
            _, _, F = mir_eval.segment.detection(ref_c, est_c, window=window, trim=True)
        except BUG_EXCEPTIONS:
            raise
        except Exception as e:
            # Skipping a delta silently shrinks the search grid and can move the
            # returned optimum, so report how much of the grid was lost.
            n_failed += 1
            first_error = first_error or f"{type(e).__name__}: {e}"
            continue
        deltas.append(float(d))
        f_scores.append(float(F))

    if n_failed:
        warnings.warn(
            f"boundary shift search skipped {n_failed} of "
            f"{n_failed + len(deltas)} candidate deltas ({first_error}); "
            "the returned shift may not be the true optimum",
            RuntimeWarning,
            stacklevel=2,
        )

    if not f_scores:
        return {"F": 0.0, "delta": 0.0}

    f_scores = np.asarray(f_scores)
    best_f = float(f_scores.max())
    if best_f <= 0.0:
        return {"F": 0.0, "delta": 0.0}

    tied = np.asarray(deltas)[f_scores >= best_f - 1e-12]
    return {"F": best_f, "delta": float(np.median(tied))}


# ---------- Chroma-based Segment Labels ----------
def _chroma_segment_labels(audio_path: str, bounds: np.ndarray) -> List[str]:
    """
    Assign repeatable segment labels based on dominant chroma pitch class.
    Unlike sequential labels (S0, S1, ...), these can repeat (e.g. A-B-A),
    making ARI meaningful.
    """
    _PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        labels = []
        for i in range(len(bounds) - 1):
            f0 = max(0, int(bounds[i] * sr / hop))
            f1 = min(chroma.shape[1], int(bounds[i + 1] * sr / hop))
            if f1 > f0:
                labels.append(_PITCH_NAMES[int(np.argmax(chroma[:, f0:f1].mean(axis=1)))])
            else:
                labels.append("C")
        return labels if labels else ["C"]
    except BUG_EXCEPTIONS:
        raise
    except Exception as e:
        # All-distinct S0..Sn labels carry no repetition structure and push
        # pairwise/ari towards 0, so this fallback must never be silent.
        warnings.warn(
            f"chroma labelling failed on {audio_path}: {type(e).__name__}: {e}; "
            "falling back to all-distinct S0..Sn labels, which drive pairwise/ari "
            "towards 0",
            RuntimeWarning,
            stacklevel=2,
        )
        return [f"S{i}" for i in range(max(0, len(bounds) - 1))]


# ---------- MSAF ----------
@contextmanager
def _msaf_private_cache():
    """Import msaf with a feature cache private to this call.

    msaf caches one file's features in ``config.features_tmp_file`` (default: the
    *relative* path ``.features_msaf_tmp.json``) and decides whether a cache entry
    applies to the requested audio by comparing **basenames** only
    (``msaf/base.py:232``). Since a reference and its edited counterpart share a
    basename (``original_audio/001.wav`` vs ``edits_audio/.../001.wav``), the edited
    file silently receives the reference's features, making the two segmentations
    identical and the score meaningless. Concurrent runs launched from the same
    directory additionally read half-written JSON from that shared file.

    A unique, deleted-on-exit cache path avoids both. The cache buys us nothing
    anyway: every audio file is processed exactly once per run.
    """
    
    if not hasattr(scipy, 'inf'):
        scipy.inf = np.inf
    import msaf

    fd, path = tempfile.mkstemp(prefix="msaf_feats_", suffix=".json")
    os.close(fd)
    os.unlink(path)  # msaf must see a missing file to treat it as a cache miss

    previous = msaf.config.features_tmp_file
    msaf.config.features_tmp_file = path
    try:
        yield msaf
    finally:
        msaf.config.features_tmp_file = previous
        try:
            os.unlink(path)
        except OSError:
            pass


def msaf_segments(audio_path: str, feature: str = "pcp", algo: str = "sf"):
    try:
        # Duration (for adding tail boundary)
        # Both readers return the same number, so falling through costs nothing
        # but speed -- this is the one handler here that may stay quiet.
        duration = None
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                duration = float(len(y)) / float(sr)
            except Exception as e:
                warnings.warn(
                    f"could not read duration of {audio_path} "
                    f"({type(e).__name__}: {e}); the tail boundary will not be "
                    "corrected",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # msaf.run.process returns a (est_times, est_labels) tuple for a single
        # file, or a list of such tuples in hierarchical mode.
        with _msaf_private_cache() as msaf:
            est = msaf.run.process(
                audio_path, feature=feature, boundaries_id=algo, labels_id="scluster"
            )
        if isinstance(est, list) and est and isinstance(est[0], (tuple, list)):
            est = est[0]
        bounds, labels = est
        bounds = np.asarray(bounds, float)

        # labels_id=None yields a list of -1, and a single repeated label carries
        # no grouping information; fall back to chroma labels in both cases.
        labels = [str(x) for x in labels] if labels is not None else None
        if labels is not None and len(set(labels)) < 2:
            labels = None

        # Normalize: >=0, sort and remove duplicates
        bounds = np.asarray(bounds, float)
        bounds = bounds[np.isfinite(bounds)]
        bounds = np.clip(bounds, 0.0, np.inf)
        bounds = np.unique(bounds)
        if bounds.size == 0 or bounds[0] > 1e-6:
            bounds = np.r_[0.0, bounds]
        if duration is not None and abs(bounds[-1] - duration) > 1e-3:
            if bounds[-1] < duration:
                bounds = np.r_[bounds, duration]
            else:
                bounds[-1] = duration

        # Align labels to number of intervals
        if labels is None:
            labels = _chroma_segment_labels(audio_path, bounds)
        else:
            labels = list(labels)
            need = max(0, len(bounds) - 1)
            if len(labels) < need:
                labels = labels + [labels[-1] if labels else "S"] * (need - len(labels))
            elif len(labels) > need:
                labels = labels[:need]

        return bounds, labels
    except BUG_EXCEPTIONS:
        # e.g. the tuple unpacking above: a bug here must not be reported as a
        # bad audio file and quietly demoted to the uniform-grid fallback.
        raise
    except Exception as e:
        warnings.warn(
            f"msaf failed on {audio_path}: {type(e).__name__}: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None


# ---------- Conversion and Cropping ----------
def crop_to_overlap(
    intervals: np.ndarray,
    labels: Optional[List],
    t_min: float,
    t_max: float
) -> Tuple[np.ndarray, Optional[List]]:
    """Crop labelled intervals to [t_min, t_max] and rebase so t_min -> 0.

    See ``_crop_overlap_for_detection`` for why rebasing is mandatory.
    """
    if intervals.size == 0:
        return intervals, ([] if labels is not None else None)

    S = np.maximum(intervals[:, 0], t_min)
    E = np.minimum(intervals[:, 1], t_max)
    keep = (E - S) > MIN_INTERVAL_DURATION
    intervals_c = np.column_stack([S[keep], E[keep]])

    if intervals_c.size == 0:
        return np.empty((0, 2), dtype=float), ([] if labels is not None else None)

    intervals_c = intervals_c - np.array([t_min, t_min], dtype=float)

    labels_c = None
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=object)
        labels_c = labels_arr[keep].tolist()
        if len(labels_c) < len(intervals_c):
            labels_c += [labels_c[-1] if labels_c else "S"] * (len(intervals_c) - len(labels_c))
        elif len(labels_c) > len(intervals_c):
            labels_c = labels_c[:len(intervals_c)]
    return intervals_c, labels_c


# ---------- Evaluation ----------
def clustering_metrics(ref_bounds, est_bounds, ref_labels, est_labels) -> Dict[str, float]:
    """Pairwise F-measure and ARI between two labelled segmentations.

    Returns NaN for a metric that cannot be computed, never 0.0 -- a real 0.0
    means "no agreement", which is a different claim from "not measurable".
    """
    out: Dict[str, float] = {"pairwise_f": float("nan"), "ari": float("nan")}

    ref_intervals = _bounds_to_intervals(ref_bounds)
    est_intervals = _bounds_to_intervals(est_bounds)
    if ref_intervals.size == 0 or est_intervals.size == 0:
        out["error"] = "empty segmentation"
        return out

    t0 = max(float(ref_intervals[0, 0]), float(est_intervals[0, 0]), 0.0)
    t1 = min(float(ref_intervals[-1, 1]), float(est_intervals[-1, 1]))
    if not (t1 > t0):
        out["error"] = "no temporal overlap"
        return out

    ref_c, ref_l = crop_to_overlap(ref_intervals, ref_labels, t0, t1)
    est_c, est_l = crop_to_overlap(est_intervals, est_labels, t0, t1)

    if ref_c.size == 0 or est_c.size == 0:
        out["error"] = "empty overlap"
        return out

    if not (ref_l is not None and est_l is not None
            and len(ref_l) == len(ref_c) and len(est_l) == len(est_c)):
        out["error"] = "missing or misaligned labels"
        return out

    # A failure here leaves the metric NaN, and the caller drops NaNs from the
    # averages -- so a systematic failure would silently average over a biased
    # subset of pairs. Warn on every one.
    try:
        _, _, f = mir_eval.segment.pairwise(ref_c, ref_l, est_c, est_l)
        out["pairwise_f"] = float(f)
    except BUG_EXCEPTIONS:
        raise
    except Exception as e:
        warnings.warn(f"pairwise failed: {type(e).__name__}: {e}; this pair is "
                      "excluded from the pairwise average",
                      RuntimeWarning, stacklevel=2)
        out["pairwise_error"] = f"{type(e).__name__}: {e}"

    try:
        out["ari"] = float(mir_eval.segment.ari(ref_c, ref_l, est_c, est_l))
    except BUG_EXCEPTIONS:
        raise
    except Exception as e:
        warnings.warn(f"ari failed: {type(e).__name__}: {e}; this pair is "
                      "excluded from the ari average",
                      RuntimeWarning, stacklevel=2)
        out["ari_error"] = f"{type(e).__name__}: {e}"

    return out


# ---------- Entry Point ----------
def librosa_segments_fallback(audio_path: str):
    """Uniform 10s grid with chroma labels, used only when msaf is unavailable.

    Boundaries carry no structural information here, so metrics computed on this
    path are not comparable to msaf-derived ones.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(len(y)) / float(sr)
        n_segments = max(2, min(12, int(round(duration / 10.0))))
        bounds = np.linspace(0.0, duration, n_segments + 1)
        labels = _chroma_segment_labels(audio_path, bounds)
        return bounds, labels
    except BUG_EXCEPTIONS:
        raise
    except Exception as e:
        warnings.warn(
            f"uniform-grid fallback failed on {audio_path}: {type(e).__name__}: "
            f"{e}; returning a single 30s dummy segment, so any score derived "
            "from this file is meaningless",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.array([0.0, 30.0]), ["S0"]


def structural_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    """Structure preservation between a reference and an edited audio file.

    Returns {"pairwise_f": float, "ari": float}; either may be NaN if the
    comparison could not be carried out.
    """
    rb, rl = msaf_segments(audio_ref)
    eb, el = msaf_segments(audio_est)
    if rb is None or eb is None:
        warnings.warn(
            f"msaf segmentation failed for {audio_ref if rb is None else audio_est}; "
            "falling back to a uniform 10s grid -- scores are not comparable to "
            "msaf-derived ones.",
            RuntimeWarning,
            stacklevel=2,
        )
        rb, rl = librosa_segments_fallback(audio_ref)
        eb, el = librosa_segments_fallback(audio_est)

    best = best_boundary_shift(rb, eb)
    eb_shifted = np.asarray(eb, float) + best["delta"]

    scores = clustering_metrics(rb, eb_shifted, rl, el)
    return {"pairwise_f": scores["pairwise_f"], "ari": scores["ari"]}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit_dir", type=str, required=True,
                        help="Path to editing subfolder")
    parser.add_argument("--orig_dir", type=str, required=True,
                        help="Path to original audio folder")
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir)
    edit_dir = Path(args.edit_dir)

    if not edit_dir.exists():
        print(f"Edit dir not found: {edit_dir}", file=sys.stderr)
        sys.exit(1)
    if not orig_dir.exists():
        print(f"Orig dir not found: {orig_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = sorted(
        (f, edit_dir / f.name)
        for f in orig_dir.glob("*.wav")
        if (edit_dir / f.name).exists()
    )

    if not pairs:
        print("No matching pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} pairs in {edit_dir}", flush=True)

    FLAT_KEYS = ["pairwise_f", "ari"]

    accumulator: dict[str, list[float]] = defaultdict(list)
    n_dropped = 0

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = structural_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            n_dropped += 1
            continue

        finite = {k: float(scores[k]) for k in FLAT_KEYS if np.isfinite(scores[k])}
        for k, v in finite.items():
            accumulator[k].append(v)
        if len(finite) < len(FLAT_KEYS):
            n_dropped += 1
            print("ok (partial: " + ", ".join(f"{k}=nan" for k in FLAT_KEYS
                                             if k not in finite) + ")")
        else:
            print("ok")

    print(f"\n--- Results: {edit_dir.name} ---")
    print(f"{'Metric':<20}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 46)
    for k in FLAT_KEYS:
        vals = accumulator.get(k, [])
        if not vals:
            print(f"{k:<20}  {'n/a':>8}  {'n/a':>8}  {0:4d}")
            continue
        print(f"{k:<20}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  {len(vals):4d}")
    if n_dropped:
        print(f"\n{n_dropped}/{len(pairs)} pairs had at least one unmeasurable metric.")
