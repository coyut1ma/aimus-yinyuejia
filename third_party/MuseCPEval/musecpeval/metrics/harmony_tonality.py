from typing import Dict, Any
import numpy as np
import librosa
from .utils import circle_of_fifths_distance, cosine_sim, dtw_cosine, _safe_float
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 22050
HOP_LENGTH = 512

CHROMA_METHOD = "cqt"          # "cqt", "stft", "cens"
CHROMA_NORM = np.inf
CHROMA_SMOOTH_FRAMES = 9

KEY_PROFILE_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
KEY_PROFILE_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _load_audio(audio_path: str, sr: int = SAMPLE_RATE):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y = np.nan_to_num(y, nan=0.0)
    return y, sr


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


def _smooth_chroma(chroma: np.ndarray, win: int = CHROMA_SMOOTH_FRAMES) -> np.ndarray:
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        return chroma
    out = np.empty_like(chroma, dtype=float)
    for i in range(12):
        out[i] = _moving_average_1d(chroma[i], win)
    return out


def _framewise_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def _extract_chroma(audio_path: str, sr: int = SAMPLE_RATE, method: str = CHROMA_METHOD) -> np.ndarray:
    y, sr = _load_audio(audio_path, sr=sr)

    if method == "stft":
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            norm=CHROMA_NORM,
        )
    elif method == "cens":
        chroma = librosa.feature.chroma_cens(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    else:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            norm=CHROMA_NORM,
        )

    chroma = np.asarray(chroma, dtype=float)
    chroma = _smooth_chroma(chroma, win=CHROMA_SMOOTH_FRAMES)
    return chroma


def _major_minor_key_from_chroma(mean_chroma: np.ndarray) -> Dict[str, Any]:
    best_key = None
    best_scale = None
    best_corr = -np.inf

    mean_chroma = np.asarray(mean_chroma, dtype=float).reshape(-1)
    if mean_chroma.size != 12 or not np.isfinite(mean_chroma).any():
        return {"error": "invalid chroma"}

    for i in range(12):
        for profile, scale_name in (
            (KEY_PROFILE_MAJOR, "major"),
            (KEY_PROFILE_MINOR, "minor"),
        ):
            rotated = np.roll(profile, i)
            corr = np.corrcoef(mean_chroma, rotated)[0, 1]
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_key = PITCH_NAMES[i]
                best_scale = scale_name

    return {
        "key": best_key,
        "scale": best_scale,
        "strength": _safe_float(best_corr, 0.0),
    }


# =========================================================
# KEY
# =========================================================

def key_scale_from_chroma(audio_path: str) -> Dict[str, Any]:
    try:
        chroma = _extract_chroma(audio_path, sr=SAMPLE_RATE, method=CHROMA_METHOD)
        mean_chroma = chroma.mean(axis=1)
        return _major_minor_key_from_chroma(mean_chroma)
    except Exception as exc:
        return {"error": f"key extraction failed: {exc}"}


def key_relatedness(ref_key: str, ref_scale: str, est_key: str, est_scale: str) -> Dict[str, Any]:
    steps, norm = circle_of_fifths_distance(ref_key, ref_scale, est_key, est_scale)
    if steps is None:
        return {"distance_steps": None, "distance_norm_0to1": None}
    return {
        "distance_steps": int(steps),
        "distance_norm_0to1": float(norm),
    }


# =========================================================
# CHROMA
# =========================================================

def chroma_similarity(
    audio_ref: str,
    audio_est: str,
    sr: int = SAMPLE_RATE,
    method: str = CHROMA_METHOD,
) -> Dict[str, float]:
    C0 = _extract_chroma(audio_ref, sr=sr, method=method).T   # [frames, 12]
    C1 = _extract_chroma(audio_est, sr=sr, method=method).T   # [frames, 12]

    if len(C0) == 0 or len(C1) == 0:
        return {
            "chroma_dtw_cosine": 0.0,
            "mean_chroma_cosine": 0.0,
        }

    C0n = _framewise_normalize(C0)
    C1n = _framewise_normalize(C1)

    mean0 = C0n.mean(axis=0)
    mean1 = C1n.mean(axis=0)

    raw_dtw = dtw_cosine(C0n, C1n)
    raw_mean = cosine_sim(mean0, mean1)

    return {
        "chroma_dtw_cosine": _safe_float(raw_dtw, 0.0),
        "mean_chroma_cosine": _safe_float(raw_mean, 0.0),
    }


# =========================================================
# MAIN API
# =========================================================

def harmony_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    key_ref = key_scale_from_chroma(audio_ref)
    key_est = key_scale_from_chroma(audio_est)

    if "key" in key_ref and "key" in key_est:
        result["key_relatedness"] = key_relatedness(
            key_ref["key"], key_ref["scale"],
            key_est["key"], key_est["scale"],
        )

    result["chroma_similarity"] = chroma_similarity(audio_ref, audio_est)

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute harmony metrics over matching original/edited wav pairs."
    )
    parser.add_argument(
        "--orig_dir",
        type=Path,
        required=True,
        help="Directory of original (reference) wav files.",
    )
    parser.add_argument(
        "--edit_dir",
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

    # Flat metric name -> (nested group, leaf key)
    METRICS = [
        ("chroma_dtw_cosine", "chroma_similarity", "chroma_dtw_cosine"),
        ("mean_chroma_cosine", "chroma_similarity", "mean_chroma_cosine"),
        ("cof_distance_norm_0to1", "key_relatedness", "distance_norm_0to1"),
    ]

    flat_keys: list[str] = [m[0] for m in METRICS]
    accumulator: dict[str, list[float]] = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = harmony_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        for flat_key, group, leaf in METRICS:
            val = _safe_float(scores.get(group, {}).get(leaf, np.nan), np.nan)
            if np.isfinite(val):
                accumulator[flat_key].append(float(val))
        print("ok")

    print("\n--- Results ---")
    print(f"{'Metric':<45}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 70)
    for k in flat_keys:
        vals = accumulator[k]
        if not vals:
            print(f"{k:<45}  {'nan':>8}  {'nan':>8}  {0:4d}")
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"{k:<45}  {mean:8.4f}  {std:8.4f}  {len(vals):4d}")
