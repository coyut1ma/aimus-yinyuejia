from typing import Dict, Any
import numpy as np
import librosa
from .utils import cosine_sim, _safe_float, EPSILON
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
MAX_DURATION = 60.0

N_MFCC = 13                 # coefficient 0 is dropped: it encodes energy, not colour
MFCC_DROP_C0 = True
MFCC_USE_DELTA = True       # first derivative, spectral movement
MFCC_USE_ACCEL = True       # second derivative, spectral acceleration
DELTA_WIDTH = 9             # librosa.feature.delta width, must be odd and >= 3

# Ridge is relative to the average feature variance: MFCC variances run in the
# hundreds, so a fixed absolute ridge is too small to regularize anything.
COV_REG_REL = 1e-6
# Divergences are heavy-tailed (negatives around 2, positives around 80, with a
# long upper tail), so exp(-skl/scale) bottoms out at 0 for every real
# instrument swap. A rational falloff keeps the whole range usable; both forms
# are monotone in the divergence, so ranking is unaffected.
SKL_SIMILARITY_SCALE = 50.0


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _load_audio(audio_path: str) -> np.ndarray:
    """Load mono audio at SAMPLE_RATE. Raises on anything unusable."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                        duration=MAX_DURATION)

    y = np.nan_to_num(y, nan=0.0)
    if len(y) == 0:
        raise ValueError(f"empty audio: {audio_path}")
    if np.max(np.abs(y)) < EPSILON:
        raise ValueError(f"silent audio: {audio_path}")
    return y


def _mfcc_features(y: np.ndarray) -> np.ndarray:
    """MFCC 1..12 stacked with delta and acceleration. Returns (T, F)."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH, n_fft=N_FFT,
    )

    if MFCC_DROP_C0:
        mfcc = mfcc[1:]

    stack = [mfcc]
    T = mfcc.shape[1]
    width = min(DELTA_WIDTH, T if T % 2 else T - 1)
    if width >= 3:
        if MFCC_USE_DELTA:
            stack.append(librosa.feature.delta(mfcc, width=width, order=1))
        if MFCC_USE_ACCEL:
            stack.append(librosa.feature.delta(mfcc, width=width, order=2))

    return np.vstack(stack).T


def _symmetric_kl_gaussian(m0, c0, m1, c1) -> float:
    """Symmetrized KL between N(m0, c0) and N(m1, c1). Closed form."""
    k = m0.shape[0]
    c0 = c0 + COV_REG_REL * (np.trace(c0) / k + EPSILON) * np.eye(k)
    c1 = c1 + COV_REG_REL * (np.trace(c1) / k + EPSILON) * np.eye(k)

    try:
        inv0 = np.linalg.inv(c0)
        inv1 = np.linalg.inv(c1)
    except np.linalg.LinAlgError:
        return np.inf

    dm = (m1 - m0).reshape(-1, 1)

    # KL(N0||N1) + KL(N1||N0); the -k and log-det terms cancel in the sum
    kl = 0.5 * (
        np.trace(inv1 @ c0) + np.trace(inv0 @ c1)
        + float((dm.T @ (inv0 + inv1) @ dm).item())
        - 2.0 * k
    )
    return float(max(kl, 0.0))


# =========================================================
# TIMBRE METRICS
# =========================================================

def mfcc_skl_similarity(mfcc_ref: np.ndarray, mfcc_est: np.ndarray) -> Dict[str, float]:
    """Bag-of-frames Gaussian distance. Returns the divergence and its [0, 1] map."""
    # A full-rank covariance needs more frames than feature dimensions.
    min_frames = mfcc_ref.shape[1] + 1
    if len(mfcc_ref) < min_frames or len(mfcc_est) < min_frames:
        return {"mfcc_skl_divergence": np.inf, "mfcc_skl_similarity": 0.0}

    skl = _symmetric_kl_gaussian(
        mfcc_ref.mean(axis=0), np.cov(mfcc_ref, rowvar=False),
        mfcc_est.mean(axis=0), np.cov(mfcc_est, rowvar=False),
    )
    sim = 1.0 / (1.0 + skl / SKL_SIMILARITY_SCALE) if np.isfinite(skl) else 0.0

    return {
        "mfcc_skl_divergence": _safe_float(skl, np.inf),
        "mfcc_skl_similarity": _safe_float(sim, 0.0),
    }


def mean_mfcc_cosine(mfcc_ref: np.ndarray, mfcc_est: np.ndarray) -> Dict[str, float]:
    """
    Cosine between the time-averaged frame vectors. The delta and acceleration
    dimensions average to roughly zero, so this is driven by the mean spectral
    envelope.
    """
    if len(mfcc_ref) == 0 or len(mfcc_est) == 0:
        return {"mean_mfcc_cosine": 0.0}

    cos = cosine_sim(mfcc_ref.mean(axis=0), mfcc_est.mean(axis=0))
    return {"mean_mfcc_cosine": _safe_float(cos, 0.0)}


# =========================================================
# MAIN API
# =========================================================

def timbre_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    mfcc_ref = _mfcc_features(_load_audio(audio_ref))
    mfcc_est = _mfcc_features(_load_audio(audio_est))

    out: Dict[str, Any] = {}
    out.update(mfcc_skl_similarity(mfcc_ref, mfcc_est))
    out.update(mean_mfcc_cosine(mfcc_ref, mfcc_est))

    return out


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute timbre metrics over matching original/edited wav pairs."
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

    flat_keys: list = []
    accumulator: dict = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = timbre_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        for k, v in scores.items():
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
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
