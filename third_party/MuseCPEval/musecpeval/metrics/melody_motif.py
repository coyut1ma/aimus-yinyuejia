from typing import Dict, List, Set, Tuple
import numpy as np
import librosa
from scipy.spatial.distance import cdist
from .utils import _safe_float
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 16000
MAX_DURATION = 60.0
CHROMA_HOP = 1600          # ~10 fps at 16 kHz — keeps the DTW matrix small

MIN_CONTOUR_FRAMES = 4     # chroma frames needed before a contour is meaningful
MIN_NOTE_FRAMES = 2        # a note must span >=2 chroma frames (~200 ms) to count

MOTIF_NGRAM_N = 3


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _load_chroma(audio_path: str) -> np.ndarray:
    """Load audio and compute chroma_cqt. Returns a (12, T) array."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
    y = np.nan_to_num(y, nan=0.0)
    if len(y) == 0:
        raise ValueError(f"empty audio: {audio_path}")
    return librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=CHROMA_HOP, norm=2)


def _dtw_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean cosine similarity along the optimal DTW path: 1/|P*| * sum(1 - D_ij)."""
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    D = cdist(X, Y, metric="cosine")
    D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float64)

    n, m = D.shape
    # Anti-diagonal vectorized DTW: O(n+m) Python iterations, all inner work in numpy.
    # plen carries the length of the optimal path reaching each cell so the final
    # cost is the mean over |P*| rather than over a fixed divisor.
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    plen = np.zeros((n + 1, m + 1), dtype=np.int32)
    acc[0, 0] = 0.0
    for k in range(1, n + m + 1):
        i_lo = max(1, k - m)
        i_hi = min(n, k - 1) + 1
        ii = np.arange(i_lo, i_hi)
        jj = k - ii                         # j = k - i, ranges over valid (i,j) pairs
        cand = np.stack((acc[ii - 1, jj - 1], acc[ii - 1, jj], acc[ii, jj - 1]))
        lens = np.stack((plen[ii - 1, jj - 1], plen[ii - 1, jj], plen[ii, jj - 1]))
        pick = np.argmin(cand, axis=0)
        r = np.arange(len(ii))
        acc[ii, jj] = D[ii - 1, jj - 1] + cand[pick, r]
        plen[ii, jj] = lens[pick, r] + 1

    cost = float(acc[n, m]) / max(int(plen[n, m]), 1)
    return float(np.clip(1.0 - cost, 0.0, 1.0))


def _note_pitch_classes(C: np.ndarray) -> List[int]:
    """
    Collapse the per-frame dominant pitch class into a note-level sequence.

    Consecutive frames sharing a pitch class form one note, so a held note
    contributes a single symbol instead of a run of zero-intervals.
    """
    pitch_seq = np.argmax(C, axis=0)
    notes: List[int] = []
    i = 0
    while i < len(pitch_seq):
        j = i
        while j < len(pitch_seq) and pitch_seq[j] == pitch_seq[i]:
            j += 1
        if (j - i) >= MIN_NOTE_FRAMES and (not notes or notes[-1] != pitch_seq[i]):
            notes.append(int(pitch_seq[i]))
        i = j
    return notes


def _interval_ngrams(C: np.ndarray, n: int) -> Set[Tuple[int, ...]]:
    notes = _note_pitch_classes(C)
    if len(notes) < 2:
        return set()
    tokens = [int(t) for t in np.diff(notes) % 12]
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# =========================================================
# CONTOUR DTW SIMILARITY
# =========================================================

def contour_dtw_similarity(C_ref: np.ndarray, C_est: np.ndarray) -> float:
    """
    Chroma-based contour DTW similarity in [0, 1].

    The first-order chroma difference encodes the direction of local pitch
    motion while tolerating polyphony. Each sequence keeps its own length —
    DTW absorbs the tempo difference, so no truncation to a common frame
    count. Transposition is normalized by searching all 12 circular chroma
    shifts and keeping the best match.
    """
    if C_ref.shape[1] < MIN_CONTOUR_FRAMES or C_est.shape[1] < MIN_CONTOUR_FRAMES:
        return 0.0

    dC_ref = np.diff(C_ref, axis=1).T   # (T_ref - 1, 12)
    dC_est = np.diff(C_est, axis=1).T   # (T_est - 1, 12)

    best = 0.0
    for shift in range(12):
        sim = _dtw_cosine(dC_ref, np.roll(dC_est, shift, axis=1))
        if sim > best:
            best = sim

    return _safe_float(best, 0.0)


# =========================================================
# MOTIF N-GRAM RECALL
# =========================================================

def motif_ngram_recall(C_ref: np.ndarray, C_est: np.ndarray, n: int = MOTIF_NGRAM_N) -> float:
    """
    Fraction of the reference's note-interval n-grams that survive the edit.

    Interval tokens are taken mod 12, which already makes them invariant to
    transposition: a circular chroma shift adds a constant pitch-class offset
    that the difference cancels.
    """
    grams_ref = _interval_ngrams(C_ref, n)
    grams_est = _interval_ngrams(C_est, n)

    if not grams_ref:
        # no motif on either side means nothing was lost
        return 1.0 if not grams_est else 0.0

    return _safe_float(len(grams_ref & grams_est) / len(grams_ref), 0.0)


# =========================================================
# MAIN API
# =========================================================

def melody_score(audio_ref: str, audio_est: str) -> Dict[str, float]:
    C_ref = _load_chroma(audio_ref)
    C_est = _load_chroma(audio_est)

    return {
        "contour_dtw_similarity": contour_dtw_similarity(C_ref, C_est),
        "motif_3gram_recall": motif_ngram_recall(C_ref, C_est, n=MOTIF_NGRAM_N),
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute melody & motif metrics over matching original/edited wav pairs."
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

    flat_keys = ["contour_dtw_similarity", "motif_3gram_recall"]
    accumulator: dict[str, list[float]] = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        scores = melody_score(str(ref_path), str(est_path))

        for k in flat_keys:
            val = _safe_float(scores[k], np.nan)
            if np.isfinite(val):
                accumulator[k].append(float(val))
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
