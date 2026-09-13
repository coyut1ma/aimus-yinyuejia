"""MuseCPEval — music context preservation metrics.

Scores how much of an original recording's harmony, rhythm, structure,
melody, and timbre survives an edit.

    from musecpeval import harmony_score, rhythm_score

    harmony_score("original.wav", "edited.wav")

The CLI in :mod:`musecpeval.runner` (installed as ``musecpeval``) wraps the same
functions for single pairs and for parallel batches.

The five scoring functions are resolved lazily: importing this package does not
pull in librosa or mir_eval, and ``python -m musecpeval.metrics.<name>`` still
runs a family's own directory-scoring CLI without double-import warnings.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    # The version lives in pyproject.toml; don't duplicate it here.
    __version__ = _dist_version("musecpeval")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0+unknown"

_LAZY = {
    "harmony_score": "musecpeval.metrics.harmony_tonality",
    "melody_score": "musecpeval.metrics.melody_motif",
    "rhythm_score": "musecpeval.metrics.rhythm_meter",
    "structural_score": "musecpeval.metrics.structural_form",
    "timbre_score": "musecpeval.metrics.timbre",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
