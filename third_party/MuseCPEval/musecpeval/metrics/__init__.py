"""The five metric families.

Each module exposes one scoring function taking ``(audio_ref, audio_est)`` and
returning a dict. They are imported lazily by name rather than re-exported here,
because :mod:`musecpeval.metrics.structural_form` pulls in msaf, which is an
optional dependency.
"""
