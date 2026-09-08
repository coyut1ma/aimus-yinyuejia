from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from music_edit_demo.audio import AudioInfo, write_wav
from music_edit_demo.evaluation import EvaluationConfig, _resolve_tempo_octave, evaluate_stems
from music_edit_demo.models import EditRegion, Stem


def _write_signal(path: Path, values: list[float], sample_rate: int = 8000) -> Path:
    samples: list[int] = []
    for value in values:
        samples.extend([max(-32768, min(32767, round(value * 32767)))] * 2)
    frames = len(values)
    info = AudioInfo(sample_rate=sample_rate, channels=2, sample_width=2, frames=frames, duration_sec=frames / sample_rate)
    return write_wav(path, info, samples)


def _pulse_signal(duration: float, sample_rate: int, frequency: float, amplitude: float = 0.3, delay: float = 0.0) -> list[float]:
    values = [0.0] * round(duration * sample_rate)
    period = 1.0 / frequency
    position = delay
    while position < duration:
        start = round(position * sample_rate)
        for offset in range(min(round(0.018 * sample_rate), len(values) - start)):
            values[start + offset] += amplitude * math.exp(-offset / max(1, round(0.004 * sample_rate)))
        position += period
    return values


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original: dict[str, Path] = {}
        self.edited: dict[str, Path] = {}
        duration = 12.0
        sr = 8000
        base = _pulse_signal(duration, sr, 2.0)
        for index, stem in enumerate(Stem):
            path = self.root / f"{stem.value}.wav"
            _write_signal(path, [value * (0.8 - index * 0.08) for value in base], sr)
            self.original[stem.value] = path
            self.edited[stem.value] = path

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_noop_has_small_volume_deviation(self) -> None:
        report = evaluate_stems(self.original, self.edited, EditRegion(start_sec=4, end_sec=8), ["drums"])
        self.assertIn(report["status"], {"ok", "low_confidence"})
        beat = report["metrics"]["beat"]
        self.assertEqual(beat["backend"], "librosa.beat.beat_track")
        self.assertGreater(beat["reference_beat_count"], 0)
        self.assertGreater(beat["edited_beat_count"], 0)
        volume = report["metrics"]["volume"]["per_stem"]["drums"]
        self.assertEqual(volume["volume_deviation_db"], 0.0)
        self.assertTrue(volume["within_good_threshold"])

    def test_volume_deviation_uses_three_db_good_threshold(self) -> None:
        source = self.original["drums"]
        values = _pulse_signal(12.0, 8000, 2.0, amplitude=0.6)
        edited = self.root / "edited-drums.wav"
        _write_signal(edited, values, 8000)
        self.edited["drums"] = edited
        report = evaluate_stems(self.original, self.edited, EditRegion(start_sec=4, end_sec=8), [Stem.DRUMS], EvaluationConfig())
        metric = report["metrics"]["volume"]["per_stem"]["drums"]
        self.assertAlmostEqual(metric["good_threshold_db"], 3.0)
        self.assertGreater(metric["volume_deviation_db"], 3.0)
        self.assertFalse(metric["within_good_threshold"])

    def test_all_edited_stems_report_missing_alignment_reference(self) -> None:
        report = evaluate_stems(self.original, self.edited, EditRegion(start_sec=4, end_sec=8), list(Stem))
        self.assertEqual(report["metrics"]["alignment"]["status"], "insufficient_reference_tracks")

    def test_resolves_clear_half_tempo_ambiguity(self) -> None:
        adjusted, factor = _resolve_tempo_octave(95.7, 191.4, EvaluationConfig())
        self.assertAlmostEqual(adjusted, 191.4)
        self.assertEqual(factor, 2.0)

        unchanged, factor = _resolve_tempo_octave(117.0, 172.0, EvaluationConfig())
        self.assertEqual(unchanged, 117.0)
        self.assertEqual(factor, 1.0)


if __name__ == "__main__":
    unittest.main()
