from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from music_edit_demo.audio import create_sine_wav, read_wav
from music_edit_demo.config import Settings
from music_edit_demo.models import JobStatus, Stem
from music_edit_demo.service import DemoService


class DemoServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        settings = Settings(runtime_dir=root / "runtime")
        self.service = DemoService(settings)
        sources = root / "sources"
        frequencies = {
            Stem.VOCALS: 220.0,
            Stem.DRUMS: 110.0,
            Stem.BASS: 55.0,
            Stem.OTHER: 330.0,
        }
        self.stems: dict[Stem, str] = {}
        for stem, frequency in frequencies.items():
            path = create_sine_wav(sources / f"{stem.value}.wav", 12, frequency)
            self.stems[stem] = str(path)
        self.project = self.service.register_project(self.stems, "test-project")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_end_to_end_fake_job_preserves_non_targets(self) -> None:
        plan = self.service.create_plan(
            self.project.project_id,
            3,
            7,
            "把鼓点加密，贝斯更有力量，其他轨保持不变",
        )
        job = self.service.submit_job(plan.plan_id)
        completed = self.service.run_job(job.job_id)
        self.assertEqual(completed.status, JobStatus.SUCCEEDED, completed.error)

        result = self.service.get_result(job.job_id)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.selected.backend, "fake-dsp-not-a-model")
        candidate_hashes = {
            hashlib.sha256(Path(candidate.edited_stems[Stem.DRUMS]).read_bytes()).hexdigest()
            for candidate in result.candidates
        }
        self.assertEqual(len(candidate_hashes), 3)
        self.assertEqual(len({candidate.score for candidate in result.candidates}), 3)

        for stem in (Stem.VOCALS, Stem.OTHER):
            original = Path(self.stems[stem]).read_bytes()
            edited = Path(result.selected.edited_stems[stem]).read_bytes()
            self.assertEqual(original, edited)

        info, original_drums = read_wav(Path(self.stems[Stem.DRUMS]))
        _, edited_drums = read_wav(Path(result.selected.edited_stems[Stem.DRUMS]))
        start = round(3 * info.sample_rate) * info.channels
        end = round(7 * info.sample_rate) * info.channels
        self.assertEqual(original_drums[:start], edited_drums[:start])
        self.assertEqual(original_drums[end:], edited_drums[end:])
        self.assertNotEqual(original_drums[start:end], edited_drums[start:end])

        evaluation = self.service.evaluate_job(job.job_id)
        self.assertEqual(evaluation["schema_version"], "evaluation.v1")
        self.assertNotIn("preservation_score", evaluation["metrics"])
        self.assertIn("beat", evaluation["metrics"])
        self.assertIn("alignment", evaluation["metrics"])
        self.assertIn("volume", evaluation["metrics"])
        self.assertEqual(evaluation["metrics"]["volume"]["good_threshold_db"], 3.0)

    def test_builds_musecpeval_manifest(self) -> None:
        plan = self.service.create_plan(
            self.project.project_id,
            3,
            7,
            "把鼓点加密，其他轨保持不变",
        )
        job = self.service.submit_job(plan.plan_id)
        completed = self.service.run_job(job.job_id)
        self.assertEqual(completed.status, JobStatus.SUCCEEDED, completed.error)

        manifest_path = self.service.settings.runtime_dir / "musecpeval" / "pairs.json"
        pairs = self.service.build_musecpeval_manifest(
            job.job_id,
            include_target_stems=True,
            output=manifest_path,
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["kind"], "mix")
        self.assertTrue(Path(str(pairs[0]["ref"])).is_file())
        self.assertTrue(Path(str(pairs[0]["est"])).is_file())
        self.assertEqual(pairs[1]["kind"], "target_stem")
        self.assertEqual(pairs[1]["stem"], "drums")
        self.assertTrue(manifest_path.is_file())

    def test_requires_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "context"):
            self.service.create_plan(
                self.project.project_id,
                0.5,
                4,
                "重新生成鼓轨",
            )


if __name__ == "__main__":
    unittest.main()
