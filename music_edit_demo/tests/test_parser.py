from __future__ import annotations

import unittest

from music_edit_demo.models import EditAttribute, EditRegion, PlanStatus, Stem
from music_edit_demo.parser import ControlledInstructionParser


class ControlledInstructionParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ControlledInstructionParser()
        self.region = EditRegion(start_sec=3, end_sec=7)

    def test_multitrack_instruction_and_preserve_clause(self) -> None:
        plan = self.parser.parse(
            "project-1",
            self.region,
            "把鼓点加密，贝斯更有力量，其他轨保持不变",
        )
        self.assertEqual(plan.status, PlanStatus.READY)
        self.assertEqual({target.stem for target in plan.targets}, {Stem.DRUMS, Stem.BASS})
        attributes = {
            target.stem: {operation.attribute for operation in target.operations}
            for target in plan.targets
        }
        self.assertIn(EditAttribute.DENSITY, attributes[Stem.DRUMS])
        self.assertIn(EditAttribute.ENERGY, attributes[Stem.BASS])

    def test_time_conflict_requires_confirmation(self) -> None:
        plan = self.parser.parse("project-1", self.region, "把第4秒到第8秒的鼓重新生成")
        self.assertEqual(plan.status, PlanStatus.NEEDS_CONFIRMATION)

    def test_exact_lyric_edit_is_unsupported(self) -> None:
        plan = self.parser.parse("project-1", self.region, "把人声歌词替换成新的句子")
        self.assertEqual(plan.status, PlanStatus.UNSUPPORTED)
        self.assertTrue(plan.unsupported_reasons)

    def test_missing_target_is_unsupported(self) -> None:
        plan = self.parser.parse("project-1", self.region, "让这一段更有力量")
        self.assertEqual(plan.status, PlanStatus.UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()

