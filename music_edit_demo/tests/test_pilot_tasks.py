from __future__ import annotations

import unittest

from music_edit_demo.models import PlanStatus
from music_edit_demo.parser import ControlledInstructionParser
from music_edit_demo.pilot_tasks import build_pilot_tasks


class PilotTaskTests(unittest.TestCase):
    def test_builds_frozen_twenty_task_manifest(self) -> None:
        tasks = build_pilot_tasks([f"song-{index:02d}" for index in range(1, 11)])
        self.assertEqual(len(tasks), 20)
        self.assertEqual(len({task.task_id for task in tasks}), 20)
        self.assertEqual(sum(task.task_type == "single_track" for task in tasks), 10)
        self.assertEqual(sum(task.task_type == "multi_track" for task in tasks), 10)
        self.assertTrue(all(task.target_stems for task in tasks))
        self.assertEqual(tasks[0].task_id, "pilot_v1_01_1")
        self.assertEqual(tasks[-1].task_id, "pilot_v1_10_2")

    def test_requires_ten_unique_song_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            build_pilot_tasks(["one"])

    def test_all_frozen_instructions_parse_declared_targets(self) -> None:
        parser = ControlledInstructionParser()
        tasks = build_pilot_tasks([f"song-{index:02d}" for index in range(1, 11)])

        for task in tasks:
            with self.subTest(task_id=task.task_id):
                plan = parser.parse(task.song_id, task.edit_region, task.instruction)
                self.assertEqual(plan.status, PlanStatus.READY)
                self.assertEqual(
                    {target.stem for target in plan.targets},
                    set(task.target_stems),
                )


if __name__ == "__main__":
    unittest.main()
