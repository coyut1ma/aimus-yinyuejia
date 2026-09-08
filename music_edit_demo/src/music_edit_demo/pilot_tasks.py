from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from .models import EditRegion, Stem


class PilotTask(BaseModel):
    """A frozen evaluation intent; audio paths are bound after dataset discovery."""

    task_id: str
    song_id: str
    task_type: str
    edit_region: EditRegion
    instruction: str
    target_stems: list[Stem] = Field(min_length=1)
    must_change: list[str] = Field(min_length=1)
    must_preserve: list[str] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303])
    input_container: str | None = None
    input_stems: dict[Stem, str] = Field(default_factory=dict)


TASK_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "task_type": "single_track",
        "target_stems": [Stem.DRUMS],
        "instruction": "把这一段鼓点加密，其他轨保持不变",
        "must_change": ["drums.density"],
    },
    {
        "task_type": "multi_track",
        "target_stems": [Stem.DRUMS, Stem.BASS],
        "instruction": "把这一段鼓点加密，贝斯更有力量，其他轨保持不变",
        "must_change": ["drums.density", "bass.energy"],
    },
    {
        "task_type": "single_track",
        "target_stems": [Stem.BASS],
        "instruction": "让这一段贝斯更有力量，保持节拍和其他轨不变",
        "must_change": ["bass.energy"],
    },
    {
        "task_type": "multi_track",
        "target_stems": [Stem.VOCALS, Stem.OTHER],
        "instruction": "重新生成这一段人声和伴奏，保持节拍、调性和鼓轨不变",
        "must_change": ["vocals.regenerate", "other.regenerate"],
    },
    {
        "task_type": "single_track",
        "target_stems": [Stem.OTHER],
        "instruction": "让这一段伴奏更有摇滚感，保留原人声和鼓点",
        "must_change": ["other.style"],
    },
    {
        "task_type": "multi_track",
        "target_stems": [Stem.DRUMS, Stem.OTHER],
        "instruction": "让鼓和伴奏更激烈，但保持人声、贝斯和速度不变",
        "must_change": ["drums.energy", "other.energy"],
    },
    {
        "task_type": "single_track",
        "target_stems": [Stem.VOCALS],
        "instruction": "重新生成这一段人声表现，保持原歌词、节拍和伴奏不变",
        "must_change": ["vocals.regenerate"],
    },
    {
        "task_type": "multi_track",
        "target_stems": [Stem.BASS, Stem.OTHER],
        "instruction": "让贝斯和伴奏更有能量，保持人声和鼓轨不变",
        "must_change": ["bass.energy", "other.energy"],
    },
    {
        "task_type": "single_track",
        "target_stems": [Stem.DRUMS],
        "instruction": "把这一段鼓点变得更稀疏，其他轨保持原样",
        "must_change": ["drums.density"],
    },
    {
        "task_type": "multi_track",
        "target_stems": [Stem.VOCALS, Stem.DRUMS, Stem.OTHER],
        "instruction": "重新生成这一段人声、鼓和伴奏，保持贝斯、速度和调性不变",
        "must_change": ["vocals.regenerate", "drums.regenerate", "other.regenerate"],
    },
)


def build_pilot_tasks(song_ids: Iterable[str]) -> list[PilotTask]:
    songs = sorted({song_id.strip() for song_id in song_ids if song_id.strip()})
    if len(songs) != 10:
        raise ValueError("the frozen pilot requires exactly 10 unique song IDs")

    tasks: list[PilotTask] = []
    for index, song_id in enumerate(songs, start=1):
        # Regions are deliberately fixed before looking at model outputs. They
        # can be shifted during dataset binding only when a song is too short.
        first_start = 20.0 + ((index - 1) % 5) * 8.0
        second_start = first_start + 32.0
        for offset, start in enumerate((first_start, second_start), start=1):
            template = TASK_TEMPLATES[((index - 1) * 2 + offset - 1) % len(TASK_TEMPLATES)]
            target_stems = list(template["target_stems"])
            must_preserve = [
                "duration",
                "tempo",
                "key",
                "outside_region",
                "non_target_stems",
            ]
            tasks.append(
                PilotTask(
                    task_id=f"pilot_v1_{index:02d}_{offset}",
                    song_id=song_id,
                    task_type=str(template["task_type"]),
                    edit_region=EditRegion(start_sec=start, end_sec=start + 8.0),
                    instruction=str(template["instruction"]),
                    target_stems=target_stems,
                    must_change=list(template["must_change"]),
                    must_preserve=must_preserve,
                )
            )
    return tasks


def write_pilot_tasks(tasks: Iterable[PilotTask], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [task.model_dump_json() for task in tasks]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def read_pilot_tasks(source: Path) -> list[PilotTask]:
    return [PilotTask.model_validate_json(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def bind_pilot_tasks(tasks: Iterable[PilotTask], containers: dict[str, str]) -> list[PilotTask]:
    bound: list[PilotTask] = []
    for task in tasks:
        try:
            container = containers[task.song_id]
        except KeyError as exc:
            raise ValueError(f"missing MUSDB18 container for {task.song_id}") from exc
        bound.append(task.model_copy(update={"input_container": str(Path(container).resolve())}))
    return bound
