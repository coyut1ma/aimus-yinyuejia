from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Stem(str, Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"


class EditAttribute(str, Enum):
    REGENERATE = "regenerate"
    ENERGY = "energy"
    DENSITY = "density"
    STYLE = "style"


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    CHANGE = "change"


class PlanStatus(str, Enum):
    READY = "ready"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNSUPPORTED = "unsupported"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EditRegion(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "EditRegion":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        if self.end_sec - self.start_sec > 15:
            raise ValueError("MVP edit regions cannot exceed 15 seconds")
        return self


class EditOperation(BaseModel):
    attribute: EditAttribute
    direction: Direction = Direction.CHANGE
    value: str | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> "EditOperation":
        if self.attribute in {EditAttribute.ENERGY, EditAttribute.DENSITY}:
            if self.direction not in {Direction.INCREASE, Direction.DECREASE}:
                raise ValueError("energy and density require increase or decrease")
        if self.attribute == EditAttribute.STYLE and not self.value:
            raise ValueError("style edits require a value")
        return self


class StemEdit(BaseModel):
    stem: Stem
    operations: list[EditOperation] = Field(min_length=1)


DEFAULT_PRESERVE = [
    "duration",
    "tempo",
    "key",
    "outside_region",
    "non_target_stems",
]


class EditPlan(BaseModel):
    schema_version: str = "1.0"
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex}")
    project_id: str
    region: EditRegion
    instruction: str = Field(min_length=1)
    targets: list[StemEdit] = Field(default_factory=list)
    generator_prompt: str = ""
    preserve: list[str] = Field(default_factory=lambda: list(DEFAULT_PRESERVE))
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303], min_length=1, max_length=5)
    status: PlanStatus = PlanStatus.READY
    unsupported_reasons: list[str] = Field(default_factory=list)
    parser_name: str = "controlled"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_plan(self) -> "EditPlan":
        stems = [target.stem for target in self.targets]
        if len(stems) != len(set(stems)):
            raise ValueError("a stem may appear only once in targets")
        if self.status == PlanStatus.READY and not self.targets:
            raise ValueError("ready plans require at least one target")
        if self.status == PlanStatus.UNSUPPORTED and not self.unsupported_reasons:
            raise ValueError("unsupported plans require reasons")
        return self


class AudioInfo(BaseModel):
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_sec: float


class MusicProject(BaseModel):
    schema_version: str = "1.0"
    project_id: str = Field(default_factory=lambda: f"project_{uuid4().hex}")
    stems: dict[Stem, str]
    audio: AudioInfo
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_four_stems(self) -> "MusicProject":
        if set(self.stems) != set(Stem):
            missing = sorted(stem.value for stem in set(Stem) - set(self.stems))
            raise ValueError(f"exactly four stems are required; missing={missing}")
        return self

    def stem_path(self, stem: Stem) -> Path:
        return Path(self.stems[stem])


class GeneratedCandidate(BaseModel):
    seed: int
    backend: str
    generated_targets: dict[Stem, str]


class CandidateResult(BaseModel):
    seed: int
    backend: str
    score: float = 0
    rank: int = 0
    edited_stems: dict[Stem, str]
    edited_mix: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class EditResult(BaseModel):
    schema_version: str = "1.0"
    job_id: str
    plan_id: str
    project_id: str
    selected_seed: int
    selected: CandidateResult
    candidates: list[CandidateResult]
    created_at: datetime = Field(default_factory=utc_now)


class JobRecord(BaseModel):
    schema_version: str = "1.0"
    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex}")
    plan_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    result_path: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

