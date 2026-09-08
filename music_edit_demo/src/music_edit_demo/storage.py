from __future__ import annotations

import threading
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import EditPlan, EditResult, JobRecord, MusicProject


ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonRepository:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._lock = threading.RLock()
        for folder in ("projects", "plans", "jobs", "results", "artifacts"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def artifacts_dir(self, job_id: str) -> Path:
        path = self.root / "artifacts" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save(self, folder: str, identifier: str, value: BaseModel) -> Path:
        destination = self.root / folder / f"{identifier}.json"
        temporary = destination.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(destination)
        return destination

    def _load(self, folder: str, identifier: str, model: type[ModelT]) -> ModelT:
        path = self.root / folder / f"{identifier}.json"
        if not path.is_file():
            raise KeyError(f"{folder[:-1]} not found: {identifier}")
        with self._lock:
            return model.model_validate_json(path.read_text(encoding="utf-8"))

    def save_project(self, project: MusicProject) -> Path:
        return self._save("projects", project.project_id, project)

    def get_project(self, project_id: str) -> MusicProject:
        return self._load("projects", project_id, MusicProject)

    def save_plan(self, plan: EditPlan) -> Path:
        return self._save("plans", plan.plan_id, plan)

    def get_plan(self, plan_id: str) -> EditPlan:
        return self._load("plans", plan_id, EditPlan)

    def save_job(self, job: JobRecord) -> Path:
        return self._save("jobs", job.job_id, job)

    def get_job(self, job_id: str) -> JobRecord:
        return self._load("jobs", job_id, JobRecord)

    def save_result(self, result: EditResult) -> Path:
        return self._save("results", result.job_id, result)

    def get_result(self, job_id: str) -> EditResult:
        return self._load("results", job_id, EditResult)

