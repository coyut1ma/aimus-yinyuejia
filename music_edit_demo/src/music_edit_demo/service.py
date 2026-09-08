from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .audio import validate_stems
from .backends import AceStepHttpGenerator, FakeGenerator, GeneratorBackend
from .compositor import render_and_rank_candidates
from .config import Settings
from .evaluation import EvaluationConfig, evaluate_candidate, save_evaluation
from .models import (
    EditPlan,
    EditRegion,
    EditResult,
    JobRecord,
    JobStatus,
    MusicProject,
    PlanStatus,
    Stem,
)
from .musdb import MusdbTrack, import_track, inspect_container, scan_musdb
from .parser import ControlledInstructionParser, InstructionParser, QwenInstructionParser
from .pilot_tasks import (
    PilotTask,
    bind_pilot_tasks,
    build_pilot_tasks,
    read_pilot_tasks,
    write_pilot_tasks,
)
from .storage import JsonRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_parser(settings: Settings) -> InstructionParser:
    if settings.parser_backend == "controlled":
        return ControlledInstructionParser()
    if settings.parser_backend == "qwen":
        return QwenInstructionParser(
            base_url=settings.qwen_base_url,
            api_key=settings.qwen_api_key,
            model=settings.qwen_model,
        )
    raise ValueError(f"unknown parser backend: {settings.parser_backend}")


def build_generator(settings: Settings) -> GeneratorBackend:
    if settings.generator_backend == "fake":
        return FakeGenerator()
    if settings.generator_backend == "ace_step_http":
        return AceStepHttpGenerator(settings.ace_step_base_url)
    raise ValueError(f"unknown generator backend: {settings.generator_backend}")


class DemoService:
    def __init__(
        self,
        settings: Settings,
        parser: InstructionParser | None = None,
        generator: GeneratorBackend | None = None,
    ):
        self.settings = settings
        self.repository = JsonRepository(settings.runtime_dir)
        self.parser = parser or build_parser(settings)
        self.generator = generator or build_generator(settings)

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "parser": self.parser.name,
            "generator": self.generator.health(),
            "runtime_dir": str(self.settings.runtime_dir),
            "musdb_root": str(self.settings.musdb_root) if self.settings.musdb_root else None,
        }

    def list_musdb_tracks(self, split: str | None = None) -> list[MusdbTrack]:
        if self.settings.musdb_root is None:
            raise ValueError("MUSDB_ROOT is not configured")
        splits = (split,) if split else ("train", "test")
        return scan_musdb(self.settings.musdb_root, splits)

    def import_musdb_track(self, track_id: str, force: bool = False) -> MusicProject:
        tracks = self.list_musdb_tracks()
        try:
            track = next(item for item in tracks if item.track_id == track_id)
        except StopIteration as exc:
            raise KeyError(f"MUSDB18 track not found: {track_id}") from exc
        project = import_track(track, self.settings.runtime_dir / "musdb_cache", force=force)
        self.repository.save_project(project)
        return project

    def build_pilot_manifest(self, destination: Path | None = None) -> list[PilotTask]:
        # The frozen task layout needs up to 92 seconds plus two seconds of
        # right-side context. Select deterministically from tracks that fit.
        tracks = sorted(
            (track for track in self.list_musdb_tracks("test") if track.duration_sec >= 94),
            key=lambda item: item.name.lower(),
        )[:10]
        if len(tracks) != 10:
            raise ValueError("the MUSDB18 test split must contain at least 10 tracks of 94 seconds or longer")
        tasks = build_pilot_tasks(track.track_id for track in tracks)
        bound = bind_pilot_tasks(tasks, {track.track_id: track.container_path for track in tracks})
        durations = {track.track_id: track.duration_sec for track in tracks}
        for task in bound:
            if task.edit_region.end_sec + 2 > durations[task.song_id]:
                raise ValueError(f"pilot region exceeds track duration: {task.task_id}")
        write_pilot_tasks(
            bound,
            destination or (self.settings.runtime_dir / "pilot_tasks_v1.jsonl"),
        )
        return bound

    def run_pilot_task(self, manifest: Path, task_id: str) -> JobRecord:
        try:
            task = next(item for item in read_pilot_tasks(manifest) if item.task_id == task_id)
        except StopIteration as exc:
            raise KeyError(f"pilot task not found: {task_id}") from exc
        if not task.input_container:
            raise ValueError("pilot task is not bound to a MUSDB18 container")
        track = inspect_container(Path(task.input_container))
        if track.track_id != task.song_id:
            raise ValueError("pilot task song_id does not match its input container")
        project = import_track(track, self.settings.runtime_dir / "musdb_cache")
        self.repository.save_project(project)
        plan = self.create_plan(
            project.project_id,
            task.edit_region.start_sec,
            task.edit_region.end_sec,
            task.instruction,
        )
        parsed_targets = {target.stem for target in plan.targets}
        if parsed_targets != set(task.target_stems):
            raise ValueError(
                f"parser target mismatch for {task.task_id}: "
                f"expected={sorted(stem.value for stem in task.target_stems)}, "
                f"actual={sorted(stem.value for stem in parsed_targets)}"
            )
        plan.seeds = list(task.seeds)
        plan.preserve = list(task.must_preserve)
        self.repository.save_plan(plan)
        job = self.submit_job(plan.plan_id)
        return self.run_job(job.job_id)

    def register_project(self, stems: dict[Stem, str], project_id: str | None = None) -> MusicProject:
        resolved = {stem: Path(path).resolve() for stem, path in stems.items()}
        info = validate_stems(resolved)
        project = MusicProject(
            **({"project_id": project_id} if project_id else {}),
            stems={stem: str(path) for stem, path in resolved.items()},
            audio=info,
        )
        self.repository.save_project(project)
        return project

    def get_project(self, project_id: str) -> MusicProject:
        return self.repository.get_project(project_id)

    def create_plan(
        self,
        project_id: str,
        start_sec: float,
        end_sec: float,
        instruction: str,
    ) -> EditPlan:
        project = self.repository.get_project(project_id)
        region = EditRegion(start_sec=start_sec, end_sec=end_sec)
        if region.end_sec > project.audio.duration_sec:
            raise ValueError("edit region exceeds project duration")
        if region.start_sec < 2 or project.audio.duration_sec - region.end_sec < 2:
            raise ValueError("the MVP requires at least 2 seconds of context on both sides")
        plan = self.parser.parse(project_id, region, instruction)
        self.repository.save_plan(plan)
        return plan

    def submit_job(self, plan_id: str) -> JobRecord:
        plan = self.repository.get_plan(plan_id)
        if plan.status != PlanStatus.READY:
            raise ValueError(f"plan is not runnable: {plan.status.value}")
        job = JobRecord(plan_id=plan_id)
        self.repository.save_job(job)
        return job

    def run_job(self, job_id: str) -> JobRecord:
        job = self.repository.get_job(job_id)
        plan = self.repository.get_plan(job.plan_id)
        project = self.repository.get_project(plan.project_id)
        artifacts = self.repository.artifacts_dir(job_id)
        job.status = JobStatus.RUNNING
        job.progress = 10
        job.updated_at = _utc_now()
        self.repository.save_job(job)

        try:
            generated = self.generator.generate(project, plan, artifacts / "backend")
            job.progress = 70
            job.updated_at = _utc_now()
            self.repository.save_job(job)
            candidates = render_and_rank_candidates(project, plan, generated, artifacts / "candidates")
            if not candidates:
                raise RuntimeError("generator returned no candidates")
            result = EditResult(
                job_id=job.job_id,
                plan_id=plan.plan_id,
                project_id=project.project_id,
                selected_seed=candidates[0].seed,
                selected=candidates[0],
                candidates=candidates,
            )
            result_path = self.repository.save_result(result)
            job.status = JobStatus.SUCCEEDED
            job.progress = 100
            job.result_path = str(result_path.resolve())
            job.error = None
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
        job.updated_at = _utc_now()
        self.repository.save_job(job)
        return job

    def get_job(self, job_id: str) -> JobRecord:
        return self.repository.get_job(job_id)

    def get_result(self, job_id: str) -> EditResult:
        job = self.repository.get_job(job_id)
        if job.status != JobStatus.SUCCEEDED:
            raise ValueError(f"job has no result: {job.status.value}")
        return self.repository.get_result(job_id)

    def evaluate_job(
        self,
        job_id: str,
        seed: int | None = None,
        all_candidates: bool = False,
        output: Path | None = None,
        config: EvaluationConfig | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        """Evaluate a candidate from a completed edit job.

        The default is the selected candidate. Passing ``seed`` evaluates
        that candidate instead; no candidate ranking or aggregate score is
        calculated by the evaluator.
        """

        job = self.get_job(job_id)
        if job.status != JobStatus.SUCCEEDED:
            raise ValueError(f"job has no result: {job.status.value}")
        result = self.get_result(job_id)
        plan = self.repository.get_plan(result.plan_id)
        project = self.repository.get_project(result.project_id)
        if all_candidates and seed is not None:
            raise ValueError("--all and --seed cannot be used together")
        if all_candidates:
            candidates = list(result.candidates)
        elif seed is None:
            candidates = [result.selected]
        else:
            candidates = [candidate for candidate in result.candidates if candidate.seed == seed]
            if not candidates:
                raise KeyError(f"candidate seed not found: {seed}")
        reports = [evaluate_candidate(project, plan, candidate, config) for candidate in candidates]
        if output is not None:
            if len(reports) == 1:
                save_evaluation(reports[0], output)
            else:
                output = output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return reports[0] if len(reports) == 1 else reports

    def get_result_file(self, job_id: str, seed: int, kind: str, stem: Stem | None = None) -> Path:
        result = self.get_result(job_id)
        try:
            candidate = next(item for item in result.candidates if item.seed == seed)
        except StopIteration as exc:
            raise KeyError(f"candidate seed not found: {seed}") from exc
        if kind == "mix":
            path = Path(candidate.edited_mix)
        elif kind == "stem" and stem is not None:
            path = Path(candidate.edited_stems[stem])
        else:
            raise ValueError("kind must be mix, or stem with a stem name")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
