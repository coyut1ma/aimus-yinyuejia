from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import Settings
from .models import EditPlan, EditResult, JobRecord, MusicProject, Stem
from .musdb import MusdbTrack
from .pilot_tasks import PilotTask
from .service import DemoService


class RegisterProjectRequest(BaseModel):
    project_id: str | None = None
    stems: dict[Stem, str]


class CreatePlanRequest(BaseModel):
    project_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    instruction: str = Field(min_length=1)


class CreateJobRequest(BaseModel):
    plan_id: str


class ImportMusdbRequest(BaseModel):
    track_id: str
    force: bool = False


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def create_app(settings: Settings | None = None) -> FastAPI:
    service = DemoService(settings or Settings.from_env())
    app = FastAPI(
        title="Music Edit Demo API",
        version="0.1.0",
        description="Natural-language local stem editing orchestration",
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.post("/v1/projects", response_model=MusicProject)
    def register_project(request: RegisterProjectRequest) -> MusicProject:
        try:
            return service.register_project(request.stems, request.project_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/projects/{project_id}", response_model=MusicProject)
    def get_project(project_id: str) -> MusicProject:
        try:
            return service.get_project(project_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/musdb/tracks", response_model=list[MusdbTrack])
    def list_musdb_tracks(split: str | None = None) -> list[MusdbTrack]:
        try:
            return service.list_musdb_tracks(split)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/musdb/import", response_model=MusicProject)
    def import_musdb_track(request: ImportMusdbRequest) -> MusicProject:
        try:
            return service.import_musdb_track(request.track_id, request.force)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/pilot/build", response_model=list[PilotTask])
    def build_pilot() -> list[PilotTask]:
        try:
            return service.build_pilot_manifest()
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/edit-plans", response_model=EditPlan)
    def create_plan(request: CreatePlanRequest) -> EditPlan:
        try:
            return service.create_plan(
                request.project_id,
                request.start_sec,
                request.end_sec,
                request.instruction,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/edit-jobs", response_model=JobRecord, status_code=202)
    def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks) -> JobRecord:
        try:
            job = service.submit_job(request.plan_id)
            background_tasks.add_task(service.run_job, job.job_id)
            return job
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/edit-jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return service.get_job(job_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/edit-jobs/{job_id}/result", response_model=EditResult)
    def get_result(job_id: str) -> EditResult:
        try:
            return service.get_result(job_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/edit-jobs/{job_id}/evaluation")
    def evaluate_job(job_id: str, seed: int | None = None, all_candidates: bool = False) -> dict[str, object] | list[dict[str, object]]:
        """Evaluate beat timing, cross-stem onset alignment, and level deviation."""
        try:
            result = service.evaluate_job(job_id, seed=seed, all_candidates=all_candidates)
            return result if isinstance(result, list) else result
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/edit-jobs/{job_id}/audio/{seed}/mix")
    def download_mix(job_id: str, seed: int) -> FileResponse:
        try:
            path = service.get_result_file(job_id, seed, "mix")
            return FileResponse(path, media_type="audio/wav", filename=f"{job_id}-{seed}-mix.wav")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/edit-jobs/{job_id}/audio/{seed}/stems/{stem}")
    def download_stem(job_id: str, seed: int, stem: Stem) -> FileResponse:
        try:
            path = service.get_result_file(job_id, seed, "stem", stem)
            return FileResponse(path, media_type="audio/wav", filename=f"{job_id}-{seed}-{stem.value}.wav")
        except Exception as exc:
            raise _http_error(exc) from exc

    return app


app = create_app()
