from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .audio import create_sine_wav
from .config import Settings
from .evaluation import EvaluationConfig
from .models import Stem
from .pilot_tasks import read_pilot_tasks
from .service import DemoService


def _print_model(value: object) -> None:
    if hasattr(value, "model_dump_json"):
        print(value.model_dump_json(indent=2))
    elif isinstance(value, list):
        print(
            json.dumps(
                [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def _service() -> DemoService:
    return DemoService(Settings.from_env())


def _cmd_synthetic(args: argparse.Namespace) -> int:
    service = _service()
    source_dir = service.settings.runtime_dir / "synthetic" / args.project_id
    frequencies = {
        Stem.VOCALS: 220.0,
        Stem.DRUMS: 110.0,
        Stem.BASS: 55.0,
        Stem.OTHER: 330.0,
    }
    stems: dict[Stem, str] = {}
    for stem, frequency in frequencies.items():
        path = create_sine_wav(source_dir / f"{stem.value}.wav", args.duration, frequency)
        stems[stem] = str(path.resolve())
    project = service.register_project(stems, args.project_id)
    _print_model(project)
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    stems = {
        Stem.VOCALS: args.vocals,
        Stem.DRUMS: args.drums,
        Stem.BASS: args.bass,
        Stem.OTHER: args.other,
    }
    project = _service().register_project(stems, args.project_id)
    _print_model(project)
    return 0


def _cmd_musdb_list(args: argparse.Namespace) -> int:
    tracks = _service().list_musdb_tracks(args.split)
    if args.limit:
        tracks = tracks[: args.limit]
    _print_model(tracks)
    return 0


def _cmd_musdb_import(args: argparse.Namespace) -> int:
    project = _service().import_musdb_track(args.track_id, args.force)
    _print_model(project)
    return 0


def _cmd_pilot_build(args: argparse.Namespace) -> int:
    service = _service()
    destination = Path(args.output).resolve() if args.output else None
    tasks = service.build_pilot_manifest(destination)
    _print_model(tasks)
    return 0


def _cmd_pilot_run(args: argparse.Namespace) -> int:
    service = _service()
    manifest = Path(args.manifest).resolve()
    if args.list:
        _print_model(read_pilot_tasks(manifest))
        return 0
    if not args.task_id:
        raise ValueError("--task-id is required unless --list is used")
    job = service.run_pilot_task(manifest, args.task_id)
    _print_model(job)
    if job.status.value == "succeeded":
        _print_model(service.get_result(job.job_id))
        return 0
    return 1


def _cmd_demo_real(args: argparse.Namespace) -> int:
    service = _service()
    manifest = service.settings.runtime_dir / "pilot_tasks_v1.jsonl"
    service.build_pilot_manifest(manifest)
    job = service.run_pilot_task(manifest, args.task_id)
    _print_model({"manifest": str(manifest.resolve()), "task_id": args.task_id})
    _print_model(job)
    if job.status.value == "succeeded":
        _print_model(service.get_result(job.job_id))
        return 0
    return 1


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = _service().create_plan(
        args.project_id,
        args.start,
        args.end,
        args.instruction,
    )
    _print_model(plan)
    return 0 if plan.status.value == "ready" else 2


def _cmd_run(args: argparse.Namespace) -> int:
    service = _service()
    job = service.submit_job(args.plan_id)
    job = service.run_job(job.job_id)
    _print_model(job)
    if job.status.value == "succeeded":
        _print_model(service.get_result(job.job_id))
        return 0
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    _print_model(_service().get_job(args.job_id))
    return 0


def _cmd_result(args: argparse.Namespace) -> int:
    _print_model(_service().get_result(args.job_id))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    service = _service()
    output = Path(args.output).resolve() if args.output else None
    report = service.evaluate_job(args.job_id, args.seed, args.all_candidates, output, EvaluationConfig())
    _print_model(report)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API dependencies with: pip install -e '.[api]'") from exc
    uvicorn.run(
        "music_edit_demo.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-edit-demo")
    parser.add_argument(
        "--runtime-dir",
        help="Override MUSIC_EDIT_RUNTIME_DIR for this command",
    )
    parser.add_argument(
        "--musdb-root",
        help="Override MUSDB_ROOT for this command",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Create a four-stem synthetic smoke project")
    synthetic.add_argument("--project-id", default="synthetic-demo")
    synthetic.add_argument("--duration", type=float, default=12.0)
    synthetic.set_defaults(handler=_cmd_synthetic)

    register = subparsers.add_parser("register", help="Register four PCM WAV stems")
    register.add_argument("--project-id")
    register.add_argument("--vocals", required=True)
    register.add_argument("--drums", required=True)
    register.add_argument("--bass", required=True)
    register.add_argument("--other", required=True)
    register.set_defaults(handler=_cmd_register)

    musdb_list = subparsers.add_parser("musdb-list", help="List tracks in a MUSDB18 root")
    musdb_list.add_argument("--split", choices=("train", "test"))
    musdb_list.add_argument("--limit", type=int)
    musdb_list.set_defaults(handler=_cmd_musdb_list)

    musdb_import = subparsers.add_parser("musdb-import", help="Decode and register a MUSDB18 track")
    musdb_import.add_argument("--track-id", required=True)
    musdb_import.add_argument("--force", action="store_true")
    musdb_import.set_defaults(handler=_cmd_musdb_import)

    pilot_build = subparsers.add_parser("pilot-build", help="Build the bound 20-task pilot manifest")
    pilot_build.add_argument("--output")
    pilot_build.set_defaults(handler=_cmd_pilot_build)

    pilot_run = subparsers.add_parser("pilot-run", help="Run or list tasks from a pilot manifest")
    pilot_run.add_argument("--manifest", required=True)
    pilot_run.add_argument("--task-id")
    pilot_run.add_argument("--list", action="store_true")
    pilot_run.set_defaults(handler=_cmd_pilot_run)

    demo_real = subparsers.add_parser(
        "demo-real",
        help="Build the pilot manifest and run one real MUSDB18 task",
    )
    demo_real.add_argument("--task-id", default="pilot_v1_01_1")
    demo_real.set_defaults(handler=_cmd_demo_real)

    plan = subparsers.add_parser("plan", help="Parse and validate an edit instruction")
    plan.add_argument("--project-id", required=True)
    plan.add_argument("--start", type=float, required=True)
    plan.add_argument("--end", type=float, required=True)
    plan.add_argument("--instruction", required=True)
    plan.set_defaults(handler=_cmd_plan)

    run = subparsers.add_parser("run", help="Run a confirmed edit plan synchronously")
    run.add_argument("--plan-id", required=True)
    run.set_defaults(handler=_cmd_run)

    status = subparsers.add_parser("status", help="Read job status")
    status.add_argument("--job-id", required=True)
    status.set_defaults(handler=_cmd_status)

    result = subparsers.add_parser("result", help="Read a completed result manifest")
    result.add_argument("--job-id", required=True)
    result.set_defaults(handler=_cmd_result)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate beat error, cross-stem onset alignment, and volume deviation",
    )
    evaluate.add_argument("--job-id", required=True)
    evaluate.add_argument("--seed", type=int, help="Candidate seed; defaults to the selected candidate")
    evaluate.add_argument("--all", dest="all_candidates", action="store_true", help="Evaluate every candidate")
    evaluate.add_argument("--output", help="Write the JSON report to this path")
    evaluate.set_defaults(handler=_cmd_evaluate)

    serve = subparsers.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runtime_dir:
        os.environ["MUSIC_EDIT_RUNTIME_DIR"] = str(Path(args.runtime_dir).resolve())
    if args.musdb_root:
        os.environ["MUSDB_ROOT"] = str(Path(args.musdb_root).resolve())
    try:
        return int(args.handler(args))
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
