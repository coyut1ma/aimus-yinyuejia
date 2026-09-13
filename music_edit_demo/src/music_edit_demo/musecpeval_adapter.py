from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .audio import mix_stems
from .models import CandidateResult, EditPlan, EditResult, MusicProject, Stem


DEFAULT_MUSECPEVAL_METRICS = ("harmony", "rhythm", "melody", "timbre")
ALL_MUSECPEVAL_METRICS = (*DEFAULT_MUSECPEVAL_METRICS, "structure")


def _select_candidates(
    result: EditResult,
    seed: int | None,
    all_candidates: bool,
) -> list[CandidateResult]:
    if all_candidates and seed is not None:
        raise ValueError("--all and --seed cannot be used together")
    if all_candidates:
        return list(result.candidates)
    if seed is None:
        return [result.selected]
    candidates = [candidate for candidate in result.candidates if candidate.seed == seed]
    if not candidates:
        raise KeyError(f"candidate seed not found: {seed}")
    return candidates


def _reference_mix(project: MusicProject, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mix_path, _ = mix_stems({stem: project.stem_path(stem) for stem in Stem}, destination)
    return mix_path.resolve()


def build_musecpeval_manifest(
    project: MusicProject,
    plan: EditPlan,
    result: EditResult,
    artifact_dir: Path,
    seed: int | None = None,
    all_candidates: bool = False,
    include_target_stems: bool = False,
    output: Path | None = None,
) -> list[dict[str, object]]:
    """Create MuseCPEval ref/est pairs for a completed edit job."""

    candidates = _select_candidates(result, seed, all_candidates)
    muse_dir = artifact_dir / "musecpeval"
    ref_mix = _reference_mix(project, muse_dir / "reference_mix.wav")
    target_stems = [target.stem for target in plan.targets]
    target_stem_names = ",".join(stem.value for stem in target_stems)
    pairs: list[dict[str, object]] = []

    for candidate in candidates:
        common = {
            "job_id": result.job_id,
            "plan_id": result.plan_id,
            "project_id": result.project_id,
            "seed": candidate.seed,
            "rank": candidate.rank,
            "target_stems": target_stem_names,
            "edit_start_sec": plan.region.start_sec,
            "edit_end_sec": plan.region.end_sec,
        }
        pairs.append(
            {
                "id": f"{result.job_id}/seed_{candidate.seed}/mix",
                "ref": str(ref_mix),
                "est": candidate.edited_mix,
                "kind": "mix",
                **common,
            }
        )
        if include_target_stems:
            for stem in target_stems:
                pairs.append(
                    {
                        "id": f"{result.job_id}/seed_{candidate.seed}/stem/{stem.value}",
                        "ref": str(project.stem_path(stem).resolve()),
                        "est": candidate.edited_stems[stem],
                        "kind": "target_stem",
                        "stem": stem.value,
                        **common,
                    }
                )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pairs


def default_musecpeval_tool_path() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "MuseCPEval"


def run_musecpeval_batch(
    manifest: Path,
    output_dir: Path,
    metrics: Sequence[str] = DEFAULT_MUSECPEVAL_METRICS,
    tool_path: Path | None = None,
    no_parallel: bool = False,
    limit: int | None = None,
) -> int:
    """Run the downloaded MuseCPEval CLI against a batch manifest."""

    selected_metrics = list(metrics)
    unknown = sorted(set(selected_metrics) - set(ALL_MUSECPEVAL_METRICS))
    if unknown:
        raise ValueError(f"unknown MuseCPEval metrics: {unknown}")

    command = [
        sys.executable,
        "-m",
        "musecpeval",
        "--batch-json",
        str(manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--metrics",
        *selected_metrics,
    ]
    if no_parallel:
        command.append("--no-parallel")
    if limit is not None:
        command.extend(["--limit", str(limit)])

    env = os.environ.copy()
    source_path = tool_path or default_musecpeval_tool_path()
    if source_path.is_dir():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(source_path.resolve()) if not existing else f"{source_path.resolve()}{os.pathsep}{existing}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=env, check=False).returncode
