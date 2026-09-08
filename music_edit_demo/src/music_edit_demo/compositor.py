from __future__ import annotations

from pathlib import Path

from .audio import mix_stems, region_rms_db, replace_region
from .models import (
    CandidateResult,
    Direction,
    EditAttribute,
    EditPlan,
    GeneratedCandidate,
    MusicProject,
    Stem,
)


def _candidate_score(plan: EditPlan, metrics: dict[str, object]) -> float:
    score = 0.0
    target_metrics = metrics["targets"]
    assert isinstance(target_metrics, dict)
    for target in plan.targets:
        stem_metrics = target_metrics[target.stem.value]
        delta_db = float(stem_metrics["rms_delta_db"])
        for operation in target.operations:
            if operation.attribute == EditAttribute.ENERGY:
                desired_positive = operation.direction == Direction.INCREASE
                direction_matches = (delta_db > 0) == desired_positive
                magnitude = min(1.0, abs(delta_db) / 6.0)
                score += (2.0 + magnitude) if direction_matches else (-2.0 - magnitude)
    score -= min(3.0, int(metrics["clipped_samples"]) / 1000)
    return round(score, 4)


def render_candidate(
    project: MusicProject,
    plan: EditPlan,
    generated: GeneratedCandidate,
    output_dir: Path,
) -> CandidateResult:
    candidate_dir = output_dir / f"seed_{generated.seed}" / "rendered"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    target_stems = {target.stem for target in plan.targets}
    edited_stems: dict[Stem, str] = {}
    metrics: dict[str, object] = {"targets": {}}

    for stem in Stem:
        source = project.stem_path(stem)
        destination = candidate_dir / f"{stem.value}.wav"
        if stem in target_stems:
            generated_path = Path(generated.generated_targets[stem])
            replace_region(source, generated_path, destination, plan.region)
            original_db = region_rms_db(source, plan.region)
            edited_db = region_rms_db(destination, plan.region)
            metrics["targets"][stem.value] = {
                "original_rms_db": round(original_db, 4),
                "edited_rms_db": round(edited_db, 4),
                "rms_delta_db": round(edited_db - original_db, 4),
            }
        else:
            destination = source
        edited_stems[stem] = str(destination.resolve())

    mix_path, clipped = mix_stems(
        {stem: Path(path) for stem, path in edited_stems.items()},
        candidate_dir / "mix.wav",
    )
    metrics["clipped_samples"] = clipped
    score = _candidate_score(plan, metrics)
    return CandidateResult(
        seed=generated.seed,
        backend=generated.backend,
        score=score,
        edited_stems=edited_stems,
        edited_mix=str(mix_path.resolve()),
        metrics=metrics,
    )


def render_and_rank_candidates(
    project: MusicProject,
    plan: EditPlan,
    generated: list[GeneratedCandidate],
    output_dir: Path,
) -> list[CandidateResult]:
    results = [render_candidate(project, plan, candidate, output_dir) for candidate in generated]
    results.sort(key=lambda item: (-item.score, item.seed))
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results
