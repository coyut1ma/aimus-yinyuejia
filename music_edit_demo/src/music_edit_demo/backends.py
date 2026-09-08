from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import _clamp_pcm16, read_wav, write_wav
from .models import (
    Direction,
    EditAttribute,
    EditPlan,
    GeneratedCandidate,
    MusicProject,
    Stem,
)


class GeneratorBackend(Protocol):
    name: str

    def health(self) -> dict[str, object]: ...

    def generate(
        self,
        project: MusicProject,
        plan: EditPlan,
        output_dir: Path,
    ) -> list[GeneratedCandidate]: ...


class FakeGenerator:
    """Deterministic DSP backend for pipeline tests, never for quality evaluation."""

    name = "fake-dsp-not-a-model"

    def health(self) -> dict[str, object]:
        return {"ok": True, "backend": self.name, "quality_evaluation_allowed": False}

    def generate(
        self,
        project: MusicProject,
        plan: EditPlan,
        output_dir: Path,
    ) -> list[GeneratedCandidate]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[GeneratedCandidate] = []
        edits = {target.stem: target for target in plan.targets}

        for seed in plan.seeds:
            generated_targets: dict[Stem, str] = {}
            seed_dir = output_dir / f"seed_{seed}" / "generated"
            for stem, target in edits.items():
                source = project.stem_path(stem)
                info, samples = read_wav(source)
                transformed = np.asarray(samples, dtype=np.int16).copy().reshape(-1, info.channels)

                energy_factor = 1.0
                density_direction: Direction | None = None
                add_texture = False
                for operation in target.operations:
                    if operation.attribute == EditAttribute.ENERGY:
                        if operation.direction == Direction.INCREASE:
                            energy_factor *= 1.34 + (seed % 7) * 0.035
                        else:
                            energy_factor *= 0.55 + (seed % 7) * 0.018
                    elif operation.attribute == EditAttribute.DENSITY:
                        density_direction = operation.direction
                    elif operation.attribute in {EditAttribute.STYLE, EditAttribute.REGENERATE}:
                        add_texture = True

                start_frame = round(plan.region.start_sec * info.sample_rate)
                end_frame = round(plan.region.end_sec * info.sample_rate)
                region = transformed[start_frame:end_frame].astype(np.float32)
                region *= energy_factor
                region_frames = np.arange(start_frame, end_frame, dtype=np.float32)
                texture_frequency = 70 + (seed % 7) * 13 + list(Stem).index(stem) * 31
                if density_direction == Direction.INCREASE:
                    pulse_period = max(1, info.sample_rate // (6 + seed % 5))
                    pulse_width = max(1, info.sample_rate // (105 + seed % 4 * 15))
                    mask = (region_frames.astype(np.int64) % pulse_period) < pulse_width
                    pulse_amplitude = 540 + (seed % 11) * 45
                    pulse_frequency = 150 + (seed % 7) * 17
                    pulse = pulse_amplitude * np.sin(
                        2 * np.pi * pulse_frequency * region_frames / info.sample_rate
                    )
                    region[mask] += pulse[mask, None]
                elif density_direction == Direction.DECREASE:
                    pulse_period = max(1, info.sample_rate // (5 + seed % 4))
                    mask = ((region_frames.astype(np.int64) // pulse_period) % 2) == 1
                    region[mask] *= 0.28 + (seed % 5) * 0.04
                if add_texture:
                    texture = 520 * np.sin(2 * np.pi * texture_frequency * region_frames / info.sample_rate)
                    region = region * 0.82 + texture[:, None]
                transformed[start_frame:end_frame] = np.clip(
                    np.rint(region), -32768, 32767
                ).astype(np.int16)

                destination = seed_dir / f"{stem.value}.wav"
                write_wav(destination, info, transformed.reshape(-1))
                generated_targets[stem] = str(destination.resolve())

            results.append(
                GeneratedCandidate(
                    seed=seed,
                    backend=self.name,
                    generated_targets=generated_targets,
                )
            )
        return results


class AceStepHttpGenerator:
    """Adapter for a shared-filesystem ACE-Step repaint service.

    The remote service must perform one joint repaint per seed and return
    extracted target stems with the same format and duration as the inputs.
    """

    name = "ace-step-http"

    def __init__(self, base_url: str, timeout_sec: int = 600):
        if not base_url:
            raise ValueError("ACE_STEP_BASE_URL is required for ace_step_http")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def health(self) -> dict[str, object]:
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "backend": self.name, "error": str(exc)}

    def generate(
        self,
        project: MusicProject,
        plan: EditPlan,
        output_dir: Path,
    ) -> list[GeneratedCandidate]:
        payload = {
            "project_id": project.project_id,
            "stems": {stem.value: path for stem, path in project.stems.items()},
            "region": plan.region.model_dump(mode="json"),
            "targets": [target.model_dump(mode="json") for target in plan.targets],
            "prompt": plan.generator_prompt,
            "preserve": plan.preserve,
            "seeds": plan.seeds,
            "output_dir": str(output_dir.resolve()),
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/repaint",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ACE-Step request failed: {exc}") from exc

        candidates = [GeneratedCandidate.model_validate(item) for item in body.get("candidates", [])]
        expected_targets = {target.stem for target in plan.targets}
        if len(candidates) != len(plan.seeds):
            raise RuntimeError("ACE-Step returned an unexpected number of candidates")
        for candidate in candidates:
            if set(candidate.generated_targets) != expected_targets:
                raise RuntimeError(f"candidate {candidate.seed} is missing requested target stems")
            for path in candidate.generated_targets.values():
                if not Path(path).is_file():
                    raise RuntimeError(f"ACE-Step output is not accessible: {path}")
        return candidates
