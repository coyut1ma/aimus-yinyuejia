"""Shared-filesystem repaint endpoint used by ``music_edit_demo``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field


STEM_ORDER = ("vocals", "drums", "bass", "other")
STEM_TO_ID = {name: index for index, name in enumerate(STEM_ORDER)}


class RepaintRegion(BaseModel):
    start_sec: float = 0.0
    end_sec: Optional[float] = None


class RepaintRequest(BaseModel):
    project_id: str = ""
    stems: Dict[str, str]
    region: RepaintRegion
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    target_stems: List[str] = Field(default_factory=list)
    prompt: str = ""
    preserve: List[str] = Field(default_factory=list)
    seeds: List[int] = Field(default_factory=list)
    output_dir: str
    joint_frontend: bool = True
    inference_steps: int = 8
    repaint_mode: str = "balanced"
    repaint_strength: float = 0.5
    repaint_wav_crossfade_sec: float = 0.0


def _extract_target_stems(req: RepaintRequest) -> List[str]:
    targets = list(req.target_stems or [])
    if not targets:
        for target in req.targets:
            stem = target.get("stem")
            if stem:
                targets.append(str(stem))
    if not targets:
        raise HTTPException(status_code=400, detail="At least one target stem is required")
    unknown = sorted(set(targets) - set(STEM_ORDER))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown target stems: {unknown}")
    return targets


def _load_four_stems(handler: Any, stems: Dict[str, str]) -> torch.Tensor:
    missing = [stem for stem in STEM_ORDER if stem not in stems]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing stems: {missing}")

    audios = []
    for stem in STEM_ORDER:
        audio = handler.process_src_audio(stems[stem])
        if audio is None:
            raise HTTPException(status_code=400, detail=f"Invalid or unreadable stem: {stem}")
        audios.append(audio)

    max_samples = max(audio.shape[-1] for audio in audios)
    aligned = [
        F.pad(audio, (0, max_samples - audio.shape[-1])) if audio.shape[-1] < max_samples else audio
        for audio in audios
    ]
    return torch.stack(aligned, dim=0).unsqueeze(0)


def _write_audio_tensor(path: Path, audio_tensor: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio_tensor.detach().cpu().float()
    if audio.ndim == 2:
        audio = audio.transpose(0, 1)
    sf.write(path, audio.numpy(), sample_rate)


def register_repaint_route(
    app: FastAPI,
    *,
    verify_api_key: Callable[..., Any],
) -> None:
    """Register synchronous ``/v1/repaint`` bridge for multitrack demo jobs."""

    @app.post("/v1/repaint")
    async def repaint_stems(req: RepaintRequest, _: None = Depends(verify_api_key)):
        handler = getattr(app.state, "handler", None)
        if handler is None or getattr(handler, "model", None) is None:
            raise HTTPException(status_code=503, detail="ACE-Step model is not initialized")
        if req.joint_frontend and not hasattr(handler.model.decoder, "multi_stem_frontend"):
            raise HTTPException(status_code=400, detail="Loaded model does not support joint_frontend")

        target_stems = _extract_target_stems(req)
        seeds = req.seeds or [-1]
        multi_stem_target_wavs = _load_four_stems(handler, req.stems)
        output_dir = Path(req.output_dir).resolve()
        candidates = []

        for seed in seeds:
            generated_targets: Dict[str, str] = {}
            for stem in target_stems:
                target_path = req.stems[stem]
                result = handler.generate_music(
                    captions=req.prompt,
                    lyrics="",
                    reference_audio=target_path,
                    src_audio=target_path,
                    task_type="repaint",
                    repainting_start=req.region.start_sec,
                    repainting_end=req.region.end_sec,
                    inference_steps=req.inference_steps,
                    guidance_scale=1.0,
                    use_random_seed=False,
                    seed=seed,
                    batch_size=1,
                    joint_frontend=req.joint_frontend,
                    target_stem_id=STEM_TO_ID[stem],
                    multi_stem_target_wavs=multi_stem_target_wavs,
                    repaint_mode=req.repaint_mode,
                    repaint_strength=req.repaint_strength,
                    repaint_wav_crossfade_sec=req.repaint_wav_crossfade_sec,
                )
                if not result.get("success"):
                    raise HTTPException(status_code=500, detail=result.get("error") or result.get("status_message"))
                audios = result.get("audios") or []
                if not audios:
                    raise HTTPException(status_code=500, detail="ACE-Step did not return audio")
                audio_payload = audios[0]
                generated_path = output_dir / f"seed_{seed}" / "generated" / f"{stem}.wav"
                _write_audio_tensor(generated_path, audio_payload["tensor"], audio_payload["sample_rate"])
                generated_targets[stem] = str(generated_path)

            candidates.append(
                {
                    "seed": seed,
                    "backend": "ace-step-joint-frontend" if req.joint_frontend else "ace-step",
                    "generated_targets": generated_targets,
                }
            )

        return {"candidates": candidates}
