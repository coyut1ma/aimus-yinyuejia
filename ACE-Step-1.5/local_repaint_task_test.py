"""Run one frozen MUSDB18 local-edit task through ACE-Step repaint.

This is an integration smoke test for pilot_v1_01_1.  ACE-Step repaint works
on a single source waveform, so the drum stem is edited in a short context
window and then spliced back into the original four-stem mix.  The other
stems are copied byte-for-byte from the imported MUSDB18 cache.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = PROJECT_ROOT.parent / "music_edit_demo"
DEFAULT_STEM_DIR = (
    DEMO_ROOT
    / "runtime_real"
    / "musdb_cache"
    / "musdb18_test_al-james-schoolboy-facination-99ffe393"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "local_repaint_task"


def _configure_environment() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    paths = {
        "ACESTEP_CHECKPOINTS_DIR": PROJECT_ROOT / "checkpoints",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "MODELSCOPE_CACHE": cache_root / "modelscope",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("PYTORCH_DEVICE", "xpu")
    os.environ.setdefault("TORCH_COMPILE_BACKEND", "eager")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem-dir", type=Path, default=DEFAULT_STEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def _write_context_clip(source: Path, destination: Path, clip_start: float, clip_end: float) -> tuple[int, int]:
    audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    start = round(clip_start * sample_rate)
    end = round(clip_end * sample_rate)
    if start < 0 or end > len(audio) or end <= start:
        raise ValueError(f"invalid clip bounds for {source}: {clip_start}-{clip_end}s")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio[start:end], sample_rate, subtype="PCM_16")
    return sample_rate, end - start


def _splice_generated_region(
    original_path: Path,
    generated_path: Path,
    destination: Path,
    original_region: tuple[float, float],
    context_region: tuple[float, float],
    edit_region_in_context: tuple[float, float],
) -> dict[str, float | int]:
    original, original_sr = sf.read(original_path, always_2d=True, dtype="float32")
    generated, generated_sr = sf.read(generated_path, always_2d=True, dtype="float32")
    if generated_sr != original_sr:
        generated = resample_poly(generated, original_sr, generated_sr, axis=0).astype(np.float32)

    # ACE-Step returns the context clip duration. Select the edited interval
    # relative to that clip, then place it at the frozen task's absolute time.
    src_start = round(edit_region_in_context[0] * original_sr)
    src_end = round(edit_region_in_context[1] * original_sr)
    dst_start = round(original_region[0] * original_sr)
    dst_end = round(original_region[1] * original_sr)
    generated_region = generated[src_start:src_end]
    target_len = dst_end - dst_start
    if len(generated_region) != target_len:
        generated_region = resample_poly(generated_region, target_len, max(1, len(generated_region)), axis=0)
        generated_region = generated_region[:target_len]
        if len(generated_region) < target_len:
            generated_region = np.pad(generated_region, ((0, target_len - len(generated_region)), (0, 0)))

    output = original.copy()
    fade = min(round(0.2 * original_sr), target_len // 2)
    alpha = np.ones(target_len, dtype=np.float32)
    if fade:
        ramp = np.sin(np.arange(fade, dtype=np.float32) / fade * np.pi / 2) ** 2
        alpha[:fade] = ramp
        alpha[-fade:] = ramp[::-1]
    output[dst_start:dst_end] = (
        original[dst_start:dst_end] * (1.0 - alpha[:, None])
        + generated_region * alpha[:, None]
    ).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(output, -1.0, 1.0), original_sr, subtype="PCM_16")

    def rms_db(x: np.ndarray) -> float:
        value = float(np.sqrt(np.mean(np.square(x))))
        return -120.0 if value <= 1e-12 else 20.0 * np.log10(value)

    return {
        "sample_rate": original_sr,
        "duration_sec": len(output) / original_sr,
        "original_region_rms_db": rms_db(original[dst_start:dst_end]),
        "generated_region_rms_db": rms_db(generated_region),
        "rms_delta_db": rms_db(generated_region) - rms_db(original[dst_start:dst_end]),
        "context_start_sec": context_region[0],
        "context_end_sec": context_region[1],
    }


def _mix_stems(stem_paths: dict[str, Path], destination: Path) -> dict[str, float | int]:
    tracks: list[np.ndarray] = []
    sample_rate: int | None = None
    for path in stem_paths.values():
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        if sr != sample_rate or not tracks or len(audio) == len(tracks[0]):
            tracks.append(audio)
        else:
            raise ValueError("stem durations/sample rates do not match")
    assert sample_rate is not None
    mixed = np.sum(np.stack(tracks), axis=0)
    clipped = int(np.count_nonzero(np.abs(mixed) > 1.0))
    sf.write(destination, np.clip(mixed, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return {"sample_rate": sample_rate, "duration_sec": len(mixed) / sample_rate, "clipped_samples": clipped}


def main() -> int:
    _configure_environment()
    args = _parse_args()
    import torch
    from loguru import logger
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.xpu.is_available():
        logger.error("Intel XPU is not available; refusing to run a mislabeled CPU edit test.")
        return 2
    stem_dir = args.stem_dir.resolve()
    source_drum = stem_dir / "drums.wav"
    if not source_drum.is_file():
        logger.error(f"Missing imported MUSDB18 drum stem: {source_drum}")
        return 3

    # pilot_v1_01_1: absolute edit [20, 28] seconds.  Use 4 seconds of
    # left/right context, reducing the model input from the 200s full song.
    context_start, context_end = 16.0, 40.0
    edit_start, edit_end = 20.0, 28.0
    edit_in_context = (edit_start - context_start, edit_end - context_start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_clip = args.output_dir / "input_context_drums.wav"
    _write_context_clip(source_drum, context_clip, context_start, context_end)
    logger.info(f"XPU device: {torch.xpu.get_device_name(0)}")
    logger.info("Task pilot_v1_01_1: densify drums in [20, 28] seconds")

    handler = AceStepHandler()
    init_started = time.perf_counter()
    _, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path="acestep-v15-turbo",
        device="xpu",
        compile_model=False,
        offload_to_cpu=True,
        offload_dit_to_cpu=False,
        quantization=None,
        prefer_source="modelscope",
        use_mlx_dit=False,
    )
    if not success:
        logger.error("ACE-Step service initialization failed")
        return 4
    logger.info(f"Model initialized in {time.perf_counter() - init_started:.1f}s")

    params = GenerationParams(
        task_type="repaint",
        src_audio=str(context_clip),
        repainting_start=edit_in_context[0],
        repainting_end=edit_in_context[1],
        chunk_mask_mode="explicit",
        caption="A denser, punchier electronic drum groove with extra rhythmic subdivisions, while preserving the original tempo and surrounding musical feel.",
        lyrics="[Instrumental]",
        duration=-1.0,
        inference_steps=8,
        guidance_scale=1.0,
        seed=args.seed,
        thinking=False,
        audio_cover_strength=1.0,
        repaint_mode="balanced",
        repaint_strength=0.5,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[args.seed], audio_format="wav")
    gen_started = time.perf_counter()
    result = generate_music(handler, None, params=params, config=config, save_dir=str(args.output_dir))
    elapsed = time.perf_counter() - gen_started
    if not result.success or not result.audios:
        logger.error(f"ACE-Step repaint failed after {elapsed:.1f}s: {result.status_message}")
        return 5
    generated_path = Path(result.audios[0]["path"]).resolve()
    logger.info(f"Repaint completed in {elapsed:.1f}s: {generated_path}")

    edited_drum = args.output_dir / "edited_drums.wav"
    metrics = _splice_generated_region(
        source_drum,
        generated_path,
        edited_drum,
        original_region=(edit_start, edit_end),
        context_region=(context_start, context_end),
        edit_region_in_context=edit_in_context,
    )
    edited_stems = {"vocals": stem_dir / "vocals.wav", "drums": edited_drum, "bass": stem_dir / "bass.wav", "other": stem_dir / "other.wav"}
    mix_path = args.output_dir / "edited_mix.wav"
    mix_metrics = _mix_stems(edited_stems, mix_path)
    logger.info(f"Edited drum: {edited_drum}")
    logger.info(f"Edited mix: {mix_path}")
    logger.info(f"Edit metrics: {metrics}")
    logger.info(f"Mix metrics: {mix_metrics}")
    print({"generated_context": str(generated_path), "edited_drum": str(edited_drum), "edited_mix": str(mix_path), "metrics": metrics, "mix_metrics": mix_metrics})
    return 0


if __name__ == "__main__":
    sys.exit(main())
