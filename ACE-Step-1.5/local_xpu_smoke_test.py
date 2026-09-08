"""Run a short ACE-Step 1.5 text-to-music smoke test on Intel XPU."""

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "local_smoke_test"


def _configure_environment() -> None:
    """Keep model caches on D: and configure the Intel XPU runtime."""
    cache_paths = {
        "ACESTEP_CHECKPOINTS_DIR": CHECKPOINT_DIR,
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HUGGINGFACE_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "MODELSCOPE_CACHE": CACHE_ROOT / "modelscope",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))

    os.environ.setdefault("PYTORCH_DEVICE", "xpu")
    os.environ.setdefault("TORCH_COMPILE_BACKEND", "eager")
    os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")
    os.environ.setdefault("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")


def _parse_args() -> argparse.Namespace:
    """Parse smoke-test generation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caption",
        default=(
            "A bright electronic instrumental with warm analog synth chords, "
            "a crisp four-on-the-floor beat, melodic bass, and an uplifting finish."
        ),
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Initialize the turbo DiT without an LM and generate one WAV file."""
    _configure_environment()
    args = _parse_args()

    import torch
    from loguru import logger

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.xpu.is_available():
        logger.error("Intel XPU is not available; refusing to label a CPU run as an XPU test.")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"XPU device: {torch.xpu.get_device_name(0)}")
    logger.info("Initializing acestep-v15-turbo with LM disabled...")

    handler = AceStepHandler()
    started = time.perf_counter()
    status, success = handler.initialize_service(
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
        logger.error(status)
        return 3
    logger.info(f"Model initialized in {time.perf_counter() - started:.1f}s")

    params = GenerationParams(
        task_type="text2music",
        thinking=False,
        caption=args.caption,
        lyrics="[Instrumental]",
        duration=max(10.0, args.duration),
        inference_steps=8,
        guidance_scale=1.0,
        seed=args.seed,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")

    started = time.perf_counter()
    result = generate_music(
        handler,
        None,
        params=params,
        config=config,
        save_dir=str(args.output_dir),
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        logger.error(f"Generation failed after {elapsed:.1f}s: {result.status_message}")
        return 4

    logger.info(f"Generation completed in {elapsed:.1f}s")
    for audio in result.audios:
        logger.info(f"Output: {audio.get('path', '(in-memory)')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
