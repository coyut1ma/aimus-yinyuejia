"""Run the frozen 20-task pilot through local ACE-Step repaint.

Each task gets its own directory containing the frozen task JSON, source and
context clips, one ACE-Step output per target stem, edited full-length stems,
the reconstructed mix, and metrics.  The script is resumable: a task is
skipped only when its metrics JSON declares status=success and every expected
output file is present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = PROJECT_ROOT.parent / "music_edit_demo"
DEFAULT_MANIFEST = DEMO_ROOT / "runtime_real" / "pilot_tasks_v1.jsonl"
DEFAULT_CACHE = DEMO_ROOT / "runtime_real" / "musdb_cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pilot20_ace_step"
CONTEXT_SEC = 4.0


def configure_environment() -> None:
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


def load_tasks(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem_dir_for_task(task: dict, cache_root: Path) -> Path:
    # The manifest's input_stems are intentionally empty; imported project
    # stems live under the deterministic MUSDB cache directory.
    return cache_root / task["song_id"]


def read_clip(source: Path, destination: Path, start: float, end: float) -> tuple[int, int]:
    audio, sr = sf.read(source, always_2d=True, dtype="float32")
    a, b = round(start * sr), round(end * sr)
    if a < 0 or b > len(audio) or b <= a:
        raise ValueError(f"invalid clip range {start}-{end} for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio[a:b], sr, subtype="PCM_16")
    return sr, b - a


def rms_db(audio: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(audio))))
    return -120.0 if value <= 1e-12 else 20.0 * np.log10(value)


def splice(
    original_path: Path,
    generated_path: Path,
    output_path: Path,
    absolute_region: tuple[float, float],
    context_start: float,
    context_end: float,
    generated_context_start: float,
    generated_context_end: float,
) -> dict[str, float | int]:
    original, sr = sf.read(original_path, always_2d=True, dtype="float32")
    generated, generated_sr = sf.read(generated_path, always_2d=True, dtype="float32")
    if generated_sr != sr:
        generated = resample_poly(generated, sr, generated_sr, axis=0).astype(np.float32)
    src_a = round(generated_context_start * sr)
    src_b = round(generated_context_end * sr)
    dst_a = round(absolute_region[0] * sr)
    dst_b = round(absolute_region[1] * sr)
    generated_region = generated[src_a:src_b]
    target_len = dst_b - dst_a
    if len(generated_region) != target_len:
        generated_region = resample_poly(generated_region, target_len, max(1, len(generated_region)), axis=0)
        generated_region = generated_region[:target_len]
        if len(generated_region) < target_len:
            generated_region = np.pad(generated_region, ((0, target_len - len(generated_region)), (0, 0)))
    original_region = original[dst_a:dst_b]
    alpha = np.ones(target_len, dtype=np.float32)
    fade = min(round(0.2 * sr), target_len // 2)
    if fade:
        ramp = np.sin(np.arange(fade, dtype=np.float32) / fade * np.pi / 2) ** 2
        alpha[:fade], alpha[-fade:] = ramp, ramp[::-1]
    mixed_region = original_region * (1.0 - alpha[:, None]) + generated_region * alpha[:, None]
    output = original.copy()
    output[dst_a:dst_b] = mixed_region
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.clip(output, -1.0, 1.0), sr, subtype="PCM_16")
    outside = np.ones(len(original), dtype=bool)
    outside[dst_a:dst_b] = False
    diff = np.abs(output - original)
    return {
        "sample_rate": sr,
        "duration_sec": len(output) / sr,
        "original_region_rms_db": rms_db(original_region),
        "generated_region_rms_db": rms_db(generated_region),
        "rms_delta_db": rms_db(generated_region) - rms_db(original_region),
        "outside_max_abs_diff": float(diff[outside].max()) if np.any(outside) else 0.0,
        "edited_changed_samples": int(np.count_nonzero(diff[dst_a:dst_b] > 1e-7)),
        "context_start_sec": context_start,
        "context_end_sec": context_end,
    }


def mix_stems(stems: dict[str, Path], destination: Path) -> dict[str, float | int]:
    tracks: list[np.ndarray] = []
    sr: int | None = None
    for path in stems.values():
        audio, current_sr = sf.read(path, always_2d=True, dtype="float32")
        if sr is None:
            sr = current_sr
        if current_sr != sr or (tracks and len(audio) != len(tracks[0])):
            raise ValueError("stem format mismatch during remix")
        tracks.append(audio)
    assert sr is not None
    summed = np.sum(np.stack(tracks), axis=0)
    clipped = int(np.count_nonzero(np.abs(summed) > 1.0))
    sf.write(destination, np.clip(summed, -1.0, 1.0), sr, subtype="PCM_16")
    return {"sample_rate": sr, "duration_sec": len(summed) / sr, "clipped_samples": clipped, "peak": float(np.abs(summed).max())}


def task_complete(task_dir: Path, task: dict) -> bool:
    metrics_path = task_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metrics.get("status") != "success":
        return False
    required = [task_dir / "edited_mix.wav"] + [task_dir / "edited_stems" / f"{stem}.wav" for stem in ("vocals", "drums", "bass", "other")]
    required += [task_dir / "ace_step" / stem / "generated_context.wav" for stem in task["target_stems"]]
    return all(path.is_file() for path in required)


def run_task(task: dict, task_dir: Path, stem_dir: Path, handler, generation_config, logger) -> dict:
    from acestep.inference import GenerationParams, generate_music

    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    start, end = task["edit_region"]["start_sec"], task["edit_region"]["end_sec"]
    context_start, context_end = max(0.0, start - CONTEXT_SEC), end + CONTEXT_SEC
    context_edit_start, context_edit_end = start - context_start, end - context_start
    all_stems = ("vocals", "drums", "bass", "other")
    edited_dir = task_dir / "edited_stems"
    ace_dir = task_dir / "ace_step"
    source_dir = task_dir / "source_stems"
    edited_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    for stem in all_stems:
        source = stem_dir / f"{stem}.wav"
        shutil.copyfile(source, source_dir / f"{stem}.wav")

    # Keep a reference mix next to the edited mix so later listening and
    # objective evaluation can compare identical full-length material.
    source_mix = task_dir / "source_mix.wav"
    if not source_mix.is_file():
        mix_stems(
            {stem: source_dir / f"{stem}.wav" for stem in all_stems},
            source_mix,
        )

    generated_by_stem: dict[str, str] = {}
    stem_metrics: dict[str, dict] = {}
    for stem in task["target_stems"]:
        stem_ace_dir = ace_dir / stem
        stem_ace_dir.mkdir(parents=True, exist_ok=True)
        source = stem_dir / f"{stem}.wav"
        context_clip = stem_ace_dir / "input_context.wav"
        read_clip(source, context_clip, context_start, context_end)
        params = GenerationParams(
            task_type="repaint",
            src_audio=str(context_clip),
            repainting_start=context_edit_start,
            repainting_end=context_edit_end,
            chunk_mask_mode="explicit",
            caption=(
                f"{task['instruction']}. Generate a coherent replacement for the selected {stem} track region, "
                "preserving tempo, key, musical timing, and the surrounding context."
            ),
            lyrics="[Instrumental]",
            duration=-1.0,
            inference_steps=8,
            guidance_scale=1.0,
            seed=int(task["seeds"][0]),
            thinking=False,
            audio_cover_strength=1.0,
            repaint_mode="balanced",
            repaint_strength=0.5,
        )
        logger.info(f"{task['task_id']} target={stem}: ACE-Step repaint {context_start:.1f}-{context_end:.1f}s")
        result = generate_music(handler, None, params=params, config=generation_config, save_dir=str(stem_ace_dir))
        if not result.success or not result.audios:
            raise RuntimeError(f"ACE-Step failed for {task['task_id']} target={stem}: {result.status_message}")
        generated_path = Path(result.audios[0]["path"]).resolve()
        copied_generated = stem_ace_dir / "generated_context.wav"
        shutil.copyfile(generated_path, copied_generated)
        generated_by_stem[stem] = str(copied_generated.resolve())
        output_path = edited_dir / f"{stem}.wav"
        stem_metrics[stem] = splice(
            source,
            copied_generated,
            output_path,
            absolute_region=(start, end),
            context_start=context_start,
            context_end=context_end,
            generated_context_start=context_edit_start,
            generated_context_end=context_edit_end,
        )

    edited_paths: dict[str, Path] = {}
    for stem in all_stems:
        path = edited_dir / f"{stem}.wav"
        if stem not in generated_by_stem:
            shutil.copyfile(stem_dir / f"{stem}.wav", path)
        edited_paths[stem] = path
    mix_metrics = mix_stems(edited_paths, task_dir / "edited_mix.wav")
    # The source context is useful for later listening and exact repro.
    for stem in task["target_stems"]:
        read_clip(stem_dir / f"{stem}.wav", task_dir / "source_stems" / f"{stem}_context.wav", context_start, context_end)
    metrics = {
        "status": "success",
        "task_id": task["task_id"],
        "song_id": task["song_id"],
        "instruction": task["instruction"],
        "target_stems": task["target_stems"],
        "edit_region": task["edit_region"],
        "context_region": {"start_sec": context_start, "end_sec": context_end},
        "backend": "ace-step-local-xpu",
        "seed": task["seeds"][0],
        "stem_metrics": stem_metrics,
        "mix_metrics": mix_metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (task_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-id", action="append", help="Run only this task; repeat for multiple tasks")
    parser.add_argument("--limit", type=int, help="Run at most N tasks in manifest order")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even if output is complete")
    parser.add_argument(
        "--backfill-source-mix",
        action="store_true",
        help="Create source_mix.wav for already generated task directories without loading ACE-Step",
    )
    args = parser.parse_args()
    configure_environment()
    tasks = load_tasks(args.manifest.resolve())
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in selected]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    if args.backfill_source_mix:
        created = 0
        skipped = 0
        for task in tasks:
            task_dir = args.output_dir / task["task_id"]
            source_dir = task_dir / "source_stems"
            source_paths = {stem: source_dir / f"{stem}.wav" for stem in ("vocals", "drums", "bass", "other")}
            destination = task_dir / "source_mix.wav"
            if destination.is_file():
                skipped += 1
                continue
            if not task_dir.is_dir() or not all(path.is_file() for path in source_paths.values()):
                skipped += 1
                continue
            mix_stems(source_paths, destination)
            created += 1
        print(json.dumps({"created": created, "skipped": skipped, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
        return 0

    import torch
    from loguru import logger
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig

    if not torch.xpu.is_available():
        logger.error("XPU is not available")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary = {"backend": "ace-step-local-xpu", "device": torch.xpu.get_device_name(0), "tasks": {}}
    if summary_path.is_file():
        try:
            summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    handler = AceStepHandler()
    logger.info(f"Initializing ACE-Step once for {len(tasks)} task(s)...")
    started = time.perf_counter()
    _, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT), config_path="acestep-v15-turbo", device="xpu",
        compile_model=False, offload_to_cpu=True, offload_dit_to_cpu=False,
        quantization=None, prefer_source="modelscope", use_mlx_dit=False,
    )
    if not success:
        logger.error("ACE-Step initialization failed")
        return 3
    logger.info(f"Model initialized in {time.perf_counter() - started:.1f}s")
    config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    completed = 0
    for index, task in enumerate(tasks, 1):
        task_dir = args.output_dir / task["task_id"]
        stem_dir = stem_dir_for_task(task, args.cache_root.resolve())
        if not stem_dir.is_dir():
            logger.warning(f"Skipping {task['task_id']}: imported stems not found at {stem_dir}")
            continue
        if not args.force and task_complete(task_dir, task):
            logger.info(f"[{index}/{len(tasks)}] {task['task_id']} already complete")
            completed += 1
            continue
        task_started = time.perf_counter()
        try:
            metrics = run_task(task, task_dir, stem_dir, handler, config, logger)
            summary["tasks"][task["task_id"]] = metrics
            completed += 1
            logger.info(f"[{index}/{len(tasks)}] {task['task_id']} complete in {time.perf_counter() - task_started:.1f}s")
        except Exception as exc:
            error = {"status": "failed", "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}
            summary["tasks"][task["task_id"]] = error
            (task_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
            (task_dir / "metrics.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.exception(f"{task['task_id']} failed")
    summary["completed"] = completed
    summary["selected"] = len(tasks)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.resolve()), "completed": completed, "selected": len(tasks)}, ensure_ascii=False))
    return 0 if completed == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
