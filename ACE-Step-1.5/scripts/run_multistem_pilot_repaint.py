#!/usr/bin/env python3
"""Run frozen music-edit demo pilot tasks with the local multi-stem adapter.

The output layout intentionally matches ``run_all_pilot_repaint.py`` so the
existing ``music_edit_demo.batch_evaluation`` evaluator can be reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.signal import resample_poly


ACE_ROOT = Path(__file__).resolve().parents[1]
MUS_ROOT = ACE_ROOT.parent
DEMO_ROOT = MUS_ROOT / "music_edit_demo"
STEM_ORDER = ("vocals", "drums", "bass", "other")
STEM_TO_ID = {stem: index for index, stem in enumerate(STEM_ORDER)}
DEFAULT_MANIFEST = DEMO_ROOT / "runtime_real" / "pilot_tasks_v1.jsonl"
DEFAULT_CACHE = DEMO_ROOT / "runtime_real" / "musdb_cache"
DEFAULT_ADAPTER = MUS_ROOT / "outputs" / "multistem_adapter" / "run_long_20260908_2125" / "adapter_latest.pt"
DEFAULT_OUTPUT = MUS_ROOT / "outputs" / "multistem_adapter" / "pilot20_joint_adapter"
CONTEXT_SEC = 4.0


def configure_imports() -> None:
    for path in (ACE_ROOT, DEMO_ROOT / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def configure_environment(project_root: Path, device: str) -> None:
    cache_root = project_root / ".cache"
    paths = {
        "ACESTEP_CHECKPOINTS_DIR": project_root / "checkpoints",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "MODELSCOPE_CACHE": cache_root / "modelscope",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("PYTORCH_DEVICE", device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")


def load_tasks(manifest: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem_dir_for_task(task: dict[str, Any], cache_root: Path) -> Path:
    return cache_root / task["song_id"]


def robust_decode_stream(
    container_path: Path,
    stream_index: int,
    destination: Path,
    expected_frames: int,
) -> dict[str, int | float]:
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".wav.tmp")
    frames_written = 0
    with av.open(str(container_path)) as container:
        stream = list(container.streams.audio)[stream_index]
        sample_rate = int(stream.codec_context.sample_rate)
        channels = len(stream.codec_context.layout.channels)
        if channels != 2:
            raise ValueError("the pilot importer requires stereo MUSDB18 streams")
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=sample_rate)
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            try:
                for decoded in container.decode(stream):
                    for frame in resampler.resample(decoded):
                        if frames_written >= expected_frames:
                            break
                        allowed = expected_frames - frames_written
                        samples = frame.to_ndarray().reshape(-1)
                        samples = samples[: allowed * channels]
                        output.writeframesraw(samples.tobytes())
                        frames_written += min(frame.samples, allowed)
            except Exception as exc:
                logger.warning(
                    "Using partial decoded audio for {} stream {}: {}",
                    container_path,
                    stream_index,
                    exc,
                )
            for frame in resampler.resample(None):
                if frames_written >= expected_frames:
                    break
                allowed = expected_frames - frames_written
                samples = frame.to_ndarray().reshape(-1)
                samples = samples[: allowed * channels]
                output.writeframesraw(samples.tobytes())
                frames_written += min(frame.samples, allowed)
            if frames_written < expected_frames:
                missing = expected_frames - frames_written
                output.writeframesraw(b"\0" * missing * channels * 2)
                frames_written = expected_frames
    temporary.replace(destination)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": 2,
        "frames": frames_written,
        "duration_sec": frames_written / sample_rate,
    }


def robust_import_track(task: dict[str, Any], cache_root: Path) -> Path:
    from music_edit_demo.audio import validate_stems
    from music_edit_demo.models import Stem
    from music_edit_demo.musdb import MUSDB_STREAMS, inspect_container

    container = task.get("input_container")
    if not container:
        raise ValueError(f"pilot task has no input_container: {task['task_id']}")
    track = inspect_container(Path(container))
    track_dir = cache_root.resolve() / track.track_id
    expected_frames = round(track.duration_sec * track.sample_rate)
    stems = {stem: track_dir / f"{stem.value}.wav" for stem in Stem}
    decoded = {
        stem.value: robust_decode_stream(Path(track.container_path), stream_index, stems[stem], expected_frames)
        for stem, stream_index in MUSDB_STREAMS.items()
    }
    audio = validate_stems(stems)
    (track_dir / "import.json").write_text(
        json.dumps(
            {
                "track": track.model_dump(mode="json"),
                "robust_partial_decode": True,
                "decoded": decoded,
                "audio": audio.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return track_dir


def ensure_imported_stems(task: dict[str, Any], cache_root: Path, force: bool = False) -> Path:
    from music_edit_demo.musdb import import_track, inspect_container

    stem_dir = stem_dir_for_task(task, cache_root)
    stem_paths = [stem_dir / f"{stem}.wav" for stem in STEM_ORDER]
    if not force and all(path.is_file() for path in stem_paths):
        return stem_dir
    container = task.get("input_container")
    if not container:
        raise ValueError(f"pilot task has no input_container: {task['task_id']}")
    track = inspect_container(Path(container))
    try:
        project = import_track(track, cache_root, force=force)
        return Path(project.stems[next(iter(project.stems))]).parent
    except Exception as exc:
        logger.warning(
            "Strict MUSDB import failed for {}; falling back to partial decode: {}",
            task["task_id"],
            exc,
        )
        return robust_import_track(task, cache_root)


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
    generated_region: tuple[float, float],
) -> dict[str, float | int]:
    original, sr = sf.read(original_path, always_2d=True, dtype="float32")
    generated, generated_sr = sf.read(generated_path, always_2d=True, dtype="float32")
    if generated_sr != sr:
        generated = resample_poly(generated, sr, generated_sr, axis=0).astype(np.float32)
    src_a = round(generated_region[0] * sr)
    src_b = round(generated_region[1] * sr)
    dst_a = round(absolute_region[0] * sr)
    dst_b = round(absolute_region[1] * sr)
    generated_region_audio = generated[src_a:src_b]
    target_len = dst_b - dst_a
    if len(generated_region_audio) != target_len:
        generated_region_audio = resample_poly(generated_region_audio, target_len, max(1, len(generated_region_audio)), axis=0)
        generated_region_audio = generated_region_audio[:target_len]
        if len(generated_region_audio) < target_len:
            generated_region_audio = np.pad(generated_region_audio, ((0, target_len - len(generated_region_audio)), (0, 0)))

    original_region = original[dst_a:dst_b]
    alpha = np.ones(target_len, dtype=np.float32)
    fade = min(round(0.2 * sr), target_len // 2)
    if fade:
        ramp = np.sin(np.arange(fade, dtype=np.float32) / fade * np.pi / 2) ** 2
        alpha[:fade], alpha[-fade:] = ramp, ramp[::-1]
    mixed_region = original_region * (1.0 - alpha[:, None]) + generated_region_audio * alpha[:, None]
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
        "generated_region_rms_db": rms_db(generated_region_audio),
        "rms_delta_db": rms_db(generated_region_audio) - rms_db(original_region),
        "outside_max_abs_diff": float(diff[outside].max()) if np.any(outside) else 0.0,
        "edited_changed_samples": int(np.count_nonzero(diff[dst_a:dst_b] > 1e-7)),
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


def task_complete(task_dir: Path, task: dict[str, Any]) -> bool:
    metrics_path = task_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metrics.get("status") != "success":
        return False
    required = [task_dir / "edited_mix.wav"] + [task_dir / "edited_stems" / f"{stem}.wav" for stem in STEM_ORDER]
    required += [task_dir / "ace_step" / stem / "generated_context.wav" for stem in task["target_stems"]]
    return all(path.is_file() for path in required)


def load_multistem_context(handler: Any, context_paths: dict[str, Path]) -> torch.Tensor:
    audios = []
    for stem in STEM_ORDER:
        audio = handler.process_src_audio(str(context_paths[stem]))
        if audio is None:
            raise RuntimeError(f"unable to load context audio for {stem}: {context_paths[stem]}")
        audios.append(audio)
    max_samples = max(audio.shape[-1] for audio in audios)
    aligned = [
        F.pad(audio, (0, max_samples - audio.shape[-1])) if audio.shape[-1] < max_samples else audio
        for audio in audios
    ]
    return torch.stack(aligned, dim=0).unsqueeze(0)


def write_audio_tensor(path: Path, audio_tensor: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio_tensor.detach().cpu().float()
    if audio.ndim == 2:
        audio = audio.transpose(0, 1)
    sf.write(path, audio.numpy(), sample_rate, subtype="PCM_16")


def run_task(
    task: dict[str, Any],
    task_dir: Path,
    stem_dir: Path,
    handler: Any,
    *,
    seed_index: int = 0,
    inference_steps: int = 8,
    repaint_mode: str = "balanced",
    repaint_strength: float = 0.5,
    cross_stem_attention: bool = True,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    start = float(task["edit_region"]["start_sec"])
    end = float(task["edit_region"]["end_sec"])
    context_start = max(0.0, start - CONTEXT_SEC)
    context_end = end + CONTEXT_SEC
    context_edit_start = start - context_start
    context_edit_end = end - context_start
    seed = int(task["seeds"][seed_index])

    source_dir = task_dir / "source_stems"
    edited_dir = task_dir / "edited_stems"
    ace_dir = task_dir / "ace_step"
    source_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)
    for stem in STEM_ORDER:
        shutil.copyfile(stem_dir / f"{stem}.wav", source_dir / f"{stem}.wav")
    mix_stems({stem: source_dir / f"{stem}.wav" for stem in STEM_ORDER}, task_dir / "source_mix.wav")

    context_paths: dict[str, Path] = {}
    for stem in STEM_ORDER:
        path = ace_dir / "context" / f"{stem}.wav"
        read_clip(stem_dir / f"{stem}.wav", path, context_start, context_end)
        context_paths[stem] = path
    multi_stem_context = load_multistem_context(handler, context_paths)

    generated_by_stem: dict[str, str] = {}
    stem_metrics: dict[str, dict[str, float | int]] = {}
    for stem in task["target_stems"]:
        stem_ace_dir = ace_dir / stem
        stem_ace_dir.mkdir(parents=True, exist_ok=True)
        context_clip = context_paths[stem]
        shutil.copyfile(context_clip, stem_ace_dir / "input_context.wav")
        caption = (
            f"{task['instruction']}. Generate a coherent replacement for the selected {stem} track region, "
            "preserving tempo, key, musical timing, and the surrounding context."
        )
        logger.info(
            "{} target={}: joint ACE-Step repaint {:.1f}-{:.1f}s seed={}",
            task["task_id"],
            stem,
            context_start,
            context_end,
            seed,
        )
        result = handler.generate_music(
            captions=caption,
            lyrics="",
            reference_audio=None,
            src_audio=str(context_clip),
            task_type="repaint",
            repainting_start=context_edit_start,
            repainting_end=context_edit_end,
            chunk_mask_mode="explicit",
            inference_steps=inference_steps,
            guidance_scale=1.0,
            use_random_seed=False,
            seed=seed,
            batch_size=1,
            audio_cover_strength=1.0,
            repaint_mode=repaint_mode,
            repaint_strength=repaint_strength,
            repaint_wav_crossfade_sec=0.0,
            joint_frontend=True,
            cross_stem_attention=cross_stem_attention,
            target_stem_id=STEM_TO_ID[stem],
            multi_stem_target_wavs=multi_stem_context,
        )
        if not result.get("success") or not result.get("audios"):
            raise RuntimeError(f"ACE-Step failed for {task['task_id']} target={stem}: {result.get('error') or result.get('status_message')}")
        audio_payload = result["audios"][0]
        generated_path = stem_ace_dir / "generated_context.wav"
        write_audio_tensor(generated_path, audio_payload["tensor"], int(audio_payload["sample_rate"]))
        generated_by_stem[stem] = str(generated_path.resolve())
        stem_metrics[stem] = splice(
            stem_dir / f"{stem}.wav",
            generated_path,
            edited_dir / f"{stem}.wav",
            absolute_region=(start, end),
            generated_region=(context_edit_start, context_edit_end),
        )

    edited_paths: dict[str, Path] = {}
    for stem in STEM_ORDER:
        path = edited_dir / f"{stem}.wav"
        if stem not in generated_by_stem:
            shutil.copyfile(stem_dir / f"{stem}.wav", path)
        edited_paths[stem] = path
    mix_metrics = mix_stems(edited_paths, task_dir / "edited_mix.wav")
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
        "backend": (
            "ace-step-cuda-joint-adapter"
            if cross_stem_attention
            else "ace-step-cuda-joint-no-cross-stem"
        ),
        "cross_stem_attention": cross_stem_attention,
        "seed": seed,
        "stem_metrics": stem_metrics,
        "mix_metrics": mix_metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (task_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return metrics


def evaluate_output(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    from music_edit_demo.batch_evaluation import evaluate_pilot_directory

    return evaluate_pilot_directory(input_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=MUS_ROOT)
    parser.add_argument("--model-name", default="acestep-v15-turbo")
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", action="append", help="Run only this task; repeat for multiple tasks")
    parser.add_argument("--limit", type=int, help="Run at most N tasks in manifest order")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even if output is complete")
    parser.add_argument("--force-import", action="store_true", help="Re-decode MUSDB stems even if cache exists")
    parser.add_argument("--seed-index", type=int, default=0, help="Use task.seeds[index]; default matches the old pilot runner")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--repaint-mode", choices=("conservative", "balanced", "aggressive"), default="balanced")
    parser.add_argument("--repaint-strength", type=float, default=0.5)
    parser.add_argument(
        "--disable-cross-stem-attention",
        action="store_true",
        help="Use the same joint frontend path but skip the cross-stem attention exchange.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run the existing demo batch evaluator after generation")
    parser.add_argument("--skip-generation", action="store_true", help="Only evaluate an existing output directory")
    parser.add_argument("--use-flash-attention", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_imports()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_environment(args.project_root.resolve(), args.device)

    tasks = load_tasks(args.manifest.resolve())
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in selected]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")
    for task in tasks:
        if args.seed_index < 0 or args.seed_index >= len(task.get("seeds", [])):
            raise SystemExit(f"--seed-index out of range for {task['task_id']}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    cross_stem_attention = not args.disable_cross_stem_attention
    summary: dict[str, Any] = {
        "backend": (
            "ace-step-cuda-joint-adapter"
            if cross_stem_attention
            else "ace-step-cuda-joint-no-cross-stem"
        ),
        "device": args.device,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "cross_stem_attention": cross_stem_attention,
        "tasks": {},
    }
    if summary_path.is_file():
        try:
            summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    completed = 0
    if not args.skip_generation:
        from acestep.handler import AceStepHandler
        from acestep.training.multistem_adapter import load_adapter_checkpoint

        if args.device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return 2
        if not args.adapter_checkpoint.is_file():
            logger.error("Adapter checkpoint not found: {}", args.adapter_checkpoint)
            return 2

        handler = AceStepHandler()
        logger.info("Initializing ACE-Step once for {} task(s)...", len(tasks))
        started = time.perf_counter()
        status, ok = handler.initialize_service(
            project_root=str(args.project_root.resolve()),
            config_path=args.model_name,
            device=args.device,
            use_flash_attention=args.use_flash_attention,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            use_mlx_dit=False,
        )
        if not ok:
            logger.error("ACE-Step initialization failed: {}", status)
            return 3
        metadata = load_adapter_checkpoint(handler.model, args.adapter_checkpoint.resolve())
        handler.model.decoder.multi_stem_frontend.to(device=handler.device, dtype=handler.dtype)
        handler.model.eval()
        logger.info("Model initialized in {:.1f}s; loaded adapter metadata={}", time.perf_counter() - started, metadata)

        for index, task in enumerate(tasks, 1):
            task_dir = output_dir / task["task_id"]
            try:
                stem_dir = ensure_imported_stems(task, args.cache_root.resolve(), force=args.force_import)
                if not args.force and task_complete(task_dir, task):
                    logger.info("[{}/{}] {} already complete", index, len(tasks), task["task_id"])
                    completed += 1
                    continue
                task_started = time.perf_counter()
                metrics = run_task(
                    task,
                    task_dir,
                    stem_dir,
                    handler,
                    seed_index=args.seed_index,
                    inference_steps=args.inference_steps,
                    repaint_mode=args.repaint_mode,
                    repaint_strength=args.repaint_strength,
                    cross_stem_attention=cross_stem_attention,
                )
                summary["tasks"][task["task_id"]] = metrics
                completed += 1
                logger.info("[{}/{}] {} complete in {:.1f}s", index, len(tasks), task["task_id"], time.perf_counter() - task_started)
            except Exception as exc:
                error = {"status": "failed", "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}
                summary["tasks"][task["task_id"]] = error
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
                (task_dir / "metrics.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.exception("{} failed", task["task_id"])

        summary["completed"] = completed
        summary["selected"] = len(tasks)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "completed": completed, "selected": len(tasks)}, ensure_ascii=False))

    eval_output_dir = args.eval_output_dir or (output_dir / "evaluation")
    if (args.evaluate or args.skip_generation) and (args.skip_generation or completed == len(tasks)):
        report = evaluate_output(output_dir, eval_output_dir.resolve())
        print(json.dumps({"evaluation": str(eval_output_dir.resolve()), "task_count": report["summary"]["task_count"]}, ensure_ascii=False))
    elif args.evaluate:
        logger.warning(
            "Skipping batch evaluation because only {}/{} selected tasks completed",
            completed,
            len(tasks),
        )

    return 0 if args.skip_generation or completed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
