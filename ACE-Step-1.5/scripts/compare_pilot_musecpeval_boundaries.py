#!/usr/bin/env python3
"""Compare two pilot repaint output directories with MuseCPEval boundary clips."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS = ("harmony", "rhythm", "melody", "timbre")
HIGHER_IS_BETTER = (
    "harmony_tonality.chroma_similarity.chroma_dtw_cosine",
    "harmony_tonality.chroma_similarity.mean_chroma_cosine",
    "melodic_content.contour_dtw_similarity",
    "melodic_content.motif_3gram_recall",
    "timbre_texture.mean_mfcc_cosine",
    "timbre_texture.mfcc_skl_similarity",
)
LOWER_IS_BETTER = (
    "harmony_tonality.key_relatedness.distance_norm_0to1",
    "rhythm_meter.delta_bpm_folded",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def crop_wav(source: Path, destination: Path, start_sec: float, end_sec: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as src:
        sample_rate = src.getframerate()
        total_frames = src.getnframes()
        start = max(0, min(total_frames, round(start_sec * sample_rate)))
        end = max(start + 1, min(total_frames, round(end_sec * sample_rate)))
        src.setpos(start)
        frames = src.readframes(end - start)
        params = src.getparams()
    with wave.open(str(destination), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)


def task_dirs(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("pilot_") and (path / "task.json").is_file()
    }


def build_manifest(args: argparse.Namespace) -> Path:
    left_tasks = task_dirs(args.left_root)
    right_tasks = task_dirs(args.right_root)
    common_ids = sorted(set(left_tasks) & set(right_tasks))
    if not common_ids:
        raise SystemExit("no matching pilot task directories found")

    clips_dir = args.output_dir / "clips"
    pairs: list[dict[str, Any]] = []
    for task_id in common_ids:
        task = read_json(left_tasks[task_id] / "task.json")
        right_task = read_json(right_tasks[task_id] / "task.json")
        if task["edit_region"] != right_task["edit_region"]:
            raise SystemExit(f"edit_region mismatch for {task_id}")
        for boundary, center in (
            ("start", float(task["edit_region"]["start_sec"])),
            ("end", float(task["edit_region"]["end_sec"])),
        ):
            start = center - args.window_radius_sec
            end = center + args.window_radius_sec
            ref_clip = clips_dir / "ref" / task_id / f"{boundary}.wav"
            crop_wav(left_tasks[task_id] / "source_mix.wav", ref_clip, start, end)
            for label, root in ((args.left_label, left_tasks[task_id]), (args.right_label, right_tasks[task_id])):
                est_clip = clips_dir / label / task_id / f"{boundary}.wav"
                crop_wav(root / "edited_mix.wav", est_clip, start, end)
                pairs.append(
                    {
                        "id": f"{label}/{task_id}/{boundary}",
                        "ref": str(ref_clip.resolve()),
                        "est": str(est_clip.resolve()),
                        "system": label,
                        "task_id": task_id,
                        "song_id": task["song_id"],
                        "boundary": boundary,
                        "target_stems": ",".join(task["target_stems"]),
                        "window_radius_sec": args.window_radius_sec,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "pairs.json"
    manifest.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest_meta.json").write_text(
        json.dumps(
            {
                "left_root": str(args.left_root.resolve()),
                "right_root": str(args.right_root.resolve()),
                "left_label": args.left_label,
                "right_label": args.right_label,
                "tasks": len(common_ids),
                "pairs": len(pairs),
                "window_radius_sec": args.window_radius_sec,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def run_musecpeval(args: argparse.Namespace, manifest: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "musecpeval",
        "--batch-json",
        str(manifest),
        "--output-dir",
        str(args.output_dir / "results"),
        "--metrics",
        *args.metrics,
        "--no-resume",
    ]
    if args.no_parallel:
        command.append("--no-parallel")
    else:
        command.extend(["--n-workers", str(args.n_workers)])
    subprocess.run(command, check=True)


def fnum(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def summarize(args: argparse.Namespace) -> None:
    summary_csv = args.output_dir / "results" / "summary.csv"
    rows = list(csv.DictReader(summary_csv.open(newline="", encoding="utf-8")))
    metrics = [item for item in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER) if item in rows[0]]
    by_system: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)

    report = args.output_dir / "COMPARISON_REPORT.md"
    lines = [
        "# MuseCPEval Boundary Comparison",
        "",
        f"- Left: `{args.left_label}`",
        f"- Right: `{args.right_label}`",
        f"- Boundary window: +/- {args.window_radius_sec:g} s",
        f"- Pairs: {len(rows)}",
        "",
        "| Metric | Direction | Left | Right | Right - Left |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        left_values = [fnum(row[metric]) for row in by_system[args.left_label]]
        right_values = [fnum(row[metric]) for row in by_system[args.right_label]]
        left_mean = mean(value for value in left_values if value is not None)
        right_mean = mean(value for value in right_values if value is not None)
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        lines.append(f"| {metric} | {direction} | {left_mean:.4f} | {right_mean:.4f} | {right_mean - left_mean:.4f} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--left-label", default="multibranch_v1")
    parser.add_argument("--right-label", default="demo_no_cross_stem")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-radius-sec", type=float, default=2.0)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--no-parallel", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.left_root = args.left_root.resolve()
    args.right_root = args.right_root.resolve()
    args.output_dir = args.output_dir.resolve()
    manifest = build_manifest(args)
    run_musecpeval(args, manifest)
    summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
