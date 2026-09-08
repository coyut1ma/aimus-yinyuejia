"""Batch evaluation for completed ACE-Step pilot task directories."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .evaluation import EvaluationConfig, evaluate_stems, save_evaluation
from .models import EditRegion, Stem


STEM_NAMES = tuple(stem.value for stem in Stem)


def _describe(values: list[float]) -> dict[str, float | int | None]:
    """Return descriptive statistics without inventing a quality score."""

    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _alignment_pairs(report: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    per_target = report.get("metrics", {}).get("alignment", {}).get("per_target", {})
    for target, target_result in per_target.items():
        for reference, pair in target_result.get("per_reference", {}).items():
            pairs.append({"target": target, "reference": reference, **pair})
    return pairs


def _flat_row(task: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    beat = report.get("metrics", {}).get("beat", {})
    volume = report.get("metrics", {}).get("volume", {})
    pairs = [pair for pair in _alignment_pairs(report) if pair.get("status") == "ok"]
    volumes = [item for item in volume.get("per_stem", {}).values() if item.get("status") == "ok"]
    return {
        "task_id": task["task_id"],
        "song_id": task["song_id"],
        "task_type": task["task_type"],
        "instruction": task["instruction"],
        "target_stems": ",".join(task["target_stems"]),
        "seed": task.get("seeds", [None])[0],
        "report_status": report.get("status"),
        "beat_status": beat.get("status"),
        "bpm_ref": beat.get("bpm_ref"),
        "bpm_edit_raw": beat.get("bpm_edit_raw"),
        "bpm_edit": beat.get("bpm_edit"),
        "tempo_octave_factor": beat.get("tempo_octave_factor", 1.0),
        "tempo_abs_error_bpm": beat.get("tempo_abs_error_bpm"),
        "tempo_rel_error_pct": beat.get("tempo_rel_error_pct"),
        "beat_mae_ms": beat.get("beat_mae_ms"),
        "beat_rmse_ms": beat.get("beat_rmse_ms"),
        "beat_p95_ms": beat.get("beat_p95_ms"),
        "beat_bias_ms": beat.get("beat_bias_ms"),
        "beat_miss_rate": beat.get("miss_rate"),
        "beat_extra_rate": beat.get("extra_rate"),
        "alignment_valid_pairs": len(pairs),
        "alignment_mae_ms_mean": round(mean(float(pair["alignment_mae_ms"]) for pair in pairs), 4) if pairs else None,
        "alignment_degradation_ms_mean": round(
            mean(float(pair["alignment_degradation_ms"]) for pair in pairs if pair.get("alignment_degradation_ms") is not None), 4
        ) if any(pair.get("alignment_degradation_ms") is not None for pair in pairs) else None,
        "onset_unmatched_rate_mean": round(mean(float(pair["unmatched_rate"]) for pair in pairs), 4) if pairs else None,
        "volume_valid_targets": len(volumes),
        "volume_abs_deviation_db_mean": round(mean(float(item["absolute_volume_deviation_db"]) for item in volumes), 4) if volumes else None,
        "volume_abs_deviation_db_max": round(max(float(item["absolute_volume_deviation_db"]) for item in volumes), 4) if volumes else None,
        "volume_targets_within_3db": sum(bool(item["within_good_threshold"]) for item in volumes),
        "volume_all_targets_within_3db": bool(volumes) and all(bool(item["within_good_threshold"]) for item in volumes),
    }


def _collect_summary(rows: list[dict[str, Any]], reports: list[dict[str, Any]]) -> dict[str, Any]:
    beat_ok = [row for row in rows if row["beat_status"] == "ok"]
    pairs = [pair for report in reports for pair in _alignment_pairs(report) if pair.get("status") == "ok"]
    volumes = [
        item
        for report in reports
        for item in report.get("metrics", {}).get("volume", {}).get("per_stem", {}).values()
        if item.get("status") == "ok"
    ]
    return {
        "task_count": len(rows),
        "report_status_counts": {status: sum(row["report_status"] == status for row in rows) for status in sorted({row["report_status"] for row in rows})},
        "beat": {
            "valid_task_count": len(beat_ok),
            "tempo_abs_error_bpm": _describe([float(row["tempo_abs_error_bpm"]) for row in beat_ok]),
            "tempo_rel_error_pct": _describe([float(row["tempo_rel_error_pct"]) for row in beat_ok]),
            "beat_mae_ms": _describe([float(row["beat_mae_ms"]) for row in beat_ok]),
            "beat_rmse_ms": _describe([float(row["beat_rmse_ms"]) for row in beat_ok]),
            "miss_rate": _describe([float(row["beat_miss_rate"]) for row in beat_ok]),
            "extra_rate": _describe([float(row["beat_extra_rate"]) for row in beat_ok]),
        },
        "alignment": {
            "valid_pair_count": len(pairs),
            "alignment_mae_ms": _describe([float(pair["alignment_mae_ms"]) for pair in pairs]),
            "alignment_degradation_ms": _describe([float(pair["alignment_degradation_ms"]) for pair in pairs if pair.get("alignment_degradation_ms") is not None]),
            "unmatched_rate": _describe([float(pair["unmatched_rate"]) for pair in pairs]),
        },
        "volume": {
            "valid_target_count": len(volumes),
            "absolute_deviation_db": _describe([float(item["absolute_volume_deviation_db"]) for item in volumes]),
            "within_3db_count": sum(bool(item["within_good_threshold"]) for item in volumes),
            "within_3db_rate": round(sum(bool(item["within_good_threshold"]) for item in volumes) / len(volumes), 4) if volumes else None,
            "all_targets_within_3db_task_count": sum(bool(row["volume_all_targets_within_3db"]) for row in rows),
        },
    }


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    beat = summary["beat"]
    alignment = summary["alignment"]
    volume = summary["volume"]
    octave_adjusted = sum(float(row.get("tempo_octave_factor") or 1.0) != 1.0 for row in rows)
    lines = [
        "# ACE-Step 20条局部编辑任务客观评测报告",
        "",
        f"- 任务数：{summary['task_count']}",
        f"- 节拍有效任务：{beat['valid_task_count']}/{summary['task_count']}",
        f"- Onset对齐有效轨道对：{alignment['valid_pair_count']}",
        f"- 音量有效目标轨：{volume['valid_target_count']}",
        f"- Beat tracker半拍/双倍拍速消歧：{octave_adjusted}条；CSV与逐任务JSON同时保留原始BPM和校正后BPM。",
        "- 本报告不计算综合分数。",
        "",
        "## 总体统计",
        "",
        f"- 节拍MAE：均值 {beat['beat_mae_ms']['mean']} ms，中位数 {beat['beat_mae_ms']['median']} ms，P95 {beat['beat_mae_ms']['p95']} ms。",
        f"- BPM绝对误差：均值 {beat['tempo_abs_error_bpm']['mean']} BPM，中位数 {beat['tempo_abs_error_bpm']['median']} BPM。",
        f"- 漏拍率：均值 {beat['miss_rate']['mean']}；多拍率：均值 {beat['extra_rate']['mean']}。",
        f"- Onset对齐MAE：均值 {alignment['alignment_mae_ms']['mean']} ms，中位数 {alignment['alignment_mae_ms']['median']} ms。",
        f"- Onset对齐退化量：均值 {alignment['alignment_degradation_ms']['mean']} ms。正数表示编辑后变差。",
        f"- Onset未匹配率：均值 {alignment['unmatched_rate']['mean']}。",
        f"- 音量绝对偏差：均值 {volume['absolute_deviation_db']['mean']} dB，中位数 {volume['absolute_deviation_db']['median']} dB。",
        f"- 音量偏差不超过3 dB：{volume['within_3db_count']}/{volume['valid_target_count']} 条目标轨（{volume['within_3db_rate']:.1%}）。",
        "",
        "## 逐任务摘要",
        "",
        "| 任务 | 目标轨 | 节拍MAE(ms) | BPM误差 | 对齐MAE均值(ms) | Onset未匹配率 | 最大音量偏差(dB) | 全部≤3dB |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['target_stems']} | {row['beat_mae_ms']} | "
            f"{row['tempo_abs_error_bpm']} | {row['alignment_mae_ms_mean']} | "
            f"{row['onset_unmatched_rate_mean']} | {row['volume_abs_deviation_db_max']} | "
            f"{'是' if row['volume_all_targets_within_3db'] else '否'} |"
        )
    lines.extend(["", "逐轨、逐参考轨的完整数据见 `tasks/*.json`，可筛选表见 `task_metrics.csv`。", ""])
    return "\n".join(lines)


def evaluate_pilot_directory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate all completed pilot task folders and write JSON/CSV/Markdown artifacts."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    task_dirs = sorted(path for path in input_dir.glob("pilot_v1_*") if path.is_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tasks").mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        original = {stem: task_dir / "source_stems" / f"{stem}.wav" for stem in STEM_NAMES}
        edited = {stem: task_dir / "edited_stems" / f"{stem}.wav" for stem in STEM_NAMES}
        report = evaluate_stems(
            original,
            edited,
            EditRegion.model_validate(task["edit_region"]),
            task["target_stems"],
            EvaluationConfig(),
        )
        report.update({
            "task_id": task["task_id"],
            "song_id": task["song_id"],
            "task_type": task["task_type"],
            "instruction": task["instruction"],
            "seed": task.get("seeds", [None])[0],
            "backend": "ace-step-local-xpu",
        })
        save_evaluation(report, output_dir / "tasks" / f"{task['task_id']}.json")
        reports.append(report)
        rows.append(_flat_row(task, report))
        print(f"evaluated {task['task_id']} ({len(rows)}/{len(task_dirs)})", flush=True)
    summary = _collect_summary(rows, reports)
    combined = {
        "schema_version": "pilot_evaluation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(input_dir),
        "summary": summary,
        "tasks": reports,
    }
    save_evaluation(combined, output_dir / "evaluation_results.json")
    with (output_dir / "task_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "EVALUATION_REPORT.md").write_text(_markdown(summary, rows), encoding="utf-8")
    return combined
