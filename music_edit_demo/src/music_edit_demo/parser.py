from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Protocol

from pydantic import BaseModel, Field

from .models import (
    Direction,
    EditAttribute,
    EditOperation,
    EditPlan,
    EditRegion,
    PlanStatus,
    Stem,
    StemEdit,
)


class InstructionParser(Protocol):
    name: str

    def parse(self, project_id: str, region: EditRegion, instruction: str) -> EditPlan: ...


class ParserOutput(BaseModel):
    targets: list[StemEdit] = Field(default_factory=list)
    generator_prompt: str = ""
    unsupported_reasons: list[str] = Field(default_factory=list)


STEM_ALIASES: dict[Stem, tuple[str, ...]] = {
    Stem.VOCALS: ("人声", "歌声", "演唱", "主唱", "vocal", "vocals"),
    Stem.DRUMS: ("鼓点", "鼓轨", "鼓", "drum", "drums"),
    Stem.BASS: ("贝斯", "低音轨", "bass"),
    Stem.OTHER: ("伴奏", "其他轨", "其他乐器", "other"),
}

STYLE_TERMS = (
    "摇滚",
    "电子",
    "爵士",
    "原声",
    "氛围",
    "复古",
    "现代",
    "温暖",
    "明亮",
    "黑暗",
    "轻柔",
    "激烈",
    "rock",
    "electronic",
    "jazz",
    "acoustic",
    "ambient",
)

TIME_RANGE_RE = re.compile(
    r"(?:第\s*)?(?P<start>\d+(?:\.\d+)?)\s*(?:秒)?\s*"
    r"(?:到|至|[-~—–])\s*(?:第\s*)?(?P<end>\d+(?:\.\d+)?)\s*秒"
)


def _find_stems(text: str) -> list[Stem]:
    found: list[Stem] = []
    lowered = text.lower()
    for stem, aliases in STEM_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            found.append(stem)
    return found


def _operation_from_clause(clause: str) -> list[EditOperation]:
    lowered = clause.lower()
    operations: list[EditOperation] = []

    density_up = ("加密", "更密", "密一些", "更丰富", "更复杂", "denser")
    density_down = ("稀疏", "更少", "简单些", "sparser")
    energy_up = ("有力量", "更有力", "增强", "更强", "冲击力", "更激烈", "energetic")
    energy_down = ("减弱", "更弱", "轻一些", "收敛", "柔和些", "less energy")

    if any(term in lowered for term in density_up):
        operations.append(EditOperation(attribute=EditAttribute.DENSITY, direction=Direction.INCREASE))
    elif any(term in lowered for term in density_down):
        operations.append(EditOperation(attribute=EditAttribute.DENSITY, direction=Direction.DECREASE))

    if any(term in lowered for term in energy_up):
        operations.append(EditOperation(attribute=EditAttribute.ENERGY, direction=Direction.INCREASE))
    elif any(term in lowered for term in energy_down):
        operations.append(EditOperation(attribute=EditAttribute.ENERGY, direction=Direction.DECREASE))

    styles = [term for term in STYLE_TERMS if term in lowered]
    if styles:
        operations.append(
            EditOperation(
                attribute=EditAttribute.STYLE,
                direction=Direction.CHANGE,
                value=styles[0],
            )
        )

    if any(term in lowered for term in ("重新生成", "重生成", "重做", "改写", "换一个版本", "regenerate")):
        operations.append(EditOperation(attribute=EditAttribute.REGENERATE))

    return operations


def _unsupported_reasons(text: str) -> list[str]:
    checks = (
        (r"(替换|修改|改成|重写).{0,8}歌词|歌词.{0,8}(替换|修改|改成|重写)", "exact lyric editing is not supported in the MVP"),
        (r"和弦|调式|升调|降调|转调", "explicit chord or key changes are not supported in the MVP"),
        (r"(?:bpm|速度).{0,8}(改成|提高|降低|加快|减慢)", "tempo changes are not supported in the MVP"),
        (r"指定音符|音高改成|旋律改成", "note-level melody editing is not supported in the MVP"),
    )
    lowered = text.lower()
    return [message for pattern, message in checks if re.search(pattern, lowered)]


def _is_preserve_clause(clause: str) -> bool:
    if any(term in clause for term in ("不要改", "不修改")):
        return True
    return bool(
        re.search(r"保持.*(?:不变|原样)", clause)
        or re.search(r"保留原", clause)
    )


class ControlledInstructionParser:
    """Deterministic fallback used for contracts, tests, and offline development."""

    name = "controlled"

    def parse(self, project_id: str, region: EditRegion, instruction: str) -> EditPlan:
        reasons = _unsupported_reasons(instruction)
        status = PlanStatus.UNSUPPORTED if reasons else PlanStatus.READY

        time_match = TIME_RANGE_RE.search(instruction)
        if time_match and not reasons:
            text_start = float(time_match.group("start"))
            text_end = float(time_match.group("end"))
            if abs(text_start - region.start_sec) > 0.05 or abs(text_end - region.end_sec) > 0.05:
                status = PlanStatus.NEEDS_CONFIRMATION
                reasons.append(
                    f"instruction time {text_start:g}-{text_end:g}s conflicts with request time "
                    f"{region.start_sec:g}-{region.end_sec:g}s"
                )

        targets: dict[Stem, list[EditOperation]] = {}
        clauses = [part.strip() for part in re.split(r"[，,；;。]|但", instruction) if part.strip()]
        for clause in clauses:
            if _is_preserve_clause(clause):
                continue
            stems = _find_stems(clause)
            operations = _operation_from_clause(clause)
            for stem in stems:
                targets.setdefault(stem, []).extend(operations or [EditOperation(attribute=EditAttribute.REGENERATE)])

        if not targets and not reasons:
            status = PlanStatus.UNSUPPORTED
            reasons.append("no supported target stem was found")

        stem_edits = [StemEdit(stem=stem, operations=operations) for stem, operations in targets.items()]
        return EditPlan(
            project_id=project_id,
            region=region,
            instruction=instruction,
            targets=stem_edits,
            generator_prompt=instruction,
            status=status,
            unsupported_reasons=reasons,
            parser_name=self.name,
        )


class QwenInstructionParser:
    """OpenAI-compatible Qwen adapter with schema-constrained output."""

    name = "qwen"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_sec: int = 30):
        if not base_url:
            raise ValueError("QWEN_BASE_URL is required when MUSIC_EDIT_PARSER=qwen")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def parse(self, project_id: str, region: EditRegion, instruction: str) -> EditPlan:
        system_prompt = (
            "Convert a Chinese or English music-editing instruction into JSON. "
            "Allowed stems: vocals, drums, bass, other. Allowed attributes: regenerate, energy, density, style. "
            "Energy and density directions must be increase or decrease. Exact lyric replacement, note-level melody, "
            "chord/key changes, and tempo changes are unsupported. Do not infer an unmentioned target stem."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "music_edit_parser_output",
                    "strict": True,
                    "schema": ParserOutput.model_json_schema(),
                },
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen parser request failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"]
        parsed = ParserOutput.model_validate_json(content)
        reasons = list(parsed.unsupported_reasons)
        status = PlanStatus.UNSUPPORTED if reasons else PlanStatus.READY

        time_match = TIME_RANGE_RE.search(instruction)
        if time_match and status == PlanStatus.READY:
            text_start = float(time_match.group("start"))
            text_end = float(time_match.group("end"))
            if abs(text_start - region.start_sec) > 0.05 or abs(text_end - region.end_sec) > 0.05:
                status = PlanStatus.NEEDS_CONFIRMATION
                reasons.append("instruction time conflicts with the structured edit region")

        if not parsed.targets and not reasons:
            status = PlanStatus.UNSUPPORTED
            reasons.append("no supported target stem was found")

        return EditPlan(
            project_id=project_id,
            region=region,
            instruction=instruction,
            targets=parsed.targets,
            generator_prompt=parsed.generator_prompt or instruction,
            status=status,
            unsupported_reasons=reasons,
            parser_name=self.name,
        )
