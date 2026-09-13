from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    runtime_dir: Path
    musdb_root: Path | None = None
    parser_backend: str = "controlled"
    generator_backend: str = "fake"
    qwen_base_url: str = ""
    qwen_api_key: str = ""
    qwen_model: str = "Qwen3-4B-Instruct"
    ace_step_base_url: str = ""
    ace_step_cross_stem_attention: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        runtime = Path(os.getenv("MUSIC_EDIT_RUNTIME_DIR", "runtime")).resolve()
        return cls(
            runtime_dir=runtime,
            musdb_root=(
                Path(value).resolve()
                if (value := os.getenv("MUSDB_ROOT", ""))
                else None
            ),
            parser_backend=os.getenv("MUSIC_EDIT_PARSER", "controlled"),
            generator_backend=os.getenv("MUSIC_EDIT_GENERATOR", "fake"),
            qwen_base_url=os.getenv("QWEN_BASE_URL", "").rstrip("/"),
            qwen_api_key=os.getenv("QWEN_API_KEY", ""),
            qwen_model=os.getenv("QWEN_MODEL", "Qwen3-4B-Instruct"),
            ace_step_base_url=os.getenv("ACE_STEP_BASE_URL", "").rstrip("/"),
            ace_step_cross_stem_attention=os.getenv("ACE_STEP_CROSS_STEM_ATTENTION", "1").lower()
            not in {"0", "false", "no", "off"},
        )
