from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:e2b"
    resume_path: str = "files/Resume_Basavaraj_Jakkannavar_v1.pdf"
    min_match_score: int = 70
    apply_mode: str = "dry_run"  # dry_run | assisted | auto
    linkedin_email: str = ""
    linkedin_password: str = ""
    naukri_email: str = ""
    naukri_password: str = ""
    playwright_headless: bool = True
    playwright_slow_mo_ms: int = 100


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_resume_path(env: EnvSettings) -> Path:
    p = Path(env.resume_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
