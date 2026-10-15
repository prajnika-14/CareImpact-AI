"""Project paths and settings.

Settings are read from ``configs/settings.yaml``. Environment variables prefixed
with ``CAREIMPACT_`` override individual keys (e.g. ``CAREIMPACT_OLLAMA_MODEL``).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
DATA = ROOT / "data"


class Paths:
    raw = DATA / "raw"
    http_cache = DATA / "raw" / "http_cache"
    interim = DATA / "interim"
    documents = DATA / "interim" / "documents"
    pairs = DATA / "interim" / "pairs.jsonl"
    processed = DATA / "processed"
    diffs = DATA / "processed" / "diffs"
    training_real = DATA / "training" / "real"
    training_synthetic = DATA / "training" / "synthetic"
    training_review = DATA / "training" / "review"
    training_final = DATA / "training" / "final"
    evaluation = DATA / "evaluation"
    eval_results = DATA / "evaluation" / "results"
    reports = DATA / "reports"
    samples = DATA / "samples"
    models = ROOT / "models"
    runs = ROOT / "models" / "runs"
    frontend = ROOT / "frontend"

    @classmethod
    def ensure(cls) -> None:
        for name, value in vars(cls).items():
            if isinstance(value, Path) and not value.suffix:
                value.mkdir(parents=True, exist_ok=True)


DEFAULTS: dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:1.5b",
    "ollama_timeout_s": 300,
    "ollama_num_ctx": 4096,
    "http_min_interval_s": 1.5,
    "http_timeout_s": 40,
    "http_max_retries": 4,
    "user_agent": "CareImpactAI-research/0.1 (student project; polite crawler)",
    "accept_cms_api_license": False,
    "prompt_max_section_chars": 1800,
}


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    merged = dict(DEFAULTS)
    path = CONFIG_DIR / "settings.yaml"
    if path.exists():
        merged.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for key, default in DEFAULTS.items():
        env = os.environ.get(f"CAREIMPACT_{key.upper()}")
        if env is None:
            continue
        if isinstance(default, bool):
            merged[key] = env.lower() in {"1", "true", "yes"}
        elif isinstance(default, int):
            merged[key] = int(env)
        elif isinstance(default, float):
            merged[key] = float(env)
        else:
            merged[key] = env
    return merged
