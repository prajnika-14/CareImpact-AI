"""Minimal Ollama HTTP client (chat + model listing + memory probe)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from careimpact.config import settings
from careimpact.schemas import PolicyAnalysis


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        cfg = settings()
        self.base_url = (base_url or cfg["ollama_url"]).rstrip("/")
        self.timeout = timeout or cfg["ollama_timeout_s"]

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/version", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError:
            return []

    def loaded(self) -> list[dict[str, Any]]:
        """Models currently in memory with their resident size (from /api/ps)."""
        try:
            r = httpx.get(f"{self.base_url}/api/ps", timeout=5)
            r.raise_for_status()
            return r.json().get("models", [])
        except httpx.HTTPError:
            return []

    def chat(self, model: str, messages: list[dict], json_schema: bool = True, num_ctx: int | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": num_ctx or settings()["ollama_num_ctx"]},
        }
        if json_schema:
            # Structured outputs: Ollama constrains decoding to this JSON schema.
            payload["format"] = PolicyAnalysis.model_json_schema()
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        if r.status_code != 200:
            raise OllamaError(f"Ollama returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        return {
            "content": data.get("message", {}).get("content", ""),
            "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "load_s": (data.get("load_duration") or 0) / 1e9,
        }
