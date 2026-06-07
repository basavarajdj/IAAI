from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx


class OllamaClient:
    """Local LLM via Ollama API."""

    def __init__(self, host: str, model: str, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return data.get("message", {}).get("content", "").strip()

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        raw = self.generate(prompt, system=system)
        return _extract_json(raw)

    def generate_from_images(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        system: str | None = None,
    ) -> str:
        images_b64: list[str] = []
        for path in image_paths:
            images_b64.append(base64.b64encode(path.read_bytes()).decode("ascii"))

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return data.get("message", {}).get("content", "").strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
