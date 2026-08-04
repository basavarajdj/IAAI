from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _build_payload(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

    def _post(self, payload: dict[str, Any]) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return data.get("message", {}).get("content", "").strip()

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = self._build_payload(prompt, system=system)
        return self._post(payload)

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
        images_b64 = [
            base64.b64encode(path.read_bytes()).decode("ascii")
            for path in image_paths
        ]
        payload = self._build_payload(prompt, system=system, images=images_b64, temperature=0.1)
        return self._post(payload)


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
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
