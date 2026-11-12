from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class LLM:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.model = cfg["model"]
        self.key = cfg["key"]
        self.url = cfg["url"].rstrip("/")
        self.temperature = float(cfg.get("temperature"))
        self.timeout = float(cfg.get("timeout", 600))

    @classmethod
    def from_file(cls, path, profile: Optional[str] = None) -> "LLM":
        payload = json.loads(path.read_text())
        if profile is None:
            # 本机 key.json 的 profile 名是 mimo_code_plan，不是 default。
            # 优先 "default"；没有就退回唯一那一个 profile。
            if "default" in payload:
                profile = "default"
            elif len(payload) == 1:
                profile = next(iter(payload))
            else:
                raise KeyError(
                    f"key.json has no 'default' profile and {len(payload)} "
                    f"candidates {list(payload)}; pass profile= explicitly"
                )
        return cls(payload[profile])

    def chat(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        return _extract_json(self.chat(system, user))


def _extract_json(text: str) -> Dict[str, Any]:
    """Tolerate ```json fences and leading prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:400]}")
    return json.loads(s[start : end + 1])
