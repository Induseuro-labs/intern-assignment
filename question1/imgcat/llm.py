"""Thin OpenRouter (OpenAI-compatible) chat client used for optional LLM steps.

Everything that calls an LLM has a deterministic fallback, so the system works
without a key; the LLM only improves natural-language understanding.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

DEFAULT_MODEL = os.environ.get("IMGCAT_LLM_MODEL", "anthropic/claude-haiku-5.5")


def _load_dotenv() -> None:
    for parent in Path(__file__).resolve().parents:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
            return


def openrouter_key() -> str | None:
    _load_dotenv()
    return os.environ.get("OPENROUTER_API_KEY") or None


def llm_available() -> bool:
    return os.environ.get("IMGCAT_OFFLINE") != "1" and openrouter_key() is not None


def chat_json(system: str, user: str, model: str = DEFAULT_MODEL, timeout: float = 60) -> dict:
    """Ask the model for a JSON object and parse it (tolerates ```json fences)."""
    key = openrouter_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(match.group(0))
