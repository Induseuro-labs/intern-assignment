"""OpenRouter (OpenAI-compatible) chat client.

Set OPENROUTER_API_KEY in the environment or a .env file at the repo root.
QGEN_OFFLINE=1 disables every LLM call (rules / review queue take over).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx

GEN_LLM = os.environ.get("QGEN_LLM_MODEL", "anthropic/claude-sonnet-5.5")


def _load_dotenv() -> None:
    for parent in Path(__file__).resolve().parents:
        env = parent / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
            return


def llm_available() -> bool:
    _load_dotenv()
    return os.environ.get("QGEN_OFFLINE") != "1" and bool(os.environ.get("OPENROUTER_API_KEY"))


def chat(messages: list[dict], model: str = GEN_LLM, temperature: float = 0.0, timeout: float = 120) -> str:
    _load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    for attempt in range(3):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "temperature": temperature, "messages": messages},
                           timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_json(text: str):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError(f"no JSON in LLM output: {text[:200]}")
    return json.loads(m.group(1))


def chat_json(system: str, user: str, model: str = GEN_LLM, temperature: float = 0.0):
    return parse_json(chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           model, temperature))
