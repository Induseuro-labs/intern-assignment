"""Thin wrapper around the Anthropic SDK used by both questions.

Every AI-backed component in this repo is optional: when no API key is
configured, callers fall back to deterministic local implementations so the
whole project (tests, demos) runs offline.
"""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")


def available() -> bool:
    """True when hosted Claude calls should be attempted."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("DISABLE_LLM") != "1"


def ask_json(content, schema: dict, system: str = "", max_tokens: int = 8000) -> dict:
    """Send one request and return the JSON object that matches ``schema``.

    ``content`` is a string or a list of Messages API content blocks (text/image).
    Structured outputs guarantee the reply parses against ``schema``.
    """
    import anthropic  # imported lazily so the offline path needs no SDK

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system or anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        # Route safety-classifier refusals to a fallback model instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"model refused the request: {response.stop_details}")
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)
