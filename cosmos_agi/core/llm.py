"""LLM interface — thin wrapper around litellm for provider-agnostic calls."""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from cosmos_agi.config.settings import LLMConfig

logger = logging.getLogger(__name__)


def completion(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    response_format: type | None = None,
    **kwargs: Any,
) -> str:
    """Call an LLM and return the text response."""
    if config is None:
        from cosmos_agi.config.settings import settings
        config = settings.llm

    model = config.model
    if config.provider.value == "ollama":
        model = f"ollama/{config.model}"

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **kwargs,
    }
    if config.api_key:
        params["api_key"] = config.api_key
    if config.base_url:
        params["api_base"] = config.base_url

    logger.debug("LLM call: model=%s messages=%d", model, len(messages))
    response = litellm.completion(**params)
    content = response.choices[0].message.content
    logger.debug("LLM response length: %d chars", len(content))
    return content


def _extract_json(raw: str) -> dict:
    """Extract JSON from LLM output, handling common wrappers.

    Handles:
    - Raw JSON
    - Markdown code fences (```json ... ```)
    - DeepSeek-style <think>...</think> preamble
    - Leading/trailing prose around a JSON object
    """
    import re

    text = raw.strip()

    # Strip <think>...</think> blocks (DeepSeek R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } or [ ... ] block via brace matching
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)


def completion_json(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> dict:
    """Call LLM and parse the response as JSON."""
    raw = completion(messages, config, **kwargs)
    return _extract_json(raw)
