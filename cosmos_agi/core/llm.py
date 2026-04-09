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


def completion_json(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> dict:
    """Call LLM and parse the response as JSON."""
    raw = completion(messages, config, **kwargs)

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)
