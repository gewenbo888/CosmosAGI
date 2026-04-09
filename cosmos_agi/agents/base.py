"""Base agent class — all specialized agents inherit from this."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from cosmos_agi.core.llm import completion, completion_json
from cosmos_agi.core.state import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all CosmosAGI agents."""

    name: str = "base"
    system_prompt: str = "You are a helpful AI assistant."

    def _build_messages(
        self,
        user_prompt: str,
        state: AgentState,
        extra_context: str = "",
    ) -> list[dict[str, str]]:
        """Build the message list for an LLM call."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if state.relevant_memories:
            mem_text = "\n".join(f"- {m}" for m in state.relevant_memories)
            messages.append({
                "role": "system",
                "content": f"Relevant memories:\n{mem_text}",
            })

        if extra_context:
            messages.append({"role": "system", "content": extra_context})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def call_llm(self, prompt: str, state: AgentState, **kwargs: Any) -> str:
        messages = self._build_messages(prompt, state)
        return completion(messages, **kwargs)

    def call_llm_json(self, prompt: str, state: AgentState, **kwargs: Any) -> dict:
        messages = self._build_messages(prompt, state)
        return completion_json(messages, **kwargs)

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute this agent's logic and return updated state."""
        ...
