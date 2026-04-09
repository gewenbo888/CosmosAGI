"""Planner Agent — decomposes a high-level task into subtasks."""

from __future__ import annotations

import logging

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentPhase, AgentState, SubTask

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    name = "planner"
    system_prompt = """You are the Planner agent in an AGI system called CosmosAGI.
Your job is to decompose a user's task into clear, actionable subtasks.

Rules:
- Each subtask must be concrete and independently executable.
- Order subtasks by dependency (earlier ones first).
- Keep the number of subtasks reasonable (2-8 typically).
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "subtasks": [
    {"id": 1, "description": "..."},
    {"id": 2, "description": "..."}
  ],
  "reasoning": "Brief explanation of the plan."
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("Planner: decomposing task: %s", state.task[:100])

        context_parts = []
        if state.errors:
            context_parts.append(
                f"Previous errors to avoid:\n"
                + "\n".join(f"- {e}" for e in state.errors[-3:])
            )
        if state.reflections:
            last = state.reflections[-1]
            context_parts.append(
                f"Last reflection (success={last.success}): {last.reasoning}\n"
                f"Suggestions: {', '.join(last.suggestions)}"
            )

        prompt = f"Task: {state.task}"
        if context_parts:
            prompt += "\n\nContext:\n" + "\n".join(context_parts)

        result = self.call_llm_json(prompt, state)

        subtasks = [
            SubTask(id=st["id"], description=st["description"])
            for st in result.get("subtasks", [])
        ]

        if not subtasks:
            state.errors.append("Planner produced no subtasks.")
            state.phase = AgentPhase.ERROR
            return state

        state.subtasks = subtasks
        state.current_subtask_index = 0
        state.phase = AgentPhase.EXECUTE
        logger.info("Planner: created %d subtasks", len(subtasks))
        return state
