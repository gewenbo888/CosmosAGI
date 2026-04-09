"""Executor Agent — carries out individual subtasks using tools and reasoning."""

from __future__ import annotations

import logging

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentPhase, AgentState

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    name = "executor"
    system_prompt = """You are the Executor agent in CosmosAGI.
Your job is to execute a specific subtask and produce a clear result.

Rules:
- Focus on the given subtask only.
- If you need a tool, describe which tool and what arguments (the orchestrator will handle actual tool calls).
- Think step by step.
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "thought": "Your chain-of-thought reasoning",
  "action": "tool_name" or "direct_answer",
  "action_input": "input for the tool or your direct answer",
  "result": "The final result of this subtask"
}"""

    def run(self, state: AgentState) -> AgentState:
        if state.current_subtask_index >= len(state.subtasks):
            state.phase = AgentPhase.REFLECT
            return state

        subtask = state.subtasks[state.current_subtask_index]
        subtask.status = "in_progress"
        subtask.attempts += 1
        logger.info("Executor: working on subtask %d: %s", subtask.id, subtask.description[:80])

        context = f"Main task: {state.task}\n\n"
        if state.current_subtask_index > 0:
            completed = [
                st for st in state.subtasks[:state.current_subtask_index]
                if st.status == "done"
            ]
            if completed:
                context += "Completed subtasks:\n"
                for st in completed:
                    context += f"  - [{st.id}] {st.description}: {st.result}\n"
                context += "\n"

        prompt = f"{context}Current subtask: {subtask.description}"

        try:
            result = self.call_llm_json(prompt, state)
            raw_result = result.get("result", result)
            subtask.result = raw_result if isinstance(raw_result, str) else str(raw_result)
            subtask.status = "done"

            state.observations.append(
                f"Subtask {subtask.id}: {subtask.result}"
            )
            state.tool_calls.append({
                "subtask_id": subtask.id,
                "action": result.get("action", "unknown"),
                "action_input": result.get("action_input", ""),
            })

            logger.info("Executor: subtask %d completed", subtask.id)

            # Move to next subtask or to reflect
            state.current_subtask_index += 1
            if state.current_subtask_index >= len(state.subtasks):
                state.phase = AgentPhase.OBSERVE
            # else stay in EXECUTE

        except Exception as e:
            subtask.status = "failed"
            error_msg = f"Subtask {subtask.id} failed: {e}"
            state.errors.append(error_msg)
            logger.error(error_msg)
            state.phase = AgentPhase.REFLECT

        return state
