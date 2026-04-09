"""Critic Agent — evaluates execution results and provides reflection."""

from __future__ import annotations

import logging

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentPhase, AgentState, Reflection

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    name = "critic"
    system_prompt = """You are the Critic agent in CosmosAGI.
Your job is to evaluate whether the executed subtasks successfully address the original task.

Rules:
- Be objective and rigorous.
- Identify what worked, what failed, and why.
- Provide concrete suggestions for improvement if needed.
- Rate your confidence in the overall result (0.0 to 1.0).
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "success": true/false,
  "reasoning": "Detailed evaluation",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "confidence": 0.85,
  "final_answer": "Synthesized answer if success is true, otherwise null"
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("Critic: evaluating %d observations", len(state.observations))

        observations_text = "\n".join(
            f"- {obs}" for obs in state.observations
        )
        errors_text = "\n".join(f"- {e}" for e in state.errors) if state.errors else "None"

        subtask_summary = "\n".join(
            f"  [{st.id}] {st.description} → status={st.status}, result={st.result}"
            for st in state.subtasks
        )

        prompt = f"""Original task: {state.task}

Subtasks and results:
{subtask_summary}

Observations:
{observations_text}

Errors:
{errors_text}

Iteration: {state.iteration}/{state.max_iterations}

Evaluate: Did we successfully complete the task? What is the final answer?"""

        try:
            result = self.call_llm_json(prompt, state)

            reflection = Reflection(
                success=result.get("success", False),
                reasoning=result.get("reasoning", "No reasoning provided"),
                suggestions=result.get("suggestions", []),
                confidence=result.get("confidence", 0.0),
            )
            state.reflections.append(reflection)

            if reflection.success and reflection.confidence >= 0.7:
                state.final_answer = result.get("final_answer", reflection.reasoning)
                state.phase = AgentPhase.COMPLETE
                logger.info("Critic: task COMPLETE (confidence=%.2f)", reflection.confidence)
            elif state.iteration >= state.max_iterations:
                state.final_answer = (
                    f"Max iterations reached. Best result: {reflection.reasoning}"
                )
                state.phase = AgentPhase.COMPLETE
                logger.warning("Critic: max iterations reached")
            else:
                state.phase = AgentPhase.IMPROVE
                logger.info(
                    "Critic: needs improvement (confidence=%.2f)",
                    reflection.confidence,
                )

        except Exception as e:
            state.errors.append(f"Critic error: {e}")
            state.phase = AgentPhase.IMPROVE
            logger.error("Critic failed: %s", e)

        return state
