"""Self-reward mechanism — the agent evaluates its own performance.

Inspired by RLHF but using the agent's own LLM as the reward model.
The agent scores its output on multiple dimensions, producing a composite
reward signal that feeds into the experience buffer and strategy evolution.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from cosmos_agi.core.state import AgentState

logger = logging.getLogger(__name__)


class RewardDimension(BaseModel):
    """A single scored dimension of performance."""

    name: str
    score: float  # -1 to 1
    weight: float = 1.0
    reasoning: str = ""


class RewardAssessment(BaseModel):
    """Complete self-assessment of an episode."""

    dimensions: list[RewardDimension] = Field(default_factory=list)
    composite_reward: float = 0.0
    lessons_learned: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    difficulty_estimate: float = 0.5


# Default evaluation dimensions
DEFAULT_DIMENSIONS = [
    ("correctness", 2.0, "Was the answer factually correct and logically sound?"),
    ("completeness", 1.5, "Did the answer fully address all aspects of the task?"),
    ("efficiency", 1.0, "Was the solution achieved with minimal wasted effort?"),
    ("clarity", 1.0, "Was the output clear, well-structured, and easy to understand?"),
    ("safety", 1.5, "Was the output safe, ethical, and aligned with human values?"),
]


class SelfRewardModel:
    """Uses the LLM to self-assess performance across multiple dimensions."""

    SYSTEM_PROMPT = """You are a self-evaluation model inside CosmosAGI.
Your job is to honestly assess the quality of an agent's work on a task.

Be rigorous and honest — do not inflate scores. A mediocre result should
get a mediocre score. Identify what went wrong and what could improve.

Score each dimension from -1.0 (terrible) to 1.0 (excellent).
0.0 = acceptable but unremarkable.

Output ONLY valid JSON:
{
  "dimensions": [
    {"name": "...", "score": 0.0, "reasoning": "..."}
  ],
  "lessons_learned": ["lesson 1"],
  "failure_modes": ["failure 1"],
  "improvement_suggestions": ["suggestion 1"],
  "difficulty_estimate": 0.5
}"""

    def __init__(self, dimensions: list[tuple[str, float, str]] | None = None):
        self.dimensions = dimensions or DEFAULT_DIMENSIONS

    def assess(self, state: AgentState) -> RewardAssessment:
        """Generate a self-assessment for a completed task."""
        from cosmos_agi.core.llm import completion_json

        dim_descriptions = "\n".join(
            f"  - {name} (weight={weight}): {desc}"
            for name, weight, desc in self.dimensions
        )

        observations_text = "\n".join(f"  - {o}" for o in state.observations[-10:])
        errors_text = "\n".join(f"  - {e}" for e in state.errors) if state.errors else "  None"

        prompt = f"""Task: {state.task}

Final answer: {state.final_answer or 'No answer produced'}

Observations:
{observations_text}

Errors:
{errors_text}

Iterations used: {state.iteration}/{state.max_iterations}

Evaluate on these dimensions:
{dim_descriptions}"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = completion_json(messages)
            return self._parse_assessment(result)
        except Exception as e:
            logger.error("Self-reward assessment failed: %s", e)
            return self._fallback_assessment(state)

    def _parse_assessment(self, result: dict[str, Any]) -> RewardAssessment:
        """Parse LLM output into a RewardAssessment."""
        dim_weights = {name: weight for name, weight, _ in self.dimensions}

        dimensions = []
        for dim_data in result.get("dimensions", []):
            name = dim_data.get("name", "unknown")
            score = max(-1.0, min(1.0, float(dim_data.get("score", 0))))
            dimensions.append(RewardDimension(
                name=name,
                score=score,
                weight=dim_weights.get(name, 1.0),
                reasoning=dim_data.get("reasoning", ""),
            ))

        # Compute weighted composite reward
        total_weight = sum(d.weight for d in dimensions) or 1.0
        composite = sum(d.score * d.weight for d in dimensions) / total_weight

        return RewardAssessment(
            dimensions=dimensions,
            composite_reward=max(-1.0, min(1.0, composite)),
            lessons_learned=result.get("lessons_learned", []),
            failure_modes=result.get("failure_modes", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            difficulty_estimate=result.get("difficulty_estimate", 0.5),
        )

    def _fallback_assessment(self, state: AgentState) -> RewardAssessment:
        """Heuristic assessment when LLM call fails."""
        has_answer = state.final_answer is not None
        has_errors = len(state.errors) > 0
        used_few_iterations = state.iteration <= state.max_iterations // 2

        score = 0.0
        if has_answer:
            score += 0.4
        if not has_errors:
            score += 0.3
        if used_few_iterations:
            score += 0.2

        return RewardAssessment(
            dimensions=[
                RewardDimension(name="heuristic", score=score, reasoning="Fallback assessment"),
            ],
            composite_reward=score,
            failure_modes=["llm_self_reward_failed"] if has_errors else [],
        )

    def compute_reward_from_scores(self, scores: dict[str, float]) -> float:
        """Compute composite reward from pre-computed dimension scores."""
        dim_weights = {name: weight for name, weight, _ in self.dimensions}
        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in scores.items():
            w = dim_weights.get(name, 1.0)
            weighted_sum += score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0
