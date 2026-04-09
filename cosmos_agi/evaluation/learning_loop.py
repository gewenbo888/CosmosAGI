"""Learning loop — connects experience, self-reward, and strategy evolution
into the agent's execution pipeline.

After each task completion:
1. Self-assess the result (SelfRewardModel)
2. Record the episode (ExperienceReplayBuffer)
3. Update strategy statistics (StrategyEvolver)
4. Periodically evolve strategies (every N episodes)
5. Inject learned guidelines into future agent prompts
"""

from __future__ import annotations

import logging
from typing import Any

from cosmos_agi.core.state import AgentState
from cosmos_agi.evaluation.experience import Episode, ExperienceReplayBuffer
from cosmos_agi.evaluation.self_reward import RewardAssessment, SelfRewardModel
from cosmos_agi.evaluation.strategy_evolution import StrategyEvolver

logger = logging.getLogger(__name__)


class LearningLoop:
    """Orchestrates the self-improvement cycle."""

    def __init__(
        self,
        experience_path: str = "./data/experience",
        strategy_path: str = "./data/strategies",
        evolve_every_n: int = 5,
    ):
        self.buffer = ExperienceReplayBuffer(persist_path=experience_path)
        self.reward_model = SelfRewardModel()
        self.evolver = StrategyEvolver(
            experience_buffer=self.buffer,
            persist_path=strategy_path,
        )
        self.evolve_every_n = evolve_every_n
        self._episode_counter = 0

    def post_task_learning(
        self,
        state: AgentState,
        strategy_used: str = "default",
        agents_used: list[str] | None = None,
        skip_llm_assessment: bool = False,
    ) -> Episode:
        """Run the full learning cycle after a task completes.

        Args:
            state: Final agent state after task execution.
            strategy_used: Name of the strategy that was used.
            agents_used: List of agent names that participated.
            skip_llm_assessment: If True, use heuristic reward (for testing).

        Returns:
            The recorded Episode with self-assessment.
        """
        logger.info("Learning loop: post-task analysis starting")

        # Step 1: Self-assess
        if skip_llm_assessment:
            assessment = self.reward_model._fallback_assessment(state)
        else:
            assessment = self.reward_model.assess(state)

        logger.info(
            "Self-reward: %.2f (dimensions=%d, lessons=%d)",
            assessment.composite_reward,
            len(assessment.dimensions),
            len(assessment.lessons_learned),
        )

        # Step 2: Create and record episode
        episode = Episode(
            task=state.task,
            subtasks=[st.model_dump() for st in state.subtasks],
            observations=state.observations,
            tool_calls=state.tool_calls,
            errors=state.errors,
            final_answer=state.final_answer,
            success=state.final_answer is not None and len(state.errors) == 0,
            iterations=state.iteration,
            self_reward=assessment.composite_reward,
            confidence=state.reflections[-1].confidence if state.reflections else 0.0,
            difficulty=assessment.difficulty_estimate,
            strategy_used=strategy_used,
            agents_used=agents_used or [],
            lessons=assessment.lessons_learned,
            failure_modes=assessment.failure_modes,
        )

        self.buffer.add(episode)

        # Step 3: Update strategy stats
        self.evolver.update_from_episode(episode)

        # Step 4: Periodically evolve strategies
        self._episode_counter += 1
        if self._episode_counter % self.evolve_every_n == 0:
            logger.info("Learning loop: triggering strategy evolution")
            actions = self.evolver.evolve_strategies()
            for action in actions:
                logger.info("Strategy evolution: %s", action)

        return episode

    def get_context_for_task(self, task: str) -> dict[str, Any]:
        """Get learning-informed context to inject into a new task.

        Returns guidelines, similar past episodes, and strategy recommendation.
        """
        # Get applicable guidelines
        guidelines = self.evolver.get_guidelines_for_task(task)

        # Find similar past episodes
        similar = self.buffer.find_similar(task, n=3)
        similar_summaries = [
            {
                "task": ep.task[:100],
                "success": ep.success,
                "reward": ep.self_reward,
                "strategy": ep.strategy_used,
                "lessons": ep.lessons[:3],
            }
            for ep in similar
        ]

        # Get recommended strategy
        best_strategy = self.evolver.get_best_strategy(task)

        return {
            "guidelines": guidelines,
            "similar_episodes": similar_summaries,
            "recommended_strategy": best_strategy.name if best_strategy else "default",
            "strategy_stats": self.evolver.buffer.get_strategy_stats(),
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance statistics."""
        return {
            "total_episodes": self.buffer.size,
            "success_rate": (
                len(self.buffer.get_successful()) / self.buffer.size
                if self.buffer.size > 0
                else 0.0
            ),
            "common_failures": self.buffer.get_common_failure_modes(),
            "strategy_stats": self.buffer.get_strategy_stats(),
            "active_strategies": [
                {"name": s.name, "success_rate": s.success_rate, "uses": s.total_uses}
                for s in self.evolver.strategies.values()
                if s.active
            ],
        }
