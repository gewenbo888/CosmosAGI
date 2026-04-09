"""Strategy evolution — meta-learning from past experiences.

Analyzes successful and failed episodes to evolve the agent's approach:
- Which strategies work for which types of tasks?
- What failure patterns keep recurring?
- How should the agent adapt its behavior?

This is a simplified form of reinforcement learning where the "policy"
is expressed as natural-language strategy guidelines that are injected
into the agent's prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cosmos_agi.evaluation.experience import Episode, ExperienceReplayBuffer

logger = logging.getLogger(__name__)


class Strategy(BaseModel):
    """A named strategy with performance statistics."""

    name: str
    description: str
    guidelines: list[str] = Field(default_factory=list)
    applicable_task_types: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    avg_reward: float = 0.0
    active: bool = True

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count


class StrategyEvolver:
    """Evolves agent strategies based on accumulated experience."""

    def __init__(
        self,
        experience_buffer: ExperienceReplayBuffer | None = None,
        persist_path: str = "./data/strategies",
    ):
        self.buffer = experience_buffer or ExperienceReplayBuffer()
        self.persist_path = Path(persist_path)
        self.strategies: dict[str, Strategy] = {}
        self._load()

    def _load(self) -> None:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        strat_file = self.persist_path / "strategies.json"
        if strat_file.exists():
            try:
                data = json.loads(strat_file.read_text())
                for s in data:
                    strat = Strategy(**s)
                    self.strategies[strat.name] = strat
                logger.info("Loaded %d strategies", len(self.strategies))
            except Exception as e:
                logger.warning("Failed to load strategies: %s", e)

    def _save(self) -> None:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        strat_file = self.persist_path / "strategies.json"
        data = [s.model_dump() for s in self.strategies.values()]
        strat_file.write_text(json.dumps(data, indent=2))

    def update_from_episode(self, episode: Episode) -> None:
        """Update strategy stats based on a new episode."""
        strategy_name = episode.strategy_used or "default"

        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = Strategy(
                name=strategy_name,
                description=f"Auto-created strategy: {strategy_name}",
            )

        strat = self.strategies[strategy_name]
        if episode.success:
            strat.success_count += 1
        else:
            strat.failure_count += 1

        # Running average of reward
        n = strat.total_uses
        strat.avg_reward = (strat.avg_reward * (n - 1) + episode.self_reward) / n

        # Add lessons as guidelines
        for lesson in episode.lessons:
            if lesson not in strat.guidelines:
                strat.guidelines.append(lesson)

        self._save()

    def evolve_strategies(self) -> list[str]:
        """Analyze experience and generate strategy improvements.

        Returns a list of evolution actions taken.
        """
        actions = []

        # Deactivate consistently failing strategies
        for name, strat in self.strategies.items():
            if strat.total_uses >= 5 and strat.success_rate < 0.2:
                strat.active = False
                actions.append(f"Deactivated '{name}' (success_rate={strat.success_rate:.2f})")
                logger.info("Deactivated strategy '%s': too many failures", name)

        # Analyze common failures and create guidelines
        common_failures = self.buffer.get_common_failure_modes(top_n=5)
        for failure_mode, count in common_failures:
            if count >= 3:
                guideline = f"AVOID: {failure_mode} (occurred {count} times)"
                # Add to all active strategies
                for strat in self.strategies.values():
                    if strat.active and guideline not in strat.guidelines:
                        strat.guidelines.append(guideline)
                        actions.append(f"Added guideline to '{strat.name}': {guideline}")

        # Use LLM to generate deeper insights if we have enough data
        if self.buffer.size >= 10:
            try:
                insights = self._llm_analyze_patterns()
                actions.extend(insights)
            except Exception as e:
                logger.warning("LLM strategy analysis failed: %s", e)

        self._save()
        return actions

    def _llm_analyze_patterns(self) -> list[str]:
        """Use LLM to find patterns in successful vs failed episodes."""
        from cosmos_agi.core.llm import completion_json

        successes = self.buffer.get_successful(min_reward=0.5)[:5]
        failures = self.buffer.get_failures()[:5]

        success_summaries = "\n".join(
            f"  - Task: {e.task[:80]}, Strategy: {e.strategy_used}, Reward: {e.self_reward:.2f}"
            for e in successes
        )
        failure_summaries = "\n".join(
            f"  - Task: {e.task[:80]}, Strategy: {e.strategy_used}, Failures: {e.failure_modes}"
            for e in failures
        )

        messages = [
            {"role": "system", "content": """You analyze agent performance patterns.
Output ONLY valid JSON:
{
  "patterns": ["pattern 1", "pattern 2"],
  "new_guidelines": ["guideline 1"],
  "strategy_suggestions": [
    {"name": "...", "description": "...", "applicable_to": ["task_type"]}
  ]
}"""},
            {"role": "user", "content": f"""Analyze these agent episodes:

Successful episodes:
{success_summaries or '  None yet'}

Failed episodes:
{failure_summaries or '  None yet'}

Current strategies: {list(self.strategies.keys())}

What patterns do you see? What new strategies or guidelines would help?"""},
        ]

        result = completion_json(messages)
        actions = []

        for guideline in result.get("new_guidelines", []):
            for strat in self.strategies.values():
                if strat.active and guideline not in strat.guidelines:
                    strat.guidelines.append(guideline)
            actions.append(f"New guideline from analysis: {guideline}")

        for suggestion in result.get("strategy_suggestions", []):
            name = suggestion.get("name", "")
            if name and name not in self.strategies:
                self.strategies[name] = Strategy(
                    name=name,
                    description=suggestion.get("description", ""),
                    applicable_task_types=suggestion.get("applicable_to", []),
                )
                actions.append(f"Created new strategy: {name}")

        return actions

    def get_best_strategy(self, task: str) -> Strategy | None:
        """Select the best strategy for a task based on past performance."""
        active = [s for s in self.strategies.values() if s.active]
        if not active:
            return None

        # Check if any strategy's applicable_task_types match
        task_lower = task.lower()
        matched = [
            s for s in active
            if any(t.lower() in task_lower for t in s.applicable_task_types)
        ]

        candidates = matched if matched else active

        # Sort by success rate (with minimum uses) then by avg reward
        candidates.sort(
            key=lambda s: (
                s.success_rate if s.total_uses >= 3 else 0,
                s.avg_reward,
            ),
            reverse=True,
        )

        return candidates[0] if candidates else None

    def get_guidelines_for_task(self, task: str) -> list[str]:
        """Get all applicable guidelines for a task."""
        strategy = self.get_best_strategy(task)
        if strategy:
            return strategy.guidelines
        # Fall back to all active strategy guidelines
        guidelines = []
        for strat in self.strategies.values():
            if strat.active:
                guidelines.extend(strat.guidelines)
        return list(set(guidelines))
