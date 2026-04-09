"""Experience replay buffer — stores and samples from past agent episodes."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Episode(BaseModel):
    """A complete record of one task execution."""

    id: str = ""
    task: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Execution trace
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # Outcome
    final_answer: str | None = None
    success: bool = False
    iterations: int = 0

    # Self-assessed scores
    self_reward: float = 0.0  # -1 to 1
    confidence: float = 0.0
    difficulty: float = 0.5  # 0-1, estimated task difficulty

    # Strategy metadata
    strategy_used: str = ""  # e.g. "sequential", "debate", "single_agent"
    agents_used: list[str] = Field(default_factory=list)

    # Lessons
    lessons: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)


class ExperienceReplayBuffer:
    """Persistent buffer of past episodes for learning.

    Supports:
    - Prioritized sampling (higher-reward episodes sampled more)
    - Similarity search (find episodes similar to a new task)
    - Failure analysis (find common failure patterns)
    - Persistence to disk
    """

    def __init__(self, max_size: int = 1000, persist_path: str = "./data/experience"):
        self.max_size = max_size
        self.persist_path = Path(persist_path)
        self.episodes: list[Episode] = []
        self._load()

    def _load(self) -> None:
        """Load episodes from disk."""
        self.persist_path.mkdir(parents=True, exist_ok=True)
        ep_file = self.persist_path / "episodes.jsonl"
        if ep_file.exists():
            try:
                for line in ep_file.read_text().strip().split("\n"):
                    if line:
                        self.episodes.append(Episode(**json.loads(line)))
                logger.info("Loaded %d episodes from disk", len(self.episodes))
            except Exception as e:
                logger.warning("Failed to load episodes: %s", e)

    def _save(self) -> None:
        """Persist all episodes to disk."""
        self.persist_path.mkdir(parents=True, exist_ok=True)
        ep_file = self.persist_path / "episodes.jsonl"
        lines = [ep.model_dump_json() for ep in self.episodes]
        ep_file.write_text("\n".join(lines))

    def add(self, episode: Episode) -> None:
        """Add an episode to the buffer."""
        if not episode.id:
            episode.id = f"ep_{len(self.episodes)}_{int(datetime.now(timezone.utc).timestamp())}"

        self.episodes.append(episode)

        # Evict oldest low-reward episodes if buffer is full
        if len(self.episodes) > self.max_size:
            self.episodes.sort(key=lambda e: e.self_reward, reverse=True)
            self.episodes = self.episodes[:self.max_size]

        self._save()
        logger.debug("Added episode %s (reward=%.2f)", episode.id, episode.self_reward)

    def sample(self, n: int = 5, prioritized: bool = True) -> list[Episode]:
        """Sample episodes from the buffer.

        If prioritized, higher-reward episodes are sampled more frequently.
        """
        if not self.episodes:
            return []

        n = min(n, len(self.episodes))

        if not prioritized:
            return random.sample(self.episodes, n)

        # Prioritized sampling: shift rewards to positive, use as weights
        min_reward = min(e.self_reward for e in self.episodes)
        weights = [e.self_reward - min_reward + 0.1 for e in self.episodes]
        return random.choices(self.episodes, weights=weights, k=n)

    def find_similar(self, task: str, n: int = 3) -> list[Episode]:
        """Find episodes with similar tasks (simple keyword overlap)."""
        task_words = set(task.lower().split())

        scored = []
        for ep in self.episodes:
            ep_words = set(ep.task.lower().split())
            overlap = len(task_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:n]]

    def get_successful(self, min_reward: float = 0.5) -> list[Episode]:
        """Get all episodes above a reward threshold."""
        return [e for e in self.episodes if e.self_reward >= min_reward]

    def get_failures(self) -> list[Episode]:
        """Get all failed episodes."""
        return [e for e in self.episodes if not e.success]

    def get_common_failure_modes(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Analyze the most common failure modes."""
        from collections import Counter
        modes: list[str] = []
        for ep in self.get_failures():
            modes.extend(ep.failure_modes)
        return Counter(modes).most_common(top_n)

    def get_strategy_stats(self) -> dict[str, dict[str, float]]:
        """Get success rate and avg reward per strategy."""
        from collections import defaultdict
        stats: dict[str, list[Episode]] = defaultdict(list)
        for ep in self.episodes:
            if ep.strategy_used:
                stats[ep.strategy_used].append(ep)

        result = {}
        for strategy, eps in stats.items():
            successes = sum(1 for e in eps if e.success)
            result[strategy] = {
                "count": len(eps),
                "success_rate": successes / len(eps) if eps else 0,
                "avg_reward": sum(e.self_reward for e in eps) / len(eps) if eps else 0,
            }
        return result

    def clear(self) -> None:
        self.episodes.clear()
        self._save()

    @property
    def size(self) -> int:
        return len(self.episodes)
