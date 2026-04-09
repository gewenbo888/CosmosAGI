"""Tests for experience replay buffer."""

import shutil
from pathlib import Path

from cosmos_agi.evaluation.experience import Episode, ExperienceReplayBuffer

TEST_PATH = "/tmp/cosmos_agi_test_experience"


def _clean():
    shutil.rmtree(TEST_PATH, ignore_errors=True)


def _make_episode(task: str, success: bool = True, reward: float = 0.5, strategy: str = "default") -> Episode:
    return Episode(
        task=task,
        success=success,
        self_reward=reward,
        strategy_used=strategy,
        final_answer="result" if success else None,
        lessons=["lesson_1"] if success else [],
        failure_modes=["timeout"] if not success else [],
    )


class TestExperienceReplayBuffer:
    def setup_method(self):
        _clean()
        self.buffer = ExperienceReplayBuffer(max_size=50, persist_path=TEST_PATH)

    def teardown_method(self):
        _clean()

    def test_add_and_size(self):
        self.buffer.add(_make_episode("task 1"))
        self.buffer.add(_make_episode("task 2"))
        assert self.buffer.size == 2

    def test_persistence(self):
        self.buffer.add(_make_episode("persistent task", reward=0.8))
        # Create new buffer pointing to same path
        buffer2 = ExperienceReplayBuffer(persist_path=TEST_PATH)
        assert buffer2.size == 1
        assert buffer2.episodes[0].task == "persistent task"

    def test_eviction_on_overflow(self):
        buf = ExperienceReplayBuffer(max_size=3, persist_path=TEST_PATH)
        buf.add(_make_episode("low", reward=-0.5))
        buf.add(_make_episode("mid", reward=0.3))
        buf.add(_make_episode("high", reward=0.9))
        buf.add(_make_episode("very_high", reward=1.0))
        assert buf.size == 3
        # Lowest reward should have been evicted
        tasks = [e.task for e in buf.episodes]
        assert "low" not in tasks

    def test_sample_basic(self):
        for i in range(10):
            self.buffer.add(_make_episode(f"task_{i}", reward=i * 0.1))
        samples = self.buffer.sample(3, prioritized=False)
        assert len(samples) == 3

    def test_sample_prioritized(self):
        for i in range(20):
            self.buffer.add(_make_episode(f"task_{i}", reward=i * 0.05))
        samples = self.buffer.sample(5, prioritized=True)
        assert len(samples) == 5
        # Higher reward episodes should appear more often in aggregate
        # (statistical, so just check it runs without error)

    def test_find_similar(self):
        self.buffer.add(_make_episode("write python code for sorting"))
        self.buffer.add(_make_episode("plan a marketing campaign"))
        self.buffer.add(_make_episode("write python function for search"))
        similar = self.buffer.find_similar("write python code")
        assert len(similar) >= 2
        assert any("sorting" in e.task for e in similar)

    def test_get_successful(self):
        self.buffer.add(_make_episode("good", success=True, reward=0.8))
        self.buffer.add(_make_episode("bad", success=False, reward=-0.3))
        successful = self.buffer.get_successful(min_reward=0.5)
        assert len(successful) == 1
        assert successful[0].task == "good"

    def test_get_failures(self):
        self.buffer.add(_make_episode("ok", success=True))
        self.buffer.add(_make_episode("fail1", success=False))
        self.buffer.add(_make_episode("fail2", success=False))
        failures = self.buffer.get_failures()
        assert len(failures) == 2

    def test_common_failure_modes(self):
        for _ in range(3):
            self.buffer.add(_make_episode("f", success=False))
        modes = self.buffer.get_common_failure_modes()
        assert modes[0] == ("timeout", 3)

    def test_strategy_stats(self):
        self.buffer.add(_make_episode("t1", success=True, reward=0.8, strategy="single"))
        self.buffer.add(_make_episode("t2", success=True, reward=0.9, strategy="single"))
        self.buffer.add(_make_episode("t3", success=False, reward=-0.2, strategy="team"))
        stats = self.buffer.get_strategy_stats()
        assert stats["single"]["success_rate"] == 1.0
        assert stats["team"]["success_rate"] == 0.0

    def test_clear(self):
        self.buffer.add(_make_episode("task"))
        self.buffer.clear()
        assert self.buffer.size == 0


class TestEpisode:
    def test_defaults(self):
        ep = Episode()
        assert ep.self_reward == 0.0
        assert ep.success is False
        assert ep.lessons == []

    def test_serialization(self):
        ep = _make_episode("test", success=True, reward=0.7)
        data = ep.model_dump()
        restored = Episode(**data)
        assert restored.task == "test"
        assert restored.self_reward == 0.7
