"""Tests for strategy evolution."""

import shutil

from cosmos_agi.evaluation.experience import Episode, ExperienceReplayBuffer
from cosmos_agi.evaluation.strategy_evolution import Strategy, StrategyEvolver

TEST_EXP_PATH = "/tmp/cosmos_agi_test_strat_exp"
TEST_STRAT_PATH = "/tmp/cosmos_agi_test_strategies"


def _clean():
    shutil.rmtree(TEST_EXP_PATH, ignore_errors=True)
    shutil.rmtree(TEST_STRAT_PATH, ignore_errors=True)


class TestStrategy:
    def test_success_rate_empty(self):
        s = Strategy(name="test", description="test")
        assert s.success_rate == 0.0
        assert s.total_uses == 0

    def test_success_rate(self):
        s = Strategy(name="test", description="test", success_count=7, failure_count=3)
        assert s.success_rate == 0.7
        assert s.total_uses == 10

    def test_active_default(self):
        s = Strategy(name="test", description="test")
        assert s.active is True


class TestStrategyEvolver:
    def setup_method(self):
        _clean()
        self.buffer = ExperienceReplayBuffer(persist_path=TEST_EXP_PATH)
        self.evolver = StrategyEvolver(
            experience_buffer=self.buffer,
            persist_path=TEST_STRAT_PATH,
        )

    def teardown_method(self):
        _clean()

    def test_update_from_success(self):
        ep = Episode(
            task="test",
            success=True,
            self_reward=0.8,
            strategy_used="approach_a",
            lessons=["be thorough"],
        )
        self.evolver.update_from_episode(ep)
        strat = self.evolver.strategies["approach_a"]
        assert strat.success_count == 1
        assert strat.failure_count == 0
        assert strat.avg_reward == 0.8
        assert "be thorough" in strat.guidelines

    def test_update_from_failure(self):
        ep = Episode(
            task="test",
            success=False,
            self_reward=-0.3,
            strategy_used="approach_b",
        )
        self.evolver.update_from_episode(ep)
        strat = self.evolver.strategies["approach_b"]
        assert strat.failure_count == 1
        assert strat.avg_reward == -0.3

    def test_running_average_reward(self):
        for reward in [0.6, 0.8, 1.0]:
            ep = Episode(task="t", success=True, self_reward=reward, strategy_used="s")
            self.evolver.update_from_episode(ep)
        strat = self.evolver.strategies["s"]
        assert abs(strat.avg_reward - 0.8) < 0.01  # mean of [0.6, 0.8, 1.0]

    def test_deactivate_failing_strategy(self):
        for _ in range(6):
            ep = Episode(task="t", success=False, self_reward=-0.5, strategy_used="bad_strategy")
            self.buffer.add(ep)
            self.evolver.update_from_episode(ep)

        actions = self.evolver.evolve_strategies()
        strat = self.evolver.strategies["bad_strategy"]
        assert not strat.active
        assert any("Deactivated" in a for a in actions)

    def test_common_failure_guidelines(self):
        for _ in range(4):
            ep = Episode(
                task="t", success=False, self_reward=-0.2,
                strategy_used="x", failure_modes=["timeout"],
            )
            self.buffer.add(ep)
            self.evolver.update_from_episode(ep)

        actions = self.evolver.evolve_strategies()
        strat = self.evolver.strategies["x"]
        has_avoid = any("AVOID" in g and "timeout" in g for g in strat.guidelines)
        assert has_avoid

    def test_get_best_strategy(self):
        # Create two strategies with different success rates
        for _ in range(5):
            self.evolver.update_from_episode(Episode(
                task="t", success=True, self_reward=0.9, strategy_used="good",
            ))
        for _ in range(5):
            self.evolver.update_from_episode(Episode(
                task="t", success=False, self_reward=-0.3, strategy_used="bad",
            ))

        best = self.evolver.get_best_strategy("any task")
        assert best is not None
        assert best.name == "good"

    def test_get_guidelines_for_task(self):
        ep = Episode(
            task="code review", success=True, self_reward=0.8,
            strategy_used="careful", lessons=["check edge cases"],
        )
        self.evolver.update_from_episode(ep)
        guidelines = self.evolver.get_guidelines_for_task("code review")
        assert "check edge cases" in guidelines

    def test_persistence(self):
        self.evolver.update_from_episode(Episode(
            task="t", success=True, self_reward=0.7,
            strategy_used="persistent_strat", lessons=["saved"],
        ))
        # Create new evolver from same path
        evolver2 = StrategyEvolver(
            experience_buffer=self.buffer,
            persist_path=TEST_STRAT_PATH,
        )
        assert "persistent_strat" in evolver2.strategies
        assert "saved" in evolver2.strategies["persistent_strat"].guidelines

    def test_applicable_task_types(self):
        self.evolver.strategies["coding"] = Strategy(
            name="coding",
            description="For coding tasks",
            applicable_task_types=["code", "python", "function"],
            success_count=5,
            failure_count=1,
            avg_reward=0.8,
        )
        self.evolver.strategies["general"] = Strategy(
            name="general",
            description="General purpose",
            success_count=3,
            failure_count=3,
            avg_reward=0.4,
        )
        best = self.evolver.get_best_strategy("write python code")
        assert best.name == "coding"
