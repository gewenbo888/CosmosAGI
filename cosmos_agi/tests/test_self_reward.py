"""Tests for self-reward mechanism."""

from cosmos_agi.core.state import AgentPhase, AgentState, Reflection
from cosmos_agi.evaluation.self_reward import (
    DEFAULT_DIMENSIONS,
    RewardAssessment,
    RewardDimension,
    SelfRewardModel,
)


class TestRewardDimension:
    def test_basic(self):
        d = RewardDimension(name="correctness", score=0.8, weight=2.0, reasoning="Good")
        assert d.score == 0.8
        assert d.weight == 2.0


class TestRewardAssessment:
    def test_defaults(self):
        a = RewardAssessment()
        assert a.composite_reward == 0.0
        assert a.dimensions == []

    def test_with_data(self):
        a = RewardAssessment(
            dimensions=[
                RewardDimension(name="a", score=0.5, weight=1.0),
                RewardDimension(name="b", score=0.9, weight=2.0),
            ],
            composite_reward=0.77,
            lessons_learned=["be more precise"],
        )
        assert len(a.dimensions) == 2
        assert len(a.lessons_learned) == 1


class TestSelfRewardModel:
    def test_fallback_assessment_success(self):
        """Test heuristic assessment for a successful task."""
        model = SelfRewardModel()
        state = AgentState(
            task="test task",
            final_answer="The answer is 42",
            iteration=2,
            max_iterations=20,
        )
        assessment = model._fallback_assessment(state)
        # Has answer (+0.4), no errors (+0.3), few iterations (+0.2) = 0.9
        assert assessment.composite_reward > 0.5

    def test_fallback_assessment_failure(self):
        """Test heuristic assessment for a failed task."""
        model = SelfRewardModel()
        state = AgentState(
            task="test task",
            errors=["something broke", "another error"],
            iteration=18,
            max_iterations=20,
        )
        assessment = model._fallback_assessment(state)
        # No answer (0), has errors (0), many iterations (0) = 0.0
        assert assessment.composite_reward == 0.0
        assert "llm_self_reward_failed" in assessment.failure_modes

    def test_fallback_assessment_partial(self):
        """Test heuristic assessment for partial success."""
        model = SelfRewardModel()
        state = AgentState(
            task="test task",
            final_answer="Partial result",
            errors=["minor issue"],
            iteration=3,
            max_iterations=20,
        )
        assessment = model._fallback_assessment(state)
        # Has answer (+0.4), has errors (0), few iterations (+0.2) = 0.6
        assert 0.3 < assessment.composite_reward < 0.8

    def test_compute_reward_from_scores(self):
        model = SelfRewardModel()
        scores = {"correctness": 0.8, "completeness": 0.6, "efficiency": 0.9}
        reward = model.compute_reward_from_scores(scores)
        # Weighted: correctness(0.8*2) + completeness(0.6*1.5) + efficiency(0.9*1)
        # = 1.6 + 0.9 + 0.9 = 3.4 / (2+1.5+1) = 3.4/4.5 ≈ 0.756
        assert 0.7 < reward < 0.8

    def test_parse_assessment(self):
        model = SelfRewardModel()
        result = {
            "dimensions": [
                {"name": "correctness", "score": 0.9, "reasoning": "Accurate"},
                {"name": "clarity", "score": 0.7, "reasoning": "Clear enough"},
            ],
            "lessons_learned": ["Check edge cases"],
            "failure_modes": [],
            "improvement_suggestions": ["Add examples"],
            "difficulty_estimate": 0.4,
        }
        assessment = model._parse_assessment(result)
        assert len(assessment.dimensions) == 2
        assert assessment.composite_reward > 0
        assert assessment.lessons_learned == ["Check edge cases"]
        assert assessment.difficulty_estimate == 0.4

    def test_score_clamping(self):
        """Scores outside [-1, 1] should be clamped."""
        model = SelfRewardModel()
        result = {
            "dimensions": [
                {"name": "x", "score": 5.0},  # should clamp to 1.0
                {"name": "y", "score": -3.0},  # should clamp to -1.0
            ],
        }
        assessment = model._parse_assessment(result)
        assert assessment.dimensions[0].score == 1.0
        assert assessment.dimensions[1].score == -1.0


class TestDefaultDimensions:
    def test_dimensions_defined(self):
        assert len(DEFAULT_DIMENSIONS) == 5

    def test_dimension_format(self):
        for name, weight, description in DEFAULT_DIMENSIONS:
            assert isinstance(name, str)
            assert isinstance(weight, float)
            assert weight > 0
            assert len(description) > 0
