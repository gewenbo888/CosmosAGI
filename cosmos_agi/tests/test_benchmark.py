"""Tests for the benchmark suite."""

from cosmos_agi.evaluation.benchmark import (
    BUILTIN_BENCHMARKS,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
)


class TestBenchmarkTask:
    def test_builtin_count(self):
        assert len(BUILTIN_BENCHMARKS) >= 7

    def test_all_have_required_fields(self):
        for task in BUILTIN_BENCHMARKS:
            assert task.id
            assert task.category
            assert task.prompt
            assert 0 <= task.difficulty <= 1


class TestBenchmarkSuite:
    def test_evaluate_keyword_match(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="test",
            category="test",
            prompt="test",
            difficulty=0.5,
            expected_keywords=["hello", "world"],
        )

        passed, score = suite.evaluate_answer(task, "Hello World!")
        assert passed
        assert score == 1.0

    def test_evaluate_partial_match(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="test",
            category="test",
            prompt="test",
            difficulty=0.5,
            expected_keywords=["alpha", "beta", "gamma", "delta"],
        )

        passed, score = suite.evaluate_answer(task, "The alpha and beta rays")
        assert passed  # 2/4 = 0.5, which meets threshold
        assert score == 0.5

    def test_evaluate_no_match(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="test",
            category="test",
            prompt="test",
            difficulty=0.5,
            expected_keywords=["specific_keyword"],
        )

        passed, score = suite.evaluate_answer(task, "nothing relevant here")
        assert not passed
        assert score == 0.0

    def test_evaluate_empty_answer(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="test", category="test", prompt="test",
            difficulty=0.5, expected_keywords=["x"],
        )
        passed, score = suite.evaluate_answer(task, "")
        assert not passed
        assert score == 0.0

    def test_evaluate_forbidden_keywords(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="safety",
            category="safety",
            prompt="test",
            difficulty=0.2,
            expected_not_keywords=["hack", "exploit"],
            expected_keywords=["ethical"],
        )

        # Contains forbidden keyword
        passed, score = suite.evaluate_answer(task, "You can hack the system")
        assert not passed
        assert score == 0.0

        # Clean answer
        passed, score = suite.evaluate_answer(task, "This is an ethical concern")
        assert passed

    def test_evaluate_no_keywords(self):
        suite = BenchmarkSuite()
        task = BenchmarkTask(
            id="test", category="test", prompt="test",
            difficulty=0.5, expected_keywords=[],
        )
        passed, score = suite.evaluate_answer(task, "any answer")
        assert passed
        assert score == 0.5

    def test_summary_empty(self):
        suite = BenchmarkSuite(tasks=[])
        summary = suite.summary()
        assert summary["total"] == 0

    def test_summary_with_results(self):
        suite = BenchmarkSuite(tasks=[])
        suite.results = [
            BenchmarkResult(
                task_id="t1", passed=True, score=0.8,
                elapsed_seconds=1.0, iterations_used=2,
                final_answer="ok", details={"category": "reasoning"},
            ),
            BenchmarkResult(
                task_id="t2", passed=False, score=0.3,
                elapsed_seconds=2.0, iterations_used=5,
                final_answer="fail", details={"category": "coding"},
            ),
        ]
        summary = suite.summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5
        assert "reasoning" in summary["by_category"]
        assert "coding" in summary["by_category"]
