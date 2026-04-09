"""CosmosAGI — Main entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from cosmos_agi.config.settings import settings
from cosmos_agi.core.graph import run_agent_loop


def setup_logging() -> None:
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.log_file),
        ],
    )


def run_benchmark(categories: list[str] | None = None) -> None:
    """Run the benchmark suite and print results."""
    from cosmos_agi.evaluation.benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    print("Running CosmosAGI benchmarks...\n")

    summary = suite.run_all(categories=categories)
    suite.save_results()

    print(f"\n{'═'*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'═'*60}")
    print(f"Total: {summary['total']}  Passed: {summary['passed']}  "
          f"Pass rate: {summary['pass_rate']:.0%}  Avg score: {summary['avg_score']:.2f}")
    print(f"Avg time: {summary['avg_time_seconds']:.1f}s")
    print(f"\nBy category:")
    for cat, stats in summary.get("by_category", {}).items():
        print(f"  {cat}: {stats['passed']}/{stats['total']} "
              f"({stats['pass_rate']:.0%}, avg={stats['avg_score']:.2f})")
    print(f"\nDetailed results:")
    for r in summary.get("results", []):
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['id']}: score={r['score']:.2f}")
    print(f"{'═'*60}")


def show_stats() -> None:
    """Show performance statistics from the learning loop."""
    from cosmos_agi.evaluation.learning_loop import LearningLoop

    loop = LearningLoop()
    stats = loop.get_performance_summary()

    print(f"\n{'═'*60}")
    print(f"COSMOSAGI PERFORMANCE STATS")
    print(f"{'═'*60}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Success rate: {stats['success_rate']:.0%}")

    if stats["common_failures"]:
        print(f"\nCommon failure modes:")
        for mode, count in stats["common_failures"]:
            print(f"  - {mode} ({count}x)")

    if stats["strategy_stats"]:
        print(f"\nStrategy performance:")
        for name, s in stats["strategy_stats"].items():
            print(f"  {name}: {s['count']} uses, "
                  f"success={s['success_rate']:.0%}, reward={s['avg_reward']:.2f}")

    if stats["active_strategies"]:
        print(f"\nActive strategies:")
        for s in stats["active_strategies"]:
            print(f"  - {s['name']}: success={s['success_rate']:.0%} ({s['uses']} uses)")

    print(f"{'═'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CosmosAGI — An open-source AGI prototype",
    )
    parser.add_argument("task", nargs="?", help="Task to execute")
    parser.add_argument(
        "--model", default=None,
        help="LLM model to use (e.g., claude-sonnet-4-20250514, gpt-4o, ollama/llama3)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20, help="Max agent loop iterations",
    )
    parser.add_argument(
        "--no-hitl", action="store_true", help="Disable human-in-the-loop",
    )
    parser.add_argument(
        "--team", action="store_true",
        help="Enable multi-agent team mode (auto-selects specialist agents)",
    )
    parser.add_argument(
        "--no-learn", action="store_true",
        help="Disable post-task self-assessment and learning",
    )
    parser.add_argument(
        "--benchmark", nargs="*", metavar="CATEGORY",
        help="Run benchmark suite (optionally filter by category: reasoning, coding, research, planning, safety)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show performance statistics from past episodes",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level",
    )
    args = parser.parse_args()

    if args.model:
        settings.llm.model = args.model
    if args.no_hitl:
        settings.safety.enable_human_in_the_loop = False
    if args.log_level:
        settings.log_level = args.log_level

    setup_logging()
    logger = logging.getLogger("cosmos_agi")

    # Special modes
    if args.benchmark is not None:
        categories = args.benchmark if args.benchmark else None
        run_benchmark(categories)
        return

    if args.stats:
        show_stats()
        return

    learn = not args.no_learn

    # Interactive mode if no task provided
    if not args.task:
        print("CosmosAGI v0.1.0 — Interactive Mode")
        print("Commands: 'quit' to exit, 'stats' for performance stats")
        print(f"Learning: {'ON' if learn else 'OFF'}  Team: {'ON' if args.team else 'OFF'}\n")
        while True:
            try:
                task = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if task.lower() == "stats":
                show_stats()
                continue

            result = run_agent_loop(
                task,
                max_iterations=args.max_iterations,
                team_mode=args.team,
                learn=learn,
            )
            print(f"\n{'─'*60}")
            print(f"Result: {result.final_answer}")
            print(f"Iterations: {result.iteration}")
            print(f"Phase: {result.phase}")
            print(f"{'─'*60}\n")
    else:
        result = run_agent_loop(
            args.task,
            max_iterations=args.max_iterations,
            team_mode=args.team,
            learn=learn,
        )
        print(f"\n{'─'*60}")
        print(f"Task: {args.task}")
        print(f"Result: {result.final_answer}")
        print(f"Iterations: {result.iteration}")
        print(f"Phase: {result.phase}")
        if result.errors:
            print(f"Errors: {result.errors}")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()
