"""LangGraph state machine — the core agent loop orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from cosmos_agi.agents.critic import CriticAgent
from cosmos_agi.agents.executor import ExecutorAgent
from cosmos_agi.agents.planner import PlannerAgent
from cosmos_agi.agents.spawner import AgentSpawner
from cosmos_agi.core.state import AgentPhase, AgentState
from cosmos_agi.evaluation.learning_loop import LearningLoop
from cosmos_agi.world_model.integration import WorldModelAgent

logger = logging.getLogger(__name__)

# Instantiate agents
planner = PlannerAgent()
executor = ExecutorAgent()
critic = CriticAgent()
world_modeler = WorldModelAgent()
spawner = AgentSpawner()

# Learning loop (shared across invocations for cumulative learning)
_learning_loop = LearningLoop()


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    """Decompose the task into subtasks."""
    agent_state = AgentState(**state)
    agent_state.iteration += 1
    result = planner.run(agent_state)
    return result.model_dump()


def simulate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run world model simulation before execution."""
    agent_state = AgentState(**state)
    result = world_modeler.run(agent_state)
    return result.model_dump()


def execute_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the current subtask."""
    agent_state = AgentState(**state)
    result = executor.run(agent_state)
    return result.model_dump()


def observe_node(state: dict[str, Any]) -> dict[str, Any]:
    """Collect and consolidate observations (pass-through for now)."""
    agent_state = AgentState(**state)
    logger.info(
        "Observe: %d observations collected",
        len(agent_state.observations),
    )
    agent_state.phase = AgentPhase.REFLECT
    return agent_state.model_dump()


def reflect_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate results via the Critic."""
    agent_state = AgentState(**state)
    result = critic.run(agent_state)
    return result.model_dump()


def improve_node(state: dict[str, Any]) -> dict[str, Any]:
    """Prepare for re-planning based on reflection."""
    agent_state = AgentState(**state)
    logger.info("Improve: re-planning based on reflection")

    # Reset subtask tracking for next iteration
    agent_state.current_subtask_index = 0
    agent_state.phase = AgentPhase.PLAN
    return agent_state.model_dump()


def route_after_plan(state: dict[str, Any]) -> str:
    phase = state.get("phase", AgentPhase.ERROR)
    if phase == AgentPhase.EXECUTE:
        return "simulate"
    return "reflect"  # error case goes to reflect


def route_after_execute(state: dict[str, Any]) -> str:
    phase = state.get("phase", AgentPhase.OBSERVE)
    if phase == AgentPhase.EXECUTE:
        return "execute"  # more subtasks to do
    if phase == AgentPhase.OBSERVE:
        return "observe"
    return "reflect"  # failure


def route_after_reflect(state: dict[str, Any]) -> str:
    phase = state.get("phase", AgentPhase.COMPLETE)
    if phase == AgentPhase.COMPLETE:
        return END
    if phase == AgentPhase.IMPROVE:
        return "improve"
    return END  # safety fallback


def build_graph() -> StateGraph:
    """Construct and compile the CosmosAGI agent loop graph."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("simulate", simulate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("observe", observe_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("improve", improve_node)

    # Entry point
    graph.set_entry_point("plan")

    # Conditional edges
    graph.add_conditional_edges("plan", route_after_plan, {
        "simulate": "simulate",
        "reflect": "reflect",
    })
    graph.add_edge("simulate", "execute")
    graph.add_conditional_edges("execute", route_after_execute, {
        "execute": "execute",
        "observe": "observe",
        "reflect": "reflect",
    })
    graph.add_edge("observe", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {
        END: END,
        "improve": "improve",
    })
    graph.add_edge("improve", "plan")

    return graph.compile()


def run_agent_loop(
    task: str,
    max_iterations: int = 20,
    team_mode: bool = False,
    learn: bool = True,
) -> AgentState:
    """Run the full agent loop on a task and return the final state.

    Args:
        task: The task to execute.
        max_iterations: Maximum number of plan-execute-reflect cycles.
        team_mode: If True, use the AgentSpawner to dynamically assemble
                   a multi-agent team instead of the single-agent pipeline.
        learn: If True, run post-task self-assessment and learning.
    """
    logger.info("Starting agent loop for task: %s (team_mode=%s)", task[:100], team_mode)

    initial_state = AgentState(
        task=task,
        max_iterations=max_iterations,
    )

    # Inject learned context from past episodes
    if learn:
        context = _learning_loop.get_context_for_task(task)
        guidelines = context.get("guidelines", [])
        if guidelines:
            initial_state.relevant_memories.append(
                "[Learned guidelines]\n" + "\n".join(f"- {g}" for g in guidelines)
            )
        similar = context.get("similar_episodes", [])
        if similar:
            summaries = [
                f"- {ep['task']} (success={ep['success']}, reward={ep['reward']:.2f})"
                for ep in similar
            ]
            initial_state.relevant_memories.append(
                "[Similar past tasks]\n" + "\n".join(summaries)
            )

    strategy = "team" if team_mode else "single_agent"

    if team_mode:
        final_state = _run_team_mode(initial_state)
    else:
        compiled_graph = build_graph()
        final_state_dict = compiled_graph.invoke(initial_state.model_dump())
        final_state = AgentState(**final_state_dict)

    logger.info(
        "Agent loop finished. Phase=%s, Iterations=%d",
        final_state.phase,
        final_state.iteration,
    )

    # Post-task learning
    if learn:
        try:
            episode = _learning_loop.post_task_learning(
                final_state,
                strategy_used=strategy,
                skip_llm_assessment=True,  # Use heuristic by default; LLM for benchmarks
            )
            logger.info(
                "Learning: recorded episode (reward=%.2f, lessons=%d)",
                episode.self_reward,
                len(episode.lessons),
            )
        except Exception as e:
            logger.warning("Post-task learning failed: %s", e)

    return final_state


def _run_team_mode(state: AgentState) -> AgentState:
    """Run the multi-agent team mode.

    The spawner analyzes the task, assembles a team, and orchestrates
    collaboration (sequential pipeline or debate).
    """
    logger.info("Team mode: spawner analyzing task")
    state = spawner.run(state)

    # If spawner didn't reach completion, fall back to critic
    if state.phase not in (AgentPhase.COMPLETE, AgentPhase.ERROR):
        logger.info("Team mode: running critic for final evaluation")
        state = critic.run(state)

    return state
