"""Dynamic agent spawner — creates agents on demand based on task needs."""

from __future__ import annotations

import logging
from typing import Any

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.agents.communication import Blackboard, MessageBus
from cosmos_agi.agents.specialists import (
    CoderAgent,
    FactCheckerAgent,
    ResearcherAgent,
    ReviewerAgent,
)
from cosmos_agi.agents.team import AgentTeam, TeamRole
from cosmos_agi.core.state import AgentState

logger = logging.getLogger(__name__)

# Registry of available agent classes
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
    "reviewer": ReviewerAgent,
    "fact_checker": FactCheckerAgent,
}


def register_agent_class(name: str, cls: type[BaseAgent]) -> None:
    """Register a new agent class for dynamic spawning."""
    AGENT_REGISTRY[name] = cls


class AgentSpawner(BaseAgent):
    """Analyzes a task and dynamically assembles the right team of agents."""

    name = "spawner"
    system_prompt = """You are the Agent Spawner in CosmosAGI.
Given a task, decide which specialist agents are needed and how they should collaborate.

Available agents:
- researcher: Information gathering, analysis, synthesis
- coder: Code writing, debugging, optimization
- reviewer: Quality review, correctness checking
- fact_checker: Claim verification, accuracy checking

Collaboration modes:
- sequential: Agents work one after another in a pipeline
- debate: Proposers generate solutions, critic evaluates and refines

Output ONLY valid JSON:
{
  "agents_needed": [
    {"role": "researcher", "description": "Why this agent is needed"}
  ],
  "collaboration_mode": "sequential" | "debate",
  "pipeline_order": ["researcher", "coder", "reviewer"],
  "debate_config": {"proposers": ["researcher", "coder"], "critic": "reviewer"},
  "reasoning": "Why this team composition"
}"""

    def analyze_task(self, state: AgentState) -> dict[str, Any]:
        """Determine the optimal team composition for a task."""
        prompt = f"Task: {state.task}\n\nWhat team of agents is needed?"
        return self.call_llm_json(prompt, state)

    def spawn_team(self, config: dict[str, Any]) -> AgentTeam:
        """Create an AgentTeam based on the spawner's analysis."""
        team = AgentTeam(name="dynamic_team")

        for agent_info in config.get("agents_needed", []):
            role_name = agent_info["role"]
            agent_cls = AGENT_REGISTRY.get(role_name)
            if agent_cls:
                team.add_role(TeamRole(
                    name=role_name,
                    agent=agent_cls(),
                    description=agent_info.get("description", ""),
                    capabilities=[role_name],
                ))
            else:
                logger.warning("Unknown agent type: %s", role_name)

        return team

    def run(self, state: AgentState) -> AgentState:
        """Analyze task, spawn team, and execute."""
        logger.info("Spawner: analyzing task for team composition")

        try:
            config = self.analyze_task(state)
            team = self.spawn_team(config)

            mode = config.get("collaboration_mode", "sequential")
            logger.info(
                "Spawner: assembled team of %d agents, mode=%s",
                len(team.roles), mode,
            )

            if mode == "debate":
                debate_cfg = config.get("debate_config", {})
                proposers = debate_cfg.get("proposers", list(team.roles.keys())[:2])
                critic = debate_cfg.get("critic", list(team.roles.keys())[-1])
                state = team.run_debate(state, proposers=proposers, critic_name=critic)
            else:
                # Reorder roles to match pipeline_order if specified
                order = config.get("pipeline_order", list(team.roles.keys()))
                ordered_roles = {}
                for name in order:
                    if name in team.roles:
                        ordered_roles[name] = team.roles[name]
                team.roles = ordered_roles
                state = team.run_sequential(state)

        except Exception as e:
            state.errors.append(f"Spawner error: {e}")
            logger.error("Spawner failed: %s", e)

        return state
