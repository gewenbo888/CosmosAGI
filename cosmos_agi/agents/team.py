"""Agent team orchestration — coordinates multiple specialized agents."""

from __future__ import annotations

import logging
from typing import Any

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.agents.communication import Blackboard, Message, MessageBus
from cosmos_agi.core.state import AgentPhase, AgentState

logger = logging.getLogger(__name__)


class TeamRole:
    """Defines a role within an agent team."""

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        description: str = "",
        capabilities: list[str] | None = None,
    ):
        self.name = name
        self.agent = agent
        self.description = description
        self.capabilities = capabilities or []


class AgentTeam:
    """Orchestrates multiple agents working together on a complex task.

    Supports two collaboration patterns:
    1. Sequential pipeline — agents process in order
    2. Debate/consensus — agents propose, critique, and converge
    """

    def __init__(
        self,
        name: str = "default_team",
        roles: list[TeamRole] | None = None,
    ):
        self.name = name
        self.roles: dict[str, TeamRole] = {}
        self.bus = MessageBus()
        self.blackboard = Blackboard()
        self._execution_log: list[dict[str, Any]] = []

        if roles:
            for role in roles:
                self.add_role(role)

    def add_role(self, role: TeamRole) -> None:
        self.roles[role.name] = role
        logger.info("Team '%s': added role '%s' (%s)", self.name, role.name, role.description)

    def remove_role(self, name: str) -> None:
        self.roles.pop(name, None)

    def get_team_description(self) -> str:
        lines = [f"Team: {self.name} ({len(self.roles)} agents)"]
        for name, role in self.roles.items():
            caps = ", ".join(role.capabilities) if role.capabilities else "general"
            lines.append(f"  - {name}: {role.description} [{caps}]")
        return "\n".join(lines)

    def run_sequential(self, state: AgentState) -> AgentState:
        """Run agents in sequence, each building on the previous result."""
        logger.info("Team '%s': sequential run with %d agents", self.name, len(self.roles))

        for name, role in self.roles.items():
            logger.info("Team: running agent '%s'", name)

            # Inject blackboard context into state
            bb_context = self.blackboard.to_text()
            if bb_context and bb_context != "Shared Blackboard:":
                state.relevant_memories.append(f"[Blackboard] {bb_context}")

            # Inject pending messages
            messages = self.bus.receive(name)
            for msg in messages:
                state.observations.append(
                    f"[Message from {msg.sender}] {msg.content}"
                )

            # Run the agent
            state = role.agent.run(state)

            # Log execution
            self._execution_log.append({
                "agent": name,
                "phase": state.phase.value,
                "observations_count": len(state.observations),
                "errors_count": len(state.errors),
            })

            # Write agent's output to blackboard
            if state.observations:
                self.blackboard.write(
                    "agent_outputs", name, state.observations[-1], author=name,
                )

            # If an agent signals error or completion, stop the pipeline
            if state.phase in (AgentPhase.ERROR, AgentPhase.COMPLETE):
                logger.info("Team: stopping pipeline at '%s' (phase=%s)", name, state.phase)
                break

        return state

    def run_debate(
        self,
        state: AgentState,
        proposers: list[str],
        critic_name: str,
        max_rounds: int = 3,
    ) -> AgentState:
        """Run a debate where proposers suggest solutions and a critic evaluates.

        Each round:
        1. Each proposer generates a proposal
        2. The critic evaluates all proposals
        3. If consensus, stop. Otherwise, proposers refine based on feedback.
        """
        logger.info(
            "Team '%s': debate with %d proposers, critic='%s'",
            self.name, len(proposers), critic_name,
        )

        critic_role = self.roles.get(critic_name)
        if not critic_role:
            state.errors.append(f"Critic '{critic_name}' not found in team")
            return state

        for round_num in range(max_rounds):
            logger.info("Debate round %d/%d", round_num + 1, max_rounds)
            proposals: dict[str, str] = {}

            # Phase 1: Collect proposals
            for proposer_name in proposers:
                role = self.roles.get(proposer_name)
                if not role:
                    continue

                # Add previous feedback if available
                feedback = self.blackboard.read("debate_feedback", proposer_name)
                if feedback:
                    state.observations.append(
                        f"[Feedback for {proposer_name}] {feedback}"
                    )

                proposal_state = role.agent.run(state)

                proposal = ""
                if proposal_state.observations:
                    proposal = proposal_state.observations[-1]
                elif proposal_state.final_answer:
                    proposal = proposal_state.final_answer

                proposals[proposer_name] = proposal
                self.blackboard.write(
                    "proposals", f"round_{round_num}_{proposer_name}",
                    proposal, author=proposer_name,
                )

            # Phase 2: Critic evaluates
            proposals_text = "\n\n".join(
                f"**{name}**: {proposal}" for name, proposal in proposals.items()
            )
            state.observations.append(
                f"[Round {round_num + 1} proposals]\n{proposals_text}"
            )

            critic_state = critic_role.agent.run(state)

            # Check if critic is satisfied
            if critic_state.reflections:
                last_reflection = critic_state.reflections[-1]
                if last_reflection.success and last_reflection.confidence >= 0.7:
                    state.final_answer = critic_state.final_answer
                    state.phase = AgentPhase.COMPLETE
                    logger.info("Debate: consensus reached in round %d", round_num + 1)
                    return state

                # Write feedback for next round
                for proposer_name in proposers:
                    self.bus.send(Message(
                        sender=critic_name,
                        recipient=proposer_name,
                        content=last_reflection.reasoning,
                        msg_type="directive",
                    ))
                    self.blackboard.write(
                        "debate_feedback", proposer_name,
                        last_reflection.reasoning, author=critic_name,
                    )

            state = critic_state

        logger.warning("Debate: no consensus after %d rounds", max_rounds)
        return state

    def select_agent_for_task(self, task_description: str) -> str | None:
        """Use LLM to select the best agent for a given task."""
        from cosmos_agi.core.llm import completion_json

        team_desc = self.get_team_description()
        messages = [
            {"role": "system", "content": """You select the best agent for a task.
Output ONLY valid JSON: {"agent": "agent_name", "reasoning": "why"}"""},
            {"role": "user", "content": f"Team:\n{team_desc}\n\nTask: {task_description}"},
        ]

        try:
            result = completion_json(messages)
            selected = result.get("agent", "")
            if selected in self.roles:
                logger.info("Selected agent '%s' for task: %s", selected, task_description[:60])
                return selected
        except Exception as e:
            logger.warning("Agent selection failed: %s", e)

        return None
