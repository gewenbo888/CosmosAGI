"""Tests for agent team orchestration."""

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.agents.team import AgentTeam, TeamRole
from cosmos_agi.core.state import AgentPhase, AgentState, Reflection


class MockAgent(BaseAgent):
    """A mock agent that appends a marker to observations."""

    def __init__(self, marker: str = "mock"):
        self.marker = marker
        self.name = marker

    def run(self, state: AgentState) -> AgentState:
        state.observations.append(f"[{self.marker}] processed")
        return state


class MockCriticAgent(BaseAgent):
    """Mock critic that approves on second call."""

    name = "mock_critic"
    _call_count = 0

    def run(self, state: AgentState) -> AgentState:
        MockCriticAgent._call_count += 1
        if MockCriticAgent._call_count >= 2:
            state.reflections.append(Reflection(
                success=True,
                reasoning="Approved after revision",
                confidence=0.9,
            ))
            state.final_answer = "Converged answer"
            state.phase = AgentPhase.COMPLETE
        else:
            state.reflections.append(Reflection(
                success=False,
                reasoning="Needs improvement",
                confidence=0.4,
            ))
        return state


class TestAgentTeam:
    def test_add_and_remove_role(self):
        team = AgentTeam(name="test_team")
        role = TeamRole(name="worker", agent=MockAgent(), description="does work")
        team.add_role(role)
        assert "worker" in team.roles
        team.remove_role("worker")
        assert "worker" not in team.roles

    def test_sequential_pipeline(self):
        team = AgentTeam(name="pipeline_test")
        team.add_role(TeamRole(name="agent_a", agent=MockAgent("A"), description="first"))
        team.add_role(TeamRole(name="agent_b", agent=MockAgent("B"), description="second"))
        team.add_role(TeamRole(name="agent_c", agent=MockAgent("C"), description="third"))

        state = AgentState(task="test pipeline")
        result = team.run_sequential(state)

        markers = [o for o in result.observations if o.startswith("[")]
        assert "[A] processed" in markers
        assert "[B] processed" in markers
        assert "[C] processed" in markers

    def test_sequential_stops_on_complete(self):
        class EarlyStopAgent(BaseAgent):
            name = "stopper"
            def run(self, state: AgentState) -> AgentState:
                state.phase = AgentPhase.COMPLETE
                state.final_answer = "Done early"
                return state

        team = AgentTeam()
        team.add_role(TeamRole(name="stopper", agent=EarlyStopAgent(), description="stops"))
        team.add_role(TeamRole(name="never_runs", agent=MockAgent("SKIP"), description="skipped"))

        state = AgentState(task="test early stop")
        result = team.run_sequential(state)

        assert result.phase == AgentPhase.COMPLETE
        # The second agent should NOT have run
        assert not any("[SKIP]" in o for o in result.observations)

    def test_blackboard_shared(self):
        team = AgentTeam(name="bb_test")

        class WriterAgent(BaseAgent):
            name = "writer"
            def run(self, state: AgentState) -> AgentState:
                state.observations.append("wrote_data")
                return state

        class ReaderAgent(BaseAgent):
            name = "reader"
            def run(self, state: AgentState) -> AgentState:
                # Blackboard context is injected as a relevant_memory
                has_bb = any("Blackboard" in m for m in state.relevant_memories)
                state.observations.append(f"bb_present={has_bb}")
                return state

        team.add_role(TeamRole(name="writer", agent=WriterAgent(), description="writes"))
        team.add_role(TeamRole(name="reader", agent=ReaderAgent(), description="reads"))

        state = AgentState(task="test blackboard")
        result = team.run_sequential(state)
        assert "wrote_data" in result.observations

    def test_debate_converges(self):
        MockCriticAgent._call_count = 0  # reset

        team = AgentTeam(name="debate_test")
        team.add_role(TeamRole(name="proposer1", agent=MockAgent("P1"), description="proposer"))
        team.add_role(TeamRole(name="proposer2", agent=MockAgent("P2"), description="proposer"))
        team.add_role(TeamRole(name="critic", agent=MockCriticAgent(), description="critic"))

        state = AgentState(task="test debate")
        result = team.run_debate(
            state,
            proposers=["proposer1", "proposer2"],
            critic_name="critic",
            max_rounds=3,
        )

        assert result.phase == AgentPhase.COMPLETE
        assert result.final_answer == "Converged answer"

    def test_get_team_description(self):
        team = AgentTeam(name="desc_test")
        team.add_role(TeamRole(
            name="researcher",
            agent=MockAgent(),
            description="Gathers info",
            capabilities=["search", "analyze"],
        ))
        desc = team.get_team_description()
        assert "researcher" in desc
        assert "search" in desc

    def test_message_passing_in_pipeline(self):
        team = AgentTeam(name="msg_test")

        class SenderAgent(BaseAgent):
            name = "sender"
            def run(self, state: AgentState) -> AgentState:
                from cosmos_agi.agents.communication import Message
                # We can't access the team's bus directly, but the team injects messages
                state.observations.append("sent_message")
                return state

        team.add_role(TeamRole(name="sender", agent=SenderAgent(), description="sends"))
        team.add_role(TeamRole(name="receiver", agent=MockAgent("receiver"), description="receives"))

        # Manually send a message via the team bus
        from cosmos_agi.agents.communication import Message
        team.bus.send(Message(sender="sender", recipient="receiver", content="hello from sender"))

        state = AgentState(task="test messaging")
        result = team.run_sequential(state)

        # The receiver should have gotten the message injected as an observation
        has_message = any("hello from sender" in o for o in result.observations)
        assert has_message
