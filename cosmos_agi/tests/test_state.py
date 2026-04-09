"""Tests for the core state module."""

from cosmos_agi.core.state import AgentPhase, AgentState, Reflection, SubTask


def test_agent_state_defaults():
    state = AgentState()
    assert state.phase == AgentPhase.PLAN
    assert state.iteration == 0
    assert state.subtasks == []
    assert state.final_answer is None


def test_agent_state_with_task():
    state = AgentState(task="Solve world hunger", max_iterations=5)
    assert state.task == "Solve world hunger"
    assert state.max_iterations == 5


def test_subtask_creation():
    st = SubTask(id=1, description="Research causes")
    assert st.status == "pending"
    assert st.attempts == 0
    assert st.result is None


def test_reflection():
    r = Reflection(
        success=True,
        reasoning="Task completed well",
        suggestions=["Could be faster"],
        confidence=0.9,
    )
    assert r.success
    assert r.confidence == 0.9


def test_state_serialization_roundtrip():
    state = AgentState(
        task="Test task",
        subtasks=[SubTask(id=1, description="sub1")],
        phase=AgentPhase.EXECUTE,
        observations=["obs1"],
    )
    dumped = state.model_dump()
    restored = AgentState(**dumped)
    assert restored.task == state.task
    assert len(restored.subtasks) == 1
    assert restored.phase == AgentPhase.EXECUTE
