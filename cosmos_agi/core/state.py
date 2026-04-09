"""LangGraph state definition for the CosmosAGI agent loop."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"
    REFLECT = "reflect"
    IMPROVE = "improve"
    COMPLETE = "complete"
    ERROR = "error"


class SubTask(BaseModel):
    id: int
    description: str
    status: str = "pending"  # pending | in_progress | done | failed
    result: str | None = None
    attempts: int = 0


class Reflection(BaseModel):
    success: bool
    reasoning: str
    suggestions: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # 0-1


class AgentState(BaseModel):
    """Central state object passed through the LangGraph state machine.

    This is the single source of truth for the agent loop. Every node
    reads from and writes to this state.
    """

    # Task
    task: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)
    current_subtask_index: int = 0

    # Execution
    phase: AgentPhase = AgentPhase.PLAN
    iteration: int = 0
    max_iterations: int = 20

    # Results
    observations: list[str] = Field(default_factory=list)
    reflections: list[Reflection] = Field(default_factory=list)
    final_answer: str | None = None

    # Tool use
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)

    # Memory context
    relevant_memories: list[str] = Field(default_factory=list)

    # Error tracking
    errors: list[str] = Field(default_factory=list)

    # Chat history for LLM calls
    messages: list[dict[str, Any]] = Field(default_factory=list)
