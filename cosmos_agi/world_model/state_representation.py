"""State representation for the world model.

A 'world state' captures the agent's understanding of the current situation
at a point in time. The world model predicts how actions transform one state
into the next.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named object/concept in the world state."""

    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    relations: list[tuple[str, str]] = Field(default_factory=list)  # (relation, target_entity)


class WorldState(BaseModel):
    """Snapshot of the agent's world model at a point in time."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entities: dict[str, Entity] = Field(default_factory=dict)
    facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    # Numeric embedding for the transformer predictor
    state_vector: list[float] | None = None

    def add_entity(self, name: str, properties: dict[str, Any] | None = None) -> Entity:
        entity = Entity(name=name, properties=properties or {})
        self.entities[name] = entity
        return entity

    def add_fact(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def add_relation(self, source: str, relation: str, target: str) -> None:
        if source in self.entities:
            self.entities[source].relations.append((relation, target))
        # Ensure target entity exists
        if target not in self.entities:
            self.add_entity(target)

    def to_text(self) -> str:
        """Serialize to natural language for LLM consumption."""
        lines = [f"World State ({self.timestamp}):"]
        if self.entities:
            lines.append("\nEntities:")
            for name, entity in self.entities.items():
                props = ", ".join(f"{k}={v}" for k, v in entity.properties.items())
                lines.append(f"  - {name}" + (f" ({props})" if props else ""))
                for rel, target in entity.relations:
                    lines.append(f"      —{rel}→ {target}")
        if self.facts:
            lines.append("\nFacts:")
            for f in self.facts:
                lines.append(f"  - {f}")
        if self.constraints:
            lines.append("\nConstraints:")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.open_questions:
            lines.append("\nOpen questions:")
            for q in self.open_questions:
                lines.append(f"  - {q}")
        return "\n".join(lines)


class Transition(BaseModel):
    """Records a state transition: action applied to a state produces a new state."""

    action: str
    before: WorldState
    after: WorldState
    reward: float = 0.0  # self-assessed quality of this transition
    predicted_after: WorldState | None = None  # what the model predicted
    prediction_error: float | None = None
