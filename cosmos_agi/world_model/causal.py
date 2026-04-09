"""Causal reasoning engine.

Maintains a directed graph of causal relationships and supports:
- Adding causal links (A causes B)
- Querying upstream/downstream effects
- Counterfactual reasoning ("what if A hadn't happened?")
- Integration with the LLM for discovering new causal relationships
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CausalLink(BaseModel):
    """A directed causal relationship."""

    cause: str
    effect: str
    strength: float = 1.0  # 0-1, how strong the causal link is
    mechanism: str = ""  # explanation of why
    evidence: list[str] = Field(default_factory=list)


class CausalGraph:
    """Directed graph of causal relationships."""

    def __init__(self) -> None:
        self._forward: dict[str, list[CausalLink]] = defaultdict(list)  # cause -> effects
        self._backward: dict[str, list[CausalLink]] = defaultdict(list)  # effect -> causes
        self._nodes: set[str] = set()

    def add_link(self, link: CausalLink) -> None:
        """Add a causal relationship."""
        self._forward[link.cause].append(link)
        self._backward[link.effect].append(link)
        self._nodes.add(link.cause)
        self._nodes.add(link.effect)
        logger.debug("Causal link: %s → %s (strength=%.2f)", link.cause, link.effect, link.strength)

    def add(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        mechanism: str = "",
    ) -> CausalLink:
        """Convenience method to add a causal link."""
        link = CausalLink(cause=cause, effect=effect, strength=strength, mechanism=mechanism)
        self.add_link(link)
        return link

    def get_effects(self, cause: str) -> list[CausalLink]:
        """Direct effects of a cause."""
        return self._forward.get(cause, [])

    def get_causes(self, effect: str) -> list[CausalLink]:
        """Direct causes of an effect."""
        return self._backward.get(effect, [])

    def trace_downstream(self, node: str, max_depth: int = 10) -> list[tuple[str, float, int]]:
        """BFS trace of all downstream effects with cumulative strength.

        Returns list of (node, cumulative_strength, depth).
        """
        visited: set[str] = set()
        queue: deque[tuple[str, float, int]] = deque([(node, 1.0, 0)])
        results: list[tuple[str, float, int]] = []

        while queue:
            current, strength, depth = queue.popleft()
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            if current != node:
                results.append((current, strength, depth))

            for link in self._forward.get(current, []):
                if link.effect not in visited:
                    queue.append((link.effect, strength * link.strength, depth + 1))

        return sorted(results, key=lambda x: -x[1])

    def trace_upstream(self, node: str, max_depth: int = 10) -> list[tuple[str, float, int]]:
        """BFS trace of all upstream causes with cumulative strength."""
        visited: set[str] = set()
        queue: deque[tuple[str, float, int]] = deque([(node, 1.0, 0)])
        results: list[tuple[str, float, int]] = []

        while queue:
            current, strength, depth = queue.popleft()
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            if current != node:
                results.append((current, strength, depth))

            for link in self._backward.get(current, []):
                if link.cause not in visited:
                    queue.append((link.cause, strength * link.strength, depth + 1))

        return sorted(results, key=lambda x: -x[1])

    def counterfactual(self, removed_node: str) -> list[str]:
        """What downstream effects would be lost if this node were removed?

        Returns nodes that have NO alternative causal path.
        """
        downstream = self.trace_downstream(removed_node)
        affected = []

        for node, _, _ in downstream:
            # Check if this node has any cause NOT downstream of removed_node
            causes = self.get_causes(node)
            has_alternative = any(
                link.cause != removed_node
                and link.cause not in {n for n, _, _ in downstream}
                for link in causes
            )
            if not has_alternative:
                affected.append(node)

        return affected

    @property
    def nodes(self) -> set[str]:
        return self._nodes.copy()

    @property
    def link_count(self) -> int:
        return sum(len(links) for links in self._forward.values())

    def to_text(self) -> str:
        """Human-readable representation."""
        lines = [f"Causal Graph ({len(self._nodes)} nodes, {self.link_count} links):"]
        for cause, links in sorted(self._forward.items()):
            for link in links:
                strength_bar = "█" * int(link.strength * 5)
                lines.append(f"  {cause} →[{strength_bar}]→ {link.effect}")
                if link.mechanism:
                    lines.append(f"      mechanism: {link.mechanism}")
        return "\n".join(lines)


class CausalReasoner:
    """Uses the causal graph + LLM to reason about cause and effect."""

    def __init__(self, graph: CausalGraph | None = None):
        self.graph = graph or CausalGraph()

    def discover_causes(self, observation: str) -> list[CausalLink]:
        """Use LLM to discover causal links from an observation."""
        from cosmos_agi.core.llm import completion_json

        existing_context = self.graph.to_text() if self.graph.nodes else "Empty graph."

        messages = [
            {"role": "system", "content": """You are a causal reasoning engine.
Given an observation and existing causal knowledge, identify new causal relationships.

Output ONLY valid JSON:
{
  "links": [
    {"cause": "A", "effect": "B", "strength": 0.9, "mechanism": "explanation"}
  ]
}"""},
            {"role": "user", "content": f"""Existing causal knowledge:
{existing_context}

New observation: {observation}

What causal relationships can be inferred?"""},
        ]

        result = completion_json(messages)
        links = []
        for link_data in result.get("links", []):
            link = CausalLink(**link_data)
            self.graph.add_link(link)
            links.append(link)

        logger.info("Discovered %d causal links from observation", len(links))
        return links

    def explain_why(self, effect: str) -> str:
        """Generate an explanation for why something happened."""
        upstream = self.graph.trace_upstream(effect, max_depth=5)
        if not upstream:
            return f"No known causes for '{effect}'."

        lines = [f"Why did '{effect}' happen?"]
        for cause, strength, depth in upstream:
            indent = "  " * depth
            lines.append(f"{indent}← {cause} (strength={strength:.2f})")
            # Include mechanism if available
            links = self.graph.get_effects(cause)
            for link in links:
                if link.mechanism:
                    lines.append(f"{indent}   ({link.mechanism})")

        return "\n".join(lines)

    def predict_consequences(self, action: str) -> list[tuple[str, float]]:
        """Predict what would happen if an action were taken."""
        downstream = self.graph.trace_downstream(action, max_depth=5)
        return [(node, strength) for node, strength, _ in downstream]
