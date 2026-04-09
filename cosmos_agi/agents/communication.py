"""Inter-agent communication — message bus and shared blackboard."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message passed between agents."""

    sender: str
    recipient: str  # agent name or "*" for broadcast
    content: str
    msg_type: str = "info"  # info | request | response | error | directive
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    in_reply_to: str | None = None  # message id for threading
    priority: int = 0  # higher = more urgent


class Blackboard:
    """Shared knowledge store that all agents can read/write.

    Organized by namespace (e.g. "facts", "plans", "results") to avoid
    collisions between agents writing to the same keys.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = defaultdict(dict)
        self._history: list[tuple[str, str, str, Any]] = []  # (timestamp, namespace, key, value)
        self._lock = threading.Lock()

    def write(self, namespace: str, key: str, value: Any, author: str = "") -> None:
        with self._lock:
            self._data[namespace][key] = value
            self._history.append((
                datetime.now(timezone.utc).isoformat(),
                namespace,
                key,
                value,
            ))
        logger.debug("Blackboard write: %s/%s by %s", namespace, key, author)

    def read(self, namespace: str, key: str, default: Any = None) -> Any:
        return self._data.get(namespace, {}).get(key, default)

    def read_namespace(self, namespace: str) -> dict[str, Any]:
        return dict(self._data.get(namespace, {}))

    def list_namespaces(self) -> list[str]:
        return list(self._data.keys())

    def search(self, query: str) -> list[tuple[str, str, Any]]:
        """Search all namespaces for keys/values matching query."""
        results = []
        q = query.lower()
        for ns, entries in self._data.items():
            for key, value in entries.items():
                if q in key.lower() or q in str(value).lower():
                    results.append((ns, key, value))
        return results

    def to_text(self, max_entries: int = 50) -> str:
        lines = ["Shared Blackboard:"]
        count = 0
        for ns in sorted(self._data):
            lines.append(f"\n  [{ns}]")
            for key, value in self._data[ns].items():
                val_str = str(value)[:200]
                lines.append(f"    {key}: {val_str}")
                count += 1
                if count >= max_entries:
                    lines.append(f"    ... (truncated)")
                    return "\n".join(lines)
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._history.clear()


class MessageBus:
    """Pub/sub message bus for agent-to-agent communication."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Message]] = defaultdict(list)
        self._subscribers: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._broadcast_log: list[Message] = []
        self._lock = threading.Lock()

    def send(self, message: Message) -> None:
        """Send a message to a specific agent or broadcast."""
        with self._lock:
            if message.recipient == "*":
                self._broadcast_log.append(message)
                for agent_name, callbacks in self._subscribers.items():
                    if agent_name != message.sender:
                        self._queues[agent_name].append(message)
                        for cb in callbacks:
                            cb(message)
            else:
                self._queues[message.recipient].append(message)
                for cb in self._subscribers.get(message.recipient, []):
                    cb(message)

        logger.debug(
            "Message: %s → %s [%s] %s",
            message.sender, message.recipient, message.msg_type,
            message.content[:80],
        )

    def receive(self, agent_name: str, msg_type: str | None = None) -> list[Message]:
        """Get all pending messages for an agent. Drains the queue."""
        with self._lock:
            messages = self._queues.pop(agent_name, [])
        if msg_type:
            messages = [m for m in messages if m.msg_type == msg_type]
        return sorted(messages, key=lambda m: -m.priority)

    def peek(self, agent_name: str) -> int:
        """Check how many messages are pending for an agent."""
        return len(self._queues.get(agent_name, []))

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]) -> None:
        """Register a callback for when messages arrive."""
        self._subscribers[agent_name].append(callback)

    def get_broadcast_log(self, limit: int = 20) -> list[Message]:
        return self._broadcast_log[-limit:]
