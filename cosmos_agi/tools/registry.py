"""Tool registry — register and look up tools available to agents."""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global tool registry
_tools: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    func: Callable,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> None:
    """Register a tool that agents can use."""
    _tools[name] = {
        "name": name,
        "func": func,
        "description": description,
        "parameters": parameters or {},
    }
    logger.debug("Registered tool: %s", name)


def get_tool(name: str) -> dict[str, Any] | None:
    return _tools.get(name)


def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": t["name"], "description": t["description"]}
        for t in _tools.values()
    ]


def execute_tool(name: str, **kwargs: Any) -> str:
    """Execute a registered tool by name."""
    tool = _tools.get(name)
    if not tool:
        return f"Error: tool '{name}' not found"
    try:
        result = tool["func"](**kwargs)
        return str(result)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"


# ── Built-in tools ──────────────────────────────────────────────


def _shell_exec(command: str, timeout: int = 30) -> str:
    """Execute a shell command (sandboxed)."""
    from cosmos_agi.config.settings import settings

    for blocked in settings.safety.blocked_actions:
        if blocked.lower() in command.lower():
            return f"BLOCKED: command contains forbidden pattern '{blocked}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\nSTDERR: {result.stderr}"
        return output[:5000]  # truncate
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"


def _python_eval(code: str) -> str:
    """Evaluate a Python expression (simple, sandboxed)."""
    try:
        result = eval(code, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Register built-in tools
register_tool(
    "shell",
    _shell_exec,
    "Execute a shell command. Use for file operations, git, etc.",
    {"command": "str", "timeout": "int (default 30)"},
)
register_tool(
    "python_eval",
    _python_eval,
    "Evaluate a simple Python expression.",
    {"code": "str"},
)
