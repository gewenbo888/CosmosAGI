"""File system tools — read, write, list files within a sandboxed workspace."""

from __future__ import annotations

import logging
from pathlib import Path

from cosmos_agi.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Workspace root — agents can only access files under this directory
WORKSPACE_ROOT = Path("./data/workspace")


def _resolve_safe_path(path: str) -> Path | None:
    """Resolve a path and ensure it's within the workspace."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = (WORKSPACE_ROOT / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_ROOT.resolve())):
        return None
    return resolved


def read_file(path: str) -> str:
    """Read a file from the workspace."""
    safe = _resolve_safe_path(path)
    if safe is None:
        return "ERROR: Path escapes workspace boundary"
    if not safe.exists():
        return f"ERROR: File not found: {path}"
    try:
        return safe.read_text(encoding="utf-8")[:50000]
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    safe = _resolve_safe_path(path)
    if safe is None:
        return "ERROR: Path escapes workspace boundary"
    try:
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files in a workspace directory."""
    safe = _resolve_safe_path(path)
    if safe is None:
        return "ERROR: Path escapes workspace boundary"
    if not safe.exists():
        return f"ERROR: Directory not found: {path}"
    try:
        entries = sorted(safe.glob(pattern))
        lines = []
        for entry in entries[:100]:
            rel = entry.relative_to(WORKSPACE_ROOT.resolve())
            kind = "d" if entry.is_dir() else "f"
            size = entry.stat().st_size if entry.is_file() else 0
            lines.append(f"[{kind}] {rel} ({size} bytes)")
        return "\n".join(lines) if lines else "(empty)"
    except Exception as e:
        return f"ERROR: {e}"


# Register tools
register_tool("read_file", read_file, "Read a file from the workspace", {"path": "str"})
register_tool("write_file", write_file, "Write content to a file", {"path": "str", "content": "str"})
register_tool("list_files", list_files, "List files in a directory", {"path": "str", "pattern": "str"})
