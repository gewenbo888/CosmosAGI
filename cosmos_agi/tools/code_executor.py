"""Sandboxed code execution tool.

Runs Python code in an isolated subprocess with resource limits.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path

from cosmos_agi.tools.registry import register_tool

logger = logging.getLogger(__name__)

SANDBOX_IMPORTS_BLOCKLIST = [
    "os.system",
    "subprocess",
    "shutil.rmtree",
    "__import__('os')",
    "eval(",
    "exec(",
]


def execute_python(code: str, timeout: int = 30) -> dict[str, str]:
    """Execute Python code in a subprocess sandbox.

    Returns {"stdout": ..., "stderr": ..., "returncode": ...}.
    """
    # Basic static safety check
    for blocked in SANDBOX_IMPORTS_BLOCKLIST:
        if blocked in code:
            return {
                "stdout": "",
                "stderr": f"BLOCKED: Code contains forbidden pattern '{blocked}'",
                "returncode": "-1",
            }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return {
            "stdout": result.stdout[:10000],
            "stderr": result.stderr[:5000],
            "returncode": str(result.returncode),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": "-1",
        }
    finally:
        Path(script_path).unlink(missing_ok=True)


def _tool_execute_python(code: str, timeout: int = 30) -> str:
    """Tool-registry wrapper."""
    result = execute_python(code, timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\nSTDERR: {result['stderr']}"
    if result["returncode"] != "0":
        output += f"\n(exit code: {result['returncode']})"
    return output


register_tool(
    "python_exec",
    _tool_execute_python,
    "Execute Python code in a sandboxed subprocess. Returns stdout/stderr.",
    {"code": "str", "timeout": "int (default 30)"},
)
