"""Tests for the tool registry."""

from cosmos_agi.tools.registry import (
    execute_tool,
    get_tool,
    list_tools,
    register_tool,
)


def test_register_and_get_tool():
    register_tool("test_tool", lambda x: x * 2, "Doubles input", {"x": "int"})
    tool = get_tool("test_tool")
    assert tool is not None
    assert tool["name"] == "test_tool"


def test_list_tools():
    tools = list_tools()
    names = [t["name"] for t in tools]
    assert "shell" in names
    assert "python_eval" in names


def test_execute_tool_python_eval():
    result = execute_tool("python_eval", code="2 + 2")
    assert result == "4"


def test_execute_missing_tool():
    result = execute_tool("nonexistent_tool")
    assert "not found" in result


def test_shell_blocked_command():
    result = execute_tool("shell", command="rm -rf /")
    assert "BLOCKED" in result
