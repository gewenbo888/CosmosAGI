"""Tests for the safety module."""

from cosmos_agi.core.safety import check_action_safety, check_output_safety


def test_safe_output():
    is_safe, reason = check_output_safety("The weather today is sunny.")
    assert is_safe


def test_unsafe_output():
    is_safe, reason = check_output_safety("Let me hack into the system and exploit the server")
    assert not is_safe


def test_safe_action():
    is_safe, reason = check_action_safety("shell", "ls -la")
    assert is_safe


def test_blocked_action():
    is_safe, reason = check_action_safety("shell", "rm -rf /")
    assert not is_safe
