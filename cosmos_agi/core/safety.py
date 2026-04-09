"""Safety and alignment layer — output filtering and human-in-the-loop gates."""

from __future__ import annotations

import logging
import re

from cosmos_agi.config.settings import settings

logger = logging.getLogger(__name__)

# Patterns that should trigger safety review
HARMFUL_PATTERNS = [
    r"(?i)\b(hack|exploit|attack|ddos|inject)\b.*(?:system|server|database)",
    r"(?i)\b(create|build|make)\b.*(?:weapon|bomb|virus|malware)",
    r"(?i)\b(steal|phish|spoof)\b.*(?:password|credential|identity)",
]

_compiled_patterns = [re.compile(p) for p in HARMFUL_PATTERNS]


def check_output_safety(text: str) -> tuple[bool, str]:
    """Check if output text passes safety filters.

    Returns (is_safe, reason).
    """
    for pattern in _compiled_patterns:
        if pattern.search(text):
            reason = f"Output matched safety pattern: {pattern.pattern[:60]}"
            logger.warning("Safety filter triggered: %s", reason)
            return False, reason
    return True, "ok"


def check_action_safety(action: str, action_input: str) -> tuple[bool, str]:
    """Check if a proposed action is safe to execute."""
    for blocked in settings.safety.blocked_actions:
        if blocked.lower() in action_input.lower():
            reason = f"Action input contains blocked pattern: {blocked}"
            logger.warning("Action blocked: %s", reason)
            return False, reason
    return True, "ok"


def human_in_the_loop(action: str, action_input: str) -> bool:
    """Prompt the human for approval before executing a potentially risky action.

    Returns True if approved.
    """
    if not settings.safety.enable_human_in_the_loop:
        return True

    print(f"\n{'='*60}")
    print(f"HUMAN APPROVAL REQUIRED")
    print(f"Action: {action}")
    print(f"Input:  {action_input[:200]}")
    print(f"{'='*60}")

    while True:
        response = input("Approve? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            logger.info("Human approved action: %s", action)
            return True
        if response in ("n", "no"):
            logger.info("Human denied action: %s", action)
            return False
        print("Please enter 'y' or 'n'.")
