"""Specialized agents for multi-agent collaboration."""

from __future__ import annotations

import logging

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentPhase, AgentState

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Gathers information, synthesizes findings, and produces research reports."""

    name = "researcher"
    system_prompt = """You are the Researcher agent in CosmosAGI.
Your job is to gather, analyze, and synthesize information about a topic.

Rules:
- Be thorough but concise.
- Cite reasoning and distinguish facts from inferences.
- Structure findings clearly.
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "findings": ["finding 1", "finding 2"],
  "synthesis": "Overall summary connecting the findings",
  "confidence": 0.8,
  "gaps": ["What we still don't know"]
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("Researcher: investigating task")

        prompt = f"Research the following:\n{state.task}"
        if state.observations:
            prompt += "\n\nExisting context:\n" + "\n".join(
                f"- {o}" for o in state.observations[-5:]
            )

        try:
            result = self.call_llm_json(prompt, state)
            findings = result.get("findings", [])
            synthesis = result.get("synthesis", "")

            state.observations.append(
                f"[Research] {synthesis}\nFindings: {'; '.join(findings)}"
            )

            for gap in result.get("gaps", []):
                state.observations.append(f"[Research gap] {gap}")

        except Exception as e:
            state.errors.append(f"Researcher error: {e}")
            logger.error("Researcher failed: %s", e)

        return state


class CoderAgent(BaseAgent):
    """Writes, reviews, and debugs code."""

    name = "coder"
    system_prompt = """You are the Coder agent in CosmosAGI.
Your job is to write clean, correct, well-structured code.

Rules:
- Write production-quality code with proper error handling.
- Follow the language's conventions and best practices.
- Include brief inline comments for non-obvious logic.
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "language": "python",
  "code": "the code here",
  "explanation": "What this code does and key design decisions",
  "tests_suggested": ["test case 1", "test case 2"]
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("Coder: writing code")

        prompt = f"Write code for:\n{state.task}"
        if state.observations:
            context = "\n".join(f"- {o}" for o in state.observations[-5:])
            prompt += f"\n\nContext:\n{context}"

        try:
            result = self.call_llm_json(prompt, state)
            code = result.get("code", "")
            explanation = result.get("explanation", "")

            state.observations.append(
                f"[Code] {explanation}\n```{result.get('language', '')}\n{code}\n```"
            )

        except Exception as e:
            state.errors.append(f"Coder error: {e}")
            logger.error("Coder failed: %s", e)

        return state


class ReviewerAgent(BaseAgent):
    """Reviews work products for quality, correctness, and completeness."""

    name = "reviewer"
    system_prompt = """You are the Reviewer agent in CosmosAGI.
Your job is to review work products (code, plans, research) for quality.

Rules:
- Be constructive but rigorous.
- Check for correctness, completeness, edge cases, and clarity.
- Score the work and provide specific actionable feedback.
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "score": 0.85,
  "verdict": "approve" | "revise" | "reject",
  "strengths": ["strength 1"],
  "issues": [{"severity": "high"|"medium"|"low", "description": "..."}],
  "suggestions": ["suggestion 1"]
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("Reviewer: reviewing work")

        work_items = "\n".join(f"- {o}" for o in state.observations[-10:])
        prompt = f"Task: {state.task}\n\nWork to review:\n{work_items}"

        try:
            result = self.call_llm_json(prompt, state)
            verdict = result.get("verdict", "revise")
            score = result.get("score", 0.5)
            issues = result.get("issues", [])

            review_summary = (
                f"[Review] verdict={verdict}, score={score:.2f}, "
                f"{len(issues)} issues found"
            )
            if issues:
                for issue in issues[:3]:
                    review_summary += f"\n  - [{issue.get('severity', '?')}] {issue.get('description', '')}"

            state.observations.append(review_summary)

            # If reviewer approves with high score, signal completion
            if verdict == "approve" and score >= 0.8:
                from cosmos_agi.core.state import Reflection
                state.reflections.append(Reflection(
                    success=True,
                    reasoning=f"Reviewer approved with score {score:.2f}",
                    confidence=score,
                ))

        except Exception as e:
            state.errors.append(f"Reviewer error: {e}")
            logger.error("Reviewer failed: %s", e)

        return state


class FactCheckerAgent(BaseAgent):
    """Verifies claims and flags potential misinformation."""

    name = "fact_checker"
    system_prompt = """You are the Fact Checker agent in CosmosAGI.
Your job is to verify claims, flag unsupported assertions, and ensure accuracy.

Rules:
- Identify each distinct claim in the input.
- Assess each claim: verified, plausible, unverified, or false.
- Explain your reasoning for each assessment.
- Output ONLY valid JSON, no markdown fences.

Output format:
{
  "claims": [
    {"claim": "...", "status": "verified"|"plausible"|"unverified"|"false", "reasoning": "..."}
  ],
  "overall_reliability": 0.8,
  "warnings": ["warning if any"]
}"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("FactChecker: verifying claims")

        content = "\n".join(state.observations[-5:]) if state.observations else state.task
        prompt = f"Fact-check the following:\n{content}"

        try:
            result = self.call_llm_json(prompt, state)
            claims = result.get("claims", [])
            reliability = result.get("overall_reliability", 0.5)

            false_claims = [c for c in claims if c.get("status") == "false"]
            unverified = [c for c in claims if c.get("status") == "unverified"]

            summary = f"[FactCheck] reliability={reliability:.2f}, {len(claims)} claims checked"
            if false_claims:
                summary += f"\n  FALSE: {'; '.join(c['claim'] for c in false_claims)}"
            if unverified:
                summary += f"\n  UNVERIFIED: {'; '.join(c['claim'] for c in unverified)}"

            state.observations.append(summary)

            for warning in result.get("warnings", []):
                state.observations.append(f"[FactCheck WARNING] {warning}")

        except Exception as e:
            state.errors.append(f"FactChecker error: {e}")

        return state
