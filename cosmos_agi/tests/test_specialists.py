"""Tests for specialist agents — unit tests that don't require LLM calls."""

from cosmos_agi.agents.specialists import (
    CoderAgent,
    FactCheckerAgent,
    ResearcherAgent,
    ReviewerAgent,
)
from cosmos_agi.agents.spawner import AGENT_REGISTRY, register_agent_class
from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentState


class TestSpecialistInstantiation:
    """Verify all specialists can be instantiated and have proper config."""

    def test_researcher_config(self):
        agent = ResearcherAgent()
        assert agent.name == "researcher"
        assert "Researcher" in agent.system_prompt

    def test_coder_config(self):
        agent = CoderAgent()
        assert agent.name == "coder"
        assert "Coder" in agent.system_prompt

    def test_reviewer_config(self):
        agent = ReviewerAgent()
        assert agent.name == "reviewer"
        assert "Reviewer" in agent.system_prompt

    def test_fact_checker_config(self):
        agent = FactCheckerAgent()
        assert agent.name == "fact_checker"
        assert "Fact Checker" in agent.system_prompt

    def test_all_have_run_method(self):
        for cls in [ResearcherAgent, CoderAgent, ReviewerAgent, FactCheckerAgent]:
            agent = cls()
            assert callable(getattr(agent, "run", None))


class TestAgentRegistry:
    def test_default_agents_registered(self):
        assert "researcher" in AGENT_REGISTRY
        assert "coder" in AGENT_REGISTRY
        assert "reviewer" in AGENT_REGISTRY
        assert "fact_checker" in AGENT_REGISTRY

    def test_register_custom_agent(self):
        class CustomAgent(BaseAgent):
            name = "custom"
            system_prompt = "Custom agent"
            def run(self, state: AgentState) -> AgentState:
                return state

        register_agent_class("custom", CustomAgent)
        assert "custom" in AGENT_REGISTRY
        assert AGENT_REGISTRY["custom"] == CustomAgent

        # Cleanup
        del AGENT_REGISTRY["custom"]

    def test_registry_values_are_classes(self):
        for name, cls in AGENT_REGISTRY.items():
            assert isinstance(cls, type), f"{name} should be a class"
            assert issubclass(cls, BaseAgent), f"{name} should subclass BaseAgent"
