"""Integration layer — connects the world model into the agent loop.

The WorldModelAgent sits between planning and execution. Before executing
a subtask, it:
1. Builds a world state from current context
2. Predicts consequences of the proposed action
3. Flags risks or unexpected outcomes
4. Updates the causal graph with observations
"""

from __future__ import annotations

import logging
from typing import Any

from cosmos_agi.agents.base import BaseAgent
from cosmos_agi.core.state import AgentPhase, AgentState
from cosmos_agi.world_model.causal import CausalGraph, CausalReasoner
from cosmos_agi.world_model.predictor import LLMPredictor, NeuralPredictor
from cosmos_agi.world_model.state_representation import Transition, WorldState

logger = logging.getLogger(__name__)


class WorldModelAgent(BaseAgent):
    """Agent that maintains and queries the world model."""

    name = "world_modeler"
    system_prompt = """You are the World Model agent in CosmosAGI.
Your job is to maintain an internal model of how the world works, predict
consequences of actions, and flag potential risks.

Given the current task context and a proposed action, you must:
1. Build a world state snapshot
2. Predict what will happen
3. Identify risks or side effects
4. Recommend whether to proceed, modify, or abort

Output ONLY valid JSON:
{
  "world_state_summary": "...",
  "prediction": "What will likely happen",
  "risks": ["risk1", "risk2"],
  "recommendation": "proceed" | "modify" | "abort",
  "modification_suggestion": "If modify, what to change",
  "confidence": 0.8
}"""

    def __init__(self) -> None:
        self.llm_predictor = LLMPredictor()
        self.neural_predictor = NeuralPredictor()
        self.causal_graph = CausalGraph()
        self.causal_reasoner = CausalReasoner(self.causal_graph)
        self.state_history: list[WorldState] = []

    def build_world_state(self, state: AgentState) -> WorldState:
        """Construct a WorldState from the current agent state."""
        world = WorldState()

        # Add the task as the central entity
        world.add_entity("task", {"description": state.task, "phase": state.phase.value})

        # Add subtasks as entities
        for st in state.subtasks:
            world.add_entity(
                f"subtask_{st.id}",
                {"description": st.description, "status": st.status},
            )
            world.add_relation("task", "has_subtask", f"subtask_{st.id}")

        # Add observations as facts
        for obs in state.observations:
            world.add_fact(obs)

        # Add errors as constraints
        for err in state.errors:
            world.add_fact(f"ERROR: {err}")

        return world

    def predict_action_outcome(
        self,
        state: AgentState,
        action: str,
    ) -> dict[str, Any]:
        """Predict the outcome of a proposed action."""
        world_state = self.build_world_state(state)

        prompt = (
            f"Task: {state.task}\n"
            f"Current world state:\n{world_state.to_text()}\n\n"
            f"Proposed action: {action}\n\n"
            f"Causal knowledge:\n{self.causal_graph.to_text()}"
        )

        result = self.call_llm_json(prompt, state)
        return result

    def run(self, state: AgentState) -> AgentState:
        """Run world model analysis on the current state.

        Called before execution to enrich the state with predictions.
        """
        logger.info("WorldModel: analyzing state (iteration %d)", state.iteration)

        world_state = self.build_world_state(state)
        self.state_history.append(world_state)

        # If we have a current subtask, predict its outcome
        if state.current_subtask_index < len(state.subtasks):
            subtask = state.subtasks[state.current_subtask_index]

            try:
                prediction = self.predict_action_outcome(state, subtask.description)

                # If the model recommends aborting, flag it
                if prediction.get("recommendation") == "abort":
                    state.observations.append(
                        f"WorldModel WARNING: Recommends aborting subtask {subtask.id}: "
                        f"{prediction.get('risks', [])}"
                    )
                    logger.warning("WorldModel recommends aborting subtask %d", subtask.id)
                elif prediction.get("recommendation") == "modify":
                    state.observations.append(
                        f"WorldModel SUGGESTION: Modify subtask {subtask.id}: "
                        f"{prediction.get('modification_suggestion', 'N/A')}"
                    )

                # Store prediction confidence
                confidence = prediction.get("confidence", 0.5)
                state.observations.append(
                    f"WorldModel prediction confidence: {confidence:.2f}"
                )

            except Exception as e:
                logger.warning("WorldModel prediction failed: %s", e)

        return state

    def update_from_observation(self, observation: str) -> None:
        """Update the causal graph based on a new observation."""
        try:
            self.causal_reasoner.discover_causes(observation)
        except Exception as e:
            logger.warning("Causal discovery failed: %s", e)

    def record_transition(
        self,
        before: AgentState,
        action: str,
        after: AgentState,
        reward: float = 0.0,
    ) -> None:
        """Record a state transition for the neural predictor."""
        before_world = self.build_world_state(before)
        after_world = self.build_world_state(after)

        transition = Transition(
            action=action,
            before=before_world,
            after=after_world,
            reward=reward,
        )
        self.neural_predictor.record_transition(transition)

    def train_neural_model(self, epochs: int = 10) -> list[float]:
        """Train the neural predictor on accumulated transitions."""
        if not self.neural_predictor.training_history:
            logger.info("No transitions to train on")
            return []
        losses = self.neural_predictor.train_on_history(epochs=epochs)
        logger.info("Neural model trained: final_loss=%.6f", losses[-1] if losses else 0)
        return losses
