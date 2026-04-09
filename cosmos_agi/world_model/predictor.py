"""Transformer-based next-state predictor.

This is a small-scale world model that learns to predict the next world state
given a current state and a proposed action. It operates in two modes:

1. **Neural mode**: A small Transformer trained on state transition history
   (suitable for domains with enough data, e.g. game environments).
2. **LLM mode**: Uses the language model itself as a world simulator,
   prompting it to predict consequences of actions (works zero-shot).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_agi.world_model.state_representation import Transition, WorldState

logger = logging.getLogger(__name__)


# ── Small Transformer for next-state prediction ────────────────────


class StateTransformer(nn.Module):
    """Minimal Transformer encoder for state-sequence prediction.

    Input: sequence of (state_vector, action_embedding) pairs
    Output: predicted next state_vector
    """

    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Project state+action into d_model
        self.input_proj = nn.Linear(state_dim + action_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to state space
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, state_action_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_action_seq: (batch, seq_len, state_dim + action_dim)
        Returns:
            predicted_state: (batch, state_dim)
        """
        batch_size, seq_len, _ = state_action_seq.shape

        x = self.input_proj(state_action_seq)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        # Use last position's output as the prediction
        return self.output_proj(x[:, -1, :])


# ── Neural predictor wrapper ───────────────────────────────────────


class NeuralPredictor:
    """Wraps the StateTransformer with training and inference methods."""

    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 32,
        lr: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = StateTransformer(state_dim=state_dim, action_dim=action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.training_history: list[Transition] = []

    def _encode_state(self, state: WorldState) -> torch.Tensor:
        """Encode a world state into a fixed-size vector."""
        if state.state_vector and len(state.state_vector) == self.state_dim:
            return torch.tensor(state.state_vector, dtype=torch.float32)
        # Fallback: hash-based encoding from text
        text = state.to_text()
        vec = [0.0] * self.state_dim
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.state_dim] += ch / 255.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
        vec = [v / norm for v in vec]
        return torch.tensor(vec, dtype=torch.float32)

    def _encode_action(self, action: str) -> torch.Tensor:
        """Encode an action string into a fixed-size vector."""
        vec = [0.0] * self.action_dim
        for i, ch in enumerate(action.encode("utf-8")):
            vec[i % self.action_dim] += ch / 255.0
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
        vec = [v / norm for v in vec]
        return torch.tensor(vec, dtype=torch.float32)

    def predict(self, state: WorldState, action: str) -> torch.Tensor:
        """Predict the next state vector given current state and action."""
        self.model.eval()
        with torch.no_grad():
            state_vec = self._encode_state(state)
            action_vec = self._encode_action(action)
            combined = torch.cat([state_vec, action_vec]).unsqueeze(0).unsqueeze(0)
            predicted = self.model(combined)
            return predicted.squeeze(0)

    def train_step(self, transitions: list[Transition]) -> float:
        """Train on a batch of transitions. Returns the loss."""
        if not transitions:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        for t in transitions:
            state_vec = self._encode_state(t.before)
            action_vec = self._encode_action(t.action)
            target_vec = self._encode_state(t.after)

            combined = torch.cat([state_vec, action_vec]).unsqueeze(0).unsqueeze(0)
            predicted = self.model(combined).squeeze(0)

            loss = F.mse_loss(predicted, target_vec)
            total_loss += loss

        avg_loss = total_loss / len(transitions)
        avg_loss.backward()
        self.optimizer.step()

        return avg_loss.item()

    def record_transition(self, transition: Transition) -> None:
        """Add a transition to training history."""
        self.training_history.append(transition)

    def train_on_history(self, epochs: int = 10, batch_size: int = 16) -> list[float]:
        """Train on accumulated history. Returns per-epoch losses."""
        losses = []
        for epoch in range(epochs):
            # Simple full-batch training for small datasets
            batch = self.training_history[-batch_size:]
            loss = self.train_step(batch)
            losses.append(loss)
            if epoch % 5 == 0:
                logger.debug("Epoch %d: loss=%.6f", epoch, loss)
        return losses


# ── LLM-based predictor ───────────────────────────────────────────


class LLMPredictor:
    """Uses an LLM to predict the next world state (zero-shot world simulation)."""

    SYSTEM_PROMPT = """You are a world simulator inside CosmosAGI.
Given a current world state and a proposed action, predict the resulting world state.

Think about:
1. Direct effects of the action
2. Side effects and cascading consequences
3. What new information would become known
4. What constraints might be violated

Output ONLY valid JSON with this structure:
{
  "predicted_entities": {"name": {"properties": {}, "relations": [["rel", "target"]]}},
  "predicted_facts": ["fact1", "fact2"],
  "predicted_constraints": ["constraint1"],
  "confidence": 0.8,
  "reasoning": "Why I predict this outcome"
}"""

    def predict(self, state: WorldState, action: str) -> dict[str, Any]:
        """Predict the next state using the LLM."""
        from cosmos_agi.core.llm import completion_json

        prompt = f"""Current world state:
{state.to_text()}

Proposed action: {action}

Predict the world state after this action is executed."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = completion_json(messages)
        logger.info(
            "LLM prediction: confidence=%.2f, %d entities, %d facts",
            result.get("confidence", 0),
            len(result.get("predicted_entities", {})),
            len(result.get("predicted_facts", [])),
        )
        return result

    def predict_as_state(self, state: WorldState, action: str) -> WorldState:
        """Predict and return a full WorldState object."""
        result = self.predict(state, action)
        new_state = WorldState()

        for name, data in result.get("predicted_entities", {}).items():
            entity = new_state.add_entity(name, data.get("properties", {}))
            for rel, target in data.get("relations", []):
                new_state.add_relation(name, rel, target)

        for fact in result.get("predicted_facts", []):
            new_state.add_fact(fact)

        new_state.constraints = result.get("predicted_constraints", [])
        return new_state
