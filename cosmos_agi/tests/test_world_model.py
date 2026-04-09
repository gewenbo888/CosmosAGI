"""Tests for the world model components."""

import torch

from cosmos_agi.world_model.state_representation import Entity, Transition, WorldState
from cosmos_agi.world_model.predictor import NeuralPredictor, StateTransformer
from cosmos_agi.world_model.causal import CausalGraph, CausalLink


# ── State representation tests ──────────────────────────────────


class TestWorldState:
    def test_create_empty(self):
        state = WorldState()
        assert state.entities == {}
        assert state.facts == []

    def test_add_entity(self):
        state = WorldState()
        e = state.add_entity("robot", {"position": [0, 0], "battery": 100})
        assert e.name == "robot"
        assert state.entities["robot"].properties["battery"] == 100

    def test_add_relation(self):
        state = WorldState()
        state.add_entity("robot")
        state.add_relation("robot", "is_in", "room_A")
        assert ("is_in", "room_A") in state.entities["robot"].relations
        assert "room_A" in state.entities  # auto-created

    def test_add_fact(self):
        state = WorldState()
        state.add_fact("The door is locked")
        state.add_fact("The door is locked")  # duplicate
        assert len(state.facts) == 1

    def test_to_text(self):
        state = WorldState()
        state.add_entity("robot", {"battery": 80})
        state.add_fact("It is raining")
        text = state.to_text()
        assert "robot" in text
        assert "raining" in text
        assert "battery=80" in text

    def test_transition(self):
        before = WorldState()
        before.add_entity("robot", {"position": "A"})
        after = WorldState()
        after.add_entity("robot", {"position": "B"})
        t = Transition(action="move_to_B", before=before, after=after, reward=1.0)
        assert t.action == "move_to_B"
        assert t.reward == 1.0


# ── Neural predictor tests ──────────────────────────────────────


class TestStateTransformer:
    def test_forward_shape(self):
        model = StateTransformer(state_dim=64, action_dim=16, d_model=64)
        x = torch.randn(2, 3, 80)  # batch=2, seq=3, state+action=64+16
        out = model(x)
        assert out.shape == (2, 64)

    def test_single_step(self):
        model = StateTransformer(state_dim=32, action_dim=8, d_model=32)
        x = torch.randn(1, 1, 40)
        out = model(x)
        assert out.shape == (1, 32)


class TestNeuralPredictor:
    def test_predict_returns_tensor(self):
        pred = NeuralPredictor(state_dim=32, action_dim=8)
        state = WorldState()
        state.add_entity("test", {"value": 42})
        result = pred.predict(state, "do_something")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32,)

    def test_train_step(self):
        pred = NeuralPredictor(state_dim=32, action_dim=8)
        before = WorldState()
        before.add_entity("x", {"v": 1})
        after = WorldState()
        after.add_entity("x", {"v": 2})

        transitions = [
            Transition(action="increment", before=before, after=after)
            for _ in range(4)
        ]
        loss = pred.train_step(transitions)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_on_history(self):
        pred = NeuralPredictor(state_dim=32, action_dim=8)
        before = WorldState()
        before.add_entity("a", {"v": 0})
        after = WorldState()
        after.add_entity("a", {"v": 1})

        for _ in range(5):
            pred.record_transition(
                Transition(action="step", before=before, after=after)
            )

        losses = pred.train_on_history(epochs=5)
        assert len(losses) == 5
        # Loss should generally decrease (or at least not explode)
        assert all(l >= 0 for l in losses)


# ── Causal graph tests ──────────────────────────────────────────


class TestCausalGraph:
    def test_add_and_get(self):
        g = CausalGraph()
        g.add("rain", "wet_road", strength=0.9, mechanism="water falls on road")
        effects = g.get_effects("rain")
        assert len(effects) == 1
        assert effects[0].effect == "wet_road"

    def test_trace_downstream(self):
        g = CausalGraph()
        g.add("rain", "wet_road", strength=0.9)
        g.add("wet_road", "car_accident", strength=0.6)
        g.add("car_accident", "traffic_jam", strength=0.8)

        downstream = g.trace_downstream("rain")
        nodes = [n for n, _, _ in downstream]
        assert "wet_road" in nodes
        assert "car_accident" in nodes
        assert "traffic_jam" in nodes

    def test_trace_upstream(self):
        g = CausalGraph()
        g.add("rain", "wet_road", strength=0.9)
        g.add("oil_spill", "wet_road", strength=0.7)

        upstream = g.trace_upstream("wet_road")
        nodes = [n for n, _, _ in upstream]
        assert "rain" in nodes
        assert "oil_spill" in nodes

    def test_counterfactual(self):
        g = CausalGraph()
        g.add("rain", "wet_road", strength=0.9)
        g.add("wet_road", "car_accident", strength=0.6)

        affected = g.counterfactual("rain")
        assert "wet_road" in affected
        assert "car_accident" in affected

    def test_counterfactual_with_alternative(self):
        g = CausalGraph()
        g.add("rain", "wet_road", strength=0.9)
        g.add("sprinkler", "wet_road", strength=0.7)
        g.add("wet_road", "car_accident", strength=0.6)

        # wet_road has alternative cause (sprinkler), so removing rain
        # should NOT make wet_road "affected"
        affected = g.counterfactual("rain")
        assert "wet_road" not in affected

    def test_cumulative_strength(self):
        g = CausalGraph()
        g.add("A", "B", strength=0.8)
        g.add("B", "C", strength=0.5)

        downstream = g.trace_downstream("A")
        strengths = {n: s for n, s, _ in downstream}
        assert abs(strengths["B"] - 0.8) < 0.01
        assert abs(strengths["C"] - 0.4) < 0.01  # 0.8 * 0.5

    def test_to_text(self):
        g = CausalGraph()
        g.add("rain", "flood", strength=0.9, mechanism="excessive water")
        text = g.to_text()
        assert "rain" in text
        assert "flood" in text
        assert "excessive water" in text
