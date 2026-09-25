import time
import torch
import pytest
from cognitive_engine.core.world_model import (
    MentalSimulator,
    LatentWorldModel,
    BackgroundWorldModelTuner,
)


def test_prospective_veto_high_risk_pruning():
    sim = MentalSimulator(risk_threshold=0.80)

    # Safe computational code
    safe_code = "x = [i * 2 for i in range(10)]\ny = sum(x)"
    is_vetoed, risk, reason = sim.prospective_veto(safe_code, horizon=5, risk_threshold=0.80)
    assert is_vetoed is False
    assert risk < 0.80

    # Destructive action with unbounded filesystem removal
    dangerous_code = "import os, shutil\nshutil.rmtree('/important_data')"
    is_vetoed_dang, risk_dang, reason_dang = sim.prospective_veto(dangerous_code, horizon=5, risk_threshold=0.80)
    assert is_vetoed_dang is True
    assert risk_dang >= 0.80


def test_latent_world_model_multi_step_rollout():
    model = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=32)
    s_0 = torch.randn(14)
    action_seq = [0, 1, 2, 0, 1]

    rollout = model.rollout_trajectory(s_0, action_seq, gamma=0.95)
    assert "latent_trajectory" in rollout
    assert len(rollout["latent_trajectory"]) == len(action_seq) + 1
    assert "step_rewards" in rollout
    assert len(rollout["step_rewards"]) == len(action_seq)
    assert "max_risk" in rollout
    assert 0.0 <= rollout["max_risk"] <= 1.0


def test_background_world_model_tuner_asynchronous():
    model = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=32)
    tuner = BackgroundWorldModelTuner(model, buffer_capacity=100)

    try:
        # Enqueue sample transitions
        for i in range(10):
            s_t = torch.randn(14)
            s_tp1 = s_t + 0.1 * torch.randn(14)
            a_idx = i % 4
            tuner.enqueue_transition(s_t, a_idx=a_idx, s_tp1=s_tp1, reward=1.0, risk=0.1)

        # Allow background worker thread to process enqueued transitions
        for _ in range(40):
            if tuner.total_training_steps >= 1:
                break
            time.sleep(0.05)

        stats = tuner.get_stats()
        assert stats["buffer_size"] >= 10
        assert stats["total_training_steps"] >= 1
    finally:
        tuner.stop()
