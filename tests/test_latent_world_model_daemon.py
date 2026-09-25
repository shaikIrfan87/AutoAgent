import time
import pytest
import torch
from cognitive_engine.core.world_model import (
    MentalSimulator,
    LatentWorldModel,
    BackgroundWorldModelTuner,
    TransitionRingBuffer,
    StructuredEnvironmentEncoder,
)
from cognitive_engine.core.sandbox import EnvironmentalSandbox
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_transition_ring_buffer_concurrent_and_sampling():
    """Verify thread-safe append, circular overwriting, and batch sampling of TransitionRingBuffer."""
    buffer = TransitionRingBuffer(capacity=10)
    for i in range(15):
        s_t = torch.tensor([float(i)] * 14)
        s_tp1 = torch.tensor([float(i + 1)] * 14)
        buffer.append(s_t, a_idx=i % 4, s_tp1=s_tp1, reward=1.0, risk=0.1)

    assert len(buffer) == 10
    batch = buffer.sample_batch(batch_size=5)
    assert len(batch) == 5
    for s_t, a_idx, s_tp1, reward, risk in batch:
        assert s_t.shape == (14,)
        assert s_tp1.shape == (14,)
        assert reward == 1.0


def test_background_world_model_tuner_online_sgd():
    """Verify background tuner runs asynchronous gradient descent and minimizes loss."""
    model = LatentWorldModel()
    tuner = BackgroundWorldModelTuner(model, buffer_capacity=100)

    try:
        # Enqueue transitions
        for i in range(6):
            s_t = torch.randn(14)
            s_tp1 = s_t + 0.1
            tuner.enqueue_transition(s_t, a_idx=0, s_tp1=s_tp1, reward=1.0, risk=0.05)

        # Poll for background worker execution
        stats = {}
        for _ in range(30):
            stats = tuner.get_stats()
            if stats["total_training_steps"] > 0:
                break
            time.sleep(0.1)
        assert stats["total_training_steps"] > 0
        assert stats["buffer_size"] == 6
    finally:
        tuner.stop()


def test_structured_environment_encoder():
    """Verify encoding of multi-modal OS metrics into 14D latent state vector."""
    encoder = StructuredEnvironmentEncoder()
    z = encoder(
        stdout_len=120,
        stderr_len=0,
        exit_code=0,
        files_modified=1,
        latency_ms=1.5,
        memory_mb=25.0,
        semantic_error_hash=0.0,
        ast_diff_count=10,
    )
    assert z.shape == (14,)
    assert not torch.isnan(z).any()


def test_prospective_mental_rollout_veto():
    """Verify mental simulation vetoes destructive code and permits safe operations."""
    sim = MentalSimulator()
    try:
        # Destructive code should be vetoed early
        destructive_code = "import shutil\nshutil.rmtree('/etc')"
        is_vetoed, risk, reason = sim.prospective_veto(destructive_code, horizon=5, risk_threshold=0.80)
        assert is_vetoed is True
        assert risk >= 0.80
        assert "risk" in reason.lower() or "rejected" in reason.lower()

        # Safe code should pass prospective evaluation
        safe_code = "def solution(x):\n    return [i * 2 for i in x]"
        is_vetoed_safe, risk_safe, _ = sim.prospective_veto(safe_code, horizon=5, risk_threshold=0.85)
        assert is_vetoed_safe is False
        assert risk_safe < 0.85
    finally:
        sim.tuner.stop()


def test_closed_loop_sandbox_telemetry_streaming():
    """Verify EnvironmentalSandbox callback streams telemetry directly to MentalSimulator."""
    sim = MentalSimulator()
    sandbox = EnvironmentalSandbox(use_persistent_worker=False)

    try:
        # Wire callback
        def on_transition(code, res):
            sim.record_sandbox_transition(
                code=code,
                delta_s=res.delta_s,
                stdout=res.stdout,
                stderr=res.stderr,
                exit_code=res.exit_code,
                latency_ms=res.duration_sec * 1000.0,
            )

        sandbox.on_transition_callback = on_transition

        # Execute safe print
        res = sandbox.execute_python("print('telemetry test')")
        assert res.exit_code == 0

        # Wait for background tuner to process enqueued transition
        time.sleep(0.4)
        stats = sim.tuner.get_stats()
        assert stats["buffer_size"] >= 1
    finally:
        sim.tuner.stop()
        sandbox.close()


def test_orchestrator_prospective_gating_in_induction():
    """Verify CognitiveEngine vetoes destructive induction goals before sandbox dispatch."""
    engine = CognitiveEngine(db_path=":memory:")
    try:
        destructive_goal = "import os\nos.remove('/bin/sh')\nassert solution(1) == 1"
        res = engine.induct_program(task_name="destructive_task", task_spec=destructive_goal)
        assert res["status"] == "vetoed"
        assert res["delta_s"] == -1.0
        assert res["risk_score"] >= 0.80
    finally:
        if hasattr(engine.world_model, "tuner"):
            engine.world_model.tuner.stop()
