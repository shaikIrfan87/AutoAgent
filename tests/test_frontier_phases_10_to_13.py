import os
import tempfile
import torch
import pytest

from cognitive_engine.core.world_model import MentalSimulator, LatentWorldModel
from cognitive_engine.core.lambda_dsl import (
    App, Prim, Var, Const, Lambda, MapNode, FoldNode,
    eval_lambda_ast, CORE_LAMBDA_PRIMITIVES, ast_size
)
from cognitive_engine.core.compression import LambdaSubtreeMiner
from cognitive_engine.core.plastic_layer import FastPlasticLinear
from cognitive_engine.core.plastic_synapse import BoundedPlasticTTTLayer
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.agent.curiosity import CuriosityDaemon


def test_phase10_latent_world_model_simulation_and_veto():
    """Phase 10: Contrastive Latent World-Model mental simulation & trajectory pruning."""
    sim = MentalSimulator()

    # 1. Safe deterministic code should pass prospective veto
    safe_code = "x = 10\ny = 20\nresult = x + y\nprint(result)"
    is_vetoed, risk, reason = sim.prospective_veto(safe_code, horizon=5, risk_threshold=0.80)
    assert not is_vetoed, f"Safe code was vetoed: {reason}"
    assert risk < 0.80

    # 2. Dangerous destructive code should be vetoed before OS child process spawn
    destructive_code = "import shutil\nshutil.rmtree('/important/data')"
    is_vetoed_dang, risk_dang, reason_dang = sim.prospective_veto(destructive_code, horizon=5, risk_threshold=0.80)
    assert is_vetoed_dang, "Destructive code bypassed prospective veto"
    assert risk_dang >= 0.80
    assert "rejected" in reason_dang.lower() or "risk" in reason_dang.lower()

    # 3. Verify continuous latent dynamics training step
    lwm = LatentWorldModel(state_dim=14, action_dim=22, hidden_dim=64)
    s_t = torch.randn(14)
    s_tp1 = torch.randn(14)
    loss = lwm.train_transition_step(s_t, a_idx=1, s_tp1=s_tp1, reward_target=1.0, risk_target=0.05)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_phase11_dreamcoder_anti_unification_and_mdl_promotion():
    """Phase 11: DreamCoder-Style Anti-Unification & MDL Macro Mining."""
    miner = LambdaSubtreeMiner(min_frequency=2, min_size=2)

    # Construct two verified programs sharing a common subtree: App(Prim("add"), Const(5))
    shared_sub = App(Prim("add"), Const(5))
    prog1 = Lambda("x", App(shared_sub, Var("x")))  # \x. add 5 x
    prog2 = Lambda("y", App(Prim("mul"), App(shared_sub, Var("y"))))  # \y. mul (add 5 y)

    # Extract subtrees
    subtrees = miner.extract_subtrees(prog1)
    assert len(subtrees) >= 1
    assert any(ast_size(s) >= 2 for s in subtrees)

    # Compute MDL compression gain for repeated pattern
    gain = miner.compute_mdl_gain(shared_sub, occurrences=3)
    assert gain > 0.0, f"Expected positive MDL gain, got {gain}"

    # Mine and promote macros across verified ASTs
    promoted = miner.mine_and_promote([prog1, prog2])
    assert len(promoted) >= 1
    macro = promoted[0]
    assert macro.frequency >= 2
    assert macro.mdl_gain > 0
    assert macro.name in CORE_LAMBDA_PRIMITIVES

    # Verify that the promoted macro is executable through CORE_LAMBDA_PRIMITIVES
    promoted_fn = CORE_LAMBDA_PRIMITIVES[macro.name]
    assert callable(promoted_fn)


def test_phase12_continuous_in_context_ttt_attention_binding():
    """Phase 12: Continuous In-Context TTT-Attention Binding."""
    # 1. Test FastPlasticLinear multi-head KV cache modulation (4D tensor)
    plastic_linear = FastPlasticLinear(in_features=64, out_features=64, num_heads=4)
    k = torch.randn(64)
    v = torch.randn(64)
    norm, _ = plastic_linear.adapt_online(k, v, delta_s=1.0)
    assert norm > 0.0

    # 4D: (batch=2, num_heads=4, seq_len=8, head_dim=16)
    k_cache = torch.randn(2, 4, 8, 16)
    v_cache = torch.randn(2, 4, 8, 16)
    k_mod, v_mod = plastic_linear.modulate_kv_cache(k_cache, v_cache, alpha=0.1)
    assert k_mod.shape == k_cache.shape
    assert v_mod.shape == v_cache.shape
    assert not torch.allclose(k_mod, k_cache), "KV cache modulation did not affect key representations"

    # Anti-Hebbian decay on failure: Delta S <= 0.0 suppresses weights
    norm_before = float(torch.linalg.norm(plastic_linear.A_fast, ord="fro").item())
    plastic_linear.adapt_online(k, v, delta_s=-1.0)
    norm_after = float(torch.linalg.norm(plastic_linear.A_fast, ord="fro").item())
    assert norm_after <= norm_before * 0.86, f"Expected 0.85 decay, got {norm_after} from {norm_before}"

    # 2. Test BoundedPlasticTTTLayer multi-head KV cache modulation
    bounded_ttt = BoundedPlasticTTTLayer(dim=64, num_heads=4)
    bounded_ttt.adapt_online(k, v, delta_s=1.0)
    k_mod2, v_mod2 = bounded_ttt.modulate_kv_cache(k_cache, v_cache, alpha=0.1)
    assert k_mod2.shape == k_cache.shape
    assert not torch.allclose(k_mod2, k_cache)


def test_phase13_autotelic_active_inference_epistemic_scan():
    """Phase 13: 24/7 Autotelic Active-Inference Life Engine."""
    from features.execution_sandbox.sandbox import safe_cleanup_temp_dir
    tmp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmp_dir, "test_curiosity_memory.db")
        engine = CognitiveEngine(db_path=db_path)
        try:
            # Seed an epistemic memory node with confidence 0.40 (< 0.60)
            engine.consolidation.write_memory(
                content="gcd algorithm computes greatest common divisor",
                confidence=0.40,
            )

            daemon = CuriosityDaemon(engine=engine, interval_sec=1.0)
            decayed = daemon.scan_decayed_memories(threshold=0.60)
            assert len(decayed) >= 1, "Failed to identify low-confidence epistemic memory node"
            assert any("gcd" in d["topic"].lower() for d in decayed)

            # Step the curiosity daemon - should trigger staged synthesis or inquiry
            step_res = daemon.step()
            assert step_res is not None
        finally:
            engine.sandbox.close()
            engine.consolidation.close()
    finally:
        safe_cleanup_temp_dir(tmp_dir)
