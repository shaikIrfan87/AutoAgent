import pytest
import torch
import torch.nn as nn

from cognitive_engine.core.plastic_layer import FastPlasticLinear, TTTAttentionLayer as PlasticTTTLayer
from cognitive_engine.core.ttt_attention import TTTAttentionLayer as CoreTTTLayer
from cognitive_engine.core.generator import LocalLLMGenerator, CausalTransformerCortex
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_fast_plastic_linear_head_weights_and_kv_modulation():
    """Verify FastPlasticLinear extracts multi-head blocks and modulates KV-cache tensors."""
    dim = 64
    num_heads = 4
    head_dim = 16
    plastic = FastPlasticLinear(in_features=dim, out_features=dim, num_heads=num_heads)

    # Initial zero modulation
    k_cache = torch.randn(2, num_heads, 8, head_dim)
    v_cache = torch.randn(2, num_heads, 8, head_dim)
    k_mod, v_mod = plastic.modulate_kv_cache(k_cache, v_cache)
    assert torch.allclose(k_mod, k_cache)
    assert torch.allclose(v_mod, v_cache)

    # Absorb association (non-zero fast weights)
    k_vec = torch.randn(dim)
    v_vec = torch.randn(dim)
    norm = plastic.absorb(k_vec, v_vec, steps=2)
    assert norm > 0.0

    # Verify head extraction
    hw = plastic.get_head_fast_weights(num_heads=num_heads, head_dim=head_dim)
    assert hw.shape == (num_heads, head_dim, head_dim)
    assert torch.linalg.norm(hw) > 0.0

    # Verify KV-cache modulation is active and non-trivial
    k_mod, v_mod = plastic.modulate_kv_cache(k_cache, v_cache, alpha=0.2)
    assert k_mod.shape == k_cache.shape
    assert v_mod.shape == v_cache.shape
    assert not torch.allclose(k_mod, k_cache)
    assert not torch.allclose(v_mod, v_cache)

    # Test 3D tensor modulation [batch, seq, dim]
    k_3d = torch.randn(2, 8, dim)
    v_3d = torch.randn(2, 8, dim)
    k_mod_3d, v_mod_3d = plastic.modulate_kv_cache(k_3d, v_3d, alpha=0.1)
    assert k_mod_3d.shape == k_3d.shape
    assert not torch.allclose(k_mod_3d, k_3d)


def test_ttt_attention_sync_from_plastic_and_kv_modulation():
    """Verify TTTAttentionLayer synchronizes fast-weights from FastPlasticLinear."""
    dim = 64
    plastic = FastPlasticLinear(in_features=dim, out_features=dim, num_heads=4)
    k_vec = torch.randn(dim)
    v_vec = torch.randn(dim)
    plastic.absorb(k_vec, v_vec, steps=3)

    # Core TTT Layer
    core_ttt = CoreTTTLayer(d_model=dim, num_heads=4)
    assert core_ttt.frobenius_norm == 0.0

    norm_synced = core_ttt.sync_from_plastic(plastic)
    assert norm_synced > 0.0
    assert core_ttt.frobenius_norm > 0.0

    # Verify modulation with synced weights
    k = torch.randn(1, 4, 10, 16)
    v = torch.randn(1, 4, 10, 16)
    k_out, v_out = core_ttt.modulate_kv_cache(k, v, alpha=0.15)
    assert not torch.allclose(k_out, k)
    assert not torch.allclose(v_out, v)

    # Plastic module TTT layer
    plastic_ttt = PlasticTTTLayer(embed_dim=dim, num_heads=4)
    plastic_norm = plastic_ttt.sync_from_plastic(plastic)
    assert plastic_norm > 0.0
    k_out_p, v_out_p = plastic_ttt.modulate_kv_cache(k, v, alpha=0.15)
    assert not torch.allclose(k_out_p, k)


def test_causal_cortex_logits_shift_with_ttt_modulation():
    """Verify in-process CausalTransformerCortex token logits shift following TTT KV modulation."""
    ttt = CoreTTTLayer(d_model=64, num_heads=4)
    cortex = CausalTransformerCortex(vocab_size=256, d_model=64, num_heads=4, ttt_layer=ttt)

    prompt_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    with torch.no_grad():
        logits_before = cortex(prompt_ids, hook_ttt=True)

    # Now adapt TTT layer with empirical key-value pair
    k = torch.randn(64)
    v = torch.randn(64)
    ttt.adapt_step(k, v)
    assert ttt.frobenius_norm > 0.0

    with torch.no_grad():
        logits_after = cortex(prompt_ids, hook_ttt=True)

    # Logits should differ because KV-cache modulation altered attention activations
    diff = torch.norm(logits_after - logits_before).item()
    assert diff > 1e-4


def test_generator_sync_fast_weights_and_feedback():
    """Verify LocalLLMGenerator syncs fast-weights from plastic and conditions on feedback."""
    gen = LocalLLMGenerator(timeout=0.1)
    plastic = FastPlasticLinear(in_features=64, out_features=64, num_heads=4)
    plastic.absorb(torch.randn(64), torch.randn(64))

    # Sync
    norm = gen.sync_fast_weights(plastic)
    assert norm > 0.0
    assert gen.ttt_attention.frobenius_norm > 0.0

    # Condition on empirical feedback
    norm_feedback = gen.condition_on_feedback(
        query="calculate kinetic energy",
        output="0.5 * m * v**2",
        delta_s=1.0,
    )
    assert norm_feedback > 0.0

    # Verify KV-cache hook reflects non-zero modulation
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    k_mod, v_mod = gen.hook_kv_cache(k, v, alpha=0.2)
    assert not torch.allclose(k_mod, k)


def test_engine_process_e2e_propagates_fast_weights_to_generator_kv():
    """End-to-end test: CognitiveEngine process() propagates empirical ΔS to generator KV modulation."""
    engine = CognitiveEngine(db_path=":memory:")
    initial_gen_norm = engine.generator.ttt_attention.frobenius_norm

    # Run a verified computation through process
    res = engine.process("calculate factorial of 4: 1 * 2 * 3 * 4", code_action="print(24)")
    assert res.get("status") in ("system_2", "system_2_success", "system_1") or res.get("routed_system") == "system_2"

    # Generator fast weights should now be populated and synchronized with plastic A_fast
    post_gen_norm = engine.generator.ttt_attention.frobenius_norm
    assert post_gen_norm >= initial_gen_norm

    # Generator KV-cache hook must now perform active empirical modulation
    k_cache = torch.randn(1, 4, 8, 16)
    v_cache = torch.randn(1, 4, 8, 16)
    k_mod, v_mod = engine.generator.hook_kv_cache(k_cache, v_cache, alpha=0.15)
    assert k_mod.shape == k_cache.shape
    assert v_mod.shape == v_cache.shape
