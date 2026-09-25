import numpy as np
import pytest
import torch

from cognitive_engine.core.generator import LocalLLMGenerator
from features.enclave_ipc import (
    CapabilityManager,
    SharedMemoryIPCRing,
    CAP_READ,
    CAP_WRITE,
    CAP_EXEC,
)
from features.protocol_guard import TypeSafeProtocolGuard
from cognitive_engine.agent.ast_policy_mcts import ASTPolicyValueNetwork, ASTGuidedMCTS
from cognitive_engine.agent.executive_loop import ExecutiveLoop


def test_phase1_generator_ttt_kv_hook():
    gen = LocalLLMGenerator(timeout=0.1)
    assert gen.ttt_attention is not None


    # Verify KV-cache hook modulation
    k_cache = torch.randn(1, 4, 16, 16)
    v_cache = torch.randn(1, 4, 16, 16)
    k_mod, v_mod = gen.hook_kv_cache(k_cache, v_cache)
    assert k_mod.shape == k_cache.shape
    assert v_mod.shape == v_cache.shape

    # Verify TTT generation with session context
    code, norm = gen.generate_code_with_ttt(
        task_prompt="Calculate kinetic energy: mass=1000, velocity=20",
        session_context="Session context: physics kinetic energy simulation"
    )
    assert isinstance(code, str) and len(code) > 0
    assert norm <= 2.0


def test_phase2_sel4_ipc_ring_and_capabilities():
    cap_mgr = CapabilityManager()
    token_write = cap_mgr.mint_token("sender_enclave", rights=CAP_WRITE)
    token_read = cap_mgr.mint_token("receiver_enclave", rights=CAP_READ)
    token_no_rights = cap_mgr.mint_token("untrusted", rights=0)

    ring = SharedMemoryIPCRing(capacity=8, cap_manager=cap_mgr)

    # Unauthorized push
    with pytest.raises(PermissionError):
        ring.push(b"data", token_no_rights)

    # Valid push
    slot = ring.push(b"secure_telemetry_payload", token_write)
    assert slot == 0
    assert ring.count == 1

    # Unauthorized pop
    with pytest.raises(PermissionError):
        ring.pop(token_write)  # token_write only has CAP_WRITE

    # Valid pop
    popped = ring.pop(token_read)
    assert popped == b"secure_telemetry_payload"
    assert ring.count == 0

    # Protocol guard capability verification
    guard = TypeSafeProtocolGuard(cap_manager=cap_mgr)
    token_exec = cap_mgr.mint_token("app", rights=CAP_EXEC)
    dec_valid = guard.evaluate_ingress("Safe query", capability_token=token_exec)
    assert dec_valid.is_compliant_noul

    cap_mgr.revoke_token(token_exec)
    dec_revoked = guard.evaluate_ingress("Safe query", capability_token=token_exec)
    assert not dec_revoked.is_compliant_noul
    assert dec_revoked.choice == "adversarial_exploit"


def test_phase3_ast_policy_guided_mcts():
    net = ASTPolicyValueNetwork(num_actions=8)
    feat = torch.randn(16)
    priors, val = net(feat)
    assert priors.shape == (8,)
    assert -1.0 <= float(val.item()) <= 1.0

    mcts = ASTGuidedMCTS(net=net, max_expansions=50)
    # Synthesize double operation: x -> 2 * x
    pairs = [(1, 2), (2, 4), (5, 10)]
    solution = mcts.synthesize(pairs)
    assert solution is not None
    assert "def solution(x):" in solution
    env = {}
    exec(solution, env)
    assert env["solution"](10) == 20


def test_phase4_executive_loop_latent_mental_rollout():
    loop = ExecutiveLoop()
    res = loop.run(
        goal="Compute area of rectangle",
        code="width = 20; height = 15; area = width * height; print(f'AREA={area}')"
    )
    assert res["success"]
    assert len(res["scratchpad"]) > 0
    sim_step = res["scratchpad"][0]
    assert "latent_mental_rollout" in sim_step
    assert not sim_step["latent_mental_rollout"]["divergent"]
