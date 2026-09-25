import time
import pytest
import torch

from cognitive_engine.core.generator import LocalLLMGenerator
from cognitive_engine.core.world_model import MentalSimulator, BackgroundWorldModelTuner, LatentWorldModel
from cognitive_engine.agent.ast_policy_mcts import ASTGuidedMCTS, ASTPolicyValueNetwork
from features.enclave_ipc import CapabilityManager, CAP_WRITE, CAP_READ


def test_frontier1_quantized_backbone_and_ttt_conditioning():
    gen = LocalLLMGenerator(timeout=0.1)
    assert gen.ttt_attention is not None

    # Initial state
    initial_norm = gen.ttt_attention.frobenius_norm

    # Condition on prompt context
    norm_after = gen.condition_on_context("def calculate_momentum(mass, velocity): return mass * velocity")
    assert norm_after <= 2.0

    # Generation conditioning update
    code = gen.generate_code("calculate kinetic energy: 1000kg at 20m/s")
    assert isinstance(code, str) and len(code) > 0
    assert gen.ttt_attention.frobenius_norm <= 2.0

    # KV-cache projection modulation
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    k_mod, v_mod = gen.hook_kv_cache(k, v)
    assert k_mod.shape == k.shape
    assert v_mod.shape == v.shape


def test_frontier2_dynamic_online_world_model_fine_tuning():
    latent_model = LatentWorldModel()
    tuner = BackgroundWorldModelTuner(latent_model)

    s_t = torch.randn(14)
    s_tp1 = torch.randn(14)

    # Enqueue real physical transitions
    tuner.enqueue_transition(s_t, a_idx=1, s_tp1=s_tp1, reward=1.0, risk=0.05)
    tuner.enqueue_transition(s_t, a_idx=2, s_tp1=s_tp1, reward=-1.0, risk=0.90)

    # Wait for background worker to process
    for _ in range(30):
        if tuner.total_training_steps >= 1:
            break
        time.sleep(0.05)
    assert tuner.total_training_steps >= 1
    assert tuner.last_loss > 0.0

    tuner.stop()

    # MentalSimulator recording
    sim = MentalSimulator()
    sim.record_sandbox_transition("ke = 0.5 * 10 * 100", delta_s=1.0)
    for _ in range(30):
        if sim.tuner.total_training_steps >= 1:
            break
        time.sleep(0.05)
    assert sim.tuner.total_training_steps >= 1
    sim.tuner.stop()



def test_frontier3_multi_statement_ast_grammar_expansion():
    net = ASTPolicyValueNetwork(num_actions=16)
    mcts = ASTGuidedMCTS(net=net, max_expansions=60)

    # Conditional parity transformation: x * 2 if x % 2 == 0 else x + 1
    io_pairs = [(2, 4), (4, 8), (1, 2), (3, 4)]
    solution = mcts.synthesize(io_pairs)
    assert solution is not None
    assert "def solution(x):" in solution

    env = {}
    exec(solution, env)
    assert env["solution"](6) == 12
    assert env["solution"](5) == 6


def test_frontier4_ephemeral_capability_leases_and_anti_replay():
    mgr = CapabilityManager()

    # 1. Short TTL lease token
    short_lease_token = mgr.mint_token("enclave_alpha", rights=CAP_WRITE, lease_sec=0.1)
    assert not short_lease_token.is_expired()
    assert mgr.verify_token(short_lease_token, required_right=CAP_WRITE)

    # Wait for expiration
    time.sleep(0.15)
    assert short_lease_token.is_expired()
    assert not mgr.verify_token(short_lease_token, required_right=CAP_WRITE)

    # 2. Token lease renewal
    valid_token = mgr.mint_token("enclave_beta", rights=CAP_READ, lease_sec=10.0)
    renewed_token = mgr.renew_lease(valid_token, extension_sec=60.0)
    assert renewed_token.seq_num == valid_token.seq_num + 1
    assert mgr.verify_token(renewed_token, required_right=CAP_READ)

    # Old token was revoked during renewal
    assert not mgr.verify_token(valid_token)

    # 3. Monotonic anti-replay counter
    assert mgr.verify_token(renewed_token, check_monotonic=True)
