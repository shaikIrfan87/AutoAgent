import time
import pytest
import numpy as np
import torch
import ast

from cognitive_engine.core.generator import (
    LocalLLMGenerator,
    CausalTransformerCortex,
    LlamaTTTLogitsProcessor,
)
from cognitive_engine.core.scm_engine import LiveTelemetrySCMEngine
from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer
from features.enclave_ipc.capabilities import CapabilityManager, CapabilityToken, CAP_READ, CAP_WRITE
from features.enclave_ipc.ring_buffer import BinaryMmapIPCRing
from features.execution_sandbox.sandbox import HardenedOSSandboxExecutor


def test_production_frontier1_cortex_and_logits():
    gen = LocalLLMGenerator()
    assert gen.cortex is not None, "CausalTransformerCortex should be instantiated"

    # Test forward pass with KV modulation
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    logits = gen.cortex(input_ids, hook_ttt=True)
    assert logits.shape == (1, 4, 256)

    # Test autoregressive generation through cortex
    res = gen.generate_with_cortex("print(42)", max_tokens=8)
    assert isinstance(res, str)

    # Test LlamaTTTLogitsProcessor
    proc = LlamaTTTLogitsProcessor(ttt_attention=gen.ttt_attention, alpha=0.2)
    raw_logits = [1.0, 2.0, 0.5, -0.5]
    mod_logits = proc([1, 2], list(raw_logits))
    assert len(mod_logits) == len(raw_logits)


def test_production_frontier2_live_telemetry_scm():
    engine = LiveTelemetrySCMEngine(window_size=32)
    # Collect telemetry stream
    for i in range(16):
        engine.sample_step(loop_latency_ms=0.4 + 0.05 * (i % 3), sandbox_delta=1.0, ttt_norm=0.15)

    assert len(engine.buffer) == 16
    W = engine.update_causal_dag(max_iter=50)
    assert W is not None
    assert W.shape == (6, 6)
    assert engine.is_dag(threshold=1e-3)

    # Pearl Level-2 do-calculus on live telemetry
    do_res = engine.intervene_telemetry({"cpu_percent": 15.0})
    assert "cpu_percent" in do_res
    assert abs(do_res["cpu_percent"] - 15.0) < 1e-4

    # Pearl Level-3 counterfactual on live telemetry
    factual = {"cpu_percent": 20.0, "mem_percent": 50.0, "thread_count": 8.0, "loop_latency_ms": 0.5, "sandbox_delta": 1.0, "ttt_norm": 0.2}
    cf_res = engine.counterfactual_telemetry(factual, {"cpu_percent": 30.0})
    assert "cpu_percent" in cf_res
    assert abs(cf_res["cpu_percent"] - 30.0) < 1e-4


def test_production_frontier3_dynamic_ast_induction():
    synth = DynamicASTSynthesizer()

    # 1. Linear task
    io_linear = [(1, 2), (2, 3), (3, 4), (5, 6)]
    code_linear = synth.synthesize_dynamic(io_linear)
    assert code_linear is not None
    ast.parse(code_linear)
    env = {}
    exec(code_linear, env)
    assert env["solution"](10) == 11

    # 2. Quadratic task: x * x + 1
    io_quad = [(0, 1), (1, 2), (2, 5), (3, 10)]
    code_quad = synth.synthesize_dynamic(io_quad)
    assert code_quad is not None
    ast.parse(code_quad)
    env_q = {}
    exec(code_quad, env_q)
    assert env_q["solution"](4) == 17


def test_production_frontier4_mmap_ring_and_hardened_sandbox():
    cap_mgr = CapabilityManager()
    token_write = cap_mgr.issue_token("enclave_alpha", CAP_WRITE)
    token_read = cap_mgr.issue_token("enclave_beta", CAP_READ)
    token_unauth = cap_mgr.issue_token("rogue_node", 0)

    # 1. Binary mmap ring buffer
    ring = BinaryMmapIPCRing(capacity=8, slot_size=1024, cap_manager=cap_mgr)
    try:
        # Unauthorized write rejection
        with pytest.raises(PermissionError):
            ring.push(b"malicious_payload", token_unauth)

        # Authorized write and read
        payload = b'{"instruction": "EVAL_AST", "id": 104}'
        slot = ring.push(payload, token_write)
        assert slot == 0

        # Unauthorized read rejection
        with pytest.raises(PermissionError):
            ring.pop(token_unauth)

        rec = ring.pop(token_read)
        assert rec == payload
    finally:
        ring.close()

    # 2. Hardened OS process sandbox
    sandbox = HardenedOSSandboxExecutor(timeout_sec=5.0)

    # Safe math execution
    delta_s, out = sandbox.execute("import math\nprint(math.sqrt(144.0))")
    assert delta_s == 1.0
    assert "12.0" in out

    # Blocked dangerous import (os)
    delta_s_bad, err = sandbox.execute("import os\nprint(os.getcwd())")
    assert delta_s_bad == -1.0
    assert "Unauthorized import" in err or "Error" in err
