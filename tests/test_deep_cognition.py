import ast
import pytest
import torch

from cognitive_engine.core.generator import LocalLLMGenerator
from cognitive_engine.core.world_model import (
    LatentWorldModel,
    StructuredEnvironmentEncoder,
)
from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer


def test_cortex_logits_steering():
    gen = LocalLLMGenerator()
    gen.condition_on_context("def critical_security_kernel(): pass")

    base_logits = [0.1, 0.5, 2.0, -1.0, 0.0]
    steered = gen.steer_logits_with_ttt(base_logits, alpha=0.3)

    assert len(steered) == len(base_logits)
    assert steered != base_logits, "Fast-weight energy should steer next-token logits"


def test_structured_env_encoder_and_latent_trajectory():
    encoder = StructuredEnvironmentEncoder(in_features=6, out_features=14)
    s_t = encoder(
        stdout_len=240,
        stderr_len=0,
        exit_code=0,
        files_modified=1,
        latency_ms=0.45,
        memory_mb=22.5
    )
    assert s_t.shape == (14,)

    world_model = LatentWorldModel()
    horizon_actions = [0, 1, 0, 2, 1]
    res = world_model.rollout_trajectory(s_t, horizon_actions, gamma=0.95)

    assert "latent_trajectory" in res
    assert len(res["latent_trajectory"]) == len(horizon_actions) + 1
    assert len(res["step_rewards"]) == len(horizon_actions)
    assert len(res["step_risks"]) == len(horizon_actions)
    assert isinstance(res["discounted_reward"], float)
    assert isinstance(res["safe"], bool)


def test_recursive_and_iterative_ast_induction():
    synth = DynamicASTSynthesizer()

    # 1. Factorial via dynamic recursive synthesis
    io_factorial = [(1, 1), (2, 2), (3, 6), (4, 24), (5, 120)]
    code_fact = synth.synthesize_dynamic(io_factorial)
    assert code_fact is not None
    ast.parse(code_fact)
    env_fact = {}
    exec(code_fact, env_fact)
    assert env_fact["solution"](6) == 720

    # 2. Fibonacci via dynamic recursive synthesis
    io_fib = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8)]
    code_fib = synth.synthesize_dynamic(io_fib)
    assert code_fib is not None
    ast.parse(code_fib)
    env_fib = {}
    exec(code_fib, env_fib)
    assert env_fib["solution"](7) == 13

    # 3. Dynamic synthesis from programmatic unit assertions
    assertions = [
        "assert solution(1) == 1",
        "assert solution(3) == 6",
        "assert solution(4) == 24",
        "assert solution(5) == 120",
    ]
    code_assert = synth.synthesize_from_assertions(assertions)
    assert code_assert is not None
    env_assert = {}
    exec(code_assert, env_assert)
    assert env_assert["solution"](5) == 120

    # 4. Binary search algorithm synthesis from assertions
    bs_assertions = [
        "assert solution(([10, 20, 30, 40], 30)) == 2",
        "assert solution(([10, 20, 30, 40], 99)) == -1",
    ]
    bs_code = synth.synthesize_from_assertions(bs_assertions)
    assert bs_code is not None
    env_bs = {}
    exec(bs_code, env_bs)
    assert env_bs["solution"](([1, 2, 3], 2)) == 1

    # 5. Graph reachability DFS synthesis from assertions
    graph_assertions = [
        "assert solution(({1: [2], 2: [3], 3: []}, 1, 3)) is True",
        "assert solution(({1: [2], 2: []}, 1, 3)) is False",
    ]
    graph_code = synth.synthesize_from_assertions(graph_assertions)
    assert graph_code is not None
    env_graph = {}
    exec(graph_code, env_graph)
    assert env_graph["solution"](({1: [2], 2: []}, 1, 2)) is True

