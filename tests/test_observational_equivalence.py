import time
import pytest
from cognitive_engine.agent.lambda_synthesizer import LambdaProgramSynthesizer
from cognitive_engine.core.lambda_dsl import eval_lambda_ast, Var, Prim, App, Const


def test_observational_equivalence_list_map():
    synth = LambdaProgramSynthesizer(max_cost=10)

    # I/O specification: increment list elements
    io_pairs = [
        ([1, 2, 3], [2, 3, 4]),
        ([10, 20], [11, 21]),
        ([0], [1]),
    ]

    t0 = time.perf_counter()
    res = synth.synthesize(io_pairs, var_name="x")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res is not None
    node, py_code = res
    assert "map" in py_code.lower() or "+" in py_code
    # Must solve in < 150ms due to signature hashing
    assert elapsed_ms < 150.0


def test_spatial_arc_grid_rotation():
    synth = LambdaProgramSynthesizer(max_cost=8)

    # 2x2 grid clockwise rotation
    g1 = [[1, 2], [3, 4]]
    g1_rot = [[3, 1], [4, 2]]

    g2 = [[5, 6], [7, 8]]
    g2_rot = [[7, 5], [8, 6]]

    t0 = time.perf_counter()
    res = synth.synthesize([(g1, g1_rot), (g2, g2_rot)], var_name="g")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res is not None
    node, py_code = res
    assert "rot90" in py_code
    assert elapsed_ms < 150.0


def test_spatial_arc_grid_vertical_flip():
    synth = LambdaProgramSynthesizer(max_cost=8)

    # 2x2 grid vertical flip
    g1 = [[1, 2], [3, 4]]
    g1_flip = [[3, 4], [1, 2]]

    t0 = time.perf_counter()
    res = synth.synthesize([(g1, g1_flip)], var_name="g")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res is not None
    node, py_code = res
    assert "flip_v" in py_code
    assert elapsed_ms < 150.0


def test_multi_object_largest_component_synthesis():
    synth = LambdaProgramSynthesizer(max_cost=8)

    # 4x4 grid: small 1-pixel object vs 2x2 4-pixel object
    inp = [
        [1, 0, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 0, 0],
    ]
    # Expected: only largest object
    out = [
        [0, 0, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 0, 0],
    ]

    t0 = time.perf_counter()
    res = synth.synthesize([(inp, out)], var_name="g")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res is not None
    node, py_code = res
    assert "largest_object" in py_code
    assert elapsed_ms < 150.0


def test_multi_object_dsl_primitives_direct_evaluation():
    inp = [
        [1, 0, 2],
        [0, 0, 2],
        [3, 3, 0],
    ]
    # 1. objects
    ast_objs = App(Prim("objects"), Var("g"))
    objs = eval_lambda_ast(ast_objs, {"g": inp})
    assert len(objs) == 3

    # 2. recolor
    ast_recolor = App(App(Prim("recolor"), Const(5)), Var("g"))
    recolored = eval_lambda_ast(ast_recolor, {"g": inp})
    assert recolored[0][0] == 5
    assert recolored[0][1] == 0
    assert recolored[0][2] == 5

    # 3. overlay
    g_a = [[1, 0], [0, 0]]
    g_b = [[0, 2], [3, 0]]
    ast_overlay = App(App(Prim("overlay"), Var("a")), Var("b"))
    overlaid = eval_lambda_ast(ast_overlay, {"a": g_a, "b": g_b})
    assert overlaid == [[1, 2], [3, 0]]


def test_bidirectional_program_synthesis():
    from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer
    synth = DynamicASTSynthesizer()

    # Composite depth-2 problem: y = (x + 3) * 2
    io_pairs = [
        (1, 8),
        (2, 10),
        (5, 16),
        (10, 26),
    ]
    t0 = time.perf_counter()
    code = synth.synthesize_bidirectional(io_pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert code is not None
    assert "def solution" in code
    assert elapsed_ms < 50.0

    scope = {}
    exec(code, scope)
    sol = scope["solution"]
    for in_v, out_v in io_pairs:
        assert sol(in_v) == out_v

    # List composite problem: reversed list scaled by 2
    list_pairs = [
        ([1, 2, 3], [6, 4, 2]),
        ([10, 20], [40, 20]),
    ]
    list_code = synth.synthesize_bidirectional(list_pairs)
    assert list_code is not None
    scope2 = {}
    exec(list_code, scope2)
    sol2 = scope2["solution"]
    for in_v, out_v in list_pairs:
        assert sol2(in_v) == out_v


def test_ttt_ast_mcts_online_adaptation():
    from cognitive_engine.agent.ast_policy_mcts import ASTGuidedMCTS
    mcts = ASTGuidedMCTS(max_expansions=50, enable_ttt=True)

    # Quadratic task: x ** 2
    io_pairs = [(1, 1), (2, 4), (3, 9), (4, 16)]

    loss = mcts.test_time_adapt(io_pairs, steps=15, lr=0.02)
    assert loss >= 0.0

    # Synthesize with adapted priors
    code = mcts.synthesize(io_pairs, enable_ttt=False)
    assert code is not None
    scope = {}
    exec(code, scope)
    sol = scope["solution"]
    for in_v, out_v in io_pairs:
        assert sol(in_v) == out_v

