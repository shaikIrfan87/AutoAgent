import os
import shutil
import pytest
from cognitive_engine.core.lambda_dsl import (
    LambdaNode, Var, Const, Prim, App, Lambda, IfThenElse,
    MapNode, FilterNode, FoldNode, WhileLoop, ForLoop,
    eval_lambda_ast, unparse_to_python, canonical_ast_hash,
)
from cognitive_engine.agent.lambda_synthesizer import LambdaProgramSynthesizer
from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_lambda_higher_order_nodes_eval_and_unparse():
    """Verify evaluation and python translation of MapNode, FilterNode, and FoldNode."""
    # 1. MapNode: [x * 2 for x in [1, 2, 3]] -> [2, 4, 6]
    double_fn = Lambda("x", App(App(Prim("*"), Var("x")), Const(2)))
    map_node = MapNode(double_fn, Const([1, 2, 3]))
    assert eval_lambda_ast(map_node, {}) == [2, 4, 6]
    py_code = unparse_to_python(map_node)
    assert "_x in [1, 2, 3]" in py_code

    # 2. FilterNode: [x for x in [1, 2, 3, 4] if x % 2 == 0] -> [2, 4]
    is_even = Lambda("x", App(App(Prim("=="), App(App(Prim("%"), Var("x")), Const(2))), Const(0)))
    filter_node = FilterNode(is_even, Const([1, 2, 3, 4]))
    assert eval_lambda_ast(filter_node, {}) == [2, 4]

    # 3. FoldNode: sum of [1, 2, 3, 4] with init 0 -> 10
    fold_node = FoldNode(Prim("+"), Const(0), Const([1, 2, 3, 4]))
    assert eval_lambda_ast(fold_node, {}) == 10


def test_lambda_loops_eval_and_unparse():
    """Verify evaluation of WhileLoop and ForLoop primitives."""
    # 1. WhileLoop: double x while x < 10, starting at 1 -> 16
    cond = Lambda("x", App(App(Prim("<"), Var("x")), Const(10)))
    step = Lambda("x", App(App(Prim("*"), Var("x")), Const(2)))
    while_node = WhileLoop(cond, step, Const(1))
    assert eval_lambda_ast(while_node, {}) == 16

    # 2. ForLoop: count loop running 5 times, incrementing val by 1 -> 5
    step_for = Lambda("v", Lambda("i", App(App(Prim("+"), Var("v")), Const(1))))
    for_node = ForLoop(Const(5), step_for, Const(0))
    assert eval_lambda_ast(for_node, {}) == 5


def test_sub_tree_equivalence_hashing():
    """Verify canonical sub-tree hashing normalizes commutative operations."""
    # (+) commutative: a + b == b + a
    node_a = App(App(Prim("+"), Var("x")), Const(1))
    node_b = App(App(Prim("+"), Const(1)), Var("x"))
    assert canonical_ast_hash(node_a) == canonical_ast_hash(node_b)

    # (*) commutative: a * b == b * a
    node_c = App(App(Prim("*"), Var("x")), Var("y"))
    node_d = App(App(Prim("*"), Var("y")), Var("x"))
    assert canonical_ast_hash(node_c) == canonical_ast_hash(node_d)

    # Non-commutative: a - b != b - a
    sub_a = App(App(Prim("-"), Var("x")), Const(1))
    sub_b = App(App(Prim("-"), Const(1)), Var("x"))
    assert canonical_ast_hash(sub_a) != canonical_ast_hash(sub_b)


def test_lambda_synthesizer_with_hof_and_hashing():
    """Verify inductive synthesis of list transformation with canonical sub-tree pruning."""
    synth = LambdaProgramSynthesizer(max_cost=8)
    io_pairs = [([1, 2], [2, 4]), ([3, 4], [6, 8])]
    res = synth.synthesize(io_pairs, var_name="x")
    assert res is not None
    node, py_code = res
    assert py_code is not None
    # Check that evaluating synthesized code matches
    env = {}
    exec(f"def solution(x):\n    return {py_code}", env)
    assert env["solution"]([5, 10]) == [10, 20]


def test_dynamic_ast_synthesizer_higher_order_and_assertions():
    """Verify DynamicASTSynthesizer discovers higher-order list solutions."""
    synth = DynamicASTSynthesizer(max_expansions=100)
    assertions = [
        "assert solution([1, 2, 3]) == [2, 3, 4]",
        "assert solution([10, 20]) == [11, 21]",
    ]
    code = synth.synthesize_from_assertions(assertions)
    assert code is not None
    assert "solution" in code
    env = {}
    exec(code, env)
    assert env["solution"]([100]) == [101]


def test_closed_loop_skill_compilation_and_orchestration(tmp_path):
    """Verify closed-loop program induction compiles directly to skills and WAL memory."""
    test_skills_dir = str(tmp_path / "skills")
    engine = CognitiveEngine(db_path=":memory:")

    # Induct a verified list summation routine
    res = engine.ast_synthesizer.synthesize_and_compile(
        task_name="vector_summer",
        assertions=["assert solution([1, 2, 3]) == 6", "assert solution([10, 20, 30]) == 60"],
        skills_dir=test_skills_dir,
        consolidation=engine.consolidation,
    )
    assert res is not None
    code, file_path = res
    assert os.path.exists(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        saved_content = f.read()
    assert "solution" in saved_content
    assert "Turing-Complete Program Induction" in saved_content

    # Check that orchestrator.induct_program returns delta_s == 1.0
    induct_res = engine.induct_program(
        task_name="vector_incrementer",
        task_spec="assert solution([1, 2]) == [2, 3]\nassert solution([5, 6]) == [6, 7]",
    )
    assert induct_res["status"] == "success"
    assert induct_res["delta_s"] == 1.0
    assert os.path.exists(induct_res["file_path"])
