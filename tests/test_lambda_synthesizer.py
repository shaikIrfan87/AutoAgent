import pytest
from cognitive_engine.core.lambda_dsl import (
    LambdaNode, Var, Const, Prim, App, Lambda, IfThenElse,
    eval_lambda_ast, unparse_to_python,
)
from cognitive_engine.agent.lambda_synthesizer import LambdaProgramSynthesizer
from cognitive_engine.core.generator import LocalLLMGenerator


def test_lambda_ast_eval_and_unparse():
    # 2 * x + 1 -> App(App(Prim("+"), App(App(Prim("*"), Var("x")), Const(2))), Const(1))
    expr = App(
        App(Prim("+"), App(App(Prim("*"), Var("x")), Const(2))),
        Const(1),
    )
    # Evaluate at x = 5 -> 2*5 + 1 = 11
    val = eval_lambda_ast(expr, {"x": 5})
    assert val == 11

    # Unparse to python
    py = unparse_to_python(expr)
    assert "x" in py
    assert "+" in py
    assert "*" in py

    # Test conditional branch
    # If x < 2 then 0 else x
    cond_expr = IfThenElse(
        App(App(Prim("<"), Var("x")), Const(2)),
        Const(0),
        Var("x"),
    )
    assert eval_lambda_ast(cond_expr, {"x": 1}) == 0
    assert eval_lambda_ast(cond_expr, {"x": 5}) == 5


def test_lambda_ast_safety_limits():
    # Unbound variable
    with pytest.raises(NameError):
        eval_lambda_ast(Var("unbound_y"), {})

    # Recursion depth limit
    infinite_app = App(Lambda("f", App(Var("f"), Var("f"))), Lambda("f", App(Var("f"), Var("f"))))
    with pytest.raises(RecursionError):
        eval_lambda_ast(infinite_app, {}, depth_limit=5)


def test_lambda_synthesize_arithmetic_linear():
    synth = LambdaProgramSynthesizer(max_cost=8)
    # f(x) = 2x
    io_pairs = [(1, 2), (2, 4), (3, 6), (4, 8)]
    res = synth.synthesize(io_pairs, var_name="x")
    assert res is not None
    node, py_code = res
    for inp, expected in io_pairs:
        assert eval_lambda_ast(node, {"x": inp}) == expected


def test_lambda_synthesize_offset():
    synth = LambdaProgramSynthesizer(max_cost=8)
    # f(x) = x + 1
    io_pairs = [(0, 1), (1, 2), (5, 6)]
    res = synth.synthesize(io_pairs, var_name="x")
    assert res is not None
    node, py_code = res
    for inp, expected in io_pairs:
        assert eval_lambda_ast(node, {"x": inp}) == expected


def test_generator_io_prompt_programmatic_induction():
    gen = LocalLLMGenerator()
    prompt = "Synthesize function for IO: [(1, 2), (2, 4), (3, 6)]"
    code = gen.generate_code(prompt)
    assert "def solution(x):" in code
    assert "return" in code

    # Execute synthesized code in local namespace
    ns = {}
    exec(code, ns)
    solution = ns["solution"]
    assert solution(1) == 2
    assert solution(5) == 10
