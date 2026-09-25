import pytest
from cognitive_engine.core.causal_ast_tracer import CausalASTTracer, CausalASTRepairEngine


def test_causal_ast_tracer_dag_construction():
    """Verify data-flow variable dependency DAG construction during AST execution."""
    code = (
        "base = 10\n"
        "multiplier = 2\n"
        "intermediate = base * multiplier\n"
        "result = intermediate + 5\n"
    )
    tracer = CausalASTTracer()
    dag = tracer.trace_execution(code, target_var="result", target_val=25)

    assert "result" in dag.nodes
    assert dag.nodes["result"].val == 25
    assert "intermediate" in dag.nodes["result"].parents
    assert ("intermediate", "result") in dag.edges
    assert ("base", "intermediate") in dag.edges


def test_causal_ast_operator_repair_single_mutation():
    """Verify counterfactual repair of an inverted arithmetic operator in exactly 1 mutation."""
    # Buggy code: computes a - b instead of a + b
    buggy_code = (
        "def compute_total(a, b):\n"
        "    return a - b\n"
    )
    test_call = "res = compute_total(15, 5)"
    expected = 20  # 15 + 5

    engine = CausalASTRepairEngine()
    repaired, code_repaired, mutations = engine.repair(
        buggy_code, test_call, expected_output=expected, target_var="res"
    )

    assert repaired, "Failed to repair inverted operator"
    assert mutations == 1, f"Expected exactly 1 targeted mutation, got {mutations}"
    assert "+" in code_repaired


def test_causal_ast_constant_off_by_one_repair():
    """Verify counterfactual repair of an off-by-one constant error in <= 2 mutations."""
    # Buggy code: uses offset 4 instead of offset 5
    buggy_code = (
        "def calculate_offset(x):\n"
        "    return x + 4\n"
    )
    test_call = "res = calculate_offset(10)"
    expected = 15  # 10 + 5

    engine = CausalASTRepairEngine()
    repaired, code_repaired, mutations = engine.repair(
        buggy_code, test_call, expected_output=expected, target_var="res"
    )

    assert repaired, "Failed to repair off-by-one constant"
    assert mutations <= 2, f"Expected <= 2 mutations, got {mutations}"
    assert "5" in code_repaired
