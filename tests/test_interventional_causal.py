import pytest
from cognitive_engine.core.causal_graph import CausalSymbolicGraph
from cognitive_engine.core.types import Triple


def test_record_intervention():
    graph = CausalSymbolicGraph()
    rec = graph.record_intervention(
        source="uncompiled_code",
        action="optimize_ast",
        target="execution_success",
        delta_s=1.0,
        metrics={"latency_ms": 1.2},
    )
    assert rec.source == "uncompiled_code"
    assert rec.action == "optimize_ast"
    assert rec.target == "execution_success"
    assert rec.delta_s == 1.0
    assert len(graph.interventions) == 1


def test_induce_interventional_edge_insufficient_samples():
    graph = CausalSymbolicGraph()
    graph.record_intervention("matrix_alloc", "strided_view", "memory_reduction", delta_s=1.0)
    success, score, msg = graph.induce_interventional_edge(
        "matrix_alloc", "strided_view", "memory_reduction", min_samples=3
    )
    assert not success
    assert "Insufficient empirical observations" in msg


def test_induce_interventional_edge_success():
    graph = CausalSymbolicGraph()
    # Record 4 trials: 3 successes, 1 failure
    for d in [1.0, 1.0, 1.0, -1.0]:
        graph.record_intervention("sparse_matrix", "csr_compress", "cache_locality", delta_s=d)

    success, score, msg = graph.induce_interventional_edge(
        "sparse_matrix", "csr_compress", "cache_locality", min_samples=3, threshold=0.5
    )
    assert success
    # (3 + 1) / (4 + 2) = 4/6 = 0.6667
    assert score > 0.6
    assert graph.get_edge_weight("sparse_matrix", "cache_locality") == pytest.approx(score, 0.01)
    assert "induced" in msg


def test_induce_interventional_edge_rejects_contradiction():
    graph = CausalSymbolicGraph()
    # Add negative physical constraint
    graph.add_axiom(Triple(subject="system_heat", relation="cannot_be", target="zero_entropy", polarity=True))

    # Attempt to induce violating edge despite positive interventions
    for _ in range(5):
        graph.record_intervention("system_heat", "passive_cool", "zero_entropy", delta_s=1.0)

    success, score, msg = graph.induce_interventional_edge(
        "system_heat", "passive_cool", "zero_entropy", min_samples=3
    )
    assert not success
    assert "contradiction" in msg.lower()


def test_pearl_level3_counterfactual_evaluation():
    graph = CausalSymbolicGraph()

    # Pre-existing knowledge: unbounded_recursion causes stack_overflow
    graph.add_edge("unbounded_recursion", "causes", "stack_overflow", is_valid=True)
    graph.add_axiom(Triple(subject="stack_overflow", relation="cannot_be", target="safe_execution", polarity=True))

    # Alternate action candidate: tail_call_optimization
    graph.add_edge("tail_call_optimization", "causes", "safe_execution", is_valid=True)
    graph.set_edge_weight("tail_call_optimization", "safe_execution", 0.92)

    # Counterfactual evaluation:
    # "Given that unbounded_recursion caused stack_overflow in recursive_algorithm,
    # would tail_call_optimization have satisfied safe_execution?"
    viable, conf, explanation = graph.evaluate_counterfactual(
        observed_state="recursive_algorithm",
        failed_action="unbounded_recursion",
        candidate_action="tail_call_optimization",
        target_invariant="safe_execution",
    )
    assert viable
    assert conf > 0.8
    assert "consistent" in explanation

    # Counterfactual evaluation with contradictory candidate
    graph.add_axiom(Triple(subject="busy_wait", relation="cannot_be", target="safe_execution", polarity=True))
    invalid_viable, _, invalid_exp = graph.evaluate_counterfactual(
        observed_state="recursive_algorithm",
        failed_action="unbounded_recursion",
        candidate_action="busy_wait",
        target_invariant="safe_execution",
    )
    assert not invalid_viable
    assert "violation" in invalid_exp.lower()
