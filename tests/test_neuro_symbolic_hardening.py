import pytest
import numpy as np
from cognitive_engine.core.intent_schema import (
    PhysicsGoalPayload,
    AlgorithmicGoalPayload,
    AmbiguityResolutionPayload,
    route_intent,
)
from cognitive_engine.core.grounding_verifier import verify_executable_grounding
from cognitive_engine.core.consolidation import ConsolidationStore
from cognitive_engine.core.lambda_dsl import (
    Prim, Const, App, eval_lambda_ast, unparse_to_python, CORE_LAMBDA_PRIMITIVES
)
from features.execution_sandbox.sandbox import IsolatedSandboxExecutor


def test_phase1_intent_schema_routing():
    # Physics query
    phys = route_intent("Calculate kinetic energy for 100 kg mass moving at 20 m/s")
    assert isinstance(phys, PhysicsGoalPayload)
    assert phys.intent_type == "physics_calculation"
    assert phys.parameters.get("mass_kg") == 100.0
    assert phys.parameters.get("velocity_mps") == 20.0

    # Algorithmic synthesis query
    algo = route_intent("Synthesize algorithm for my_filter_fn with io_examples")
    assert isinstance(algo, AlgorithmicGoalPayload)
    assert algo.intent_type == "algorithmic_synthesis"

    # Ambiguity resolution query
    amb = route_intent("quantum superposition paradox")
    assert isinstance(amb, AmbiguityResolutionPayload)
    assert amb.intent_type == "ambiguity_resolution"
    assert len(amb.candidate_interpretations) >= 2


def test_phase2_grounding_verifier_falsifiable_invariants():
    sandbox = IsolatedSandboxExecutor()

    # Trivial assignment without assert should be rejected (-1.0)
    delta_s, msg = verify_executable_grounding("fact = 'Gravity pulls objects down'", sandbox)
    assert delta_s == -1.0
    assert "no falsifiable assertions" in msg

    # Valid falsifiable assertion should pass (+1.0)
    delta_s, msg = verify_executable_grounding("assert 2 + 2 == 4\nassert len('hello') == 5", sandbox)
    assert delta_s == 1.0
    assert "Verified" in msg

    # Failing assertion should return -1.0
    delta_s, msg = verify_executable_grounding("assert 2 + 2 == 5", sandbox)
    assert delta_s == -1.0


def test_phase2_wal_commit_guard():
    store = ConsolidationStore(db_path=":memory:", dim=16)
    vec = np.zeros(16, dtype=np.float32)

    # Ingestion with delta_s <= 0.0 must raise ValueError
    with pytest.raises(ValueError, match="Ungrounded memory write rejected"):
        store.write_memory(content="Ungrounded fact", vector=vec, delta_s=-1.0)

    # Ingestion from external source without delta_s == 1.0 must raise ValueError
    with pytest.raises(ValueError, match="External ingestion memory write requires verified execution"):
        store.write_memory(content="External scrape", vector=vec, source="web_ingestor", delta_s=None)

    # Grounded write with delta_s == 1.0 succeeds
    mid = store.write_memory(content="Verified fact", vector=vec, source="web_ingestor", delta_s=1.0)
    assert mid is not None


def test_phase4_isolated_sandbox_subprocess():
    sandbox = IsolatedSandboxExecutor()
    delta_s, msg = sandbox.execute("x = 10\ny = 20\nassert x + y == 30")
    assert delta_s == 1.0
    assert msg == "Success"

    # Crash handling
    delta_s, msg = sandbox.execute("raise ZeroDivisionError('denied')")
    assert delta_s == -1.0
    assert "ZeroDivisionError" in msg


def test_phase5_general_lambda_primitives():
    assert "add" in CORE_LAMBDA_PRIMITIVES
    assert "sub" in CORE_LAMBDA_PRIMITIVES
    assert "mul" in CORE_LAMBDA_PRIMITIVES
    assert "branch" in CORE_LAMBDA_PRIMITIVES

    # Test branch: branch(cond)(t)(f)
    ast_branch = App(App(App(Prim("branch"), Const(True)), Const(42)), Const(0))
    res = eval_lambda_ast(ast_branch, {})
    assert res == 42

    # Test unparse
    ast_add = App(App(Prim("add"), Const(10)), Const(25))
    py_code = unparse_to_python(ast_add)
    assert "(10 + 25)" in py_code
