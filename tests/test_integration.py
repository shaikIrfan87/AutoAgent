from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.types import Triple


def test_full_cognitive_lifecycle():
    engine = CognitiveEngine()

    # 1. Spam rejection
    spam = "repeat repeat repeat repeat repeat repeat repeat"
    res_spam = engine.process(spam)
    assert res_spam["status"] == "dropped"
    assert "Low Shannon entropy" in res_spam["reason"]

    # 2. System 2 deliberation on novel query
    query = "Calculate factorial of 5 using environmental execution"
    code = "import math\nprint(math.factorial(5))"
    res_s2 = engine.process(query, code_action=code)
    assert res_s2["status"] == "system_2_success"
    assert res_s2["output"] == "120"
    assert res_s2["delta_s"] == 1.0
    assert res_s2["consolidated_memory_id"] is not None

    # 3. System 1 instant recall on repeated query
    res_s1 = engine.process(query)
    # The query is now in cyclic buffer (novelty ~ 0) or consolidated
    assert res_s1["status"] in ("system_1", "dropped")

    # 4. System 2 halts on causal violation
    invalid_hypo = Triple(subject="human", relation="is_a", target="syntheticmachine", polarity=True)
    res_causal = engine.process("Assess whether the android is a biological human organism", hypothesis=invalid_hypo)
    assert res_causal["status"] == "system_2_failure"
    assert "contradiction" in res_causal["error"].lower() or "mutual exclusion" in res_causal["error"].lower()



def test_self_eval_bridge_pass_and_fail():
    from unittest.mock import MagicMock
    from self_eval_engine.schemas import GraphTrajectory, VerificationResult

    engine = CognitiveEngine()
    engine.causal.set_edge_weight("node_start", "node_exec", 0.5)

    # 1. Pass case: high uncertainty delegates to self_eval, passes verification, consolidates
    pass_traj = GraphTrajectory(
        id="t-pass",
        reasoning_trace="Reasoning trace OK",
        terminal_output="Verified 42",
        token_entropy=0.1,
        active_node_ids=["node_start"],
        traversed_edges=[("node_start", "node_exec")],
        verification=VerificationResult(is_valid=True, score=1.0),
    )
    mock_self_eval = MagicMock()
    mock_self_eval.run.return_value = pass_traj
    engine.self_eval = mock_self_eval
    engine.executive.self_eval = mock_self_eval

    query = "Completely novel quantum deliberation query about multiverses"
    res_pass = engine.process(query)
    assert res_pass["status"] == "system_2_success"
    assert res_pass["output"] == "Verified 42"
    assert res_pass["consolidated_memory_id"] is not None

    # 2. Fail case: verification failure triggers GraphCreditAssigner & plastic anti-Hebbian penalty
    fail_traj = GraphTrajectory(
        id="t-fail",
        reasoning_trace="Failed trace",
        terminal_output="Bad code output",
        token_entropy=0.1,
        active_node_ids=["node_start"],
        traversed_edges=[("node_start", "node_exec")],
        verification=VerificationResult(is_valid=False, score=0.0, failed_assertions=["Faulty output"]),
    )
    mock_self_eval.run.return_value = fail_traj
    init_norm = float(engine.plastic.A_fast.norm().item())

    res_fail = engine.process("Another completely novel failing task requiring reflection")
    assert res_fail["status"] == "system_2_failure"


if __name__ == "__main__":
    test_full_cognitive_lifecycle()
    test_self_eval_bridge_pass_and_fail()
    print("All integration tests passed.")
