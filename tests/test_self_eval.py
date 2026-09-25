import pytest
import numpy as np
from self_eval_engine import (
    CritiquePayload,
    ConsensusEngine,
    DisentangledJudge,
    GraphCreditAssigner,
    GraphTrajectory,
    PythonExecutionSandbox,
    RubricCriteria,
    SelfEvaluationOrchestrator,
)


class DummyNode:
    def __init__(self, activation_threshold: float = 0.5):
        self.activation_threshold = activation_threshold


class MockGraphMemory:
    def __init__(self):
        self.edges = {("node_A", "node_B"): 0.2}
        self.nodes = {
            "node_A": DummyNode(0.5),
            "node_B": DummyNode(0.5),
        }

    def get_edge_weight(self, u: str, v: str) -> float:
        return self.edges.get((u, v), 0.0)

    def set_edge_weight(self, u: str, v: str, weight: float) -> None:
        self.edges[(u, v)] = weight


def test_python_execution_sandbox():
    sandbox = PythonExecutionSandbox(timeout_sec=3)
    # Valid syntax & passing assertion
    res = sandbox.verify_code("x = 40 + 2", "assert x == 42")
    assert res.is_valid
    assert res.score == 1.0

    # AST syntax error caught without spawning process
    bad_syntax = sandbox.verify_code("def syntax_err(")
    assert not bad_syntax.is_valid
    assert "SyntaxError" in bad_syntax.failed_assertions[0]

    # Failing assertion
    fail_res = sandbox.verify_code("x = 1", "assert x == 2")
    assert not fail_res.is_valid
    assert fail_res.score == 0.0


def test_consensus_engine():
    t1 = GraphTrajectory(id="1", reasoning_trace="r1", terminal_output="ans_a")
    t2 = GraphTrajectory(id="2", reasoning_trace="r2", terminal_output="ans_a")
    t3 = GraphTrajectory(id="3", reasoning_trace="r3", terminal_output="ans_b")

    winner, ratio = ConsensusEngine.resolve_majority([t1, t2, t3])
    assert winner.terminal_output == "ans_a"
    assert pytest.approx(ratio, 0.01) == 2 / 3


def test_graph_credit_assigner_hebbian_and_threshold():
    graph = MockGraphMemory()
    assigner = GraphCreditAssigner(graph, decay_rate=0.05)

    traj = GraphTrajectory(
        id="t1",
        reasoning_trace="trace",
        terminal_output="x = 1",
        active_node_ids=["node_A"],
        traversed_edges=[("node_A", "node_B")],
    )

    # 1. Pass reinforcement: edge increases
    critique_pass = CritiquePayload(aggregate_score=1.0)
    assigner.apply_feedback(traj, critique_pass, passed=True)
    assert graph.get_edge_weight("node_A", "node_B") == pytest.approx(0.3)
    assert graph.nodes["node_A"].activation_threshold == 0.5

    # 2. Fail punishment: edge decreases, threshold increases
    critique_fail = CritiquePayload(aggregate_score=0.2)
    assigner.apply_feedback(traj, critique_fail, passed=False)
    # delta = -0.2 * (1 - 0.2) = -0.16 -> 0.3 - 0.16 = 0.14
    assert graph.get_edge_weight("node_A", "node_B") == pytest.approx(0.14)
    assert graph.nodes["node_A"].activation_threshold == pytest.approx(0.55)


def test_orchestrator_speculative_and_short_circuit():
    import asyncio

    async def _run():
        graph = MockGraphMemory()
        orchestrator = SelfEvaluationOrchestrator(
            graph_memory=graph,
            mode="closed_loop",
            entropy_threshold=0.5,
        )

        # Speculative execution path (entropy <= 0.5 clears immediately without consensus beam)
        res = await orchestrator.run_async(
            task="def add(): return 42",
            deterministic_unit_test="assert solution() is not None",
        )
        assert res is not None
        assert res.verification.is_valid
        assert graph.get_edge_weight("node_start", "node_exec") > 0.0

    asyncio.run(_run())


def test_rollout_gates():
    graph = MockGraphMemory()
    # Gate 1: Shadow mode -> no mutations
    shadow_orch = SelfEvaluationOrchestrator(graph_memory=graph, mode="shadow")
    w_before = graph.get_edge_weight("node_start", "node_exec")
    _ = shadow_orch.run(task="x = 1")
    assert graph.get_edge_weight("node_start", "node_exec") == w_before


def test_consensus_tie_break_and_formatting_strip():
    # Tie-break prefers higher verification score
    t1 = GraphTrajectory(id="1", reasoning_trace="r1", terminal_output="cand_1", token_entropy=0.8)
    t1.verification = PythonExecutionSandbox().verify_code("x = 1", "assert x == 1")
    t2 = GraphTrajectory(id="2", reasoning_trace="r2", terminal_output="cand_2", token_entropy=0.1)
    t2.verification = PythonExecutionSandbox().verify_code("x = 1", "assert x == 2")

    winner, ratio = ConsensusEngine.resolve_majority([t1, t2])
    assert winner.terminal_output == "cand_1"
    assert ratio == 0.5

    # Formatting stripper in judge
    judge = DisentangledJudge()
    prompt = judge._construct_judge_prompt("task", GraphTrajectory(id="3", reasoning_trace="r3", terminal_output="```python\nx = 1\n```"))
    assert "```" not in prompt


def test_consensus_sha256_tie_break_deterministic():
    t1 = GraphTrajectory(id="1", reasoning_trace="r1", terminal_output="cand_alpha", token_entropy=0.5)
    t2 = GraphTrajectory(id="2", reasoning_trace="r2", terminal_output="cand_beta", token_entropy=0.5)
    winner_12, _ = ConsensusEngine.resolve_majority([t1, t2])
    winner_21, _ = ConsensusEngine.resolve_majority([t2, t1])
    assert winner_12.terminal_output == winner_21.terminal_output



def test_homeostatic_decay_on_inactive_nodes():
    graph = MockGraphMemory()
    graph.nodes["node_B"].activation_threshold = 0.60
    assigner = GraphCreditAssigner(graph, decay_rate=0.05, homeostatic_decay=0.02)
    traj = GraphTrajectory(id="t", reasoning_trace="", terminal_output="a", active_node_ids=["node_A"])
    assigner.apply_feedback(traj, CritiquePayload(aggregate_score=1.0), passed=True)
    # node_B is inactive -> relaxes towards 0.50
    assert graph.nodes["node_B"].activation_threshold == pytest.approx(0.58)


def test_edge_pruning_and_timeout_cleanup():
    # 1. Edge pruning drops near-zero transitions
    graph = MockGraphMemory()
    graph.edges[("node_A", "node_C")] = 0.005
    assigner = GraphCreditAssigner(graph)
    pruned_count = assigner.prune_near_zero_edges(epsilon=0.01)
    assert pruned_count == 1
    assert ("node_A", "node_C") not in graph.edges

    # 2. Timeout cleanup kills looping subprocess without resource leaks
    sandbox = PythonExecutionSandbox(timeout_sec=1)
    res = sandbox.verify_code("while True: pass", "assert True")
    assert not res.is_valid
    assert "timed out" in res.failed_assertions[0].lower()


def test_graph_credit_assigner_sqlite_persistence(tmp_path):
    db_file = str(tmp_path / "graph_state.db")
    graph1 = MockGraphMemory()
    assigner1 = GraphCreditAssigner(graph1, db_path=db_file)

    traj = GraphTrajectory(
        id="persist_traj",
        reasoning_trace="r",
        terminal_output="o",
        active_node_ids=["node_A"],
        traversed_edges=[("node_A", "node_B")],
    )
    # Fail feedback: threshold increases, edge weight decreases
    critique = CritiquePayload(aggregate_score=0.2)
    assigner1.apply_feedback(traj, critique, passed=False)

    w1 = graph1.get_edge_weight("node_A", "node_B")
    t1 = graph1.nodes["node_A"].activation_threshold

    # Instantiate brand new graph & assigner targeting the same SQLite db (simulating restart)
    graph2 = MockGraphMemory()
    _ = GraphCreditAssigner(graph2, db_path=db_file)

    assert graph2.get_edge_weight("node_A", "node_B") == pytest.approx(w1)
    assert graph2.nodes["node_A"].activation_threshold == pytest.approx(t1)

