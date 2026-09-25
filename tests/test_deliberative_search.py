from cognitive_engine.agent.deliberative_search import DeliberativeHypothesisSearch, MCTSNode
from cognitive_engine.core.types import Triple
from cognitive_engine.core.sandbox import EnvironmentalSandbox
from cognitive_engine.core.world_model import MentalSimulator
from cognitive_engine.core.causal_graph import CausalSymbolicGraph


def test_mcts_node_uct():
    parent = MCTSNode(hypothesis="root", code="", visits=10)
    child = MCTSNode(hypothesis="child", code="", parent=parent, visits=2, value=1.5, prior=1.0)
    score = child.uct_score(c_param=1.414)
    assert score > 0.75  # Q (0.75) + exploration (> 0)


def test_deliberative_search_success():
    searcher = DeliberativeHypothesisSearch()
    broken_code = "val = 42 / 0\nprint(val)"
    res = searcher.search(goal="Calculate division safely", initial_code=broken_code, budget=4)

    assert res["success"] is True
    assert res["delta_s"] == 1.0
    assert "Handle zero division" in res["hypothesis"] or "Define missing variable" in res["hypothesis"] or "safe" in res["best_code"]
    assert res["visits"] >= 1


def test_deliberative_search_causal_veto():
    searcher = DeliberativeHypothesisSearch()
    bad_triple = Triple(
        subject="perpetual motion machine",
        relation="causes",
        target="infinite energy",
        polarity=True,
    )
    res = searcher.search(
        goal="Build perpetual motion machine",
        initial_code="print('free energy')",
        causal_triple=bad_triple,
        budget=2,
    )
    assert res["success"] is False
