import pytest
import sqlite3
from unittest.mock import MagicMock

from cognitive_engine.core.autotelic import AutotelicExperimentLoop, OpenWorldHypothesis
from cognitive_engine.core.consolidation import ConsolidationStore, HybridConsolidationStore
from cognitive_engine.core.graph_epistemic_scanner import GraphEpistemicScanner, EpistemicTarget
from cognitive_engine.agent.executive_loop import ExecutiveLoop
from cognitive_engine.agent.curiosity_explorer import AutonomousCuriosityExplorer
from cognitive_engine.agent.curiosity import CuriosityDaemon


def test_epistemic_gap_formulate_open_world_hypothesis():
    """Verify conversion of epistemic gap into OpenWorldHypothesis with search query and test template."""
    autotelic = AutotelicExperimentLoop()
    hypo = autotelic.formulate_open_world_hypothesis("quicksort")

    assert isinstance(hypo, OpenWorldHypothesis)
    assert hypo.concept == "quicksort"
    assert "quicksort" in hypo.search_query.lower()
    assert "def verify_concept" in hypo.verification_code_template
    assert "VERIFIED_QUICKSORT" in hypo.expected_predicate


def test_autonomous_curiosity_explorer_step_inquiry(tmp_path):
    """Verify full inquiry cycle: gap detection, web search, sandbox verification, and memory consolidation."""
    db_file = str(tmp_path / "test_curiosity.db")
    store = HybridConsolidationStore(db_path=db_file)

    # Populate a stale/low-confidence memory to serve as epistemic target
    store.write_memory(
        mem_id="concept_graph_coloring",
        content="Graph coloring algorithm needs empirical verification",
        vector=[0.1] * 384,
        confidence=0.15,
    )

    # Mock executive loop to avoid live external internet dependencies in unit test
    mock_loop = MagicMock(spec=ExecutiveLoop)
    mock_loop.execute_open_world_action.return_value = {
        "action": "http_get",
        "url": "https://api.duckduckgo.com/?q=Python+graph+coloring&format=json",
        "status": "success",
        "success": True,
        "status_code": 200,
    }
    mock_loop.execute_algorithmic_task.return_value = {
        "task": "algorithmic_task",
        "status": "success",
        "success": True,
        "output": "VERIFIED_CONCEPT_GRAPH_COLORING",
        "execution_time_ms": 12.5,
    }

    explorer = AutonomousCuriosityExplorer(
        executive_loop=mock_loop,
        store=store,
    )

    cycle = explorer.step_inquiry()

    assert cycle is not None
    assert cycle["concept"] == "concept_graph_coloring"
    assert cycle["search_status"] == "success"
    assert cycle["execution_success"] is True
    assert len(explorer.history) == 1

    # Verify automatic persistence of verified concepts into SQLite HybridConsolidationStore
    cur = store.conn.execute("SELECT id, content, confidence FROM memories WHERE id = ?", ("emp_concept_graph_coloring",))
    row = cur.fetchone()
    assert row is not None
    assert "Empirically verified concept: concept_graph_coloring" in row[1]
    assert row[2] >= 0.8  # Default confidence

    store.close()


def test_autonomous_curiosity_explorer_live_sandbox(tmp_path):
    """Verify live sandbox execution of generated hypothesis template."""
    db_file = str(tmp_path / "test_sandbox_curiosity.db")
    store = HybridConsolidationStore(db_path=db_file)
    store.write_memory(
        mem_id="matrix_multiplication",
        content="Matrix multiplication kernel verification",
        vector=[0.2] * 384,
        confidence=0.20,
    )

    real_loop = ExecutiveLoop()
    # Mock only the web request to avoid network flake
    real_loop.execute_open_world_action = MagicMock(return_value={
        "action": "http_get",
        "status": "success",
        "success": True,
    })

    explorer = AutonomousCuriosityExplorer(
        executive_loop=real_loop,
        store=store,
    )

    cycle = explorer.step_inquiry()

    assert cycle is not None
    assert cycle["execution_success"] is True
    assert "VERIFIED_MATRIX_MULTIPLICATION" in cycle["output"]

    store.close()


def test_curiosity_daemon_explorer_alternation():
    """Verify CuriosityDaemon step alternates between open-world inquiry and autotelic experiments."""
    mock_engine = MagicMock()
    mock_store = MagicMock()
    mock_engine.consolidation = mock_store

    mock_explorer = MagicMock(spec=AutonomousCuriosityExplorer)
    mock_explorer.step_inquiry.return_value = {"concept": "fft", "execution_success": True}

    mock_autotelic = MagicMock(spec=AutotelicExperimentLoop)
    mock_autotelic.active_hypotheses = []

    daemon = CuriosityDaemon(
        engine=mock_engine,
        autotelic=mock_autotelic,
        explorer=mock_explorer,
    )

    # First cycle: _cycle_counter = 1 (odd) -> explorer.step_inquiry()
    res1 = daemon.step()
    assert res1 == {"concept": "fft", "execution_success": True}
    assert mock_explorer.step_inquiry.call_count == 1
