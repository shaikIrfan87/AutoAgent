import sqlite3
import time
import pytest
from cognitive_engine.core.graph_epistemic_scanner import EpistemicTarget, GraphEpistemicScanner
from cognitive_engine.agent.curiosity import CuriosityDaemon
from cognitive_engine.core.causal_graph import CausalSymbolicGraph


def _create_test_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
            session_id TEXT NOT NULL DEFAULT 'default_session',
            content TEXT NOT NULL,
            vector BLOB NOT NULL,
            confidence REAL DEFAULT 0.8,
            access_count INTEGER DEFAULT 1,
            last_accessed REAL NOT NULL
        )
    """)
    return conn


def test_scanner_detects_decaying_memories():
    conn = _create_test_db()
    t0 = 1000000.0

    # Fresh high confidence record
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("mem_fresh", "default_tenant", "s1", "Fresh knowledge", b"0", 0.90, 1, t0),
    )
    # Stale record with decayed effective confidence
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("mem_stale", "default_tenant", "s1", "Stale decaying knowledge", b"0", 0.50, 1, t0 - 20 * 3600),
    )

    scanner = GraphEpistemicScanner(db_conn=conn, decay_threshold=0.40, lambda_decay=0.05)
    targets = scanner.scan_stale_or_uncertain_nodes(top_k=5, current_time=t0)

    assert len(targets) == 1
    assert targets[0].node_id == "mem_stale"
    assert targets[0].target_type == "decaying_confidence"
    assert targets[0].priority > 0.50


def test_scanner_detects_topological_holes_and_unverified_edges():
    causal = CausalSymbolicGraph()
    # Add an isolated node and a low-weight edge
    causal.add_edge("node_a", "causes", "node_b", is_valid=True)
    causal.set_edge_weight("node_a", "node_b", 0.20)  # low weight
    causal.graph.add_node("orphan_node")  # degree 0

    scanner = GraphEpistemicScanner(causal_graph=causal)
    targets = scanner.scan_stale_or_uncertain_nodes(top_k=20)

    types = [t.target_type for t in targets]
    assert "topological_hole" in types
    assert "unverified_edge" in types


def test_curiosity_step_dispatches_scanner_target():
    conn = _create_test_db()
    t0 = time.time()
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("mem_decayed", "default_tenant", "s1", "Unverified memory concept", b"0", 0.20, 1, t0),
    )

    dispatched = []

    class MockEngine:
        def __init__(self):
            self.consolidation = None
            self.causal = None

        def process(self, query: str):
            dispatched.append(query)
            return {"status": "success", "query": query}

    engine = MockEngine()
    scanner = GraphEpistemicScanner(db_conn=conn, decay_threshold=0.40)
    daemon = CuriosityDaemon(engine=engine, scanner=scanner)

    res = daemon.step()
    assert res is not None
    assert len(dispatched) == 1
    assert "mem_decayed" in dispatched[0]
    assert "decaying_confidence" in dispatched[0]
