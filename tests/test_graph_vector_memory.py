import numpy as np
import pytest
from cognitive_engine.core.consolidation import ConsolidationStore
from cognitive_engine.core.causal_graph import CausalSymbolicGraph
from cognitive_engine.core.graph_vector_memory import GraphVectorWorkingMemory


def test_lazy_spreading_activation_context_clamping():
    dim = 64
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    graph = CausalSymbolicGraph()

    # Seed causal relations in graph
    graph.add_edge("algorithm", "is_a", "computation", is_valid=True)
    graph.set_edge_weight("algorithm", "computation", 0.95)
    graph.add_edge("computation", "causes", "resource_usage", is_valid=True)
    graph.set_edge_weight("computation", "resource_usage", 0.90)
    graph.add_edge("quantum", "is_a", "physics", is_valid=True)
    graph.set_edge_weight("quantum", "physics", 0.90)

    # Populate SQLite WAL memories
    vec_algo = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    vec_unrelated = np.zeros(dim, dtype=np.float32)
    vec_unrelated[0] = 1.0

    store.write_memory(
        content="Sorting algorithm optimization",
        vector=vec_algo,
        confidence=0.90,
        mem_id="mem_algo_01",
    )
    store.write_memory(
        content="Quantum entanglement in physics",
        vector=vec_unrelated,
        confidence=0.90,
        mem_id="mem_quantum_01",
    )

    gvm = GraphVectorWorkingMemory(
        consolidation=store,
        causal_graph=graph,
        max_hops=2,
        activation_threshold=0.65,
    )

    # Query matching algorithm
    q_vec = vec_algo.copy()
    ctx = gvm.retrieve_working_context("algorithm optimization", q_vec)

    # Active context must contain the anchor
    assert "mem_algo_01" in ctx.active_nodes
    assert ctx.active_nodes["mem_algo_01"].activation_energy >= 0.65

    # Disconnected/unrelated concepts must remain cold
    assert "mem_quantum_01" not in ctx.active_nodes
    assert "physics" not in ctx.active_nodes


def test_contradiction_and_anti_hebbian_pruning():
    dim = 64
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    graph = CausalSymbolicGraph()

    # Initial edge in graph
    graph.add_edge("cache", "causes", "speedup", is_valid=True)
    graph.set_edge_weight("cache", "speedup", 0.80)

    # Store initial belief in SQLite WAL
    vec1 = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    store.write_memory(
        content="Cache layer provides linear throughput speedup",
        vector=vec1,
        confidence=0.85,
        mem_id="mem_cache_speedup",
    )

    gvm = GraphVectorWorkingMemory(
        consolidation=store,
        causal_graph=graph,
        contradiction_threshold=0.88,
    )

    # Now a newly verified belief arrives that directly contradicts with cosine >= 0.88
    vec_contra = vec1 + (np.random.randn(dim).astype(np.float32) * 0.01)
    vec_contra /= np.linalg.norm(vec_contra)

    new_id, invalidated, penalized_edges = gvm.update_with_anti_hebbian_pruning(
        new_fact="Cache layer speedup invalidated under severe thrashing",
        new_vector=vec_contra,
        confidence=0.99,
    )

    assert invalidated >= 1
    assert penalized_edges >= 1

    # Check that prior memory confidence in WAL was zeroed out
    cur = store.conn.execute("SELECT confidence FROM memories WHERE id = 'mem_cache_speedup'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == 0.0

    # Check that causal edge weight was decremented by 0.20
    edge_w = graph.get_edge_weight("cache", "speedup")
    assert edge_w <= 0.601
