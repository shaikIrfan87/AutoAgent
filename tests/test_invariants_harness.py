import time
import threading
from pathlib import Path
import sys
import numpy as np
import torch
import pytest

# Ensure cognitive_engine package root is on sys.path
pkg_dir = str(Path(__file__).parent.parent / "cognitive_engine")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from core.saliency import SaliencyGate, FeatureEncoder
from core.causal_graph import CausalSymbolicGraph, EdgeType
from core.sandbox import PersistentREPLDaemon
from core.plastic_layer import MultiHeadPlasticLayer
from core.consolidation import ConsolidationStore


# =====================================================================
# INVARIANT 1: Sensory Gating & Anti-Masking
# =====================================================================
def test_invariant_1_gating_and_anomaly():
    gate = SaliencyGate()

    # 1. Low entropy spam must be dropped before embedding
    low_entropy = "spam " * 40
    res_spam = gate.evaluate(low_entropy)
    assert not res_spam.pass_filter, "Failed: Low entropy spam was not dropped."
    assert "Low information entropy" in res_spam.reason

    # 2. Adversarial semantic masking (Hex/Base64 masked in prose)
    masked_payload = "The secret code is 4a6f686e20446f6520536563726574204b65792121"
    res_masked = gate.evaluate(masked_payload)
    assert res_masked.is_anomaly, "Failed: OOV/Hex anomaly was not detected."

    # 3. High throughput check (>800 ops/sec in general test environment)
    sample_text = "Quantum computing uses superposition to accelerate calculations."
    for _ in range(50):
        _ = gate.evaluate(sample_text)
    t0 = time.time()
    for _ in range(1000):
        _ = gate.evaluate(sample_text)
    elapsed = time.time() - t0
    ops_per_sec = 1000 / elapsed
    assert ops_per_sec > 800, f"Throughput too low: {ops_per_sec:.2f} ops/sec."


# =====================================================================
# INVARIANT 2: Pearl-Compliant Causal Calculus & Cycle Safety
# =====================================================================
def test_invariant_2_causal_soundness():
    graph = CausalSymbolicGraph()

    # Structural Transitive: Dog -> Mammal -> Animal => Dog is Animal
    graph.add_edge("Dog", "is_a", "Mammal", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)
    graph.add_edge("Mammal", "is_a", "Animal", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)

    # Negative Constraint
    graph.add_edge("Dog", "cannot_be", "Robot", is_valid=False)

    # Valid transitive deduction
    valid, _ = graph.verify_hypothesis("Dog", "is_a", "Animal")
    assert valid, "Failed: Valid transitive deduction rejected."

    # Direct contradiction
    valid_contra, _ = graph.verify_hypothesis("Dog", "is_a", "Robot")
    assert not valid_contra, "Failed: Direct negative constraint was not enforced."

    # Cycle Detection: A -> B -> C -> A (Must not enter infinite recursion)
    graph.add_edge("A", "leads_to", "B", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)
    graph.add_edge("B", "leads_to", "C", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)
    graph.add_edge("C", "leads_to", "A", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)

    t0 = time.time()
    valid_cycle, _ = graph.verify_hypothesis("A", "leads_to", "C")
    assert time.time() - t0 < 0.001, "Cycle verification exceeded 1ms target."


# =====================================================================
# INVARIANT 3: Persistent Sandbox, POSIX Limits & Rollback
# =====================================================================
def test_invariant_3_sandbox_isolation():
    daemon = PersistentREPLDaemon()
    daemon.start()

    try:
        # Step 1: Valid execution (<10ms)
        t0 = time.time()
        res_ok = daemon.execute("x = 42\ny = x * 2\nprint(y)")
        lat = (time.time() - t0) * 1000
        assert res_ok["status"] == "success" and res_ok["output"] == "84"
        assert lat < 20, f"IPC execution latency too high: {lat:.2f}ms"

        # Step 2: Failure with state rollback (Namespace should not leak)
        res_fail = daemon.execute("z = 100\n1 / 0")
        assert res_fail["status"] == "failed"
        assert res_fail["delta"] == -1.0

        # Verify 'z' did not leak into global namespace
        res_leak_check = daemon.execute("print('z' in globals())")
        assert res_leak_check["output"] == "False", "Failed: Dirty namespace state was not rolled back!"

    finally:
        daemon.terminate()


# =====================================================================
# INVARIANT 4: Multi-Head Plastic Layer & Oja Norm Bounding
# =====================================================================
def test_invariant_4_plastic_bounds_and_crosstalk():
    layer = MultiHeadPlasticLayer(dim=32, num_heads=4, decay_rate=0.92, eta=0.05)

    # 1. Ingest 500 orthogonal keys to test saturation & numerical stability
    for i in range(500):
        k = torch.randn(1, 32)
        v = torch.randn(1, 32)
        layer.adapt(k, v)

    # Assert Frobenius norm is strictly bounded by Oja's rule
    f_norm = torch.norm(layer.A_fast).item()
    assert f_norm <= 2.0, f"Plastic matrix exploded! Frobenius norm: {f_norm}"
    assert not torch.isnan(layer.A_fast).any(), "NaN detected in plastic weights."

    # 2. Test Associative Recall
    test_k = torch.randn(1, 32)
    test_v = torch.randn(1, 32)
    layer.adapt(test_k, test_v)
    recalled_v = layer.recall(test_k)

    cosine_sim = torch.cosine_similarity(recalled_v, test_v).item()
    assert cosine_sim > 0.85, f"Associative recall degraded: {cosine_sim:.3f}"


# =====================================================================
# INVARIANT 5: SQLite WAL Concurrency & RRF Decay
# =====================================================================
def test_invariant_5_consolidation_concurrency():
    store = ConsolidationStore(db_path=":memory:")

    # Concurrent write/read stress test (50 threads)
    errors = []

    def worker(idx):
        try:
            vec = np.random.randn(64).astype(np.float32)
            store.write_memory(f"entity_{idx}", f"Content for entity {idx}", vec)
            _ = store.hybrid_search(f"entity {idx}", vec, top_k=2)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store.close()
    assert len(errors) == 0, f"SQLite concurrency failed with errors: {errors}"
