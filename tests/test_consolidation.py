import numpy as np
from cognitive_engine.core.consolidation import ConsolidationStore


def test_consolidation_decay_and_pruning():
    dim = 32
    store = ConsolidationStore(db_path=":memory:", lambda_decay=0.05, prune_threshold=0.20, dim=dim)

    t0 = 1000000.0
    rng = np.random.RandomState(42)

    # Insert 50 records
    ids = []
    for i in range(50):
        vec = rng.randn(dim).astype(np.float32)
        mid = store.write_memory(
            content=f"Knowledge record item number {i} regarding cognitive architecture",
            vector=vec,
            confidence=0.80,
            timestamp=t0,
            mem_id=f"mem_{i}",
        )
        ids.append(mid)

    # Reinforce record 0 multiple times across the window
    for h in [20, 40, 60, 80, 95]:
        store.reinforce("mem_0", current_time=t0 + h * 3600)

    # Simulate 100 hours elapsed
    t_later = t0 + 100 * 3600

    # Effective confidence of unreinforced records:
    # C_eff = 0.8 * exp(-0.05 * 100) = 0.8 * exp(-5.0) = 0.8 * 0.00673 = ~0.0054 < 0.20
    pruned_count = store.prune_dead_memories(current_time=t_later)

    print(f"\nPruned {pruned_count}/50 memories after 100 hours decay.")
    assert pruned_count == 49, f"Expected 49 dead memories to be pruned, got {pruned_count}"

    # Verify mem_0 remains intact
    cur = store.conn.execute("SELECT id, confidence FROM memories WHERE id = 'mem_0'")
    row = cur.fetchone()
    assert row is not None, "Reinforced memory mem_0 should not be pruned"
    assert row[0] == "mem_0"


def test_hybrid_search():
    dim = 16
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    vec1 = np.ones(dim, dtype=np.float32)
    vec2 = -np.ones(dim, dtype=np.float32)

    store.write_memory("Alpha synaptic plasticity and neuromorphic computing", vec1, mem_id="alpha")
    store.write_memory("Beta quantum entanglement and superconducting qubits", vec2, mem_id="beta")

    # Search with term matching alpha and vector close to vec1
    results = store.hybrid_search("neuromorphic", vec1, top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "alpha"


import os
import tempfile
import threading


def test_concurrent_read_write_stress_wal():
    # Test on a file-backed DB to exercise true WAL mode under concurrent threading
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "stress_wal.db")
        dim = 16
        store = ConsolidationStore(db_path=db_file, dim=dim)

        errors = []

        def worker_write(idx: int):
            try:
                vec = np.random.randn(dim).astype(np.float32)
                store.write_memory(f"Concurrent thread memory item {idx}", vec, confidence=0.75)
            except Exception as e:
                errors.append(f"Write error in thread {idx}: {e}")

        def worker_read(idx: int):
            try:
                q_vec = np.random.randn(dim).astype(np.float32)
                store.hybrid_search(f"thread item {idx}", q_vec, top_k=2)
                store.prune_dead_memories()
            except Exception as e:
                errors.append(f"Read error in thread {idx}: {e}")

        threads = []
        # 25 writer threads and 25 reader threads running concurrently
        for i in range(25):
            tw = threading.Thread(target=worker_write, args=(i,))
            tr = threading.Thread(target=worker_read, args=(i,))
            threads.extend([tw, tr])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        store.close()
        assert len(errors) == 0, f"Encountered database lock / concurrency errors: {errors}"
        print(f"\nConcurrent Read/Write Stress: 50 threads completed with 0 errors in WAL mode.")



def test_rrf_rank_inversion_prevention():
    dim = 16
    t0 = 2000000.0
    store = ConsolidationStore(db_path=":memory:", dim=dim, lambda_decay=0.05)

    v_query = np.ones(dim, dtype=np.float32)

    # 1. Ancient decayed memory with a solitary keyword match
    # Aged 60 hours: C_eff = 0.8 * exp(-0.05 * 60) = 0.8 * exp(-3) = 0.8 * 0.0498 = ~0.039
    store.write_memory(
        content="RareKeywordProtocol older specification",
        vector=-v_query,  # Orthogonal/opposite vector
        confidence=0.8,
        timestamp=t0 - 60 * 3600,
        mem_id="ancient_keyword",
    )

    # 2. Fresh memory with strong vector similarity and semantic content
    # Aged 0 hours: C_eff = 0.95
    store.write_memory(
        content="Contemporary modern active system architecture",
        vector=v_query,  # Exact vector match
        confidence=0.95,
        timestamp=t0,
        mem_id="fresh_vector",
    )

    # Search with keyword + vector simultaneously
    results = store.hybrid_search("RareKeywordProtocol architecture", v_query, top_k=2, current_time=t0)
    assert len(results) == 2
    # Fresh high-confidence memory must rank higher than the severely decayed keyword match
    assert results[0][0].id == "fresh_vector", "Rank inversion occurred: decayed keyword defeated fresh vector memory"


def test_asynchronous_queue_writer_worker():
    import time
    dim = 8
    store = ConsolidationStore(db_path=":memory:", dim=dim)

    # Dispatch 20 rapid non-blocking flushes
    mids = []
    for i in range(20):
        v = np.random.randn(dim).astype(np.float32)
        mid = store.enqueue_memory_flush(f"Queued memory item {i}", v, confidence=0.85)
        mids.append(mid)

    # Wait briefly for background worker thread to commit
    time.sleep(0.3)

    cur = store.conn.execute("SELECT count(*) FROM memories")
    count = cur.fetchone()[0]
    assert count == 20, f"Expected 20 queued memories to be committed, got {count}"
    store.close()


def test_semantic_deduplication_invalidation():
    dim = 32
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    rng = np.random.RandomState(42)
    vec1 = rng.randn(dim).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    # Initial state
    mid1 = store.write_memory("Entity order status is Pending", vec1, confidence=0.9, mem_id="order_123_v1")

    # Conflicting update with identical semantic vector
    vec2 = vec1.copy()
    invalidated = store.invalidate_conflicting_memories("order_123", vec2, similarity_threshold=0.92)
    assert invalidated == 1, f"Expected 1 memory to be invalidated, got {invalidated}"

    # Verify mid1 confidence is zeroed out
    cur = store.conn.execute("SELECT confidence FROM memories WHERE id = 'order_123_v1'")
    conf = cur.fetchone()[0]
    assert conf == 0.0, f"Expected confidence 0.0, got {conf}"
    store.close()


# Alias for test manifest sync
def test_deduplication_invalidation():
    test_semantic_deduplication_invalidation()


def test_fts_deletion_trigger_synchronization():
    store = ConsolidationStore(db_path=":memory:", dim=16)
    store.write_memory("UniqueTriggerKeyword", np.ones(16), mem_id="mem_trg_1")

    cur = store.conn.execute("SELECT id FROM memories_fts WHERE id = 'mem_trg_1'")
    assert cur.fetchone() is not None

    with store.conn:
        store.conn.execute("DELETE FROM memories WHERE id = 'mem_trg_1'")

    cur = store.conn.execute("SELECT id FROM memories_fts WHERE id = 'mem_trg_1'")
    assert cur.fetchone() is None
    store.close()


def test_quarantine_staging_and_promotion():
    store = ConsolidationStore(db_path=":memory:", dim=16)
    vec = np.ones(16, dtype=np.float32)

    # 1. Stage memory in quarantine
    mid = store.quarantine_memory(
        content="Quarantined unverified empirical conjecture",
        vector=vec,
        confidence=0.65,
        source="autonomous_curiosity",
    )
    assert mid is not None

    # Verify not yet visible in main memories
    cur = store.conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (mid,))
    assert cur.fetchone()[0] == 0

    # 2. List pending quarantined items
    pending = store.list_quarantined(status="PENDING")
    assert any(item["id"] == mid for item in pending)

    # 3. Promote item to main episodic store
    promoted = store.promote_quarantined(mid, reviewer_notes="manually validated")
    assert promoted is True

    # Verify now in main memories & FTS5
    cur = store.conn.execute("SELECT id, content FROM memories WHERE id = ?", (mid,))
    row = cur.fetchone()
    assert row is not None
    assert row[1] == "Quarantined unverified empirical conjecture"

    # 4. Reject another item
    mid2 = store.quarantine_memory("Bad synthetic noise", vector=vec)
    rejected = store.reject_quarantined(mid2, reason="unverified AST noise")
    assert rejected is True
    cur = store.conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (mid2,))
    assert cur.fetchone()[0] == 0
    store.close()


if __name__ == "__main__":
    test_consolidation_decay_and_pruning()
    test_hybrid_search()
    test_concurrent_read_write_stress_wal()
    test_rrf_rank_inversion_prevention()
    test_asynchronous_queue_writer_worker()
    test_semantic_deduplication_invalidation()
    test_quarantine_staging_and_promotion()
    print("All consolidation tests passed.")
