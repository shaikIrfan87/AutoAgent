import numpy as np
import pytest
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.consolidation import ConsolidationStore


def test_consolidation_overwrite_belief_zeroes_confidence():
    """Verify overwrite_belief zeroes out confidence of semantically conflicting memories (cos_sim >= 0.88)."""
    dim = 32
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    try:
        rng = np.random.RandomState(42)
        base_vec = rng.randn(dim).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)

        # 1. Store initial belief
        m1 = store.write_memory("The system status is OFFLINE", base_vec, confidence=0.9, mem_id="status_mem_1")

        # 2. Add slightly perturbed vector with cos_sim >= 0.88
        perturbed_vec = base_vec + (rng.randn(dim).astype(np.float32) * 0.05)
        perturbed_vec /= np.linalg.norm(perturbed_vec)
        cos_sim = float(np.dot(base_vec, perturbed_vec))
        assert cos_sim >= 0.88, f"Expected cos_sim >= 0.88, got {cos_sim}"

        # 3. Overwrite belief
        m2, invalidated_count = store.overwrite_belief(
            "The system status is ONLINE",
            perturbed_vec,
            confidence=1.0,
            similarity_threshold=0.88,
            prune_immediately=False,
        )

        assert invalidated_count == 1
        assert m2 != m1

        # 4. Check that m1 confidence was zeroed out, m2 is 1.0
        cur = store.conn.execute("SELECT confidence FROM memories WHERE id = ?", (m1,))
        assert cur.fetchone()[0] == 0.0

        cur = store.conn.execute("SELECT confidence FROM memories WHERE id = ?", (m2,))
        assert cur.fetchone()[0] == 1.0
    finally:
        store.close()


def test_consolidation_overwrite_belief_prunes_immediately():
    """Verify overwrite_belief with prune_immediately=True removes contradicting memory from memories and FTS."""
    dim = 32
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    try:
        rng = np.random.RandomState(123)
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        m1 = store.write_memory("Alpha cluster host is 192.168.1.10", vec, confidence=0.85, mem_id="cluster_alpha_v1")

        # Overwrite with same vector and prune_immediately=True
        m2, count = store.overwrite_belief(
            "Alpha cluster host is 192.168.1.50",
            vec,
            confidence=1.0,
            prune_immediately=True,
        )
        assert count == 1

        # m1 must be deleted from memories and memories_fts
        cur = store.conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (m1,))
        assert cur.fetchone()[0] == 0

        cur = store.conn.execute("SELECT COUNT(*) FROM memories_fts WHERE id = ?", (m1,))
        assert cur.fetchone()[0] == 0
    finally:
        store.close()


def test_consolidation_unrelated_memory_preserved():
    """Verify memories with cosine similarity < 0.88 are unaffected by overwrite_belief."""
    dim = 32
    store = ConsolidationStore(db_path=":memory:", dim=dim)
    try:
        rng = np.random.RandomState(99)
        vec_a = np.zeros(dim, dtype=np.float32)
        vec_a[0] = 1.0  # Orthogonal vector A

        vec_b = np.zeros(dim, dtype=np.float32)
        vec_b[1] = 1.0  # Orthogonal vector B

        m_unrelated = store.write_memory("Unrelated fact about botany", vec_a, confidence=0.95, mem_id="unrelated_1")

        # Write memory with orthogonal vector B
        m_new, count = store.overwrite_belief("Fact about astrophysics", vec_b, confidence=1.0)
        assert count == 0

        # Unrelated memory retains full confidence
        cur = store.conn.execute("SELECT confidence FROM memories WHERE id = ?", (m_unrelated,))
        assert cur.fetchone()[0] == 0.95
    finally:
        store.close()


def test_engine_update_belief_end_to_end():
    """Verify CognitiveEngine update_belief embeds and updates beliefs in SQLite WAL."""
    engine = CognitiveEngine(db_path=":memory:")
    try:
        # Step 1: Initial belief
        engine.consolidation.write_memory(
            "System state: OFFLINE",
            engine.saliency._embed("System state: OFFLINE"),
            confidence=0.9,
            mem_id="status_doc_1",
        )

        # Step 2: Update belief with contradicting fact (sim ~0.87 >= 0.85)
        new_id, invalidated = engine.update_belief(
            "System state: ONLINE",
            confidence=1.0,
            similarity_threshold=0.85,
        )
        assert invalidated >= 1

        # Verify old status confidence is zeroed out
        cur = engine.consolidation.conn.execute("SELECT confidence FROM memories WHERE id = 'status_doc_1'")
        assert cur.fetchone()[0] == 0.0
    finally:
        if hasattr(engine, "world_model") and hasattr(engine.world_model, "tuner"):
            engine.world_model.tuner.stop()
