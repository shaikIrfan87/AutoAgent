"""
Tests for 2026 Production Endurance Gates:
1. Paraphrase Gating (Semantic ONNX embedding vs false novelty)
2. Workspace Checkpointing & Rollback Determinism
3. Active Temporal Contradiction Resolution
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from cognitive_engine.core.embeddings import get_semantic_embedding
from cognitive_engine.core.sandbox import SandboxCheckpoint, EnvironmentalSandbox
from cognitive_engine.core.consolidation import ConsolidationStore


def test_gate_1_paraphrase_semantic_embedding():
    # Paraphrases with distinct phrasing must have high cosine similarity (>= 0.65)
    t1 = "downloading a payload"
    t2 = "fetch the data"

    v1 = get_semantic_embedding(t1)
    v2 = get_semantic_embedding(t2)

    assert len(v1) == 384
    assert len(v2) == 384

    cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert cos_sim >= 0.65, f"Expected high cosine similarity for synonyms, got {cos_sim:.4f}"


def test_gate_2_workspace_checkpoint_and_rollback():
    # Setup temporary test workspace
    ws = tempfile.mkdtemp(prefix="test_workspace_")
    file_a = os.path.join(ws, "config.json")
    file_b = os.path.join(ws, "script.py")

    with open(file_a, "w") as f:
        f.write('{"status": "pristine"}')
    with open(file_b, "w") as f:
        f.write("print('hello world')")

    # Create atomic checkpoint
    sandbox = EnvironmentalSandbox()
    chk = sandbox.create_checkpoint(ws)

    try:
        # Simulate destructive agent action
        os.remove(file_a)
        with open(file_b, "w") as f:
            f.write("corrupted malicious payload")
        with open(os.path.join(ws, "rogue_file.tmp"), "w") as f:
            f.write("untracked residue")

        assert not os.path.exists(file_a)
        assert os.path.exists(os.path.join(ws, "rogue_file.tmp"))

        # Trigger restore
        chk.restore()

        # Verify byte-for-byte restoration
        assert os.path.exists(file_a)
        with open(file_a, "r") as f:
            assert f.read() == '{"status": "pristine"}'

        with open(file_b, "r") as f:
            assert f.read() == "print('hello world')"

        assert not os.path.exists(os.path.join(ws, "rogue_file.tmp"))
    finally:
        chk.close()
        sandbox.close()
        if os.path.exists(ws):
            shutil.rmtree(ws, ignore_errors=True)


def test_gate_3_active_temporal_contradiction_resolution():
    store = ConsolidationStore(db_path=":memory:", dim=384)
    v1 = get_semantic_embedding("API endpoint route is v1")

    # 1. Insert original fact
    mid_v1 = store.write_memory(
        content="API endpoint route is v1",
        vector=v1,
        confidence=0.95,
        mem_id="api_route_v1",
        entity_id="api_route",
    )

    # 2. Insert temporal contradiction with matching entity_id
    v2 = get_semantic_embedding("API endpoint route is v2")
    mid_v2 = store.write_memory(
        content="API endpoint route is v2",
        vector=v2,
        confidence=0.95,
        mem_id="api_route_v2",
        entity_id="api_route",
    )

    # Assert v1 confidence was actively zeroed out
    cur = store.conn.execute("SELECT confidence FROM memories WHERE id = 'api_route_v1'")
    conf_v1 = cur.fetchone()[0]
    assert conf_v1 == 0.0, f"Expected v1 confidence to be zeroed, got {conf_v1}"

    # Assert v2 is fully active
    cur = store.conn.execute("SELECT confidence FROM memories WHERE id = 'api_route_v2'")
    conf_v2 = cur.fetchone()[0]
    assert conf_v2 == 0.95

    # Hybrid search must retrieve v2, not poisoned by v1
    results = store.hybrid_search("API endpoint route", v2, top_k=1)
    assert len(results) >= 1
    top_record = results[0][0]
    assert top_record.id == "api_route_v2"
    assert "v2" in top_record.content

    store.close()
