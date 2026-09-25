import threading
import numpy as np
import pytest
from cognitive_engine.core.consolidation import HybridConsolidationStore

def test_concurrent_contradiction_hammer():
    store = HybridConsolidationStore(db_path="assets/test_concurrency.db")
    errors = []
    
    # Two worker groups asserting contradictory states of the same entity concurrently
    def write_worker(entity_val: str, conf: float):
        try:
            vec = np.random.randn(64).astype(np.float32)
            for _ in range(20):
                store.write_memory(
                    content=f"System state: server_cluster_status is {entity_val}",
                    vector=vec,
                    confidence=conf
                )
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=write_worker, args=("ONLINE_ACTIVE", 0.95)),
        threading.Thread(target=write_worker, args=("OFFLINE_MAINTENANCE", 0.99)),
        threading.Thread(target=write_worker, args=("DECOMMISSIONED", 0.40)),
    ]
    
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(errors) == 0, f"SQLite WAL write lock contention failed: {errors}"
    
    # Read final state
    results = store.search_semantic(np.random.randn(64).astype(np.float32), top_k=5)
    assert len(results) > 0, "No records committed during stress run"
    store.close()
