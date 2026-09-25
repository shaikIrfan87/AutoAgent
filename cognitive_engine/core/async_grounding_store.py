import sqlite3
import queue
import threading
import time
import math
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("AsyncGroundingStore")

class DecayedRRFConsolidationStore:
    """
    Invariant 5: Hybrid FTS5 + Vector SQLite storage backed by
    decayed Reciprocal Rank Fusion (RRF) and asynchronous thread execution.
    """

    def __init__(self, db_path: str = "assets/cognitive_memory.db", dim: int = 64, decay_lambda: float = 0.05):
        self.db_path = db_path
        self.dim = dim
        self.decay_lambda = decay_lambda
        self.write_queue = queue.Queue()
        self.running = True
        
        self._init_db()
        self.worker_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self.worker_thread.start()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 1,
                last_accessed REAL NOT NULL
            );
        """)
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts 
            USING fts5(id UNINDEXED, content);
        """)
        conn.commit()
        conn.close()

    def _process_write_queue(self):
        """Dedicated writer avoiding sqlite3.OperationalError database locks."""
        conn = sqlite3.connect(self.db_path)
        while self.running:
            try:
                task = self.write_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                cur = conn.cursor()
                vec = task["vector"]
                vec_bytes = vec.tobytes() if hasattr(vec, "tobytes") else np.asarray(vec, dtype=np.float32).tobytes()

                cur.execute("""
                    INSERT INTO memories (id, content, vector, confidence, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content=excluded.content,
                        vector=excluded.vector,
                        confidence=MIN(1.0, memories.confidence + 0.15),
                        access_count=memories.access_count + 1,
                        last_accessed=excluded.last_accessed;
                """, (
                    task["id"], task["content"], vec_bytes,
                    task.get("confidence", 0.95), 1, time.time()
                ))
                cur.execute("""
                    INSERT OR IGNORE INTO memories_fts (id, content) VALUES (?, ?);
                """, (task["id"], task["content"]))
                conn.commit()
            except Exception as ex:
                logger.error(f"Commit failed in SQLite writer: {ex}")
            finally:
                self.write_queue.task_done()
        conn.close()

    def hybrid_search(self, query: str, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion with time-decayed scoring."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # 1. Lexical BM25 search via FTS5
        fts_ranks = {}
        try:
            fts_rows = cur.execute(
                "SELECT id, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT 20",
                (query,)
            ).fetchall()
            for rank_idx, (m_id, _) in enumerate(fts_rows):
                fts_ranks[m_id] = rank_idx + 1
        except sqlite3.OperationalError:
            pass

        # 2. Vector scan
        rows = cur.execute(
            "SELECT id, content, vector, confidence, access_count, last_accessed FROM memories"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        scored_candidates = []
        now = time.time()

        for m_id, content, vec_blob, conf, access_cnt, last_acc in rows:
            # Reconstruct vector and calculate cosine similarity
            stored_vec = np.frombuffer(vec_blob, dtype=np.float32)
            if stored_vec.shape != query_vec.shape:
                continue
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)) + 1e-8
            cos_sim = float(np.dot(query_vec, stored_vec) / denom)

            # Exponential Ebbinghaus Decay
            elapsed_hours = (now - last_acc) / 3600.0
            c_eff = conf * math.exp(-self.decay_lambda * elapsed_hours)

            # Drop degraded memories
            if c_eff < 0.20:
                continue

            scored_candidates.append({
                "id": m_id,
                "content": content,
                "cos_sim": cos_sim,
                "c_eff": c_eff,
                "access_count": access_cnt
            })

        # Rank vectors
        scored_candidates.sort(key=lambda x: x["cos_sim"], reverse=True)
        for rank_idx, cand in enumerate(scored_candidates):
            cand["vec_rank"] = rank_idx + 1

        # Calculate Rank Inversion-hardened score: Final = RRF * C_eff
        results = []
        for cand in scored_candidates:
            m_id = cand["id"]
            r_fts = fts_ranks.get(m_id, 100)
            r_vec = cand["vec_rank"]
            
            rrf_base = (0.40 / (60 + r_fts)) + (0.60 / (60 + r_vec))
            cand["final_score"] = rrf_base * cand["c_eff"]
            results.append(cand)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def queue_async_write(self, item_id: str, content: str, vector: np.ndarray, confidence: float = 0.95):
        self.write_queue.put({
            "id": item_id,
            "content": content,
            "vector": vector,
            "confidence": confidence
        })

    def close(self):
        self.running = False
        self.worker_thread.join(timeout=2.0)
