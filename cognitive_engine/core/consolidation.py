import math
import os
import queue
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from .types import MemoryRecord
except ImportError:
    from types import MemoryRecord


class ConsolidationStore:
    """Invariant 5: Dual-Tier Long-Term Storage & Continual Decay.
    SQLite + FTS5 + BLOB vector storage with exponential temporal decay,
    serialized background queue worker, WAL-mode concurrency, and RRF hybrid retrieval.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        lambda_decay: float = 0.05,
        prune_threshold: float = 0.20,
        boost_factor: float = 0.15,
        dim: int = 384,
    ):
        self.db_path = db_path
        self.lambda_decay = lambda_decay
        self.prune_threshold = prune_threshold
        self.boost_factor = boost_factor
        self.dim = dim
        self.quarantine_mode = bool(os.environ.get("AUTOAGENT_MEMORY_QUARANTINE", False))
        self._lock = threading.RLock()
        self._write_queue: queue.Queue = queue.Queue()
        if self.db_path != ":memory:" and os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._init_db()

        # Start dedicated background worker to serialize asynchronous write bursts
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Sequential writer thread consuming from queue to eliminate sqlite3.BusyError."""
        while not self._stop_event.is_set():
            try:
                task = self._write_queue.get(timeout=0.1)
                if task is None:
                    break
                content, vec_blob, conf, ts, mid, tenant_id, session_id, done_evt = task
                with self._lock:
                    with self.conn:
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO memories (id, tenant_id, session_id, content, vector, confidence, access_count, last_accessed)
                            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                            """,
                            (mid, tenant_id, session_id, content, vec_blob, conf, ts),
                        )
                        self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
                        self.conn.execute("INSERT INTO memories_fts (id, content) VALUES (?, ?)", (mid, content))
                if done_evt:
                    done_evt.set()
                self._write_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def _init_db(self) -> None:
        with self._lock:
            with self.conn:
                if self.db_path != ":memory:":
                    self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA busy_timeout=5000;")
                self.conn.execute("PRAGMA synchronous=NORMAL;")
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                        session_id TEXT NOT NULL DEFAULT 'default_session',
                        source TEXT NOT NULL DEFAULT 'cognitive_core',
                        content TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        confidence REAL DEFAULT 0.8,
                        access_count INTEGER DEFAULT 1,
                        last_accessed REAL NOT NULL
                    );
                """)
                # Automatic schema migration for existing SQLite databases
                cur = self.conn.execute("PRAGMA table_info(memories);")
                cols = [row[1] for row in cur.fetchall()]
                if "tenant_id" not in cols:
                    self.conn.execute("ALTER TABLE memories ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default_tenant';")
                if "session_id" not in cols:
                    self.conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default_session';")
                if "source" not in cols:
                    self.conn.execute("ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'cognitive_core';")

                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_tenant_session ON memories(tenant_id, session_id);
                """)
                self.conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, content);
                """)
                self.conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_memories_delete AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_fts WHERE id = old.id;
                    END;
                """)
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS plastic_weights (
                        tenant_id TEXT,
                        session_id TEXT,
                        weights BLOB NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (tenant_id, session_id)
                    );
                """)
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_quarantine (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
                        session_id TEXT NOT NULL DEFAULT 'default_session',
                        source TEXT NOT NULL DEFAULT 'autonomous_writer',
                        content TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        confidence REAL DEFAULT 0.8,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        created_at REAL NOT NULL,
                        reviewer_notes TEXT
                    );
                """)

    def save_plastic_state(self, tenant_id: str, session_id: str, weights_blob: bytes) -> None:
        """ACID snapshot of fast-weight tensor into SQLite WAL."""
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO plastic_weights (tenant_id, session_id, weights, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(tenant_id, session_id) DO UPDATE SET
                        weights = excluded.weights,
                        last_updated = CURRENT_TIMESTAMP
                    """,
                    (tenant_id, session_id, weights_blob),
                )

    def load_plastic_state(self, tenant_id: str, session_id: str) -> Optional[bytes]:
        """Restore fast-weight tensor blob from SQLite."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT weights FROM plastic_weights WHERE tenant_id = ? AND session_id = ?",
                (tenant_id, session_id),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def prune_stale_plastic_weights(self, max_age_days: int = 7) -> int:
        """Purge abandoned working session tensors older than max_age_days."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "DELETE FROM plastic_weights WHERE last_updated < datetime('now', ?)",
                (f"-{max_age_days} days",),
            )
            return cur.rowcount

    def add_memory(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        confidence: float = 0.8,
        vector: Optional[np.ndarray] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        **kwargs,
    ) -> str:
        """Convenience method to write episodic memory with optional metadata tags."""
        tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
        return self.write_memory(
            content=f"{content}{tag_str}",
            confidence=confidence,
            vector=vector,
            tenant_id=tenant_id,
            session_id=session_id,
            **kwargs,
        )

    def write_memory(
        self,
        arg1: Optional[str] = None,
        arg2: Any = None,
        arg3: Any = None,
        content: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        confidence: float = 0.8,
        timestamp: Optional[float] = None,
        mem_id: Optional[str] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        **kwargs,
    ) -> str:
        # Support both keyword calls and flexible positional combinations
        if content is not None and vector is not None:
            cnt = content
            vec = vector
            mid = mem_id or (arg1 if isinstance(arg1, str) else str(uuid.uuid4()))
        elif isinstance(arg2, str) and isinstance(arg3, np.ndarray):
            mid = str(arg1)
            cnt = arg2
            vec = arg3
        elif isinstance(arg2, np.ndarray):
            cnt = str(arg1)
            vec = arg2
            mid = mem_id or str(uuid.uuid4())
            if isinstance(arg3, (int, float)):
                confidence = float(arg3)
        else:
            cnt = str(content or arg1 or "")
            vec = vector if vector is not None else arg2
            if not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)
            mid = mem_id or str(uuid.uuid4())

        t_id = kwargs.get("tenant_id", tenant_id)
        s_id = kwargs.get("session_id", session_id)
        src = kwargs.get("source", "cognitive_core")
        ts = timestamp if timestamp is not None else time.time()
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=np.float32)
        vec_blob = vec.astype(np.float32).tobytes()

        # Guard SQLite WAL Commit Invariant: verified execution delta_s == 1.0
        delta_s = kwargs.get("delta_s")
        if delta_s is not None and delta_s <= 0.0:
            raise ValueError(f"Ungrounded memory write rejected: delta_s={delta_s} <= 0.0")
        if src in ("web_ingestor", "external_scraper") and delta_s != 1.0:
            raise ValueError(f"External ingestion memory write requires verified execution delta_s == 1.0 (got {delta_s})")

        # Active temporal contradiction resolution
        entity_id = kwargs.get("entity_id")
        if entity_id:
            sim_thresh = kwargs.get("similarity_threshold", 0.85)
            self.invalidate_conflicting_memories(entity_id, vec, similarity_threshold=sim_thresh)
        elif kwargs.get("auto_invalidate", True) and isinstance(cnt, str) and (cnt.startswith("Query:") or "Output:" in cnt):
            self.invalidate_conflicting_memories(cnt, vec, similarity_threshold=0.88)

        # Autonomous quarantine staging if quarantine mode is active or requested
        if kwargs.get("quarantine") or (self.quarantine_mode and src not in ("user", "ground_truth", "manual")):
            return self.quarantine_memory(
                content=cnt,
                vector=vec,
                confidence=confidence,
                source=src,
                mem_id=mid,
                tenant_id=t_id,
                session_id=s_id,
            )

        # Synchronous write via lock
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO memories (id, tenant_id, session_id, source, content, vector, confidence, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (mid, t_id, s_id, src, cnt, vec_blob, confidence, ts),
                )
                self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
                self.conn.execute("INSERT INTO memories_fts (id, content) VALUES (?, ?)", (mid, cnt))

        return mid

    store_memory = write_memory
    add_memory = write_memory

    def quarantine_memory(
        self,
        content: str,
        vector: Optional[np.ndarray] = None,
        confidence: float = 0.8,
        source: str = "autonomous_writer",
        mem_id: Optional[str] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> str:
        """Stages an unverified autonomous memory for offline/human-in-the-loop validation."""
        mid = mem_id or str(uuid.uuid4())
        vec = vector if vector is not None else np.zeros(self.dim, dtype=np.float32)
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=np.float32)
        vec_blob = vec.astype(np.float32).tobytes()
        now = time.time()
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memory_quarantine
                (id, tenant_id, session_id, source, content, vector, confidence, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (mid, tenant_id, session_id, source, content, vec_blob, confidence, now),
            )
        return mid

    def list_quarantined(self, status: str = "PENDING", limit: int = 100) -> List[Dict[str, Any]]:
        """Lists staged memories awaiting validation."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "SELECT id, tenant_id, session_id, source, content, confidence, status, created_at, reviewer_notes FROM memory_quarantine WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
            cols = [col[0] for col in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def promote_quarantined(self, mem_id: str, reviewer_notes: Optional[str] = None) -> bool:
        """Promotes a verified quarantined memory record into primary episodic storage."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "SELECT id, tenant_id, session_id, source, content, vector, confidence FROM memory_quarantine WHERE id = ? AND status = 'PENDING'",
                (mem_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            mid, t_id, s_id, src, content, vec_blob, conf = row
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memories (id, tenant_id, session_id, source, content, vector, confidence, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (mid, t_id, s_id, src, content, vec_blob, conf, time.time()),
            )
            self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
            self.conn.execute("INSERT INTO memories_fts (id, content) VALUES (?, ?)", (mid, content))
            self.conn.execute(
                "UPDATE memory_quarantine SET status = 'APPROVED', reviewer_notes = ? WHERE id = ?",
                (reviewer_notes or "promoted_to_episodic", mid),
            )
            return True

    def reject_quarantined(self, mem_id: str, reason: str = "rejected") -> bool:
        """Rejects a quarantined memory record without polluting episodic memory."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "UPDATE memory_quarantine SET status = 'REJECTED', reviewer_notes = ? WHERE id = ? AND status = 'PENDING'",
                (reason, mem_id),
            )
            return cur.rowcount > 0

    def invalidate_conflicting_memories(
        self,
        entity_id_or_content: str,
        new_vector: np.ndarray,
        similarity_threshold: float = 0.88,
        prune_immediately: bool = False,
    ) -> int:
        """Zeroes out confidence (and optionally prunes) outdated records with identical entity signatures or overlapping semantics."""
        invalidated = 0
        if not isinstance(new_vector, np.ndarray) or new_vector.size == 0:
            return 0
        v_norm = np.linalg.norm(new_vector)
        if v_norm == 0:
            return 0

        with self._lock:
            with self.conn:
                cursor = self.conn.execute("SELECT id, vector FROM memories WHERE id LIKE ? AND confidence > 0.0", (f"{entity_id_or_content}%",))
                rows = cursor.fetchall()
                if not rows:
                    cursor = self.conn.execute("SELECT id, vector FROM memories WHERE confidence > 0.0")
                    rows = cursor.fetchall()

                invalidated_ids = []
                for m_id, vec_blob in rows:
                    if m_id == entity_id_or_content or m_id in entity_id_or_content:
                        continue
                    old_vec = np.frombuffer(vec_blob, dtype=np.float32)
                    if old_vec.shape != new_vector.shape:
                        continue
                    denom = (v_norm * np.linalg.norm(old_vec)) + 1e-9
                    sim = float(np.dot(new_vector, old_vec) / denom)
                    if sim >= similarity_threshold:
                        invalidated_ids.append(m_id)

                for mid in invalidated_ids:
                    if prune_immediately:
                        self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                        self.conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
                    else:
                        self.conn.execute("UPDATE memories SET confidence = 0.0 WHERE id = ?", (mid,))
                    invalidated += 1

        return invalidated

    def overwrite_belief(
        self,
        content: str,
        vector: np.ndarray,
        confidence: float = 1.0,
        similarity_threshold: float = 0.88,
        prune_immediately: bool = False,
        **kwargs,
    ) -> Tuple[str, int]:
        """
        Active belief invalidation: Stores a newly verified belief in SQLite WAL while
        identifying and invalidating (or pruning) prior contradicting records with cosine similarity >= threshold.
        Returns (new_mem_id, num_invalidated).
        """
        invalidated = self.invalidate_conflicting_memories(
            entity_id_or_content=content,
            new_vector=vector,
            similarity_threshold=similarity_threshold,
            prune_immediately=prune_immediately,
        )

        mid = self.write_memory(
            content=content,
            vector=vector,
            confidence=confidence,
            auto_invalidate=False,
            **kwargs,
        )
        return mid, invalidated

    def enqueue_memory_flush(
        self,
        content: str,
        vector: np.ndarray,
        confidence: float = 0.9,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> str:
        """Asynchronous non-blocking queue enqueue for high-frequency plastic flushes."""
        mem_id = str(uuid.uuid4())
        ts = time.time()
        vec_blob = vector.astype(np.float32).tobytes()
        self._write_queue.put((content, vec_blob, confidence, ts, mem_id, tenant_id, session_id, None))
        return mem_id


    def compute_effective_confidence(self, confidence: float, last_accessed: float, current_time: float) -> float:
        """C_eff = C_t * exp(-lambda * delta_t / 3600)"""
        delta_hours = max(0.0, (current_time - last_accessed) / 3600.0)
        return confidence * math.exp(-self.lambda_decay * delta_hours)

    def reinforce(self, mem_id: str, current_time: Optional[float] = None) -> None:
        """C_{t+1} = min(1.0, C_t + 0.15), access_count += 1"""
        ts = current_time if current_time is not None else time.time()
        with self._lock:
            with self.conn:
                cur = self.conn.execute("SELECT confidence, access_count FROM memories WHERE id = ?", (mem_id,))
                row = cur.fetchone()
                if row:
                    c, cnt = row
                    new_c = min(1.0, c + self.boost_factor)
                    self.conn.execute(
                        "UPDATE memories SET confidence = ?, access_count = ?, last_accessed = ? WHERE id = ?",
                        (new_c, cnt + 1, ts, mem_id),
                    )

    def prune_dead_memories(self, current_time: Optional[float] = None) -> int:
        """Purge records where effective confidence < prune_threshold."""
        ts = current_time if current_time is not None else time.time()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, confidence, last_accessed FROM memories")
            rows = cur.fetchall()

            dead_ids = []
            for mid, conf, last_acc in rows:
                c_eff = self.compute_effective_confidence(conf, last_acc, ts)
                if c_eff < self.prune_threshold:
                    dead_ids.append(mid)

            if dead_ids:
                with self.conn:
                    self.conn.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in dead_ids])
                    self.conn.executemany("DELETE FROM memories_fts WHERE id = ?", [(i,) for i in dead_ids])

            return len(dead_ids)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        current_time: Optional[float] = None,
        tenant_id: str = "default_tenant",
        session_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Reciprocal Rank Fusion (RRF) combining BM25 FTS rank and vector cosine similarity."""
        ts = current_time if current_time is not None else time.time()
        rrf_scores: dict[str, float] = {}

        with self._lock:
            # 1. BM25 Search scoped by tenant
            fts_ids = []
            try:
                clean_query = "".join(c for c in query_text if c.isalnum() or c.isspace()).strip()
                if clean_query:
                    match_expr = " OR ".join(clean_query.split())
                    if session_id:
                        cur = self.conn.execute(
                            """
                            SELECT fts.id FROM memories_fts fts
                            JOIN memories m ON fts.id = m.id
                            WHERE memories_fts MATCH ? AND m.tenant_id = ? AND m.session_id = ?
                            ORDER BY rank LIMIT 50
                            """,
                            (match_expr, tenant_id, session_id),
                        )
                    else:
                        cur = self.conn.execute(
                            """
                            SELECT fts.id FROM memories_fts fts
                            JOIN memories m ON fts.id = m.id
                            WHERE memories_fts MATCH ? AND m.tenant_id = ?
                            ORDER BY rank LIMIT 50
                            """,
                            (match_expr, tenant_id),
                        )
                    fts_ids = [r[0] for r in cur.fetchall()]
            except Exception:
                fts_ids = []

            for rank, mid in enumerate(fts_ids):
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (60.0 + rank + 1)

            # 2. Vector search over active records scoped by tenant
            if session_id:
                cur = self.conn.execute(
                    "SELECT id, content, vector, confidence, access_count, last_accessed, tenant_id, session_id FROM memories WHERE tenant_id = ? AND session_id = ?",
                    (tenant_id, session_id),
                )
            else:
                cur = self.conn.execute(
                    "SELECT id, content, vector, confidence, access_count, last_accessed, tenant_id, session_id FROM memories WHERE tenant_id = ?",
                    (tenant_id,),
                )
            all_rows = cur.fetchall()
            if not all_rows:
                return []

            q_norm = np.linalg.norm(query_vector)
            q_vec = query_vector / (q_norm + 1e-9)

            vec_sims = []
            records_by_id = {}
            for mid, content, blob, conf, cnt, last_acc, t_id, s_id in all_rows:
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape != q_vec.shape:
                    continue
                v_norm = np.linalg.norm(v)
                cos_sim = float(np.dot(q_vec, v / (v_norm + 1e-9)))
                vec_sims.append((mid, cos_sim))
                records_by_id[mid] = MemoryRecord(
                    id=mid,
                    content=content,
                    confidence=conf,
                    access_count=cnt,
                    last_accessed=last_acc,
                    tenant_id=t_id,
                    session_id=s_id,
                )

            vec_sims.sort(key=lambda x: x[1], reverse=True)
            for rank, (mid, _) in enumerate(vec_sims[:50]):
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (60.0 + rank + 1)

            # Filter: enforce that candidate either has high cosine similarity (>= 0.70) or verified FTS match
            cos_sim_dict = dict(vec_sims)
            filtered_scores = {}
            for mid, rrf_score in rrf_scores.items():
                if mid in records_by_id:
                    sim = cos_sim_dict.get(mid, 0.0)
                    if sim >= 0.70 or mid in fts_ids:
                        rec = records_by_id[mid]
                        c_eff = self.compute_effective_confidence(rec.confidence, rec.last_accessed, ts)
                        filtered_scores[mid] = rrf_score * c_eff

            if not filtered_scores:
                return []

            # Sort by confidence-weighted score
            sorted_mids = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for mid, score in sorted_mids:
                if mid in records_by_id:
                    rec = records_by_id[mid]
                    self.reinforce(mid, ts)
                    results.append((rec, score))

            return results

    def retrieve_similar(
        self,
        query_text: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 1,
        threshold: float = 0.85,
        tenant_id: str = "default_tenant",
    ) -> List[Dict[str, Any]]:
        """Sub-millisecond retrieval of verified memories matching query."""
        if query_vector is None:
            query_vector = np.zeros(self.dim, dtype=np.float32)
        results = self.hybrid_search(query_text, query_vector, top_k=top_k, tenant_id=tenant_id)
        hits = []
        for rec, score in results:
            if rec.confidence >= threshold or score >= threshold:
                hits.append({
                    "id": rec.id,
                    "content": rec.content,
                    "confidence": rec.confidence,
                    "score": score,
                })
        return hits

    def search_semantic(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        tenant_id: str = "default_tenant",
        session_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Pure semantic vector search using cosine similarity over committed memory embeddings."""
        with self._lock:
            if session_id:
                cur = self.conn.execute(
                    "SELECT id, content, vector, confidence, access_count, last_accessed, tenant_id, session_id FROM memories WHERE tenant_id = ? AND session_id = ?",
                    (tenant_id, session_id),
                )
            else:
                cur = self.conn.execute(
                    "SELECT id, content, vector, confidence, access_count, last_accessed, tenant_id, session_id FROM memories WHERE tenant_id = ?",
                    (tenant_id,),
                )
            rows = cur.fetchall()
            if not rows:
                return []

            q_norm = np.linalg.norm(query_vector)
            q_vec = query_vector / (q_norm + 1e-9)

            scored = []
            for mid, content, blob, conf, cnt, last_acc, t_id, s_id in rows:
                v = np.frombuffer(blob, dtype=np.float32)
                if v.shape != q_vec.shape:
                    continue
                v_norm = np.linalg.norm(v)
                cos_sim = float(np.dot(q_vec, v / (v_norm + 1e-9))) if v_norm > 0 else 0.0
                rec = MemoryRecord(
                    id=mid,
                    content=content,
                    confidence=conf,
                    access_count=cnt,
                    last_accessed=last_acc,
                    tenant_id=t_id,
                    session_id=s_id,
                )
                scored.append((rec, cos_sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def store_memory(
        self,
        memory_id: str,
        content: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Any] = None,
        confidence: float = 0.9,
    ) -> str:
        """Store verified fact or observation into long-term memory."""
        if vector is None:
            vector = np.zeros(self.dim, dtype=np.float32)
        return self.write_memory(
            mem_id=memory_id,
            content=content,
            vector=vector,
            confidence=confidence,
        )

    def count_memories(self, tenant_id: Optional[str] = None) -> int:
        """Returns the total number of consolidated memory records."""
        with self._lock:
            try:
                if tenant_id:
                    cur = self.conn.execute("SELECT COUNT(*) FROM memories WHERE tenant_id = ?", (tenant_id,))
                else:
                    cur = self.conn.execute("SELECT COUNT(*) FROM memories")
                row = cur.fetchone()
                return row[0] if row else 0
            except Exception:
                return 0

    def close(self) -> None:
        """Cleanly stop background worker and close SQLite connection."""
        self._stop_event.set()
        if hasattr(self, "_writer_thread") and self._writer_thread.is_alive():
            self._write_queue.put(None)
            self._writer_thread.join(timeout=1.0)
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass


HybridConsolidationStore = ConsolidationStore



