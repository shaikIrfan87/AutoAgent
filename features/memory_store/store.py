import sqlite3
from typing import Optional, Dict, Any

class DualTierWalMemoryStore:
    """
    Invariant 5: Dual-Tier Memory Consolidation Store.
    Manages persistent memory via SQLite in WAL mode with confidence weighting.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_store()

    def _init_store(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL,
                last_accessed REAL
            );
        """)
        self.conn.commit()

    def commit_longterm(self, memory_id: str, content: str, confidence: float = 0.95):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO memories (id, content, confidence, last_accessed) VALUES (?, ?, ?, 1.0);",
            (memory_id, content, confidence)
        )
        self.conn.commit()

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, content, confidence, last_accessed FROM memories WHERE id = ?;", (memory_id,))
        row = cur.fetchone()
        if row:
            return {"id": row[0], "content": row[1], "confidence": row[2], "last_accessed": row[3]}
        return None

    def close(self):
        self.conn.close()
