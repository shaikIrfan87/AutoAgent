import sqlite3
from pathlib import Path


def sanitize_cognitive_memory(db_path: str = "assets/cognitive_memory.db") -> int:
    path = Path(db_path)
    if not path.exists():
        print(f"Database {db_path} not found.")
        return 0

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    # 1. Delete synthetic curiosity artifacts, diagnostic nodes, low-confidence web scrapes, and junk
    cursor.execute("""
        DELETE FROM memories 
        WHERE id LIKE 'emp_%' 
           OR id LIKE 'gap_%'
           OR id LIKE 'calc_%'
           OR id LIKE 'curiosity_%'
           OR id LIKE 'diagnostic_%'
           OR (id LIKE 'web_%' AND confidence < 0.7)
           OR content LIKE '%Verify empirical causal properties%'
           OR content LIKE '%node_start->node_exec%'
           OR content LIKE '%def solution(): return%'
           OR content LIKE '%Ground truth (Query)%'
           OR content LIKE '%Ground truth (%'
           OR content LIKE '%Look up query in%'
           OR content LIKE '%Ground truth (algorithm)%'
           OR content LIKE '%Empirical gap identified%'
           OR LENGTH(TRIM(content)) < 15
    """)
    deleted = cursor.rowcount

    # 1b. Deduplicate: keep only the newest record for identical content
    cursor.execute("""
        DELETE FROM memories
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (PARTITION BY content ORDER BY last_accessed DESC) as rn
                FROM memories
            ) WHERE rn = 1
        )
    """)
    deleted += cursor.rowcount

    conn.commit()

    # 2. Synchronize FTS5 virtual table if present, then commit before VACUUM
    try:
        cursor.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    remaining = cursor.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.execute("VACUUM;")
    conn.close()
    print(f"Sanitized {deleted} polluted memory records from {db_path}.")
    print(f"Remaining memories: {remaining}")
    return deleted


if __name__ == "__main__":
    sanitize_cognitive_memory()

