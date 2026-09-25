"""
Endurance Soak Runner — monitors RAM, handle leaks, Ebbinghaus decay health,
curriculum frontier advancement, and Ground truth re-pollution.
"""
import os, time, math, sqlite3
import psutil
from pathlib import Path
from cognitive_engine.agent.orchestrator import CognitiveEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT_DIR / "assets" / "cognitive_memory.db")
CYCLES = 100
REPORT_EVERY = 25

proc = psutil.Process(os.getpid())
engine = CognitiveEngine(db_path=DB_PATH)
start_mem = proc.memory_info().rss / (1024 * 1024)

print(f"Soak start — Baseline RAM: {start_mem:.2f} MB\n")

def _count_pollution():
    try:
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE 'Ground truth (%'"
        ).fetchone()[0]
        conn.close()
        return n
    except Exception:
        return -1

def _decay_health():
    """Returns fraction of memories with C_eff >= 0.20 (healthy)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT confidence, last_accessed FROM memories").fetchall()
        conn.close()
        now = time.time()
        if not rows:
            return 1.0, 0
        healthy = sum(
            1 for conf, la in rows
            if float(conf) * math.exp(-0.05 * (now - float(la)) / 3600.0) >= 0.20
        )
        return healthy / len(rows), len(rows)
    except Exception:
        return -1.0, 0

for i in range(1, CYCLES + 1):
    _ = engine.interact(f"Calculate kinetic energy of {1000 + i}kg car moving at {10 + (i % 5)}m/s")

    if i % REPORT_EVERY == 0:
        curr_mem = proc.memory_info().rss / (1024 * 1024)
        num_fds = proc.num_handles() if hasattr(proc, "num_handles") else proc.num_fds()
        health_ratio, total_mem = _decay_health()
        pollution = _count_pollution()
        print(
            f"Cycle [{i:>3}/{CYCLES}] | RAM: {curr_mem:.1f} MB (+{curr_mem - start_mem:.1f}) "
            f"| Handles: {num_fds} | Memories: {total_mem} | "
            f"Decay-healthy: {health_ratio:.0%} | Pollution: {pollution}"
        )
        if pollution > 0:
            print(f"  [WARNING] Ground truth re-pollution detected ({pollution} records) -- run purge_memory_pollution.py")

engine.sandbox.close()
engine.consolidation.close()
print("\nEndurance soak complete.")

