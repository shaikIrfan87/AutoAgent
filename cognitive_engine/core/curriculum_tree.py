import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurriculumNode:
    id: str
    domain: str             # e.g., "mathematics", "computer_science", "physics"
    tier: int               # 0 = beginner, 1 = intermediate, 2 = advanced, 3 = research
    topic: str
    prerequisites: List[str]
    mastery_threshold: float = 0.85


class AutonomousCurriculumManager:
    """Manages the step-by-step developmental ladder from beginner to advanced."""

    # Universal developmental bootstrap tree
    DEFAULT_CURRICULUM = [
        # Mathematics Branch
        ("math_arithmetic", "mathematics", 0, "Basic Arithmetic & Number Theory", []),
        ("math_algebra", "mathematics", 1, "Linear Algebra & Symbolic Equations", ["math_arithmetic"]),
        ("math_calculus", "mathematics", 2, "Derivatives, Integrals & Optimization", ["math_algebra"]),
        ("math_diff_eq", "mathematics", 3, "Differential Equations & Dynamic Systems", ["math_calculus"]),

        # Computer Science Branch
        ("cs_primitives", "computer_science", 0, "Control Flow, Primitives & Types", []),
        ("cs_algorithms", "computer_science", 1, "Searching, Sorting & Graph Traversal", ["cs_primitives", "math_arithmetic"]),
        ("cs_structures", "computer_science", 2, "Trees, Memory Layouts & Hash Arrays", ["cs_algorithms"]),
        ("cs_systems", "computer_science", 3, "Distributed Architecture & IPC Concurrency", ["cs_structures"]),

        # Physics Branch
        ("phys_kinematics", "physics", 0, "Classical Mechanics & Motion Equations", ["math_algebra"]),
        ("phys_thermo", "physics", 1, "Thermodynamics & Statistical Mechanics", ["phys_kinematics", "math_calculus"]),
        ("phys_electromag", "physics", 2, "Electromagnetism & Maxwell Equations", ["phys_kinematics", "math_calculus"]),
        ("phys_quantum", "physics", 3, "Quantum Mechanics & Hilbert Vector Spaces", ["phys_electromag", "math_diff_eq"]),

        # Tier 4 Capstone Frontiers
        ("cs_compiler_codegen", "computer_science", 4, "Compiler IR Optimization & Native Codegen", ["cs_systems"]),
        ("math_lie_groups", "mathematics", 4, "Lie Groups, Algebras & Gauge Symmetries", ["math_diff_eq"]),
        ("phys_general_relativity", "physics", 4, "General Relativity & Spacetime Curvature Tensors", ["phys_quantum", "math_lie_groups"]),
    ]

    def __init__(self, db_path: str = "assets/cognitive_memory.db"):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_curriculum_db()

    def _init_curriculum_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS curriculum_state (
                    node_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    prerequisites TEXT,
                    mastery_score REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'LOCKED', -- LOCKED, ACTIVE, MASTERED
                    last_explored REAL DEFAULT 0.0
                )
            """)
            # Seed default tree and ensure any new tier nodes are registered
            for n_id, dom, tier, topic, prereqs in self.DEFAULT_CURRICULUM:
                conn.execute("""
                    INSERT OR IGNORE INTO curriculum_state (node_id, domain, tier, topic, prerequisites, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (n_id, dom, tier, topic, ",".join(prereqs), "ACTIVE" if not prereqs else "LOCKED"))
            conn.commit()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Returns all curriculum nodes with their current status and tier layout index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM curriculum_state ORDER BY tier ASC, node_id ASC").fetchall()
            nodes = []
            tier_counters: Dict[int, int] = {}
            for r in rows:
                d = dict(r)
                tier = d["tier"]
                idx = tier_counters.get(tier, 0)
                tier_counters[tier] = idx + 1
                d["index_in_tier"] = idx
                d["id"] = d["node_id"]
                d["prerequisites"] = [p.strip() for p in d["prerequisites"].split(",") if p.strip()]
                nodes.append(d)
            return nodes

    def get_autonomous_frontier(self) -> Optional[Dict[str, Any]]:
        """Finds the next unmastered node whose prerequisites are 100% satisfied."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            nodes = conn.execute("""
                SELECT * FROM curriculum_state 
                ORDER BY tier ASC, 
                         CASE WHEN last_explored IS NULL THEN 0 ELSE 1 END ASC,
                         last_explored ASC, 
                         node_id ASC
            """).fetchall()

            mastered_ids = {n["node_id"] for n in nodes if n["status"] == "MASTERED"}

            for node in nodes:
                if node["status"] == "MASTERED":
                    continue
                prereqs = [p.strip() for p in node["prerequisites"].split(",") if p.strip()]
                if all(p in mastered_ids for p in prereqs):
                    return dict(node)
        return None

    def update_mastery(self, node_id: str, success: bool):
        """Updates empirical progress and unlocks downstream frontier topics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT mastery_score, prerequisites FROM curriculum_state WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()
            if not row:
                return

            score = row[0]
            # Success grants +0.35; repeated exploration grants +0.12 to prevent infinite deadlock
            score = min(1.0, score + 0.35) if success else min(1.0, score + 0.12)
            new_status = "MASTERED" if score >= 0.70 else "ACTIVE"

            conn.execute("""
                UPDATE curriculum_state 
                SET mastery_score = ?, status = ?, last_explored = strftime('%s', 'now')
                WHERE node_id = ?
            """, (score, new_status, node_id))

            # If node was mastered, activate dependent downstream nodes whose prerequisites are now complete
            if new_status == "MASTERED":
                cursor = conn.execute("SELECT node_id, prerequisites FROM curriculum_state WHERE status = 'LOCKED'")
                locked_nodes = cursor.fetchall()
                # Recalculate mastered set
                cursor = conn.execute("SELECT node_id FROM curriculum_state WHERE status = 'MASTERED'")
                all_mastered = {r[0] for r in cursor.fetchall()}
                for l_id, l_prereqs in locked_nodes:
                    reqs = [p.strip() for p in l_prereqs.split(",") if p.strip()]
                    if all(req in all_mastered for req in reqs):
                        conn.execute("UPDATE curriculum_state SET status = 'ACTIVE' WHERE node_id = ?", (l_id,))

            conn.commit()
