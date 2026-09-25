import sqlite3
from typing import Any, Optional
import numpy as np
from .schemas import CritiquePayload, GraphTrajectory


class GraphCreditAssigner:
    def __init__(
        self,
        graph_memory: Any,
        decay_rate: float = 0.05,
        homeostatic_decay: float = 0.01,
        db_path: Optional[str] = None,
    ):
        self.graph = graph_memory
        self.decay_rate = decay_rate
        self.homeostatic_decay = homeostatic_decay
        self.db_path = db_path
        if self.db_path:
            self._init_db()
            self.load_state()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dynamic_edges (
                    tenant_id TEXT DEFAULT 'default_tenant',
                    source TEXT,
                    target TEXT,
                    weight REAL CHECK(weight >= -1.0 AND weight <= 1.0),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, source, target)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_thresholds (
                    tenant_id TEXT DEFAULT 'default_tenant',
                    node_id TEXT,
                    threshold REAL,
                    base_threshold REAL,
                    PRIMARY KEY (tenant_id, node_id)
                )
                """
            )
            # Schema migration for existing tables without tenant_id
            cur = conn.execute("PRAGMA table_info(dynamic_edges);")
            cols = [row[1] for row in cur.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE dynamic_edges ADD COLUMN tenant_id TEXT DEFAULT 'default_tenant';")
            cur = conn.execute("PRAGMA table_info(node_thresholds);")
            cols = [row[1] for row in cur.fetchall()]
            if "tenant_id" not in cols:
                conn.execute("ALTER TABLE node_thresholds ADD COLUMN tenant_id TEXT DEFAULT 'default_tenant';")
            conn.commit()

    def load_state(self, tenant_id: str = "default_tenant") -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source, target, weight FROM dynamic_edges WHERE tenant_id = ?", (tenant_id,))
            for u, v, w in cursor.fetchall():
                self.graph.set_edge_weight(u, v, float(w))

            cursor.execute("SELECT node_id, threshold, base_threshold FROM node_thresholds WHERE tenant_id = ?", (tenant_id,))
            nodes = getattr(self.graph, "nodes", {})
            for node_id, threshold, base_threshold in cursor.fetchall():
                if isinstance(nodes, dict):
                    if node_id not in nodes:
                        nodes[node_id] = {"activation_threshold": float(threshold), "base_threshold": float(base_threshold)}
                    else:
                        node = nodes[node_id]
                        if isinstance(node, dict):
                            node["activation_threshold"] = float(threshold)
                            node["base_threshold"] = float(base_threshold)
                        else:
                            node.activation_threshold = float(threshold)
                            if hasattr(node, "base_threshold"):
                                node.base_threshold = float(base_threshold)

    def save_state(self, tenant_id: str = "default_tenant") -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            edges = getattr(self.graph, "edges", {})
            if isinstance(edges, dict):
                edge_records = [
                    (tenant_id, k[0], k[1], float(w))
                    for k, w in edges.items()
                    if isinstance(k, tuple) and len(k) == 2
                ]
                if edge_records:
                    conn.executemany(
                        """
                        INSERT INTO dynamic_edges (tenant_id, source, target, weight, last_updated)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(tenant_id, source, target) DO UPDATE SET
                            weight = excluded.weight,
                            last_updated = CURRENT_TIMESTAMP
                        """,
                        edge_records,
                    )

            nodes = getattr(self.graph, "nodes", {})
            if isinstance(nodes, dict):
                node_records = []
                for node_id, node in nodes.items():
                    thresh = (
                        node.get("activation_threshold", 0.5)
                        if isinstance(node, dict)
                        else getattr(node, "activation_threshold", 0.5)
                    )
                    base = (
                        node.get("base_threshold", 0.5)
                        if isinstance(node, dict)
                        else getattr(node, "base_threshold", 0.5)
                    )
                    node_records.append((tenant_id, str(node_id), float(thresh), float(base)))
                if node_records:
                    conn.executemany(
                        """
                        INSERT INTO node_thresholds (tenant_id, node_id, threshold, base_threshold)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tenant_id, node_id) DO UPDATE SET
                            threshold = excluded.threshold,
                            base_threshold = excluded.base_threshold
                        """,
                        node_records,
                    )
            conn.commit()

    def apply_feedback(
        self,
        trajectory: GraphTrajectory,
        critique: CritiquePayload,
        passed: bool,
        tenant_id: str = "default_tenant",
    ) -> None:
        edge_delta = (
            0.1 * critique.aggregate_score
            if passed
            else -0.2 * (1.0 - critique.aggregate_score)
        )

        for u, v in trajectory.traversed_edges:
            current_weight = self.graph.get_edge_weight(u, v)
            # Bound edge weights within [-1.0, 1.0]
            new_weight = float(np.clip(current_weight + edge_delta, -1.0, 1.0))
            self.graph.set_edge_weight(u, v, new_weight)

        # Dynamic Threshold Adaptation
        if not passed:
            for node_id in trajectory.active_node_ids:
                node = self.graph.nodes[node_id]
                if hasattr(node, "activation_threshold"):
                    node.activation_threshold += self.decay_rate
                elif isinstance(node, dict):
                    node["activation_threshold"] = (
                        node.get("activation_threshold", 0.5) + self.decay_rate
                    )

        # Homeostatic recovery: passive relaxation of non-active nodes towards baseline
        nodes = getattr(self.graph, "nodes", {})
        if isinstance(nodes, dict):
            for node_id, node in nodes.items():
                if node_id not in trajectory.active_node_ids:
                    base = getattr(node, "base_threshold", 0.5) if not isinstance(node, dict) else node.get("base_threshold", 0.5)
                    if hasattr(node, "activation_threshold") and node.activation_threshold > base:
                        node.activation_threshold = max(base, node.activation_threshold - self.homeostatic_decay)
                    elif isinstance(node, dict) and node.get("activation_threshold", 0.5) > base:
                        node["activation_threshold"] = max(base, node["activation_threshold"] - self.homeostatic_decay)

        self.save_state(tenant_id=tenant_id)

    def prune_near_zero_edges(self, epsilon: float = 0.01, tenant_id: str = "default_tenant") -> int:
        """Drop near-zero transitions to prevent sparse graph memory bloat."""
        edges = getattr(self.graph, "edges", {})
        if isinstance(edges, dict):
            pruned = [k for k, w in edges.items() if abs(w) < epsilon]
            for k in pruned:
                del edges[k]
            if pruned and self.db_path:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        "DELETE FROM dynamic_edges WHERE tenant_id = ? AND source = ? AND target = ?",
                        [(tenant_id, p[0], p[1]) for p in pruned if isinstance(p, tuple) and len(p) == 2],
                    )
                    conn.commit()
            return len(pruned)
        return 0

    def get_outgoing_edges(self, source: str, tenant_id: str = "default_tenant") -> dict[str, float]:
        """Retrieve outgoing edge weights scoped by tenant to prevent cross-tenant leak."""
        if not self.db_path:
            edges = getattr(self.graph, "edges", {})
            return {k[1]: float(w) for k, w in edges.items() if isinstance(k, tuple) and len(k) == 2 and k[0] == source}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT target, weight FROM dynamic_edges WHERE tenant_id = ? AND source = ?",
                (tenant_id, source),
            )
            return {target: float(weight) for target, weight in cursor.fetchall()}


