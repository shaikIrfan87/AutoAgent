from dataclasses import dataclass
import math
import sqlite3
import time
from typing import Any, List, Optional


@dataclass
class EpistemicTarget:
    node_id: str
    target_type: str  # "decaying_confidence", "unverified_edge", "topological_hole"
    priority: float
    context: str


class GraphEpistemicScanner:
    """Scans consolidated memory and causal graphs for epistemic frontiers and decaying beliefs."""

    def __init__(
        self,
        db_conn: Optional[sqlite3.Connection] = None,
        causal_graph: Optional[Any] = None,
        decay_threshold: float = 0.40,
        lambda_decay: float = 0.05,
    ):
        self.conn = db_conn
        self.causal_graph = causal_graph
        self.decay_threshold = decay_threshold
        self.lambda_decay = lambda_decay

    def scan_stale_or_uncertain_nodes(
        self,
        top_k: int = 5,
        current_time: Optional[float] = None,
        tenant_id: str = "default_tenant",
    ) -> List[EpistemicTarget]:
        """Queries SQLite store for decaying memories and incomplete causal links."""
        targets: List[EpistemicTarget] = []
        now = current_time or time.time()

        # 1. Scan SQLite memories for decaying confidence
        if self.conn:
            try:
                cur = self.conn.execute(
                    """
                    SELECT id, content, confidence, last_accessed 
                    FROM memories 
                    WHERE tenant_id = ?
                    ORDER BY confidence ASC, last_accessed ASC
                    LIMIT 50
                    """,
                    (tenant_id,),
                )
                rows = cur.fetchall()
                for mem_id, content, base_conf, last_accessed in rows:
                    hours_elapsed = max(0.0, (now - last_accessed) / 3600.0)
                    eff_conf = base_conf * math.exp(-self.lambda_decay * hours_elapsed)
                    if eff_conf < self.decay_threshold or base_conf < self.decay_threshold:
                        priority = max(0.0, 1.0 - eff_conf)
                        targets.append(
                            EpistemicTarget(
                                node_id=mem_id,
                                target_type="decaying_confidence",
                                priority=priority,
                                context=content[:120],
                            )
                        )
            except Exception:
                pass

        # 2. Scan causal graph for topological holes and low-weight edges
        if self.causal_graph and hasattr(self.causal_graph, "graph"):
            graph = self.causal_graph.graph
            for node, deg in graph.degree():
                if deg <= 1:
                    targets.append(
                        EpistemicTarget(
                            node_id=str(node),
                            target_type="topological_hole",
                            priority=0.85,
                            context=f"Isolated or leaf node with degree {deg}",
                        )
                    )

            # Check unverified or low-weight edges
            if hasattr(graph, "edges"):
                for u, v, data in graph.edges(data=True):
                    weight = float(data.get("weight", 1.0))
                    is_valid = data.get("polarity", True)
                    if weight < 0.5 or not is_valid:
                        targets.append(
                            EpistemicTarget(
                                node_id=f"{u}->{v}",
                                target_type="unverified_edge",
                                priority=max(0.0, 1.0 - max(0.0, weight)),
                                context=f"Edge {u} {data.get('relation', 'causes')} {v} weight={weight:.2f}",
                            )
                        )

        # Sort by priority descending and slice top_k
        targets.sort(key=lambda t: t.priority, reverse=True)
        return targets[:top_k]
