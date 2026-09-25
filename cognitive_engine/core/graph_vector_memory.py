"""
Graph-Vector Associative Working Memory & Lazy Causal Spreading Context.
Bridges SQLite WAL episodic storage with NetworkX causal graph topology.
Only reachable, causally relevant subgraphs with activation energy >= 0.65 are
retained in active working memory; unreferenced concepts remain cold on disk.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    from .consolidation import ConsolidationStore
    from .causal_graph import CausalSymbolicGraph
    from .types import MemoryRecord
except ImportError:
    from cognitive_engine.core.consolidation import ConsolidationStore
    from cognitive_engine.core.causal_graph import CausalSymbolicGraph
    from cognitive_engine.core.types import MemoryRecord


@dataclass
class ActiveGraphNode:
    node_id: str
    content: str
    activation_energy: float
    vector: np.ndarray
    confidence: float
    depth: int
    source_relation: Optional[str] = None


@dataclass
class WorkingMemoryContext:
    query: str
    active_nodes: Dict[str, ActiveGraphNode] = field(default_factory=dict)
    traversed_edges: List[Tuple[str, str, str, float]] = field(default_factory=list)
    anchors: List[str] = field(default_factory=list)
    pruned_count: int = 0


class GraphVectorWorkingMemory:
    """
    Two-Tier Hybrid Memory Architecture:
    - Tier 1: Cold Persistent Storage in SQLite WAL (FTS5 + 384D BLOB vectors).
    - Tier 2: Dynamic Ephemeral Working Context populated via Lazy Spreading Activation.
    """

    def __init__(
        self,
        consolidation: ConsolidationStore,
        causal_graph: CausalSymbolicGraph,
        max_hops: int = 2,
        activation_threshold: float = 0.65,
        decay_factor: float = 0.85,
        contradiction_threshold: float = 0.88,
    ):
        self.consolidation = consolidation
        self.causal_graph = causal_graph
        self.max_hops = max_hops
        self.activation_threshold = activation_threshold
        self.decay_factor = decay_factor
        self.contradiction_threshold = contradiction_threshold
        self.active_context: Optional[WorkingMemoryContext] = None

    def retrieve_working_context(
        self,
        query: str,
        query_vector: np.ndarray,
        tenant_id: str = "default_tenant",
        session_id: Optional[str] = None,
    ) -> WorkingMemoryContext:
        """
        1. Entry Jump: Top-3 anchor nodes via hybrid RRF search (BM25 + Cosine).
        2. Spreading Activation: Breadth-first graph traversal up to max_hops along structural & causal edges.
        3. Energy Gating: A_{next} = A_{curr} * EdgeWeight(u, v) * max(0.0, cos(q_vec, v_vec)).
        4. Context Clamping: Retain only nodes with A_{next} >= activation_threshold.
        """
        ctx = WorkingMemoryContext(query=query)

        # 1. Entry Jump via Hybrid RRF search
        anchors_raw = self.consolidation.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=3,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        if not anchors_raw:
            self.active_context = ctx
            return ctx

        # Normalize query vector for cosine checks
        q_norm = np.linalg.norm(query_vector)
        q_unit = query_vector / (q_norm + 1e-9)

        queue: List[Tuple[str, float, int]] = []
        visited: Set[str] = set()

        for rec, rrf_score in anchors_raw:
            # Anchor starts with base activation proportional to confidence and RRF rank
            node_vec = self._get_record_vector(rec.id)
            if node_vec is None:
                node_vec = query_vector
            cos_sim = float(np.dot(q_unit, node_vec / (np.linalg.norm(node_vec) + 1e-9)))
            init_energy = max(0.65, min(1.0, 0.5 + (0.5 * cos_sim)))

            ctx.anchors.append(rec.id)
            ctx.active_nodes[rec.id] = ActiveGraphNode(
                node_id=rec.id,
                content=rec.content,
                activation_energy=init_energy,
                vector=node_vec,
                confidence=rec.confidence,
                depth=0,
            )
            queue.append((rec.id, init_energy, 0))
            visited.add(rec.id)

        # 2. Spreading Activation
        causal_g = self.causal_graph.graph
        ALLOWED_RELATIONS = {"is_a", "causes", "part_of", "associated_with", "leads_to"}

        while queue:
            curr_id, curr_energy, depth = queue.pop(0)
            if depth >= self.max_hops:
                continue

            # Look up causal neighbors in causal graph using lowercase match
            curr_node_key = curr_id.lower()
            if curr_node_key not in causal_g:
                # Attempt to extract subject keywords
                tokens = curr_node_key.replace("_", " ").split()
                matched_key = None
                for t in tokens:
                    if t in causal_g:
                        matched_key = t
                        break
                if not matched_key:
                    continue
                curr_node_key = matched_key

            neighbors = list(causal_g.successors(curr_node_key))
            for nbr in neighbors:
                for edge_data in causal_g[curr_node_key][nbr].values():
                    rel = edge_data.get("relation", "associated_with")
                    if rel not in ALLOWED_RELATIONS:
                        continue

                    edge_w = float(edge_data.get("weight", 0.8))
                    nbr_content = str(nbr)
                    nbr_vec = self._get_concept_vector(nbr_content)
                    cos_sim = float(np.dot(q_unit, nbr_vec / (np.linalg.norm(nbr_vec) + 1e-9)))

                    # 3. Energy Decay Gating
                    a_next = curr_energy * edge_w * max(0.0, cos_sim)

                    # 4. Context Clamping (A_{next} >= 0.65)
                    if a_next >= self.activation_threshold:
                        ctx.traversed_edges.append((curr_node_key, nbr, rel, a_next))
                        if nbr not in visited:
                            visited.add(nbr)
                            ctx.active_nodes[nbr] = ActiveGraphNode(
                                node_id=nbr,
                                content=nbr_content,
                                activation_energy=a_next,
                                vector=nbr_vec,
                                confidence=0.8,
                                depth=depth + 1,
                                source_relation=rel,
                            )
                            queue.append((nbr, a_next, depth + 1))
                        elif nbr in ctx.active_nodes and a_next > ctx.active_nodes[nbr].activation_energy:
                            ctx.active_nodes[nbr].activation_energy = a_next

        self.active_context = ctx
        return ctx

    def update_with_anti_hebbian_pruning(
        self,
        new_fact: str,
        new_vector: np.ndarray,
        confidence: float = 0.99,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> Tuple[str, int, int]:
        """
        Contradiction & Anti-Hebbian Pruning:
        - If verified belief has cosine similarity >= 0.88 with prior memory:
          1. Prune or zero out contradicted record in SQLite WAL.
          2. Decrement associated causal edge weights by 0.20: W(u, v) <- W(u, v) - 0.20.
        Returns: (new_mem_id, num_invalidated_memories, num_penalized_edges).
        """
        # Step 1: Active Contradiction Pruning in SQLite WAL
        new_mem_id, num_invalidated = self.consolidation.overwrite_belief(
            content=new_fact,
            vector=new_vector,
            confidence=confidence,
            similarity_threshold=self.contradiction_threshold,
            prune_immediately=False,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        # Step 2: Anti-Hebbian Edge Weight Penalization
        penalized_edges = 0
        fact_lower = new_fact.lower()
        causal_g = self.causal_graph.graph

        for u, v, data in list(causal_g.edges(data=True)):
            u_str, v_str = str(u).lower(), str(v).lower()
            if u_str in fact_lower or v_str in fact_lower:
                old_w = float(data.get("weight", 0.5))
                new_w = round(max(0.0, old_w - 0.20), 4)
                data["weight"] = new_w
                penalized_edges += 1

        return new_mem_id, num_invalidated, penalized_edges

    def _get_record_vector(self, mem_id: str) -> Optional[np.ndarray]:
        """Fetches vector blob from SQLite WAL for a given memory id."""
        try:
            cur = self.consolidation.conn.execute("SELECT vector FROM memories WHERE id = ?", (mem_id,))
            row = cur.fetchone()
            if row and row[0]:
                return np.frombuffer(row[0], dtype=np.float32).copy()
        except Exception:
            pass
        return None

    def _get_concept_vector(self, concept_text: str) -> np.ndarray:
        """Retrieves or creates embedding vector for a concept node."""
        if hasattr(self.consolidation, "dim"):
            dim = self.consolidation.dim
        else:
            dim = 384
        try:
            from .saliency import _semantic_topological_embed
            return _semantic_topological_embed(concept_text, dim=dim)
        except Exception:
            return np.zeros(dim, dtype=np.float32)
