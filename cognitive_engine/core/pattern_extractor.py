"""
Universal Structural Pattern Extractor & Cross-Domain ADAG Representation.
Parses arbitrary cross-domain observations and constraint pairs (x_in -> y_out)
into Attributed Directed Acyclic Graphs (ADAG) and discrete invariant vectors,
decoupling cognitive induction from foundation models and natural language text.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class StructuralDomain(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    GRID = "grid"
    SEQUENCE = "sequence"
    GRAPH = "graph"


@dataclass
class ScalarInvariants:
    domain: StructuralDomain = StructuralDomain.SCALAR
    value: float = 0.0
    is_integer: bool = False
    is_positive: bool = False
    is_zero: bool = False
    is_even: bool = False
    is_prime: bool = False

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.is_integer),
            float(self.is_positive),
            float(self.is_zero),
            float(self.is_even),
            float(self.is_prime),
            math.tanh(self.value / 100.0),
        ], dtype=np.float32)


@dataclass
class VectorInvariants:
    domain: StructuralDomain = StructuralDomain.VECTOR
    length: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    is_monotonic_increasing: bool = False
    is_monotonic_decreasing: bool = False
    has_duplicates: bool = False

    def to_vector(self) -> np.ndarray:
        return np.array([
            min(self.length / 100.0, 1.0),
            math.tanh(self.mean / 50.0),
            math.tanh(self.std / 50.0),
            math.tanh(self.min_val / 50.0),
            math.tanh(self.max_val / 50.0),
            float(self.is_monotonic_increasing),
            float(self.is_monotonic_decreasing),
            float(self.has_duplicates),
        ], dtype=np.float32)


@dataclass
class GridInvariants:
    domain: StructuralDomain = StructuralDomain.GRID
    height: int = 0
    width: int = 0
    aspect_ratio: float = 1.0
    is_square: bool = False
    unique_symbols: int = 0
    symmetry_horizontal: bool = False
    symmetry_vertical: bool = False
    symmetry_diagonal: bool = False
    connected_components: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([
            min(self.height / 30.0, 1.0),
            min(self.width / 30.0, 1.0),
            min(self.aspect_ratio / 5.0, 1.0),
            float(self.is_square),
            min(self.unique_symbols / 10.0, 1.0),
            float(self.symmetry_horizontal),
            float(self.symmetry_vertical),
            float(self.symmetry_diagonal),
            min(self.connected_components / 20.0, 1.0),
        ], dtype=np.float32)


@dataclass
class SequenceInvariants:
    domain: StructuralDomain = StructuralDomain.SEQUENCE
    length: int = 0
    unique_tokens: int = 0
    is_palindrome: bool = False
    has_periodicity: bool = False
    period_length: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([
            min(self.length / 100.0, 1.0),
            min(self.unique_tokens / 50.0, 1.0),
            float(self.is_palindrome),
            float(self.has_periodicity),
            min(self.period_length / 20.0, 1.0),
        ], dtype=np.float32)


@dataclass
class GraphInvariants:
    domain: StructuralDomain = StructuralDomain.GRAPH
    num_nodes: int = 0
    num_edges: int = 0
    is_directed: bool = False
    min_degree: int = 0
    max_degree: int = 0
    avg_degree: float = 0.0
    has_cycles: bool = False

    def to_vector(self) -> np.ndarray:
        return np.array([
            min(self.num_nodes / 50.0, 1.0),
            min(self.num_edges / 100.0, 1.0),
            float(self.is_directed),
            min(self.min_degree / 10.0, 1.0),
            min(self.max_degree / 10.0, 1.0),
            min(self.avg_degree / 10.0, 1.0),
            float(self.has_cycles),
        ], dtype=np.float32)


@dataclass
class ADAGNode:
    node_id: str
    domain: StructuralDomain
    attributes: Dict[str, Any] = field(default_factory=dict)
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))


@dataclass
class ADAGEdge:
    source_id: str
    target_id: str
    relation: str  # e.g., "transforms_to", "contains", "subgraph_of"
    weight: float = 1.0


@dataclass
class AttributedDAG:
    """Attributed Directed Acyclic Graph encoding cross-domain structural relations."""
    nodes: Dict[str, ADAGNode] = field(default_factory=dict)
    edges: List[ADAGEdge] = field(default_factory=list)

    def add_node(self, node: ADAGNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        self.edges.append(ADAGEdge(source_id=source_id, target_id=target_id, relation=relation, weight=weight))


@dataclass
class StructuralSpec:
    """Canonical specification extracted from input-output problem examples."""
    input_domain: StructuralDomain
    output_domain: StructuralDomain
    is_isomorphic_domain: bool
    input_invariants: Union[ScalarInvariants, VectorInvariants, GridInvariants, SequenceInvariants, GraphInvariants]
    output_invariants: Union[ScalarInvariants, VectorInvariants, GridInvariants, SequenceInvariants, GraphInvariants]
    adag: AttributedDAG
    state_vector: np.ndarray  # Fixed-dimension 32-dim feature vector for MCTS/RLCD conditioning


def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class UniversalPatternExtractor:
    """
    Zero-LLM Universal Pattern Extractor.
    Translates raw cross-domain data into attributed graph invariants and fixed feature vectors.
    """

    @classmethod
    def classify_domain(cls, obj: Any) -> StructuralDomain:
        """Determines the canonical structural domain of an arbitrary Python entity."""
        if isinstance(obj, (int, float, bool, np.integer, np.floating)):
            return StructuralDomain.SCALAR

        # 2D Grid / Matrix
        if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], (list, tuple)):
            # Check rectangular consistency
            if all(isinstance(row, (list, tuple)) and len(row) == len(obj[0]) for row in obj):
                return StructuralDomain.GRID

        if isinstance(obj, np.ndarray):
            if obj.ndim == 2:
                return StructuralDomain.GRID
            elif obj.ndim == 1:
                return StructuralDomain.VECTOR
            elif obj.ndim == 0:
                return StructuralDomain.SCALAR

        # Graph (adjacency dictionary or edge list)
        if isinstance(obj, dict):
            # Check if dict represents an adjacency list (node -> list of neighbors)
            if all(isinstance(v, (list, tuple, set)) for v in obj.values()):
                return StructuralDomain.GRAPH

        # 1D Vector (homogeneous numbers) vs 1D Sequence (general discrete tokens/strings)
        if isinstance(obj, (list, tuple)):
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in obj):
                return StructuralDomain.VECTOR
            return StructuralDomain.SEQUENCE

        if isinstance(obj, str):
            return StructuralDomain.SEQUENCE

        return StructuralDomain.SCALAR

    @classmethod
    def extract_scalar_invariants(cls, val: Any) -> ScalarInvariants:
        f_val = float(val)
        is_int = float(val).is_integer() if isinstance(val, float) else isinstance(val, (int, np.integer))
        i_val = int(f_val) if is_int else 0
        return ScalarInvariants(
            value=f_val,
            is_integer=is_int,
            is_positive=f_val > 0,
            is_zero=f_val == 0,
            is_even=is_int and (i_val % 2 == 0),
            is_prime=is_int and _is_prime(i_val),
        )

    @classmethod
    def extract_vector_invariants(cls, vec: Any) -> VectorInvariants:
        arr = np.array(vec, dtype=np.float64) if len(vec) > 0 else np.array([], dtype=np.float64)
        n = len(arr)
        if n == 0:
            return VectorInvariants(length=0)

        is_inc = bool(np.all(arr[1:] >= arr[:-1])) if n > 1 else True
        is_dec = bool(np.all(arr[1:] <= arr[:-1])) if n > 1 else True
        has_dup = len(arr) != len(set(arr.tolist()))

        return VectorInvariants(
            length=n,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            is_monotonic_increasing=is_inc,
            is_monotonic_decreasing=is_dec,
            has_duplicates=has_dup,
        )

    @classmethod
    def extract_grid_invariants(cls, grid: Any) -> GridInvariants:
        arr = np.array(grid)
        if arr.ndim != 2:
            return GridInvariants()
        h, w = arr.shape
        if h == 0 or w == 0:
            return GridInvariants(height=h, width=w)

        sym_h = bool(np.array_equal(arr, np.flipud(arr)))
        sym_v = bool(np.array_equal(arr, np.fliplr(arr)))
        sym_d = bool(np.array_equal(arr, arr.T)) if h == w else False

        # Connected components (4-connected on foreground > 0)
        comps = 0
        visited = set()
        for r in range(h):
            for c in range(w):
                if arr[r, c] != 0 and (r, c) not in visited:
                    comps += 1
                    queue = [(r, c)]
                    visited.add((r, c))
                    color = arr[r, c]
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if arr[nr, nc] == color and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))

        return GridInvariants(
            height=h,
            width=w,
            aspect_ratio=round(w / h, 3) if h > 0 else 1.0,
            is_square=(h == w),
            unique_symbols=len(np.unique(arr)),
            symmetry_horizontal=sym_h,
            symmetry_vertical=sym_v,
            symmetry_diagonal=sym_d,
            connected_components=comps,
        )

    @classmethod
    def extract_sequence_invariants(cls, seq: Any) -> SequenceInvariants:
        elements = list(seq)
        n = len(elements)
        if n == 0:
            return SequenceInvariants()

        is_pal = (elements == elements[::-1])

        # Periodicity detector (checks prefixes up to n // 2)
        has_period = False
        p_len = 0
        for p in range(1, (n // 2) + 1):
            pattern = elements[:p]
            repeats = n // p
            remainder = n % p
            if elements == (pattern * repeats + pattern[:remainder]):
                has_period = True
                p_len = p
                break

        return SequenceInvariants(
            length=n,
            unique_tokens=len(set(elements)),
            is_palindrome=is_pal,
            has_periodicity=has_period,
            period_length=p_len,
        )

    @classmethod
    def extract_graph_invariants(cls, graph: Dict[Any, List[Any]]) -> GraphInvariants:
        num_nodes = len(graph)
        num_edges = sum(len(neighbors) for neighbors in graph.values())
        degrees = [len(neighbors) for neighbors in graph.values()] if graph else [0]
        min_deg = min(degrees) if degrees else 0
        max_deg = max(degrees) if degrees else 0
        avg_deg = sum(degrees) / num_nodes if num_nodes > 0 else 0.0

        # Cycle check using DFS
        visited: Set[Any] = set()
        rec_stack: Set[Any] = set()
        has_cycles = False

        def _dfs(u: Any) -> bool:
            visited.add(u)
            rec_stack.add(u)
            for v in graph.get(u, []):
                if v not in visited:
                    if _dfs(v):
                        return True
                elif v in rec_stack:
                    return True
            rec_stack.remove(u)
            return False

        for node in graph:
            if node not in visited:
                if _dfs(node):
                    has_cycles = True
                    break

        return GraphInvariants(
            num_nodes=num_nodes,
            num_edges=num_edges,
            is_directed=True,
            min_degree=min_deg,
            max_degree=max_deg,
            avg_degree=round(avg_deg, 2),
            has_cycles=has_cycles,
        )

    @classmethod
    def extract_invariants(cls, obj: Any) -> Tuple[StructuralDomain, Any]:
        """Extracts the domain and corresponding invariant dataclass for an object."""
        domain = cls.classify_domain(obj)
        if domain == StructuralDomain.SCALAR:
            return domain, cls.extract_scalar_invariants(obj)
        elif domain == StructuralDomain.VECTOR:
            return domain, cls.extract_vector_invariants(obj)
        elif domain == StructuralDomain.GRID:
            return domain, cls.extract_grid_invariants(obj)
        elif domain == StructuralDomain.SEQUENCE:
            return domain, cls.extract_sequence_invariants(obj)
        elif domain == StructuralDomain.GRAPH:
            return domain, cls.extract_graph_invariants(obj)
        return domain, cls.extract_scalar_invariants(obj)

    @classmethod
    def build_adag(cls, x_in: Any, y_out: Any) -> AttributedDAG:
        """Constructs an Attributed Directed Acyclic Graph connecting input and output representations."""
        adag = AttributedDAG()
        in_domain, in_inv = cls.extract_invariants(x_in)
        out_domain, out_inv = cls.extract_invariants(y_out)

        in_vec = in_inv.to_vector()
        out_vec = out_inv.to_vector()

        node_in = ADAGNode(
            node_id="input_0",
            domain=in_domain,
            attributes={"invariants": in_inv},
            feature_vector=in_vec,
        )
        node_out = ADAGNode(
            node_id="output_0",
            domain=out_domain,
            attributes={"invariants": out_inv},
            feature_vector=out_vec,
        )

        adag.add_node(node_in)
        adag.add_node(node_out)
        adag.add_edge("input_0", "output_0", relation="transforms_to", weight=1.0)
        return adag

    @classmethod
    def extract_spec(cls, io_pairs: List[Tuple[Any, Any]]) -> StructuralSpec:
        """
        Formulates a formal StructuralSpec across multiple (x_in, y_out) observations.
        Produces a 32-dimensional invariant state vector s_feat to condition downstream MCTS/RLCD.
        """
        if not io_pairs:
            raise ValueError("Cannot extract structural spec from empty observation pairs.")

        first_in, first_out = io_pairs[0]
        in_domain, in_inv = cls.extract_invariants(first_in)
        out_domain, out_inv = cls.extract_invariants(first_out)

        adag = cls.build_adag(first_in, first_out)

        # Assemble fixed-dimension 32-dim state vector
        s_feat = np.zeros(32, dtype=np.float32)

        # 0-4: One-hot input domain
        domain_order = [StructuralDomain.SCALAR, StructuralDomain.VECTOR, StructuralDomain.GRID, StructuralDomain.SEQUENCE, StructuralDomain.GRAPH]
        s_feat[domain_order.index(in_domain)] = 1.0

        # 5-9: One-hot output domain
        s_feat[5 + domain_order.index(out_domain)] = 1.0

        # 10: Domain isomorphism flag
        s_feat[10] = 1.0 if in_domain == out_domain else 0.0

        # 11-20: Input invariant slice
        in_vec = in_inv.to_vector()
        slice_len_in = min(len(in_vec), 10)
        s_feat[11: 11 + slice_len_in] = in_vec[:slice_len_in]

        # 21-30: Output invariant slice
        out_vec = out_inv.to_vector()
        slice_len_out = min(len(out_vec), 10)
        s_feat[21: 21 + slice_len_out] = out_vec[:slice_len_out]

        # 31: Number of training exemplars normalized
        s_feat[31] = min(len(io_pairs) / 10.0, 1.0)

        return StructuralSpec(
            input_domain=in_domain,
            output_domain=out_domain,
            is_isomorphic_domain=(in_domain == out_domain),
            input_invariants=in_inv,
            output_invariants=out_inv,
            adag=adag,
            state_vector=s_feat,
        )


class InvariantFunctor(str, Enum):
    MONOTONIC_ORDERING = "monotonic_ordering"
    PARTITION_EQUIVALENCE = "partition_equivalence"
    SELECTION_FILTER = "selection_filter"
    ISOMORPHIC_MAPPING = "isomorphic_mapping"
    AXIS_REVERSAL = "axis_reversal"
    DIMENSIONAL_PROJECTION = "dimensional_projection"


@dataclass
class AbstractDomainSchema:
    schema_id: str
    functor: InvariantFunctor
    source_domain: StructuralDomain
    invariant_signature: np.ndarray
    canonical_operator: str


class DomainInvariantProjector:
    """
    Cross-Domain Conceptual Transfer & Invariant Projection:
    Lifts learned algorithmic solutions into domain-independent functors,
    enabling transfer of abstract patterns (e.g., partitioning, monotonicity, filtering)
    across disparate problem domains (e.g., spatial grids -> task scheduling -> inventory lists).
    """

    def __init__(self):
        self.schema_registry: Dict[str, AbstractDomainSchema] = {}
        self._init_core_schemas()

    def _init_core_schemas(self):
        # Monotonic ordering schema (sorting / prioritization)
        self.register_schema(
            schema_id="schema_monotonic_order",
            functor=InvariantFunctor.MONOTONIC_ORDERING,
            source_domain=StructuralDomain.VECTOR,
            canonical_operator="lambda seq: sorted(seq, key=lambda x: x if not isinstance(x, dict) else list(x.values())[0])",
        )
        # Partition equivalence schema (clustering / grouping by key)
        self.register_schema(
            schema_id="schema_partition_eq",
            functor=InvariantFunctor.PARTITION_EQUIVALENCE,
            source_domain=StructuralDomain.GRID,
            canonical_operator="lambda seq, key_fn=None: {k: [item for item in seq if (key_fn(item) if key_fn else item) == k] for k in set(key_fn(x) if key_fn else x for x in seq)}",
        )
        # Selection filter schema
        self.register_schema(
            schema_id="schema_selection_filter",
            functor=InvariantFunctor.SELECTION_FILTER,
            source_domain=StructuralDomain.SEQUENCE,
            canonical_operator="lambda seq, pred=bool: [x for x in seq if pred(x)]",
        )
        # Axis reversal schema (reflection / LIFO)
        self.register_schema(
            schema_id="schema_axis_reversal",
            functor=InvariantFunctor.AXIS_REVERSAL,
            source_domain=StructuralDomain.GRID,
            canonical_operator="lambda seq: seq[::-1]",
        )

    def register_schema(
        self,
        schema_id: str,
        functor: InvariantFunctor,
        source_domain: StructuralDomain,
        canonical_operator: str,
        invariant_sig: Optional[np.ndarray] = None,
    ):
        sig = invariant_sig if invariant_sig is not None else np.zeros(32, dtype=np.float32)
        self.schema_registry[schema_id] = AbstractDomainSchema(
            schema_id=schema_id,
            functor=functor,
            source_domain=source_domain,
            invariant_signature=sig,
            canonical_operator=canonical_operator,
        )

    def infer_functor_with_params(self, io_pairs: List[Tuple[Any, Any]]) -> Tuple[InvariantFunctor, Optional[Any]]:
        if not io_pairs:
            return InvariantFunctor.ISOMORPHIC_MAPPING, None
        first_in, first_out = io_pairs[0]
        if isinstance(first_in, (list, tuple)) and isinstance(first_out, (list, tuple)):
            if len(first_in) == len(first_out):
                try:
                    if list(first_out) == sorted(first_in):
                        return InvariantFunctor.MONOTONIC_ORDERING, None
                    if list(first_out) == list(reversed(first_in)):
                        return InvariantFunctor.AXIS_REVERSAL, None
                    if first_in and isinstance(first_in[0], (tuple, list)):
                        n_dims = len(first_in[0])
                        for dim_idx in range(n_dims):
                            if list(first_out) == sorted(first_in, key=lambda x: x[dim_idx]):
                                return InvariantFunctor.MONOTONIC_ORDERING, dim_idx
                except Exception:
                    pass
                return InvariantFunctor.ISOMORPHIC_MAPPING, None
            elif len(first_out) < len(first_in):
                return InvariantFunctor.SELECTION_FILTER, None
        elif isinstance(first_out, dict):
            return InvariantFunctor.PARTITION_EQUIVALENCE, None
        return InvariantFunctor.ISOMORPHIC_MAPPING, None

    def infer_functor(self, io_pairs: List[Tuple[Any, Any]]) -> InvariantFunctor:
        """Infers the abstract invariant functor governing the transformation."""
        functor, _ = self.infer_functor_with_params(io_pairs)
        return functor

    def project_to_target_domain(
        self,
        target_io_pairs: List[Tuple[Any, Any]],
        target_domain: StructuralDomain,
    ) -> Optional[Dict[str, Any]]:
        """
        Projects an abstract schema onto a target domain problem.
        Returns matching schema, projected Python code, and validation status.
        """
        functor, param = self.infer_functor_with_params(target_io_pairs)
        matching_schema = None
        for schema in self.schema_registry.values():
            if schema.functor == functor:
                matching_schema = schema
                break

        if not matching_schema:
            return None

        if functor == InvariantFunctor.MONOTONIC_ORDERING and param is not None:
            op_code = f"lambda seq: sorted(seq, key=lambda x: x[{param}])"
        else:
            op_code = matching_schema.canonical_operator

        return {
            "schema_id": matching_schema.schema_id,
            "functor": functor.value,
            "source_domain": matching_schema.source_domain.value,
            "target_domain": target_domain.value,
            "executable_code": f"transform = {op_code}",
        }


