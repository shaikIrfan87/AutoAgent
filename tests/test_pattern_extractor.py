import numpy as np
import pytest
from cognitive_engine.core.pattern_extractor import (
    UniversalPatternExtractor,
    StructuralDomain,
    ScalarInvariants,
    VectorInvariants,
    GridInvariants,
    SequenceInvariants,
    GraphInvariants,
    StructuralSpec,
)


def test_scalar_invariant_extraction():
    # Prime odd scalar
    domain, inv = UniversalPatternExtractor.extract_invariants(17)
    assert domain == StructuralDomain.SCALAR
    assert isinstance(inv, ScalarInvariants)
    assert inv.is_integer is True
    assert inv.is_prime is True
    assert inv.is_even is False
    assert inv.is_positive is True

    # Even zero scalar
    domain, inv_zero = UniversalPatternExtractor.extract_invariants(0)
    assert inv_zero.is_zero is True
    assert inv_zero.is_even is True
    assert inv_zero.is_prime is False

    # Float vector
    vec = inv.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32


def test_vector_invariant_extraction():
    # Monotonic increasing vector
    domain, inv = UniversalPatternExtractor.extract_invariants([1, 2, 4, 8, 16])
    assert domain == StructuralDomain.VECTOR
    assert isinstance(inv, VectorInvariants)
    assert inv.length == 5
    assert inv.is_monotonic_increasing is True
    assert inv.is_monotonic_decreasing is False
    assert inv.has_duplicates is False

    # Monotonic decreasing with duplicates
    _, inv_dup = UniversalPatternExtractor.extract_invariants([10, 5, 5, 2, 1])
    assert inv_dup.is_monotonic_decreasing is True
    assert inv_dup.has_duplicates is True

    vec = inv.to_vector()
    assert len(vec) == 8


def test_grid_invariant_extraction():
    # 3x3 symmetric grid
    grid = [
        [1, 2, 1],
        [3, 4, 3],
        [1, 2, 1],
    ]
    domain, inv = UniversalPatternExtractor.extract_invariants(grid)
    assert domain == StructuralDomain.GRID
    assert isinstance(inv, GridInvariants)
    assert inv.height == 3
    assert inv.width == 3
    assert inv.is_square is True
    assert inv.symmetry_horizontal is True
    assert inv.symmetry_vertical is True
    assert inv.unique_symbols == 4

    vec = inv.to_vector()
    assert len(vec) == 9


def test_sequence_invariant_extraction():
    # Palindrome sequence
    domain, inv = UniversalPatternExtractor.extract_invariants("radar")
    assert domain == StructuralDomain.SEQUENCE
    assert isinstance(inv, SequenceInvariants)
    assert inv.length == 5
    assert inv.is_palindrome is True

    # Periodic sequence
    _, inv_period = UniversalPatternExtractor.extract_invariants(["A", "B", "A", "B", "A", "B"])
    assert inv_period.has_periodicity is True
    assert inv_period.period_length == 2

    vec = inv.to_vector()
    assert len(vec) == 5


def test_graph_invariant_extraction():
    # Cyclic directed graph
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
    }
    domain, inv = UniversalPatternExtractor.extract_invariants(graph)
    assert domain == StructuralDomain.GRAPH
    assert isinstance(inv, GraphInvariants)
    assert inv.num_nodes == 3
    assert inv.num_edges == 3
    assert inv.has_cycles is True
    assert inv.min_degree == 1
    assert inv.max_degree == 1

    vec = inv.to_vector()
    assert len(vec) == 7


def test_adag_and_structural_spec_formulation():
    # Example: Vector -> Scalar (e.g., list sum or length)
    io_pairs = [
        ([1, 2, 3], 6),
        ([4, 5, 6], 15),
    ]

    spec = UniversalPatternExtractor.extract_spec(io_pairs)
    assert isinstance(spec, StructuralSpec)
    assert spec.input_domain == StructuralDomain.VECTOR
    assert spec.output_domain == StructuralDomain.SCALAR
    assert spec.is_isomorphic_domain is False
    assert spec.adag is not None
    assert len(spec.adag.nodes) == 2
    assert len(spec.adag.edges) == 1
    assert spec.adag.edges[0].relation == "transforms_to"

    # State vector must be exactly 32-dimensional float32
    assert isinstance(spec.state_vector, np.ndarray)
    assert spec.state_vector.shape == (32,)
    assert spec.state_vector.dtype == np.float32

    # Verify one-hot input is VECTOR (index 1) and output is SCALAR (index 5)
    assert spec.state_vector[1] == 1.0
    assert spec.state_vector[5] == 1.0


def test_isomorphic_grid_spec():
    # Example: Grid -> Grid (e.g., ARC transformation / flip)
    g_in = [[1, 0], [0, 1]]
    g_out = [[0, 1], [1, 0]]
    spec = UniversalPatternExtractor.extract_spec([(g_in, g_out)])

    assert spec.input_domain == StructuralDomain.GRID
    assert spec.output_domain == StructuralDomain.GRID
    assert spec.is_isomorphic_domain is True
    assert spec.state_vector[2] == 1.0   # Grid input
    assert spec.state_vector[7] == 1.0   # Grid output
    assert spec.state_vector[10] == 1.0  # Isomorphic flag
