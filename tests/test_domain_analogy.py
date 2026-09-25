import pytest
from cognitive_engine.core.pattern_extractor import (
    DomainInvariantProjector,
    InvariantFunctor,
    StructuralDomain,
)


def test_cross_domain_monotonic_ordering_projection():
    projector = DomainInvariantProjector()

    # Target problem in inventory / scheduling domain: priority sorting
    # input: [(job_id, priority)], output: ordered by priority
    target_io = [
        ([(1, 30), (2, 10), (3, 20)], [(2, 10), (3, 20), (1, 30)]),
    ]

    functor = projector.infer_functor(target_io)
    assert functor == InvariantFunctor.MONOTONIC_ORDERING

    projection = projector.project_to_target_domain(
        target_io_pairs=target_io,
        target_domain=StructuralDomain.SEQUENCE,
    )
    assert projection is not None
    assert projection["functor"] == InvariantFunctor.MONOTONIC_ORDERING.value
    assert "transform =" in projection["executable_code"]

    # Execute projected transformation code in sandbox-style namespace
    local_ns = {}
    exec(projection["executable_code"], {}, local_ns)
    transform = local_ns["transform"]

    test_input = [(9, 50), (4, 15), (7, 2)]
    transformed = transform(test_input)
    assert transformed == [(7, 2), (4, 15), (9, 50)]


def test_cross_domain_axis_reversal_projection():
    projector = DomainInvariantProjector()

    target_io = [
        ([1, 2, 3, 4], [4, 3, 2, 1]),
    ]
    functor = projector.infer_functor(target_io)
    assert functor == InvariantFunctor.AXIS_REVERSAL

    projection = projector.project_to_target_domain(
        target_io_pairs=target_io,
        target_domain=StructuralDomain.VECTOR,
    )
    assert projection is not None
    assert projection["functor"] == InvariantFunctor.AXIS_REVERSAL.value
