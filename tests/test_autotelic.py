import pytest
from cognitive_engine.core.autotelic import AutotelicExperimentLoop, EpistemicGap
from cognitive_engine.core.dsl import to_grid


def test_epistemic_gap_detection():
    loop = AutotelicExperimentLoop(uncertainty_threshold=0.30)
    mock_memory = [
        {"id": "concept_a", "confidence": 0.50, "last_accessed": 0.0, "content": "test A"},
        {"id": "concept_b", "confidence": 0.95, "last_accessed": 0.0, "content": "test B"},
    ]
    mock_rules = {"concept_b": [("is_a", "entity", True)]}

    gaps = loop.scan_epistemic_gaps(mock_memory, mock_rules)
    assert len(gaps) >= 1
    assert any(g.concept == "concept_a" for g in gaps)


def test_autotelic_successful_synthesis_experiment():
    loop = AutotelicExperimentLoop()
    gap = EpistemicGap(gap_id="g1", concept="spatial_inversion", uncertainty=0.7)

    # Invert/flip vertical task
    ex1_in = ((1, 2), (0, 0))
    ex1_out = ((0, 0), (1, 2))
    examples = [(to_grid(ex1_in), to_grid(ex1_out))]

    result = loop.run_experiment(gap, custom_examples=examples)
    assert result.success is True
    assert result.delta == 1.0
    assert result.program is not None
    assert "flip_v" in result.program.code


def test_autotelic_unsolvable_experiment_delta():
    loop = AutotelicExperimentLoop()
    gap = EpistemicGap(gap_id="g2", concept="impossible_mapping", uncertainty=0.9)

    # Contradictory mapping impossible within unary depth 3
    ex1_in = ((1, 1), (1, 1))
    ex1_out = ((2, 3), (4, 5))
    examples = [(to_grid(ex1_in), to_grid(ex1_out))]

    result = loop.run_experiment(gap, custom_examples=examples)
    assert result.success is False
    assert result.delta == -1.0
    assert result.program is None
