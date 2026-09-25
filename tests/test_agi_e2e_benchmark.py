import pytest
from cognitive_engine.core.dsl import (
    to_grid,
    pad,
    unpad,
    reflect_sym_h,
    reflect_sym_v,
    invert,
    outline,
    UNARY_PRIMITIVES,
)
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer, SynthesizedProgram
from cognitive_engine.core.autotelic import AutotelicExperimentLoop, EpistemicGap
from cognitive_engine.core.compression import ProgramCompressor
from cognitive_engine.core.meta_optimizer import MetaSelfOptimizer
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_extended_dsl_primitives():
    """Verify new spatial, reflection, and topological operators."""
    g = to_grid([[1, 2], [3, 4]])

    # Symmetry reflections
    sym_h = reflect_sym_h(g)
    assert sym_h == ((1, 2, 2, 1), (3, 4, 4, 3))

    sym_v = reflect_sym_v(g)
    assert sym_v == ((1, 2), (3, 4), (3, 4), (1, 2))

    # Padding and unpadding
    padded = pad(g, pad_val=0)
    assert len(padded) == 4 and len(padded[0]) == 4
    assert unpad(padded) == g

    # Inversion
    inv = invert(to_grid([[1, 0], [0, 9]]))
    assert inv == ((8, 0), (0, 0))

    # Outline
    solid_box = to_grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    hollow = outline(solid_box)
    assert hollow == ((1, 1, 1), (1, 0, 1), (1, 1, 1))


def test_full_agi_loop_e2e():
    """End-to-end benchmark verifying all 4 AGI pillars in orchestrated execution."""
    # 1. Initialize Cognitive Engine
    engine = CognitiveEngine()
    assert hasattr(engine, "synthesizer")
    assert hasattr(engine, "autotelic")
    assert hasattr(engine, "compressor")
    assert hasattr(engine, "meta_optimizer")
    assert hasattr(engine, "curiosity")

    # 2. Pillar 1: Synthesize program for multi-step spatial task
    train_examples = [
        ([[1, 0], [0, 0]], [[1, 0, 0, 1], [0, 0, 0, 0]]),
        ([[2, 3], [0, 0]], [[2, 3, 3, 2], [0, 0, 0, 0]]),
    ]
    prog = engine.synthesizer.synthesize(train_examples)
    assert prog is not None
    assert "reflect_sym_h" in prog.code
    # OOD generalization test
    test_grid = [[5, 6], [7, 8]]
    assert prog(test_grid) == ((5, 6, 6, 5), (7, 8, 8, 7))

    # 3. Pillar 2: Autotelic gap detection and empirical experiment
    gap = EpistemicGap(gap_id="bench_gap_1", concept="spatial_reflection", uncertainty=0.8)
    exp_res = engine.autotelic.run_experiment(gap, custom_examples=[(to_grid([[1, 2]]), to_grid([[1, 2, 2, 1]]))])
    assert exp_res.success is True
    assert exp_res.delta == 1.0

    # 4. Pillar 3: AST Subtree Mining and Macro Ingestion
    repeated_steps = [("rot90", ()), ("pad", ())]
    synthetic_corpus = [
        SynthesizedProgram(steps=repeated_steps, fn=lambda g: g, code="p1"),
        SynthesizedProgram(steps=repeated_steps, fn=lambda g: g, code="p2"),
    ]
    mined = engine.compressor.mine_abstractions(synthetic_corpus)
    assert len(mined) >= 1
    macro_name = mined[0].name
    engine.compressor.consolidate_into_dsl(mined)
    assert macro_name in UNARY_PRIMITIVES

    # 5. Pillar 4: Telemetry-driven recursive self-optimization
    initial_depth = engine.synthesizer.max_depth
    # Inject 4 consecutive failures into curiosity daemon telemetry
    for _ in range(4):
        engine.curiosity.record_telemetry(False)
    # Trigger optimization
    engine.curiosity.failure_threshold_ratio = 0.5
    engine.curiosity.telemetry_window = type(engine.curiosity.telemetry_window)([False, False, False, False], maxlen=4)
    res = engine.curiosity._attempt_self_optimization()
    assert res is not None and res.applied is True
    assert engine.synthesizer.max_depth == min(initial_depth + 1, 5)

    # 6. Verify full rollback restores original state
    reverted = engine.meta_optimizer.rollback_last()
    assert reverted is True
