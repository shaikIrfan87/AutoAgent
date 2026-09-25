import pytest
from cognitive_engine.core.compression import ProgramCompressor
from cognitive_engine.core.dsl import UNARY_PRIMITIVES, rot90, flip_h, to_grid
from cognitive_engine.agent.program_synthesizer import SynthesizedProgram


def test_subtree_mining_and_utility_ranking():
    compressor = ProgramCompressor(min_frequency=2, max_macro_len=2)

    # 3 programs sharing identical (rot90 -> flip_h) sub-sequence
    steps_a = [("rot90", ()), ("flip_h", ()), ("rot180", ())]
    steps_b = [("gravity", ()), ("rot90", ()), ("flip_h", ())]
    steps_c = [("rot90", ()), ("flip_h", ())]

    progs = [
        SynthesizedProgram(steps=steps_a, fn=lambda g: g, code="prog_a"),
        SynthesizedProgram(steps=steps_b, fn=lambda g: g, code="prog_b"),
        SynthesizedProgram(steps=steps_c, fn=lambda g: g, code="prog_c"),
    ]

    mined = compressor.mine_abstractions(progs)
    assert len(mined) >= 1
    top_macro = mined[0]
    assert top_macro.operations == (("rot90", ()), ("flip_h", ()))
    assert top_macro.utility_score == 3 * (2 - 1.0)


def test_macro_dsl_consolidation_and_execution():
    compressor = ProgramCompressor(min_frequency=1, max_macro_len=2)
    steps = [("rot90", ()), ("flip_h", ())]
    progs = [SynthesizedProgram(steps=steps, fn=lambda g: g, code="prog")]

    mined = compressor.mine_abstractions(progs)
    assert len(mined) == 1

    added_count = compressor.consolidate_into_dsl(mined)
    macro_name = mined[0].name
    assert added_count == 1
    assert macro_name in UNARY_PRIMITIVES

    # Verify semantic behavior: (flip_h after rot90)
    grid = to_grid(((1, 2), (3, 4)))
    expected = flip_h(rot90(grid))
    actual = UNARY_PRIMITIVES[macro_name](grid)
    assert actual == expected
