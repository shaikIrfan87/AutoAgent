import pytest
from cognitive_engine.core.dsl import to_grid, rot90, flip_h, replace_color, crop_nonzero, gravity
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer


def test_dsl_primitives():
    g = to_grid([[1, 2], [3, 4]])
    assert rot90(g) == ((3, 1), (4, 2))
    assert flip_h(g) == ((2, 1), (4, 3))
    assert replace_color(g, 1, 9) == ((9, 2), (3, 4))

    sparse = to_grid([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert crop_nonzero(sparse) == ((5,),)

    falling = to_grid([[1, 0], [0, 2], [0, 0]])
    assert gravity(falling) == ((0, 0), (0, 0), (1, 2))


def test_inductive_synthesis_single_step():
    synthesizer = ProgramSynthesizer(max_depth=2)
    # Examples: horizontal flip
    train = [
        ([[1, 2], [3, 4]], [[2, 1], [4, 3]]),
        ([[5, 6], [7, 8]], [[6, 5], [8, 7]]),
    ]
    prog = synthesizer.synthesize(train)
    assert prog is not None
    assert prog([[9, 0], [1, 2]]) == ((0, 9), (2, 1))


def test_inductive_synthesis_multi_step_composition():
    synthesizer = ProgramSynthesizer(max_depth=3)
    # Target rule: rot90 then replace_color(1, 7)
    train = [
        ([[1, 0], [0, 0]], [[0, 7], [0, 0]]),
        ([[1, 1], [0, 0]], [[0, 7], [0, 7]]),
    ]
    prog = synthesizer.synthesize(train)
    assert prog is not None
    # Unseen test input evaluation
    test_input = [[0, 1], [0, 0]]
    result = prog(test_input)
    assert result == ((0, 0), [0, 7]) or result == ((0, 0), (0, 7))


def test_synthesizer_benchmark_and_guided_depth4():
    synthesizer = ProgramSynthesizer(max_depth=4)
    # Benchmark suite
    report = synthesizer.benchmark()
    assert report["total"] == 5
    assert report["solved"] == 5
    assert report["accuracy"] == 1.0

    # Depth 3/4 composition: flip_v then rot90 then replace_color(1, 8)
    train = [
        ([[1, 2], [0, 0]], [[0, 8], [0, 2]]),
        ([[1, 0], [3, 0]], [[3, 8], [0, 0]]),
    ]
    prog = synthesizer.synthesize(train)
    assert prog is not None
    assert prog([[0, 1], [0, 0]]) == ((0, 0), (0, 8))


def test_topological_dsl_primitives():
    from cognitive_engine.core.dsl import label_components, keep_largest_object, keep_smallest_object, shift_right
    # Two disconnected objects: size 3 (color 5) and size 1 (color 5)
    g = to_grid([
        [5, 5, 0, 0],
        [5, 0, 0, 0],
        [0, 0, 0, 5],
    ])
    labeled = label_components(g)
    assert labeled[0][0] == labeled[0][1] == labeled[1][0] == 1
    assert labeled[2][3] == 2

    largest = keep_largest_object(g)
    assert largest == ((5, 5, 0, 0), (5, 0, 0, 0), (0, 0, 0, 0))

    smallest = keep_smallest_object(g)
    assert smallest == ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 5))

    shifted = shift_right(largest)
    assert shifted == ((0, 5, 5, 0), (0, 5, 0, 0), (0, 0, 0, 0))

    # Synthesis of topological task
    synthesizer = ProgramSynthesizer(max_depth=2)
    prog = synthesizer.synthesize([(g, largest)])
    assert prog is not None
    assert prog.code == "keep_largest_object(g)"


def test_mcts_program_synthesizer_deep_composition():
    from cognitive_engine.agent.program_synthesizer import MCTSProgramSynthesizer
    mcts_synth = MCTSProgramSynthesizer(max_depth=5, max_expansions=1000)

    # Multi-step composition: keep_largest_object then rot90 then replace_color(5, 7)
    train = [
        (
            [[5, 5, 0], [5, 0, 0], [0, 0, 9]],
            [[0, 7, 7], [0, 0, 7], [0, 0, 0]],
        ),
        (
            [[5, 0, 0], [5, 5, 0], [0, 0, 4]],
            [[0, 7, 7], [0, 7, 0], [0, 0, 0]],
        ),
    ]
    prog = mcts_synth.synthesize(train)
    assert prog is not None
    assert prog([[5, 5, 0], [5, 0, 0], [0, 0, 3]]) == ((0, 7, 7), (0, 0, 7), (0, 0, 0))



