import pytest
from cognitive_engine.agent.arc_benchmark import ARCBenchmarkHarness, get_canonical_arc_tasks
from cognitive_engine.agent.program_synthesizer import MCTSProgramSynthesizer, ProgramSynthesizer


def test_arc_benchmark_canonical_tasks():
    tasks = get_canonical_arc_tasks()
    assert len(tasks) >= 5
    assert all(task.train and task.test_in and task.expected_test_out for task in tasks)


def test_arc_benchmark_harness_execution():
    harness = ARCBenchmarkHarness()
    synth = MCTSProgramSynthesizer(max_depth=5, max_expansions=1500)
    report = harness.run_benchmark(synth)

    assert report["total_tasks"] == 5
    assert report["solved_train"] == 5
    assert report["generalized_test"] == 5
    assert report["generalization_rate"] == 1.0
    assert report["mean_latency_ms"] < 2000.0  # Fast sub-second average per task
    assert len(report["results"]) == 5


def test_functional_lambda_ast_combinators():
    from cognitive_engine.core.dsl import (
        to_grid,
        rot90,
        flip_h,
        has_color,
        is_symmetric_h,
        if_then_else,
        repeat_until,
        shift_down,
    )
    g_sym = to_grid([[1, 2, 2, 1], [3, 4, 4, 3]])
    g_asym = to_grid([[1, 2, 0, 0], [3, 4, 0, 0]])

    assert is_symmetric_h(g_sym) is True
    assert is_symmetric_h(g_asym) is False
    assert has_color(3)(g_sym) is True
    assert has_color(9)(g_sym) is False

    # Conditional lambda branch: if symmetric rotate 90, else flip horizontal
    branch_fn = if_then_else(is_symmetric_h, rot90, flip_h)
    assert branch_fn(g_sym) == rot90(g_sym)
    assert branch_fn(g_asym) == flip_h(g_asym)

    # Iterative repeat loop: shift down until row 2 is populated
    step_fn = repeat_until(shift_down, cond_fn=lambda g: any(g[2]), max_steps=4)
    g_top = to_grid([[1, 0], [0, 0], [0, 0]])
    stepped = step_fn(g_top)
    assert stepped[2][0] == 1


def test_arc_benchmark_json_streaming(tmp_path):
    import json
    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]}
        ]
    }
    json_file = tmp_path / "sample_arc_task.json"
    json_file.write_text(json.dumps(task_data), encoding="utf-8")

    harness = ARCBenchmarkHarness.from_json_file(str(json_file))
    assert len(harness.tasks) == 1
    assert harness.tasks[0].name == "sample_arc_task_test0"

    report = harness.run_benchmark()
    assert report["total_tasks"] == 1
    assert report["solved_train"] == 1
    assert report["generalized_test"] == 1


def test_relational_predicate_branch_synthesis():
    from cognitive_engine.core.dsl import to_grid, rot90, flip_h
    # Example 1: rot90 output
    ex1_in = to_grid([[1, 0], [1, 1]])
    ex1_out = rot90(ex1_in)
    # Example 2: flip_h output
    ex2_in = to_grid([[2, 2], [1, 0]])
    ex2_out = flip_h(ex2_in)

    synth = ProgramSynthesizer(max_depth=2)
    prog = synth.synthesize([(ex1_in, ex1_out), (ex2_in, ex2_out)])
    assert prog is not None
    assert "if_then_else" in prog.code

    # Verify predictions on training inputs match targets
    assert prog(ex1_in) == ex1_out
    assert prog(ex2_in) == ex2_out


def test_multistep_composite_conditional_branching():
    from cognitive_engine.core.dsl import to_grid, rot90, flip_h, invert
    # Subset 1 (has color 1): 2-step composite rot90(flip_h(g))
    ex1_in = to_grid([[1, 2], [3, 4]])
    ex1_out = rot90(flip_h(ex1_in))

    # Subset 2 (no color 1): 1-step invert(g)
    ex2_in = to_grid([[0, 5], [0, 0]])
    ex2_out = invert(ex2_in)

    synth = ProgramSynthesizer(max_depth=2)
    prog = synth.synthesize([(ex1_in, ex1_out), (ex2_in, ex2_out)])
    assert prog is not None
    assert "if_then_else" in prog.code
    assert prog(ex1_in) == ex1_out
    assert prog(ex2_in) == ex2_out





