import pytest
from benchmarks.run_algorithmic_eval import run_algorithmic_eval, get_canonical_algorithmic_benchmarks
from cognitive_engine.agent.executive_loop import ExecutiveLoop
from cognitive_engine.core.sandbox import EnvironmentalSandbox


def test_open_algorithmic_benchmarks_structure():
    benchmarks = get_canonical_algorithmic_benchmarks()
    assert len(benchmarks) >= 4
    for b in benchmarks:
        assert "name" in b and "code" in b and "assertions" in b
        assert len(b["assertions"]) > 0


def test_executive_loop_algorithmic_eval_execution():
    report = run_algorithmic_eval()
    assert report["total"] == 4
    assert report["passed"] == 4
    assert report["success_rate"] == 1.0
    assert report["elapsed_sec"] < 2.0


def test_scene_graph_rewriting_operators():
    from cognitive_engine.core.dsl import to_grid, remove_isolated_objects
    from cognitive_engine.core.scene_graph import SceneGraphExtractor
    # 2 touching blocks + 1 isolated block
    g = to_grid([
        [1, 1, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [0, 0, 0, 0, 9],  # isolated 9
    ])
    # Remove isolated objects
    filtered = SceneGraphExtractor.remove_isolated_objects(g)
    assert filtered[2][4] == 0
    assert filtered[0][0] == 1
    assert filtered[1][1] == 2

    # Verify DSL wrapper
    dsl_filtered = remove_isolated_objects(g)
    assert dsl_filtered == filtered

    # Recolor touching objects
    recolored = SceneGraphExtractor.recolor_touching_objects(g, touch_color=2, new_color=7)
    assert recolored[0][0] == 7
    assert recolored[0][1] == 7
