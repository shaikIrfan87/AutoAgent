import pytest
from cognitive_engine.agent.executive_loop import ExecutiveLoop
from cognitive_engine.agent.goal_tree import GoalTree, GoalNode
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.world_model import MentalSimulator


def test_decompose_goal_numbered_and_bulleted():
    """Verify decomposition of numbered and bulleted compound goals into structured subgoals."""
    loop = ExecutiveLoop()

    # 1. Numbered steps
    numbered_goal = "1. Fetch raw sensor telemetry\n2. Filter out-of-bounds outliers\n3. Compute rolling average"
    subgoals = loop.decompose_goal(numbered_goal)
    assert len(subgoals) == 3
    assert subgoals[0][0] == "subgoal_1"
    assert "Fetch" in subgoals[0][1]
    assert "rolling" in subgoals[2][1]

    # 2. Bulleted steps
    bullet_goal = "- Load historical weights\n- Calculate Frobenius norm\n- Update SQLite database"
    bullet_subgoals = loop.decompose_goal(bullet_goal)
    assert len(bullet_subgoals) == 3
    assert "Load" in bullet_subgoals[0][1]
    assert "SQLite" in bullet_subgoals[2][1]


def test_decompose_goal_temporal_conjunctions():
    """Verify decomposition of sequential natural language goals connected by 'then'."""
    loop = ExecutiveLoop()
    temporal_goal = "Initialize the fast weights, then modulate the KV cache, and then verify bounded norm"
    subgoals = loop.decompose_goal(temporal_goal)
    assert len(subgoals) == 3
    assert "Initialize" in subgoals[0][1]
    assert "modulate" in subgoals[1][1]
    assert "verify" in subgoals[2][1]


def test_goal_tree_context_and_progress():
    """Verify GoalTree progress calculation, context retention, and local resetting."""
    tree = GoalTree(root_id="root", root_description="Root Task")
    tree.decompose("root", [("sg_1", "Step 1"), ("sg_2", "Step 2"), ("sg_3", "Step 3")])

    progress_initial = tree.get_execution_progress()
    assert progress_initial["total"] == 3
    assert progress_initial["completed"] == 0
    assert progress_initial["pct_complete"] == 0.0

    # Complete Step 1 with context
    tree.mark_completed("sg_1", result="output_1", context={"var_a": 42})
    assert tree.nodes["sg_1"].context["var_a"] == 42
    progress_step1 = tree.get_execution_progress()
    assert progress_step1["completed"] == 1
    assert progress_step1["pct_complete"] == 33.3

    # Mark step 2 failed, then reset for local retry
    tree.mark_failed("sg_2", error="Execution timeout")
    assert tree.nodes["sg_2"].status == "failed"
    tree.reset_subgoal("sg_2")
    assert tree.nodes["sg_2"].status == "pending"
    assert tree.nodes["sg_2"].error is None


def test_execute_hierarchical_chained_state_success():
    """Verify multi-step hierarchical execution chains state across subgoals."""
    loop = ExecutiveLoop()
    try:
        compound_goal = (
            "1. data = [10, 20, 30]\n"
            "2. doubled = [x * 2 for x in data]\n"
            "3. print('RESULT:', sum(doubled))"
        )
        res = loop.execute_hierarchical(compound_goal)
        assert res["success"] is True
        assert res["completed_count"] == 3
        assert res["delta_s"] == 1.0
        assert "RESULT: 120" in res["final_output"]
    finally:
        if hasattr(loop, "world_model") and hasattr(loop.world_model, "tuner"):
            loop.world_model.tuner.stop()


def test_execute_hierarchical_local_repair():
    """Verify local sub-step repair fixes transient step failures without invalidating prior steps."""
    loop = ExecutiveLoop()
    try:
        # Step 2 has a deliberate division by zero that the code_corrector auto-heals
        compound_goal = (
            "1. base_val = 100\n"
            "2. adjusted = base_val / 0\n"
            "3. print('HEALED:', adjusted)"
        )
        res = loop.execute_hierarchical(compound_goal)
        assert res["success"] is True
        assert res["completed_count"] == 3
        assert "HEALED: 100" in res["final_output"]
    finally:
        if hasattr(loop, "world_model") and hasattr(loop.world_model, "tuner"):
            loop.world_model.tuner.stop()


def test_execute_hierarchical_prospective_mental_veto():
    """Verify prospective mental rollout vetoes compound goals containing destructive subgoals."""
    loop = ExecutiveLoop()
    try:
        destructive_goal = (
            "1. data = [1, 2, 3]\n"
            "2. import shutil\nshutil.rmtree('/etc')\n"
            "3. print('DONE')"
        )
        res = loop.execute_hierarchical(destructive_goal)
        assert res["success"] is False
        assert res["delta_s"] == -1.0
        assert "veto" in res.get("reason", "").lower() or "risk" in res.get("reason", "").lower()
    finally:
        if hasattr(loop, "world_model") and hasattr(loop.world_model, "tuner"):
            loop.world_model.tuner.stop()


def test_orchestrator_execute_decomposed_plan():
    """Verify CognitiveEngine exposes execute_decomposed_plan endpoint."""
    engine = CognitiveEngine(db_path=":memory:")
    try:
        compound_plan = (
            "1. x = 5\n"
            "2. y = x * 10\n"
            "3. print('TOTAL:', y)"
        )
        res = engine.execute_decomposed_plan(compound_plan)
        assert res["success"] is True
        assert res["completed_count"] == 3
        assert "TOTAL: 50" in res["final_output"]
    finally:
        if hasattr(engine, "world_model") and hasattr(engine.world_model, "tuner"):
            engine.world_model.tuner.stop()
