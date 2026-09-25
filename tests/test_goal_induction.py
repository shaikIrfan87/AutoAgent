import pytest
from cognitive_engine.agent.goal_tree import UnsupervisedGoalInducer, GoalTree
from cognitive_engine.agent.lambda_synthesizer import LambdaProgramSynthesizer


def test_unsupervised_sorting_goal_induction():
    # Initial unsorted sequence vs sorted target
    s_init = [5, 2, 8, 1, 9]
    s_target = [1, 2, 5, 8, 9]

    invariants = UnsupervisedGoalInducer.extract_state_invariants(s_init, s_target)
    assert invariants["is_permutation"] is True
    assert invariants["is_monotonic_increase"] is True

    tree = UnsupervisedGoalInducer.frame_problem_to_htn(s_init, s_target, goal_name="Unsupervised Sort")
    assert isinstance(tree, GoalTree)
    assert len(tree.nodes["root"].subgoal_ids) == 2


def test_unsupervised_filter_goal_induction():
    s_init = [1, 2, 3, 4, 5, 6]
    s_target = [2, 4, 6]

    invariants = UnsupervisedGoalInducer.extract_state_invariants(s_init, s_target)
    assert invariants["is_filter"] is True

    tree = UnsupervisedGoalInducer.frame_problem_to_htn(s_init, s_target, goal_name="Unsupervised Filter")
    assert "sg_1_filter" in tree.nodes["root"].subgoal_ids


def test_unsupervised_synthesizer_coupling():
    # Observation history of raw state transitions
    obs_history = [
        ([1, 2, 3], [2, 3, 4]),
        ([10, 20], [11, 21]),
        ([0], [1]),
    ]
    io_pairs = UnsupervisedGoalInducer.frame_unsupervised_io_pairs(obs_history)
    assert len(io_pairs) == 3

    synth = LambdaProgramSynthesizer(max_cost=8)
    res = synth.synthesize(io_pairs, var_name="x")
    assert res is not None
    node, code = res
    assert "+" in code or "map" in code
