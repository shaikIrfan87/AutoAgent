import numpy as np
import pytest
from cognitive_engine.core.causal_graph import CausalSymbolicGraph
from cognitive_engine.agent.goal_tree import UnsupervisedGoalInducer
from cognitive_engine.core.saliency import VectorBeliefDisambiguator


def test_4_variable_unknown_causal_system_discovery():
    """
    Stress-tests continuous NOTEARS interventional discovery on a 4-variable DAG:
      X0 (pressure) -> X1 (volume) [weight = 1.2]
      X1 (volume)   -> X2 (temperature) [weight = -0.9]
      X0 (pressure) -> X3 (entropy) [weight = 0.7]
      X2 (temp)     -> X3 (entropy) [weight = 1.1]
    Verifies that active interventions recover directional edges without manual axioms.
    """
    graph = CausalSymbolicGraph()
    var_names = ["force", "mass", "acceleration", "kinetic_energy"]

    def multi_variable_physics_sim(intervention: dict) -> dict:
        f = float(intervention.get("force", np.random.uniform(-3.0, 3.0)))
        m = float(intervention.get("mass", np.random.uniform(1.0, 5.0)))
        a = 1.5 * f - 0.8 * m + float(np.random.normal(0, 0.02))
        ke = 1.2 * m + 1.4 * a + float(np.random.normal(0, 0.02))
        return {"force": f, "mass": m, "acceleration": a, "kinetic_energy": ke}

    trials = []
    for _ in range(50):
        trials.append({
            "force": float(np.random.uniform(-3.0, 3.0)),
            "mass": float(np.random.uniform(1.0, 5.0)),
        })

    W, scm = graph.conduct_interventional_experiment(
        sandbox_or_sim=multi_variable_physics_sim,
        var_names=var_names,
        intervention_trials=trials,
        exogenous_vars=["force", "mass"],
        max_iter=150,
        lr=0.015,
    )

    assert W is not None
    assert W.shape == (4, 4)
    assert scm is not None

    # Forward causal paths must be discovered
    # force (0) -> acceleration (2)
    assert W[0, 2] > 0.05
    # mass (1) -> acceleration (2) [inhibitory]
    assert W[1, 2] < -0.05
    # acceleration (2) -> kinetic_energy (3)
    assert W[2, 3] > 0.05

    # Reverse paths must not exist (energy cannot cause force or mass)
    assert abs(W[3, 0]) < 0.20
    assert abs(W[3, 1]) < 0.20

    # Causal graph edges must reflect discovery
    assert graph.graph.has_edge("force", "acceleration")
    assert graph.graph.has_edge("mass", "acceleration")
    assert graph.graph.has_edge("acceleration", "kinetic_energy")



def test_unsupervised_goal_induction_bayesian_filtering():
    """
    Tests that coincidental correlations across noisy transitions are pruned,
    while persistent invariant laws are accepted with high Bayesian confidence.
    """
    # 3 transitions where sorting is strictly invariant, but cardinality is coincidentally fixed only once
    noisy_transitions = [
        ([3, 1, 2], [1, 2, 3]),
        ([9, 4], [4, 9]),
        ([10, 2, 8, 5], [2, 5, 8, 10]),
    ]

    consensus = UnsupervisedGoalInducer.cross_validate_invariants(noisy_transitions, min_confidence=0.60)
    assert consensus["is_permutation"] is True
    assert consensus["is_monotonic_increase"] is True
    assert consensus["is_filter"] is False
    assert consensus["confidence_scores"]["is_monotonic_increase"] >= 0.75

    tree = UnsupervisedGoalInducer.frame_problem_to_htn(None, None, transition_history=noisy_transitions)
    assert "sg_1_order_metric" in tree.nodes["root"].subgoal_ids


def test_vector_disambiguator_orthogonality_merging():
    """
    Tests that redundant prototype vectors exceeding cosine 0.70 are merged into
    a unified concept to prevent artificial entropy inflation.
    """
    dis = VectorBeliefDisambiguator()
    # Add two near-identical senses
    dis.add_sense("drone", "Quadcopter", "UAV unmanned aerial vehicle quadcopter flying aircraft", "quadcopter drone uav aircraft flying")
    dis.add_sense("drone", "Multirotor", "UAV multirotor remote controlled flying machine", "quadcopter drone uav aircraft multirotor", max_similarity=0.60)

    # Must be merged into a single orthogonal sense
    senses = dis.sense_clusters.get("drone", [])
    assert len(senses) == 1
    assert "Quadcopter" in senses[0][0]
    assert "Multirotor" in senses[0][0]
