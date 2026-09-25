import numpy as np
import pytest
from cognitive_engine.core.causal_graph import CausalSymbolicGraph


def test_interventional_causal_discovery():
    graph = CausalSymbolicGraph()
    var_names = ["voltage", "resistance", "current"]

    # Ground-truth physics generator:
    # current = 1.5 * voltage - 0.8 * resistance + noise
    # (voltage -> current, resistance -> current, no feedback from current)
    def physics_simulator(intervention: dict) -> dict:
        v = float(intervention.get("voltage", np.random.uniform(1.0, 10.0)))
        r = float(intervention.get("resistance", np.random.uniform(1.0, 5.0)))
        noise = float(np.random.normal(0, 0.02))
        i = 1.5 * v - 0.8 * r + noise
        return {"voltage": v, "resistance": r, "current": i}

    # Generate targeted intervention trials
    trials = []
    for _ in range(40):
        trials.append({"voltage": float(np.random.uniform(-3.0, 3.0)), "resistance": float(np.random.uniform(-2.0, 2.0))})

    W, scm = graph.conduct_interventional_experiment(
        sandbox_or_sim=physics_simulator,
        var_names=var_names,
        intervention_trials=trials,
        max_iter=120,
        lr=0.01,
    )

    assert W is not None
    assert W.shape == (3, 3)
    assert scm is not None

    # Voltage (idx 0) and Resistance (idx 1) must cause/inhibit Current (idx 2)
    # W[0, 2] should be positive (voltage causes current)
    # W[1, 2] should be negative (resistance inhibits current)
    # W[2, 0] should be ~0 (no backwards feedback)
    assert W[0, 2] > 0.05
    assert W[1, 2] < -0.05
    assert abs(W[2, 0]) < 0.20

    # Causal graph edges must reflect discovery
    assert graph.graph.has_edge("voltage", "current")
    assert graph.graph.has_edge("resistance", "current")


def test_nonlinear_mlp_notears_causal_discovery():
    from cognitive_engine.core.scm_engine import ContinuousSCMEngine

    # Ground truth non-linear kinetic energy: KE = 0.5 * mass * velocity^2
    # mass (idx 0), velocity (idx 1), kinetic_energy (idx 2)
    np.random.seed(42)
    n = 60
    masses = np.random.uniform(1.0, 5.0, size=(n, 1)).astype(np.float32)
    velocities = np.random.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float32)
    noise = np.random.normal(0, 0.05, size=(n, 1)).astype(np.float32)
    ke = 0.5 * masses * (velocities ** 2) + noise

    X = np.hstack([masses, velocities, ke])
    engine = ContinuousSCMEngine(dim=3, lambda_l1=0.01)

    # Fit using non-linear MLP NOTEARS formulation with exogenous intervention masks on mass & velocity
    W = engine.fit_nonlinear(X, max_iter=160, lr=0.02, exogenous_indices=[0, 1], hidden_dim=16)

    assert W.shape == (3, 3)
    # Mass and velocity must have causal influence on kinetic energy
    assert abs(W[0, 2]) > 0.05
    assert abs(W[1, 2]) > 0.05
    # Kinetic energy must not have backwards causal link to mass or velocity
    assert abs(W[2, 0]) == 0.0
    assert abs(W[2, 1]) == 0.0
    assert engine.is_dag()

    # Test non-linear counterfactual/interventional propagation
    state = engine.intervene({0: 2.0, 1: 3.0})
    assert len(state) == 3
    assert state[0] == 2.0
    assert state[1] == 3.0
    # Expected KE ~ 9.0; verify non-linear output is positive and reasonable
    assert state[2] > 2.0
