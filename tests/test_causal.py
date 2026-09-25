import time
from cognitive_engine.core.causal_graph import CausalSymbolicGraph
from cognitive_engine.core.types import Triple


def test_transitive_contradiction():
    graph = CausalSymbolicGraph()
    # Axioms already in default:
    # human is_a biologicalorganism
    # biologicalorganism mutually_exclusive syntheticmachine
    # android is_a syntheticmachine

    # Hypothesis: human is_a syntheticmachine -> Should fail transitive contradiction
    hypo = Triple(subject="human", relation="is_a", target="syntheticmachine", polarity=True)
    valid, diag = graph.verify_hypothesis(hypo)
    assert not valid
    assert "transitive contradiction" in diag.lower() or "mutual exclusion" in diag.lower()


def test_direct_contradiction():
    graph = CausalSymbolicGraph()
    # Axiom: vacuum cannot_be air
    hypo = Triple(subject="vacuum", relation="is_a", target="air", polarity=True)
    valid, diag = graph.verify_hypothesis(hypo)
    assert not valid
    assert "negative constraint" in diag.lower() or "contradicts" in diag.lower()


def test_latency_benchmark():
    graph = CausalSymbolicGraph()
    hypo = Triple(subject="human", relation="is_a", target="biologicalorganism", polarity=True)

    t0 = time.perf_counter()
    n_queries = 2000
    for _ in range(n_queries):
        graph.verify_hypothesis(hypo)
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_queries) * 1000.0
    print(f"\nCausal Rule Check Latency: {latency_ms:.4f} ms (target: <1.0 ms)")
    assert latency_ms < 1.0, f"Expected <1ms per query, got {latency_ms:.4f} ms"


def test_cyclic_contradictions():
    graph = CausalSymbolicGraph()
    # A -> causes -> B
    graph.add_axiom(Triple(subject="event_a", relation="causes", target="event_b", polarity=True))
    # B -> causes -> C
    graph.add_axiom(Triple(subject="event_b", relation="causes", target="event_c", polarity=True))

    # Hypothesis: C -> cannot_be -> A
    # Since event_a transitively causes event_c, asserting C cannot_be A closes a cyclic contradiction
    hypo = Triple(subject="event_c", relation="cannot_be", target="event_a", polarity=True)
    valid, diag = graph.verify_hypothesis(hypo)
    assert not valid
    assert "cyclic contradiction" in diag.lower()

    # Conversely, assert C causes A when A cannot_be C
    graph.add_axiom(Triple(subject="event_a", relation="cannot_be", target="event_c", polarity=True))
    hypo2 = Triple(subject="event_c", relation="causes", target="event_a", polarity=True)
    valid2, diag2 = graph.verify_hypothesis(hypo2)
    assert not valid2


def test_compositional_causal_non_collapse():
    graph = CausalSymbolicGraph()
    # spark causes flame, flame is_a thermal_event
    graph.add_axiom(Triple(subject="spark", relation="causes", target="flame", polarity=True))
    graph.add_axiom(Triple(subject="flame", relation="is_a", target="thermal_event", polarity=True))

    # In Pearl's do-calculus: spark causing flame does not mean spark causes all thermal_event categories
    # So asserting spark causes flame is valid, but spark cannot_be flame is prohibited
    hypo_invalid = Triple(subject="spark", relation="cannot_be", target="flame", polarity=True)
    valid, diag = graph.verify_hypothesis(hypo_invalid)
    assert not valid
    assert "cyclic contradiction" in diag.lower()


def test_notears_differentiable_causal_discovery_and_counterfactual_scm():
    import torch

    # Ground truth DAG: X0 -> X1 -> X2
    torch.manual_seed(42)
    n_samples = 250
    u0 = torch.randn(n_samples) * 0.5
    x0 = u0
    u1 = torch.randn(n_samples) * 0.2
    x1 = 2.0 * x0 + u1
    u2 = torch.randn(n_samples) * 0.2
    x2 = 1.5 * x1 + u2
    X = torch.stack([x0, x1, x2], dim=1)

    graph = CausalSymbolicGraph()
    scm = graph.fit_continuous_scm(X, var_names=["x0", "x1", "x2"], epochs=120, lr=0.03)

    # 1. Verify learned matrix is a true DAG
    h = torch.trace(torch.matrix_exp(scm.W * scm.W)) - 3.0
    assert abs(float(h.item())) < 1e-3

    # 2. Test Pearl Level 3 Counterfactual: Abduction -> Action -> Prediction
    factual_obs = torch.tensor([1.0, 2.1, 3.25])
    # Intervene do(x0 = 3.0)
    cf_obs, u_abducted = graph.evaluate_counterfactual_scm(factual_obs, {"x0": 3.0})

    assert cf_obs.shape == (1, 3)
    # x0 is set to 3.0
    assert abs(float(cf_obs[0, 0].item()) - 3.0) < 1e-4
    # Downstream x1 and x2 should scale causally based on learned edge weights
    assert float(cf_obs[0, 1].item()) > float(factual_obs[1])
    assert float(cf_obs[0, 2].item()) > float(factual_obs[2])


if __name__ == "__main__":
    test_transitive_contradiction()
    test_direct_contradiction()
    test_cyclic_contradictions()
    test_compositional_causal_non_collapse()
    test_notears_differentiable_causal_discovery_and_counterfactual_scm()
    test_latency_benchmark()
    print("All causal tests passed.")


