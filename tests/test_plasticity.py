import torch
from cognitive_engine.core.plastic_layer import FastPlasticLinear, MultiHeadPlasticLayer


def test_associative_retention_under_interference():
    # Invariant 4: Correlated noise stress test on FastPlasticLinear
    dim = 32
    layer = FastPlasticLinear(in_features=dim, out_features=dim, gamma=0.99, eta=0.08)

    torch.manual_seed(42)
    # Target memory anchor
    target_k = torch.randn(dim)
    target_k = target_k / torch.linalg.norm(target_k)
    target_v = torch.randn(dim)
    target_v = target_v / torch.linalg.norm(target_v)
    layer.absorb(target_k, target_v, steps=6)

    # Inject 25 intervening correlated noise facts (cos ~ 0.40 - 0.60)
    for _ in range(25):
        rho = 0.50
        z = torch.randn(dim)
        z = z - torch.dot(z, target_k) * target_k
        z = z / torch.linalg.norm(z)
        noise_k = rho * target_k + ((1.0 - rho**2) ** 0.5) * z
        noise_v = torch.randn(dim)
        noise_v = noise_v / torch.linalg.norm(noise_v)
        layer.absorb(noise_k, noise_v, steps=1)

    recalled_v = layer.recall_fast(target_k)
    sim = torch.cosine_similarity(recalled_v.unsqueeze(0), target_v.unsqueeze(0)).item()
    norm = torch.linalg.norm(layer.A_fast, ord="fro").item()

    # Dispersed retention score cos approx 0.70 - 0.85 under subspace interference
    assert 0.68 <= sim <= 0.85, f"Expected dispersed retention cos in [0.68, 0.85], got {sim:.3f}"
    assert norm > 0.40, f"Frobenius norm too constrained under correlated updates: {norm:.3f}"



def test_frobenius_norm_bounded_under_200_updates():
    dim = 64
    layer = FastPlasticLinear(in_features=dim, out_features=dim, frobenius_limit=2.0, eta=0.04)

    torch.manual_seed(42)
    for _ in range(200):
        k = torch.randn(dim)
        v = torch.randn(dim)
        norm = layer.absorb(k, v, steps=1)
        assert norm <= 2.0001, f"Frobenius norm exploded: {norm}"

    final_norm = torch.linalg.norm(layer.A_fast, ord="fro").item()
    assert final_norm <= 2.0, f"Final Frobenius norm exceeded bound: {final_norm}"


def test_associative_recall_accuracy():
    dim = 64
    layer = FastPlasticLinear(in_features=dim, out_features=dim, gamma=0.98, eta=0.10)
    layer.reset_session()

    torch.manual_seed(123)
    k = torch.randn(dim)
    k = k / torch.linalg.norm(k)
    v = torch.randn(dim)
    v = v / torch.linalg.norm(v)

    # Ingest association over a few delta rule updates
    layer.absorb(k, v, steps=15)

    # Recall
    recalled = layer.recall_fast(k)
    cos_sim = torch.cosine_similarity(recalled.unsqueeze(0), v.unsqueeze(0)).item()

    print(f"\nAssociative Recall Cosine Similarity: {cos_sim:.4f} (target: >0.85)")
    assert cos_sim > 0.85, f"Expected cosine similarity >0.85, got {cos_sim:.4f}"


def test_reset_session():
    dim = 32
    layer = FastPlasticLinear(in_features=dim, out_features=dim)
    k = torch.randn(dim)
    v = torch.randn(dim)
    layer.absorb(k, v)
    assert torch.linalg.norm(layer.A_fast) > 0.0

    layer.reset_session()
    assert torch.all(layer.A_fast == 0.0)


def test_orthogonal_saturation_500_keys():
    dim = 128
    layer = FastPlasticLinear(in_features=dim, out_features=dim, gamma=0.96, eta=0.03, frobenius_limit=2.0)
    layer.reset_session()

    torch.manual_seed(999)
    first_k = None
    first_v = None

    # Ingest 500 normalized pseudo-orthogonal keys
    for i in range(500):
        k = torch.randn(dim)
        k = k / torch.linalg.norm(k)
        v = torch.randn(dim)
        v = v / torch.linalg.norm(v)

        if i == 0:
            first_k = k.clone()
            first_v = v.clone()

        norm = layer.absorb(k, v, steps=1)
        assert norm <= 2.0001, f"Frobenius norm exceeded 2.0 during saturation at step {i}: {norm}"
        assert not torch.isnan(layer.A_fast).any(), "NaN encountered in A_fast"

    final_norm = torch.linalg.norm(layer.A_fast, ord="fro").item()
    assert final_norm <= 2.0

    # Oldest association decays smoothly without numerical underflow / NaN
    recalled_first = layer.recall_fast(first_k)
    assert not torch.isnan(recalled_first).any()
    assert torch.isfinite(recalled_first).all()


def test_threshold_gated_flush():
    dim = 64
    layer = FastPlasticLinear(in_features=dim, out_features=dim, flush_threshold=0.30)
    layer.reset_session()

    k = torch.randn(dim)
    k = k / torch.linalg.norm(k)
    v = torch.randn(dim)
    v = v / torch.linalg.norm(v)

    # First update has high error: trace magnitude exceeds flush threshold
    norm, flush_req, trace_mag = layer.absorb_with_flush_check(k, v, steps=1)
    assert flush_req is True
    assert trace_mag >= 0.30

    # Repeated identical update on already learned key: prediction error is now small
    norm2, flush_req2, trace_mag2 = layer.absorb_with_flush_check(k, v, steps=10)
    assert trace_mag2 < trace_mag
    print(f"\nTrace magnitude reduced from {trace_mag:.4f} to {trace_mag2:.4f} (flush gating operational).")


def test_multi_head_decoupled_capacity():
    dim = 64
    num_heads = 4
    head_dim = dim // num_heads
    layer = FastPlasticLinear(in_features=dim, out_features=dim, num_heads=num_heads)
    layer.reset_session()

    # Create key/value active strictly in Head 0
    k0 = torch.zeros(dim)
    k0[:head_dim] = torch.randn(head_dim)
    k0[:head_dim] /= torch.linalg.norm(k0[:head_dim])
    v0 = torch.zeros(dim)
    v0[:head_dim] = torch.randn(head_dim)

    layer.absorb(k0, v0, steps=5)

    # Verify Head 0 block is active
    assert torch.linalg.norm(layer.A_fast[:head_dim, :head_dim]) > 0.0

    # Verify other heads (e.g. Head 1 block) have strictly ZERO cross-talk
    head1_block = layer.A_fast[head_dim : 2 * head_dim, head_dim : 2 * head_dim]
    assert torch.all(head1_block == 0.0), "Cross-talk detected in decoupled multi-head projection"

    # Off-diagonal cross-head blocks must remain zero
    cross_block = layer.A_fast[:head_dim, head_dim : 2 * head_dim]
    assert torch.all(cross_block == 0.0), "Off-diagonal leakage across heads detected"

    # Multi-head decoupled penalization must also preserve block-diagonal isolation
    layer.penalize(k0, v0, penalty=0.05)
    assert torch.all(layer.A_fast[:head_dim, head_dim : 2 * head_dim] == 0.0), "Anti-Hebbian cross-talk detected across heads"
    assert torch.all(layer.A_fast[head_dim : 2 * head_dim, head_dim : 2 * head_dim] == 0.0)


if __name__ == "__main__":
    test_frobenius_norm_bounded_under_200_updates()
    test_associative_recall_accuracy()
    test_reset_session()
    test_orthogonal_saturation_500_keys()
    test_threshold_gated_flush()
    test_multi_head_decoupled_capacity()
    print("All plasticity tests passed.")



