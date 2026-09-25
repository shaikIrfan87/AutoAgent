import torch
import pytest
from cognitive_engine.core.plastic_layer import MultiHeadPlasticLayer

def test_associative_interference_under_continuous_stream():
    layer = MultiHeadPlasticLayer(dim=32, num_heads=4, decay_rate=0.98, eta=0.05)
    
    # 1. Anchor Key-Value pair
    anchor_k = torch.randn(1, 32)
    anchor_v = torch.randn(1, 32)
    layer.adapt(anchor_k, anchor_v)
    
    initial_recall = layer.recall(anchor_k)
    init_sim = torch.cosine_similarity(initial_recall, anchor_v).item()
    assert init_sim > 0.85, f"Initial recall too low: {init_sim}"
    
    # 2. Ingest 150 intervening distracting task associations
    for _ in range(150):
        distract_k = torch.randn(1, 32)
        distract_v = torch.randn(1, 32)
        layer.adapt(distract_k, distract_v)
    
    # 3. Test retention of anchor
    decayed_recall = layer.recall(anchor_k)
    final_sim = torch.cosine_similarity(decayed_recall, anchor_v).item()
    
    print(f"\nPlastic Retention: {init_sim:.4f} -> {final_sim:.4f} (Frobenius Norm: {torch.norm(layer.A_fast).item():.4f})")
    assert not torch.isnan(layer.A_fast).any(), "NaN found in weights"
    assert torch.norm(layer.A_fast).item() <= 2.0, "Frobenius norm violated ceiling"
    # Fails if fast memory suffers catastrophic erasure (< 0.20 similarity)
    assert final_sim >= 0.20, f"Catastrophic forgetting occurred: final cosine sim {final_sim:.4f} < 0.20"
