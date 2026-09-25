import time
from cognitive_engine.core.saliency import SaliencyGate


def test_low_entropy_gibberish_dropped():
    gate = SaliencyGate()
    # Extremely repetitive low-entropy string
    gibberish = "spam spam spam spam spam spam spam spam spam spam spam"
    decision = gate.evaluate(gibberish)
    assert not decision.pass_filter
    assert decision.entropy < 2.0
    assert "Low Shannon entropy" in decision.reason


def test_duplicate_text_blocks_dropped():
    gate = SaliencyGate()
    informative_text = "The quantum fluctuations in cosmological inflation seed large-scale galaxy formations."

    # First presentation passes
    d1 = gate.evaluate(informative_text)
    assert d1.pass_filter

    # Duplicate presentation
    d2 = gate.evaluate(informative_text)
    assert not d2.pass_filter
    assert d2.novelty < 0.10
    assert "duplicate" in d2.reason.lower()


def test_throughput_benchmark():
    gate = SaliencyGate()
    # Benchmark front-end filter rejecting incoming low-entropy streams before ONNX
    test_chunks = [
        f"sensor noise noise noise noise noise {i % 2}"
        for i in range(2000)
    ]


    t0 = time.perf_counter()
    for chunk in test_chunks:
        res = gate.evaluate(chunk)
        assert not res.pass_filter
    elapsed = time.perf_counter() - t0

    throughput = len(test_chunks) / elapsed
    print(f"\nSaliency Filter Front-End Throughput: {throughput:.2f} chunks/sec (target: >1,000 chunks/sec)")
    assert throughput > 1000.0, f"Expected >1000 chunks/sec, got {throughput:.2f}"


def test_high_entropy_random_noise():
    gate = SaliencyGate()
    # Hex dump / encrypted base64-like high-entropy noise
    noise = "4f3a9b1c7e8d2f0a 8b7c6d5e4f3a2b10 a9b8c7d6e5f4a3b2 1c2d3e4f5a6b7c8d e9f0a1b2c3d4e5f6"
    decision = gate.evaluate(noise)

    # Entropy check passes because tokens are varied/random
    assert decision.pass_filter is True
    assert decision.entropy > 2.0
    # Epistemic novelty & uncertainty trigger anomaly flag
    assert decision.requires_deliberation is True
    assert decision.is_anomaly is True
    assert "Epistemic anomaly" in decision.reason


def test_semantic_neighborhood_preservation():
    import numpy as np
    gate = SaliencyGate()

    vec_canine = gate._embed("domestic canine hound")
    vec_dog = gate._embed("friendly dog puppy")
    vec_quantum = gate._embed("quantum qubit entanglement")

    # Metric space neighborhood check
    cos_synonyms = float(np.dot(vec_canine, vec_dog))
    cos_unrelated = float(np.dot(vec_canine, vec_quantum))

    print(f"\nSemantic Similarity (canine vs dog): {cos_synonyms:.4f} (target: >0.50)")
    print(f"Semantic Similarity (canine vs quantum): {cos_unrelated:.4f} (target: <0.45)")

    assert cos_synonyms > 0.50, f"Expected synonym similarity >0.50, got {cos_synonyms:.4f}"
    assert cos_unrelated < 0.45, f"Expected unrelated similarity <0.45, got {cos_unrelated:.4f}"
    assert (cos_synonyms - cos_unrelated) > 0.20, f"Expected margin >0.20, got {cos_synonyms - cos_unrelated:.4f}"



def test_semantic_masking_polysemy():
    gate = SaliencyGate()
    # Grammatically valid wrapper masking hex/hash tokens
    masked_prompt = "The key token is a8f9c2d109b8f7e6 and next hash 19b2c3d4e5f6a7b8 with block 99f8e7d6c5b4a321"
    decision = gate.evaluate(masked_prompt)

    assert decision.pass_filter is True
    assert decision.is_anomaly is True
    assert decision.requires_deliberation is True
    assert "Epistemic anomaly" in decision.reason


def test_metacognitive_gate_routing_and_familiarity():
    import numpy as np
    from cognitive_engine.core.saliency import MetacognitiveGate

    gate = MetacognitiveGate(uncertainty_threshold=0.45)

    # 1. Test familiarity evaluation against centroids
    q_known = np.array([1.0, 0.0, 0.0])
    c_known = np.array([0.98, 0.02, 0.0])
    u_known = gate.evaluate_familiarity(q_known, [c_known])
    assert u_known < 0.10

    q_novel = np.array([0.0, 1.0, 0.0])
    u_novel = gate.evaluate_familiarity(q_novel, [c_known])
    assert u_novel > 0.90

    # 2. Test dynamic routing between System 1 and System 2
    r_s1 = gate.route_deliberation("Known factual recall", uncertainty=0.20)
    assert r_s1 == "SYSTEM_1_IMMEDIATE_RECALL"

    r_s2 = gate.route_deliberation("Complex unknown anomaly", uncertainty=0.75)
    assert r_s2 == "SYSTEM_2_DEEP_DELIBERATIVE_THINKING"


if __name__ == "__main__":
    test_low_entropy_gibberish_dropped()
    test_duplicate_text_blocks_dropped()
    test_throughput_benchmark()
    test_high_entropy_random_noise()
    test_semantic_neighborhood_preservation()
    test_semantic_masking_polysemy()
    test_metacognitive_gate_routing_and_familiarity()
    print("All saliency tests passed.")



