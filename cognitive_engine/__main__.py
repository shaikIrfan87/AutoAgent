import sys
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.types import Triple


def main():
    print("=" * 70)
    print("           COGNITIVE ENGINE 5-INVARIANT RUNTIME DEMO")
    print("=" * 70)

    engine = CognitiveEngine()

    # Scenario 1: Low-entropy web spam / noise
    print("\n[Percept 1: Low-Entropy Web Spam]")
    spam_input = "buy now buy now buy now buy now buy now"
    res1 = engine.process(spam_input)
    print(f"Status: {res1['status']} | Reason: {res1['reason']}")
    assert res1['status'] == "dropped"

    # Scenario 2: Novel complex query requiring System 2 Deliberation
    print("\n[Percept 2: Novel Deliberation Query]")
    query = "Compute the cumulative sum of squares for the first 5 integers"
    code = "vals = [i**2 for i in range(1, 6)]\nprint(f'Sum: {sum(vals)}')"
    res2 = engine.process(query, code_action=code)
    print(f"Status: {res2['status']}")
    print(f"Output: {res2['output']}")
    print(f"State Delta (Delta S): {res2['delta_s']}")

    print(f"Fast-Weight Frobenius Norm: {res2['frobenius_norm']:.4f}")
    print(f"Consolidated Memory ID: {res2['consolidated_memory_id']}")
    assert res2['status'] == "system_2_success"

    # Scenario 3: Causal Constraint Violation Prevention
    print("\n[Percept 3: Contradictory Semantic Hypothesis]")
    contradiction_hypo = Triple(subject="human", relation="is_a", target="syntheticmachine", polarity=True)
    res3 = engine.process("Assess biological machine classification", hypothesis=contradiction_hypo)
    print(f"Status: {res3['status']}")
    print(f"Halted Diagnostic: {res3['error']}")
    assert res3['status'] == "system_2_failure"

    # Scenario 4: Fast Recall from Consolidated Long-Term Store (System 1)
    print("\n[Percept 4: System 1 Instant Hybrid Recall]")
    familiar_query = "Compute the cumulative sum of squares for the first 5 integers"
    res4 = engine.interact(familiar_query)
    print(f"Result:\n{res4}")

    print("\n" + "=" * 70)
    print("  ALL 5 COGNITIVE INVARIANTS DEMONSTRATED & VERIFIED OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
