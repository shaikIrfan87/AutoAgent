import time
from pathlib import Path
import sys

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cognitive_engine.agent.orchestrator import CognitiveEngine


def run_cognitive_live_test():
    engine = CognitiveEngine()

    print("=========================================================")
    print("STEP 1: Testing Noise Rejection (Saliency)")
    print("=========================================================")
    noisy_input = "buy now cheap pills discount free " * 5
    res = engine.interact(noisy_input)
    print(f"Result: {res}\n")

    print("=========================================================")
    print("STEP 2: High Uncertainty Query -> Deliberation (System 2)")
    print("=========================================================")
    # Query something not in memory to force web/tool search + Python verification
    query_unknown = "Calculate the kinetic energy of a 1500kg car traveling at 28 m/s."
    res = engine.interact(query_unknown)
    print(f"System 2 Output:\n{res}\n")

    print("=========================================================")
    print("STEP 3: Testing Instant Recall (System 1)")
    print("=========================================================")
    # Repeating the exact same context should trigger instant System 1 retrieval
    # because it was absorbed into memory during Step 2.
    query_known = "kinetic energy of 1500kg car at 28 m/s"
    t0 = time.time()
    res_fast = engine.interact(query_known)
    lat = (time.time() - t0) * 1000
    print(f"System 1 Fast Recall ({lat:.2f}ms):\n{res_fast}\n")

    print("=========================================================")
    print("STEP 4: Neuro-Symbolic Causal Guardrail Verification")
    print("=========================================================")
    # System 2 attempts an action that violates physical axioms
    hallucination_attempt = "Verify if a perpetual motion machine generates infinite energy."
    res_causal = engine.interact(hallucination_attempt)
    print(f"Causal Guardrail Response:\n{res_causal}\n")

    # Clean up background sandbox daemons and stores
    engine.sandbox.close()
    engine.consolidation.close()


if __name__ == "__main__":
    run_cognitive_live_test()
