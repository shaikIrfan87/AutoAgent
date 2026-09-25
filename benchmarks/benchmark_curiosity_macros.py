import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.dsl import UNARY_PRIMITIVES


def run_curiosity_benchmark(cycles: int = 100):
    print(f"Starting Autotelic Curiosity Benchmark ({cycles} cycles)...")
    engine = CognitiveEngine()
    initial_macros = len(UNARY_PRIMITIVES)

    t0 = time.perf_counter()
    successes = 0
    failures = 0

    for i in range(cycles):
        res = engine.curiosity.step()
        if res:
            delta_val = getattr(res, "delta", 0.0)
            if getattr(res, "success", False) or (delta_val is not None and delta_val > 0.0):
                successes += 1
            else:
                failures += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{cycles}] cycles complete | Succeeded: {successes} | Failed: {failures}")

    elapsed = time.perf_counter() - t0
    final_macros = len(UNARY_PRIMITIVES)
    macros_mined = final_macros - initial_macros

    print("\n=== Curiosity & Compression Benchmark Results ===")
    print(f"Cycles Executed:     {cycles}")
    print(f"Elapsed Time:        {elapsed:.3f} s ({cycles / elapsed:.1f} cycles/sec)")
    print(f"Initial Primitives:  {initial_macros}")
    print(f"Final Primitives:    {final_macros} (+{macros_mined} macros mined)")
    print(f"Synthesized Tasks:   {successes} successes, {failures} failures")
    print(f"Current Search Depth:{engine.synthesizer.max_depth}")


if __name__ == "__main__":
    run_curiosity_benchmark(cycles=100)
