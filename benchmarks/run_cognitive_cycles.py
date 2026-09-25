"""
Continuous benchmark runner profiling autotelic inquiry, library compression, and meta-optimization.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.dsl import UNARY_PRIMITIVES, to_grid


def run_benchmark(cycles: int = 100, verbose: bool = False) -> Dict[str, Any]:
    engine = CognitiveEngine()
    initial_macros = len(UNARY_PRIMITIVES)
    start_time = time.time()

    # Pre-populate synthetic epistemic gaps for continuous exploration
    mock_memory = [
        {"id": f"spatial_concept_{i}", "confidence": 0.45, "last_accessed": 0.0, "content": f"concept {i}"}
        for i in range(min(cycles, 50))
    ]
    engine.autotelic.scan_epistemic_gaps(mock_memory, {})

    successes = 0
    failures = 0
    mutations_applied = 0

    for step_i in range(1, cycles + 1):
        res = engine.curiosity.step()
        if res is not None:
            if getattr(res, "success", False):
                successes += 1
            else:
                failures += 1

        if engine.curiosity.last_optimization and engine.curiosity.last_optimization.applied:
            mutations_applied += 1

        if verbose and step_i % 10 == 0:
            print(f"[Cycle {step_i}/{cycles}] Macros: {len(UNARY_PRIMITIVES)} | "
                  f"Successes: {successes} | Failures: {failures} | "
                  f"Search Depth: {engine.synthesizer.max_depth}")

    elapsed = time.time() - start_time
    final_macros = len(UNARY_PRIMITIVES)
    macros_mined = final_macros - initial_macros

    stats = {
        "cycles": cycles,
        "elapsed_sec": round(elapsed, 2),
        "cycles_per_sec": round(cycles / max(elapsed, 0.001), 2),
        "initial_macros": initial_macros,
        "final_macros": final_macros,
        "macros_mined": macros_mined,
        "successes": successes,
        "failures": failures,
        "current_search_depth": engine.synthesizer.max_depth,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAgent AGI 4-Pillar Continuous Runner")
    parser.add_argument("--cycles", type=int, default=100, help="Number of idle cycles to simulate")
    parser.add_argument("--verbose", action="store_true", help="Print progress telemetry")
    args = parser.parse_args()

    results = run_benchmark(cycles=args.cycles, verbose=args.verbose)
    print("\n=== Benchmark Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
