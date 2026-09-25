import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

# Core Architecture Subsystems
from cognitive_engine.core.saliency import SaliencyGate
from cognitive_engine.core.causal_graph import CausalSymbolicGraph, EdgeType
from cognitive_engine.core.sandbox import PersistentREPLDaemon
from cognitive_engine.core.plastic_layer import MultiHeadPlasticLayer
from cognitive_engine.core.consolidation import ConsolidationStore
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer, MCTSProgramSynthesizer
from cognitive_engine.core.dsl import to_grid, UNARY_PRIMITIVES


def benchmark_saliency_throughput() -> Dict[str, Any]:
    gate = SaliencyGate()
    sample_text = "Autonomous neuro-symbolic cognitive runtime executing inductive synthesis."
    warmup_n = 200
    for _ in range(warmup_n):
        _ = gate.evaluate(sample_text)

    test_n = 5000
    t0 = time.perf_counter()
    for _ in range(test_n):
        _ = gate.evaluate(sample_text)
    elapsed = time.perf_counter() - t0
    ops_sec = test_n / elapsed

    return {
        "benchmark": "Invariant 1: Saliency Gate Throughput",
        "iterations": test_n,
        "elapsed_sec": round(elapsed, 4),
        "throughput_ops_sec": round(ops_sec, 2),
        "target_met": ops_sec > 1000.0,
    }


def benchmark_causal_cycle_resolution() -> Dict[str, Any]:
    graph = CausalSymbolicGraph()
    # Construct 10-node cycle: N0 -> N1 -> ... -> N9 -> N0
    for i in range(9):
        graph.add_edge(f"N{i}", "leads_to", f"N{i+1}", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)
    graph.add_edge("N9", "leads_to", "N0", edge_type=EdgeType.STRUCTURAL_TRANSITIVE)

    t0 = time.perf_counter()
    valid, reason = graph.verify_hypothesis("N0", "leads_to", "N5")
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "benchmark": "Invariant 2: Causal Cycle Contradiction Check",
        "latency_ms": round(lat_ms, 4),
        "cycle_handled": valid is not None,
        "target_met": lat_ms < 1.0,
    }


def benchmark_sandbox_ipc_and_rollback() -> Dict[str, Any]:
    daemon = PersistentREPLDaemon()
    daemon.start()
    try:
        # Step 1: Sub-millisecond execution
        t0 = time.perf_counter()
        res_ok = daemon.execute("res = sum(i**2 for i in range(100))\nprint(res)")
        exec_lat_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Namespace rollback on deliberate failure
        res_fail = daemon.execute("dirty_state = 99999\nraise ValueError('deliberate_abort')")
        check_leak = daemon.execute("print('dirty_state' in globals())")

        rolled_back = check_leak.get("output", "").strip() == "False"

        return {
            "benchmark": "Invariant 3: Sandbox Latency & Rollback",
            "ipc_latency_ms": round(exec_lat_ms, 3),
            "output": res_ok.get("output", "").strip(),
            "failure_intercepted": res_fail.get("status") == "failed",
            "namespace_rolled_back": rolled_back,
            "target_met": exec_lat_ms < 15.0 and rolled_back,
        }
    finally:
        daemon.terminate()


def benchmark_plastic_layer_correlated_noise() -> Dict[str, Any]:
    dim = 32
    layer = MultiHeadPlasticLayer(dim=dim, num_heads=4, decay_rate=0.95, eta=0.05)

    # 1. Establish anchor association
    target_k = torch.randn(1, dim)
    target_k = target_k / torch.linalg.norm(target_k)
    target_v = torch.randn(1, dim)
    target_v = target_v / torch.linalg.norm(target_v)
    layer.adapt(target_k, target_v)

    init_recall = layer.recall(target_k)
    init_sim = torch.cosine_similarity(init_recall, target_v).item()

    # 2. Inject 50 correlated noise vectors (cosine overlap ~ 0.40 - 0.60)
    for _ in range(50):
        noise = torch.randn(1, dim)
        proj = torch.dot(noise.squeeze(), target_k.squeeze()) * target_k
        ortho = noise - proj
        ortho = ortho / torch.linalg.norm(ortho)
        correlated_k = 0.5 * target_k + 0.866 * ortho
        layer.adapt(correlated_k, torch.randn(1, dim))

    final_recall = layer.recall(target_k)
    final_sim = torch.cosine_similarity(final_recall, target_v).item()
    norm = torch.norm(layer.A_fast).item()

    return {
        "benchmark": "Invariant 4: Multi-Head Plastic Weight Bounding",
        "initial_cosine": round(init_sim, 4),
        "post_noise_cosine": round(final_sim, 4),
        "frobenius_norm": round(norm, 4),
        "no_nan": not bool(torch.isnan(layer.A_fast).any()),
        "target_met": norm <= 2.0 and final_sim >= 0.50,
    }


def benchmark_arc_program_induction() -> Dict[str, Any]:
    # Multi-step transformation task: rot90 -> flip_h
    in_grid = to_grid([[1, 2], [3, 4]])
    out_grid = to_grid([[3, 1], [4, 2]])  # rot90(flip_h)

    synth = ProgramSynthesizer(max_depth=3)
    t0 = time.perf_counter()
    prog = synth.synthesize([(in_grid, out_grid)])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    solved = prog is not None and prog(in_grid) == out_grid

    return {
        "benchmark": "Pillar 1: Inductive Program Synthesis",
        "synthesis_time_ms": round(elapsed_ms, 2),
        "solved": solved,
        "induced_code": prog.code if prog else None,
        "target_met": solved and elapsed_ms < 100.0,
    }


def run_all_benchmarks():
    print("=" * 65)
    print("       AUTOAGENT COGNITIVE ENGINE BENCHMARK SUITE")
    print("=" * 65)

    benchmarks = [
        benchmark_saliency_throughput,
        benchmark_causal_cycle_resolution,
        benchmark_sandbox_ipc_and_rollback,
        benchmark_plastic_layer_correlated_noise,
        benchmark_arc_program_induction,
    ]

    all_passed = True
    for bm in benchmarks:
        res = bm()
        status = "[PASS]" if res.get("target_met") else "[FAIL]"
        if not res.get("target_met"):
            all_passed = False
        print(f"\n{status} {res['benchmark']}")
        for k, v in res.items():
            if k not in ["benchmark", "target_met"]:
                print(f"      • {k}: {v}")

    print("\n" + "=" * 65)
    print(f"Overall Result: {'ALL BENCHMARKS PASSED' if all_passed else 'SOME BENCHMARKS FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_benchmarks()
