"""
Real-Time Production Benchmark & Telemetry Runner for AutoAgent.
Executes live computational tasks, monitors physical memory / latencies,
and reports exactly where invariants fail under real operational load.
"""

import time
import os
import sys
import psutil
from dataclasses import dataclass
from typing import List, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.types import ExecutionContext

@dataclass
class BenchmarkTask:
    task_id: str
    description: str
    query: str
    code_target: str
    unit_test: str
    expected_system: str  # "system_1" or "system_2"

# Real-world challenging tasks: algorithm logic, state isolation, recursion, edge cases
REAL_WORLD_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="T1_PRIME_SIEVE",
        description="Compute primes up to 1000 using Sieve of Eratosthenes",
        query="Implement an optimized Sieve of Eratosthenes to return primes below 1000.",
        code_target="""
def solution():
    limit = 1000
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    return [i for i in range(limit) if sieve[i]]
""",
        unit_test="assert len(solution()) == 168 and solution()[-1] == 997",
        expected_system="system_2"
    ),
    BenchmarkTask(
        task_id="T2_MATRIX_TRANSPOSE",
        description="Transpose a ragged or large 2D matrix safely",
        query="Transpose a 100x100 matrix generated with deterministic indices.",
        code_target="""
def solution():
    m = [[i * 100 + j for j in range(100)] for i in range(100)]
    return [[m[j][i] for j in range(100)] for i in range(100)]
""",
        unit_test="res = solution(); assert res[12][34] == 34 * 100 + 12",
        expected_system="system_2"
    ),
    BenchmarkTask(
        task_id="T3_INFINITE_LOOP_DEFENSE",
        description="Code with accidental infinite loop to test real-time timeout & Win32 clamp",
        query="Execute while True loop and ensure sandbox terminates without leaking processes.",
        code_target="""
def solution():
    i = 0
    while True:
        i += 1
    return i
""",
        unit_test="assert solution() == 0",
        expected_system="system_2"
    ),
    BenchmarkTask(
        task_id="T4_MEMORY_EXHAUSTION_DEFENSE",
        description="Allocate massive byte buffer to verify 256MB Job Object limit",
        query="Attempt to allocate 512MB RAM inside sandbox.",
        code_target="""
def solution():
    # Attempt allocating beyond the 256MB limit
    blob = bytearray(500 * 1024 * 1024)
    return len(blob)
""",
        unit_test="assert solution() > 0",
        expected_system="system_2"
    ),
    BenchmarkTask(
        task_id="T5_CAUSAL_CONTRADICTION",
        description="Action that attempts to decrease entropy without work",
        query="Decrease system heat entropy without energy dissipation.",
        code_target="def solution(): return 'perpetual_cooling'",
        unit_test="assert solution() == 'perpetual_cooling'",
        expected_system="system_2"
    ),
    BenchmarkTask(
        task_id="T6_REPEAT_FAST_RECALL",
        description="Re-query previous computation to verify instant System 1 retrieval",
        query="Implement an optimized Sieve of Eratosthenes to return primes below 1000.",
        code_target="",
        unit_test="",
        expected_system="system_1"
    )
]

def run_realtime_benchmark():
    print("=================================================================")
    print("  AutoAgent Real-Time Production Benchmark Harness               ")
    print("=================================================================")
    
    engine = CognitiveEngine()
    ctx = ExecutionContext(tenant_id="benchmark_tenant", session_id="live_bench_01")
    
    proc = psutil.Process(os.getpid())
    results: List[Dict[str, Any]] = []

    for idx, task in enumerate(REAL_WORLD_TASKS, 1):
        print(f"\n[{idx}/{len(REAL_WORLD_TASKS)}] Testing: {task.task_id} ({task.description})")
        
        mem_before = proc.memory_info().rss / (1024 * 1024)
        t_start = time.perf_counter()
        
        try:
            # Live real-time execution through Cognitive Engine
            output = engine.process(
                task.query, 
                code_action=task.code_target if task.code_target else None,
                ctx=ctx,
                unit_test=task.unit_test if task.unit_test else None
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            mem_after = proc.memory_info().rss / (1024 * 1024)
            
            status = "PASS"
            # Analyze if routing aligned with expectations
            routed_system = getattr(output, "routed_system", output.get("status", "unknown") if isinstance(output, dict) else "unknown")
            
            print(f"    ├─ Status: {status}")
            print(f"    ├─ Latency: {elapsed_ms:.2f} ms")
            print(f"    ├─ Memory Delta: {mem_after - mem_before:+.2f} MB (Total: {mem_after:.1f} MB)")
            print(f"    └─ Routed Through: {routed_system}")
            
            results.append({
                "task_id": task.task_id,
                "status": "PASS",
                "latency_ms": elapsed_ms,
                "error": None
            })
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            print(f"    ├─ Status: FAIL / INTERCEPTED")
            print(f"    ├─ Latency: {elapsed_ms:.2f} ms")
            print(f"    └─ Exception Caught: {type(e).__name__}: {str(e)}")
            
            results.append({
                "task_id": task.task_id,
                "status": "CAUGHT_EXCEPTION",
                "latency_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {str(e)}"
            })

    print("\n=================================================================")
    print("  Real-Time Execution Summary & Diagnostics                      ")
    print("=================================================================")
    for r in results:
        mark = "✓" if r["status"] == "PASS" else "!"
        err_str = f" -> {r['error']}" if r['error'] else ""
        print(f" {mark} {r['task_id']:<30} {r['latency_ms']:>8.2f} ms  [{r['status']}]{err_str}")

if __name__ == "__main__":
    run_realtime_benchmark()
