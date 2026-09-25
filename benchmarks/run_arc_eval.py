"""
ARC-400 Evaluation Dataset Streaming Benchmark Runner.
"""
import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_engine.agent.arc_benchmark import ARCBenchmarkHarness
from cognitive_engine.agent.program_synthesizer import MCTSProgramSynthesizer
from cognitive_engine.core.sharded_replay_buffer import ShardedReplayBuffer


def evaluate_arc_split(
    dataset_dir: str = "benchmarks/data/evaluation",
    max_tasks: int = 100,
    max_expansions: int = 300,
    enable_replay_distill: bool = True,
):
    print(f"Loading ARC tasks from: {dataset_dir}")
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        try:
            from .download_arc_data import download_and_extract_arc
        except ImportError:
            from download_arc_data import download_and_extract_arc
        download_and_extract_arc(dataset_dir)

    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        harness = ARCBenchmarkHarness.from_json_file(dataset_dir)
        harness.tasks = harness.tasks[:max_tasks]
    else:
        print(f"[Notice] Directory '{dataset_dir}' not found. Falling back to canonical evaluation suite.")
        harness = ARCBenchmarkHarness()

    from cognitive_engine.core.macro_store import PersistentMacroStore
    from cognitive_engine.core.dsl import UNARY_PRIMITIVES

    macro_store = PersistentMacroStore()
    loaded_macros = macro_store.load_into_dsl()
    if loaded_macros > 0:
        print(f"[DreamCoder] Injected {loaded_macros} persistent macros into active DSL primitives (Total: {len(UNARY_PRIMITIVES)}).")

    synthesizer = MCTSProgramSynthesizer(max_depth=5, max_expansions=max_expansions)
    replay_buffer = ShardedReplayBuffer(num_shards=4, max_shard_capacity=100)

    report = harness.run_benchmark(synthesizer)

    # Trajectory pooling into ShardedReplayBuffer
    if enable_replay_distill:
        for r in report.get("results", []):
            if r["solved_train"] and r["code"]:
                pos_code = r["code"]
                pos_probs = torch.full((5,), 0.8, dtype=torch.float32)
                neg_code = "identity(g)"
                neg_probs = torch.full((5,), 0.2, dtype=torch.float32)
                margin = 2.0 if r["generalized"] else 1.0
                replay_buffer.push(
                    pos_code=pos_code,
                    pos_log_probs=pos_probs,
                    neg_code=neg_code,
                    neg_log_probs=neg_probs,
                    margin=margin,
                    domain="arc_visual_induction",
                )
        report["replay_trajectories_pooled"] = replay_buffer.total_size()

    print("\n================ ARC Evaluation Summary ================")
    print(f"Total Evaluated:      {report['total_tasks']}")
    print(f"Solved (Train):       {report['solved_train']}")
    print(f"Generalized (Test):   {report['generalized_test']}")
    print(f"Generalization Rate:  {report['generalization_rate'] * 100:.2f}%")
    print(f"Mean Latency:         {report['mean_latency_ms']} ms")
    print(f"Elapsed Time:         {report['total_elapsed_sec']} s")
    if enable_replay_distill:
        print(f"Pooled Trajectories:  {report.get('replay_trajectories_pooled', 0)}")
    print("========================================================\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MCTSProgramSynthesizer on ARC dataset.")
    parser.add_argument("--dataset_dir", type=str, default="benchmarks/data/evaluation")
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--max_expansions", type=int, default=300)
    parser.add_argument("--no_distill", action="store_true", help="Disable trajectory pooling")
    args = parser.parse_args()
    evaluate_arc_split(
        dataset_dir=args.dataset_dir,
        max_tasks=args.max_tasks,
        max_expansions=args.max_expansions,
        enable_replay_distill=not args.no_distill,
    )
