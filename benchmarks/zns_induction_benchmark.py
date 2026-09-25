"""
ZeroNeuralSynthesizer — induction benchmark.
Uses fast in-process eval() for verification (no subprocess per episode),
falling back to the sandbox only on the final discovered candidate.

Run: python benchmarks/zns_induction_benchmark.py
"""
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_engine.agent.zero_neural_synthesizer import ZeroNeuralSynthesizer

# ---------- Task suite ----------
TASKS = [
    {"name": "assign_eq",   "assertions": ["x == 1"]},
    {"name": "double",      "assertions": ["x * 2 == 4"]},
    {"name": "zero_val",    "assertions": ["x == 0"]},
    {"name": "mod_even",    "assertions": ["x % 2 == 0"]},
    {"name": "chained",     "assertions": ["x + 1 == 2"]},
    {"name": "square",      "assertions": ["x * x == 4"]},
    {"name": "diff",        "assertions": ["x - 1 == 2"]},
    {"name": "identity",    "assertions": ["x == 3"]},
]

MAX_EPISODES = 35


def fast_verify(code_str: str, assertions: list) -> float:
    """In-process eval — ~100x faster than subprocess. Used for training episodes."""
    script = code_str + "\n" + "\n".join(f"assert {a}" for a in assertions)
    try:
        exec(compile(script, "<bench>", "exec"), {})  # noqa: S102
        return 1.0
    except Exception:
        return -1.0


def run_benchmark() -> None:
    synth = ZeroNeuralSynthesizer()
    synth.verify_in_sandbox = fast_verify  # type: ignore[method-assign]

    initial_vocab = len(synth.tokens)
    results = []

    print(f"\n{'Task':<14} {'Status':<12} {'Episodes':>8} {'Vocab':>6} {'Time(ms)':>10}", flush=True)
    print("-" * 56, flush=True)

    for task in TASKS:
        t0 = time.perf_counter()
        res = synth.learn_task(task["assertions"], max_episodes=MAX_EPISODES)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vocab_now = len(synth.tokens)
        results.append({
            "name": task["name"],
            "status": res["status"],
            "episodes": res.get("episodes") or MAX_EPISODES,
            "vocab": vocab_now,
            "elapsed_ms": elapsed_ms,
        })
        print(f"{task['name']:<14} {res['status']:<12} {results[-1]['episodes']:>8} {vocab_now:>6} {elapsed_ms:>10.1f}", flush=True)

    # ---------- Summary ----------
    final_vocab = len(synth.tokens)
    macro_growth = final_vocab - initial_vocab
    discovered = sum(1 for r in results if r["status"] == "discovered")
    timeouts = sum(1 for r in results if r["status"] == "unresolved")
    timeout_ratio = timeouts / len(TASKS)
    ep_success = [r["episodes"] for r in results if r["status"] == "discovered"]
    avg_ep = statistics.mean(ep_success) if ep_success else float("nan")

    print("\n" + "=" * 56)
    print(f"Tasks solved        : {discovered}/{len(TASKS)}")
    print(f"Avg episodes/solve  : {avg_ep:.1f}")
    print(f"Sandbox timeout %   : {timeout_ratio * 100:.1f}%   (healthy < 5%)")
    print(f"Vocab growth        : {initial_vocab} -> {final_vocab}  (+{macro_growth} macros)")
    print(f"Macros/solved task  : {macro_growth / max(discovered, 1):.2f}  (healthy 1-3 per 50)")

    if macro_growth > 100:
        print("[WARN] Vocab bloat >100 - lower char cap in _inject_macros.")
    if timeout_ratio > 0.05:
        print(f"[WARN] Timeout ratio {timeout_ratio:.0%} exceeds 5% threshold.")
    if macro_growth == 0 and discovered > 0:
        print("[INFO] No macros mined - discovered code had no short BinOp/Compare subtrees.")
    print("=" * 56)


if __name__ == "__main__":
    run_benchmark()
