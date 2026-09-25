"""
Open Algorithmic Programming Benchmark Runner (HumanEval / Code Grounding style).
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_engine.agent.executive_loop import ExecutiveLoop
from cognitive_engine.core.sandbox import EnvironmentalSandbox


def get_canonical_algorithmic_benchmarks() -> List[Dict[str, Any]]:
    return [
        {
            "name": "is_palindrome",
            "code": "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n",
            "assertions": [
                "assert is_palindrome('A man a plan a canal Panama') is True",
                "assert is_palindrome('race a car') is False",
                "assert is_palindrome('') is True",
            ],
        },
        {
            "name": "binary_search",
            "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n",
            "assertions": [
                "assert binary_search([1, 2, 3, 4, 5], 3) == 2",
                "assert binary_search([1, 2, 3, 4, 5], 6) == -1",
                "assert binary_search([], 1) == -1",
            ],
        },
        {
            "name": "sieve_of_eratosthenes",
            "code": "def primes_up_to(n):\n    if n < 2: return []\n    sieve = [True] * (n + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, n + 1, i):\n                sieve[j] = False\n    return [i for i, p in enumerate(sieve) if p]\n",
            "assertions": [
                "assert primes_up_to(10) == [2, 3, 5, 7]",
                "assert primes_up_to(1) == []",
                "assert len(primes_up_to(20)) == 8",
            ],
        },
        {
            "name": "two_sum",
            "code": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        diff = target - n\n        if diff in seen: return [seen[diff], i]\n        seen[n] = i\n    return []\n",
            "assertions": [
                "assert two_sum([2, 7, 11, 15], 9) == [0, 1]",
                "assert two_sum([3, 2, 4], 6) == [1, 2]",
                "assert two_sum([3, 3], 6) == [0, 1]",
            ],
        },
    ]


def run_algorithmic_eval() -> Dict[str, Any]:
    sandbox = EnvironmentalSandbox()
    loop = ExecutiveLoop(sandbox=sandbox)
    tasks = get_canonical_algorithmic_benchmarks()

    print(f"Running Open Algorithmic Benchmark across {len(tasks)} programming tasks...")
    results = []
    t_start = time.perf_counter()

    for t in tasks:
        res = loop.execute_algorithmic_task(
            task_name=t["name"],
            solution_code=t["code"],
            test_assertions=t["assertions"],
        )
        results.append(res)

    elapsed = time.perf_counter() - t_start
    total = len(tasks)
    passed = sum(1 for r in results if r["success"])

    print("\n============== Open Algorithmic Benchmark Results ==============")
    print(f"Total Tasks:        {total}")
    print(f"Passed:             {passed}")
    print(f"Success Rate:       {(passed / total) * 100:.2f}%")
    print(f"Total Latency:      {elapsed * 1000:.2f} ms")
    print(f"Mean per Task:      {(elapsed * 1000) / total:.2f} ms")
    print("===============================================================\n")

    return {
        "total": total,
        "passed": passed,
        "success_rate": passed / total,
        "elapsed_sec": round(elapsed, 4),
        "results": results,
    }


if __name__ == "__main__":
    run_algorithmic_eval()
