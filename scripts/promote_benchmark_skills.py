"""
Synthesize all 9 benchmark paradigms and commit each through HostCommitGate
into skills/default_tenant/.  Run from the project root.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognitive_engine.core.combinator_dsl import TypeDirectedCombinatorSynthesizer
from cognitive_engine.core.host_commit_gate import HostCommitGate

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills", "default_tenant")

_Y = (
    "    def _y_comb(f, max_depth=500):\n"
    "        depth = 0\n"
    "        def fix(*args):\n"
    "            nonlocal depth\n"
    "            if depth >= max_depth: raise RecursionError('Y recursion limit')\n"
    "            depth += 1\n"
    "            try: return f(fix)(*args)\n"
    "            finally: depth -= 1\n"
    "        return fix\n"
)

BENCHMARK_IO = {
    "list_reverse": [([1, 2, 3, 4], [4, 3, 2, 1]), ([42], [42]), ([], [])],
    "quicksort":    [([5, 2, 8, 1], [1, 2, 5, 8]), ([3, 3, 1], [1, 3, 3]), ([], [])],
    "gcd":          [((48, 18), 6), ((101, 10), 1), ((54, 24), 6)],
    "list_flatten": [([[1, 2], [3, 4]], [1, 2, 3, 4]), ([[1], [], [2, 3]], [1, 2, 3])],
    "linear_f":     [(1, 3), (2, 5), (4, 9), (0, 1), (10, 21)],
    "pow_recursive":[((2, 3), 8), ((3, 4), 81), ((5, 0), 1), ((2, 10), 1024)],
    "euclidean_sq": [((3, 4), 25), ((0, 5), 25), ((1, 1), 2)],
    "lin_comb":     [((1, 1), 7), ((2, 0), 6), ((0, 2), 8), ((1, 2), 11)],
    "vec_add":      [(([1, 2, 3], [4, 5, 6]), [5, 7, 9]), (([0, 0], [1, 1]), [1, 1])],
}

def promote_all() -> None:
    synth = TypeDirectedCombinatorSynthesizer()
    gate  = HostCommitGate()
    results = []

    for name, io in BENCHMARK_IO.items():
        res = synth.synthesize(io)
        if res is None:
            results.append((name, False, "synthesis returned None"))
            continue

        _, py_code = res
        target = os.path.join(SKILLS_DIR, f"{name}.py")
        ok, reason = gate.attest_and_commit(target, py_code, delta_s=1.0)
        results.append((name, ok, reason))

    # Report
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nSkill promotion: {passed}/{len(results)} committed\n")
    print(f"{'Skill':<20} {'Status':<8} Reason")
    print("-" * 70)
    for name, ok, reason in results:
        status = "OK" if ok else "FAIL"
        print(f"{name:<20} {status:<8} {reason}")

    if passed < len(results):
        sys.exit(1)

if __name__ == "__main__":
    promote_all()
