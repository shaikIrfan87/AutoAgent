"""
Standardized Algorithmic Test Suite for TypeDirectedCombinatorSynthesizer.
Benchmarks inductive synthesis across fundamental algorithmic paradigms:
- Reversal
- QuickSort
- Greatest Common Divisor (GCD)
- Flattening nested sequences
- Factorial / Fold product
"""
import pytest
from cognitive_engine.core.combinator_dsl import TypeDirectedCombinatorSynthesizer


BENCHMARK_SUITE = {
    "list_reverse": [
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([42], [42]),
        ([], []),
        ([7, 8, 9], [9, 8, 7]),
    ],
    "quicksort": [
        ([5, 2, 8, 1, 9], [1, 2, 5, 8, 9]),
        ([3, 3, 1], [1, 3, 3]),
        ([-5, 0, -10], [-10, -5, 0]),
        ([], []),
    ],
    "gcd": [
        ((48, 18), 6),
        ((101, 10), 1),
        ((54, 24), 6),
        ((100, 25), 25),
    ],
    "list_flatten": [
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([[1], [], [2, 3]], [1, 2, 3]),
        ([], []),
    ],
    # Multi-argument paradigms
    "linear_f": [
        (1, 3),
        (2, 5),
        (4, 9),
        (0, 1),
        (10, 21),
    ],
    "pow_recursive": [
        ((2, 3), 8),
        ((3, 4), 81),
        ((5, 0), 1),
        ((2, 10), 1024),
    ],
    "euclidean_sq": [
        ((3, 4), 25),
        ((0, 5), 25),
        ((1, 1), 2),
        ((6, 8), 100),
    ],
    "lin_comb": [
        ((1, 1), 7),
        ((2, 0), 6),
        ((0, 2), 8),
        ((1, 2), 11),
    ],
    "vec_add": [
        (([1, 2, 3], [4, 5, 6]), [5, 7, 9]),
        (([0, 0], [1, 1]), [1, 1]),
        (([-1, 2], [1, -2]), [0, 0]),
    ],
}



def test_standardized_algorithmic_benchmark():
    synth = TypeDirectedCombinatorSynthesizer()
    results = {}

    for task_name, io_pairs in BENCHMARK_SUITE.items():
        res = synth.synthesize(io_pairs)
        assert res is not None, f"Failed to synthesize {task_name}"
        ast_repr, py_code = res
        assert py_code and len(py_code.strip()) > 0

        # Dynamic execution check
        scope = {}
        exec(py_code, scope)
        fn_names = [k for k in scope if callable(scope[k]) and not k.startswith("__")]
        assert len(fn_names) > 0, f"No callable function generated for {task_name}"
        fn = scope[fn_names[0]]

        # Verify on all test pairs
        for inp, expected in io_pairs:
            if isinstance(inp, tuple):
                actual = fn(*inp)
            else:
                actual = fn(inp)
            assert actual == expected, f"{task_name} failed on {inp}: expected {expected}, got {actual}"

        results[task_name] = "PASSED"

    assert len(results) == len(BENCHMARK_SUITE)
