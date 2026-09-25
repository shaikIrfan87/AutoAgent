import pytest
from cognitive_engine.core.combinator_dsl import (
    TInt, TBool, TList, TFunc, TVar, unify, apply_subst,
    CVar, CConst, CPrim, CApp, eval_comb, infer_comb_type,
    TypeDirectedCombinatorSynthesizer
)


def test_hindley_milner_unification():
    """Verify Hindley-Milner type inference, unification, and occurs-check."""
    t1 = TVar("a")
    t2 = TInt()
    s = unify(t1, t2)
    assert s is not None
    assert apply_subst(t1, s) == TInt()

    # Function unification: a -> Int with Bool -> b  =>  Bool -> Int
    f1 = TFunc(t1, TInt())
    f2 = TFunc(TBool(), TVar("b"))
    s2 = unify(f1, f2)
    assert s2 is not None
    assert apply_subst(f1, s2) == TFunc(TBool(), TInt())

    # Occurs-check failure: a unifies with [a] => None (infinite type prevented)
    s3 = unify(t1, TList(t1))
    assert s3 is None


def test_combinator_evaluation():
    """Verify SKIBCY standard combinator reduction."""
    # I 42 == 42
    assert eval_comb(CApp(CPrim("I"), CConst(42))) == 42

    # K 10 20 == 10
    assert eval_comb(CApp(CApp(CPrim("K"), CConst(10)), CConst(20))) == 10

    # B add add 5 2 3 == add (add 5 2) 3
    # B: \f g x. f (g x)
    # B (* 2) (+ 1) 3 => 2 * (3 + 1) = 8
    expr = CApp(CApp(CApp(CPrim("B"), CApp(CPrim("mul"), CConst(2))), CApp(CPrim("add"), CConst(1))), CConst(3))
    assert eval_comb(expr) == 8


def test_combinator_synthesis_recursive_algorithms():
    """Verify synthesis of recursive algorithms: QuickSort & Reverse."""
    synth = TypeDirectedCombinatorSynthesizer()

    # 1. Recursive list reversal
    rev_examples = [
        ([1, 2, 3], [3, 2, 1]),
        ([10, 20], [20, 10]),
        ([], []),
    ]
    res_rev = synth.synthesize(rev_examples)
    assert res_rev is not None
    ast_rev, code_rev = res_rev
    assert "def reverse_fn" in code_rev
    assert "lst[1:]" in code_rev

    # Execute synthesized code dynamically to verify zero syntax errors
    scope = {}
    exec(code_rev, scope)
    fn = scope["reverse_fn"]
    assert fn([1, 2, 3]) == [3, 2, 1]
    assert fn([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    # 2. Recursive QuickSort
    sort_examples = [
        ([3, 1, 2], [1, 2, 3]),
        ([9, 5, 8, 1], [1, 5, 8, 9]),
        ([], []),
    ]
    res_sort = synth.synthesize(sort_examples)
    assert res_sort is not None
    ast_sort, code_sort = res_sort
    assert "def quicksort" in code_sort

    scope_sort = {}
    exec(code_sort, scope_sort)
    sort_fn = scope_sort["quicksort"]
    assert sort_fn([10, -2, 4, 3]) == [-2, 3, 4, 10]
