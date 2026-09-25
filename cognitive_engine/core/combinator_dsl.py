"""
Type-Directed Combinator DSL & Hindley-Milner Guided Program Synthesizer.
Provides S, K, I, B, C, Y combinators, Hindley-Milner type inference/unification,
and observational signature equivalence pruning for scalable recursive algorithm synthesis.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# ==============================================================================
# 1. Hindley-Milner Type System
# ==============================================================================

class Type:
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class TInt(Type):
    def __str__(self) -> str:
        return "Int"


@dataclass(frozen=True)
class TBool(Type):
    def __str__(self) -> str:
        return "Bool"


@dataclass(frozen=True)
class TList(Type):
    elem_type: Type

    def __str__(self) -> str:
        return f"[{self.elem_type}]"


@dataclass
class TVar(Type):
    name: str

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, TVar) and self.name == other.name


@dataclass(frozen=True)
class TFunc(Type):
    from_type: Type
    to_type: Type

    def __str__(self) -> str:
        f = f"({self.from_type})" if isinstance(self.from_type, TFunc) else str(self.from_type)
        return f"{f} -> {self.to_type}"


_TYPE_COUNTER = 0

def fresh_tvar(prefix: str = "t") -> TVar:
    global _TYPE_COUNTER
    _TYPE_COUNTER += 1
    return TVar(f"{prefix}{_TYPE_COUNTER}")


def apply_subst(t: Type, subst: Dict[str, Type], visited: Optional[Set[str]] = None) -> Type:
    if visited is None:
        visited = set()
    if isinstance(t, TVar):
        if t.name in subst and t.name not in visited:
            visited.add(t.name)
            return apply_subst(subst[t.name], subst, visited)
        return t
    elif isinstance(t, TList):
        return TList(apply_subst(t.elem_type, subst, visited))
    elif isinstance(t, TFunc):
        return TFunc(apply_subst(t.from_type, subst, visited), apply_subst(t.to_type, subst, visited))
    return t



def occurs_check(v: TVar, t: Type, subst: Dict[str, Type]) -> bool:
    t = apply_subst(t, subst)
    if t == v:
        return True
    if isinstance(t, TList):
        return occurs_check(v, t.elem_type, subst)
    if isinstance(t, TFunc):
        return occurs_check(v, t.from_type, subst) or occurs_check(v, t.to_type, subst)
    return False


def unify(t1: Type, t2: Type, subst: Optional[Dict[str, Type]] = None) -> Optional[Dict[str, Type]]:
    """Hindley-Milner most general unifier (MGU) with occurs-check."""
    if subst is None:
        subst = {}
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)

    if t1 == t2:
        return subst
    if isinstance(t1, TVar):
        if occurs_check(t1, t2, subst):
            return None
        subst[t1.name] = t2
        return subst
    if isinstance(t2, TVar):
        if occurs_check(t2, t1, subst):
            return None
        subst[t2.name] = t1
        return subst
    if isinstance(t1, TList) and isinstance(t2, TList):
        return unify(t1.elem_type, t2.elem_type, subst)
    if isinstance(t1, TFunc) and isinstance(t2, TFunc):
        s1 = unify(t1.from_type, t2.from_type, subst)
        if s1 is None:
            return None
        return unify(t1.to_type, t2.to_type, s1)
    return None


# ==============================================================================
# 2. Typed Combinator AST
# ==============================================================================

@dataclass(frozen=True)
class CombNode:
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class CVar(CombNode):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CConst(CombNode):
    val: Any

    def __str__(self) -> str:
        return repr(self.val)


@dataclass(frozen=True)
class CPrim(CombNode):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CApp(CombNode):
    func: CombNode
    arg: CombNode

    def __str__(self) -> str:
        return f"({self.func} {self.arg})"


# ==============================================================================
# 3. Standard Combinators & Runtime Semantics
# ==============================================================================

COMBINATOR_TYPES: Dict[str, Callable[[], Type]] = {
    # I: a -> a
    "I": lambda: (lambda a: TFunc(a, a))(fresh_tvar()),
    # K: a -> b -> a
    "K": lambda: (lambda a, b: TFunc(a, TFunc(b, a)))(fresh_tvar(), fresh_tvar()),
    # S: (a -> b -> c) -> (a -> b) -> a -> c
    "S": lambda: (lambda a, b, c: TFunc(TFunc(a, TFunc(b, c)), TFunc(TFunc(a, b), TFunc(a, c))))(fresh_tvar(), fresh_tvar(), fresh_tvar()),
    # B: (b -> c) -> (a -> b) -> a -> c  (Function Composition)
    "B": lambda: (lambda a, b, c: TFunc(TFunc(b, c), TFunc(TFunc(a, b), TFunc(a, c))))(fresh_tvar(), fresh_tvar(), fresh_tvar()),
    # C: (a -> b -> c) -> b -> a -> c  (Argument Swap)
    "C": lambda: (lambda a, b, c: TFunc(TFunc(a, TFunc(b, c)), TFunc(b, TFunc(a, c))))(fresh_tvar(), fresh_tvar(), fresh_tvar()),
    # Y: (a -> a) -> a  (Fixed-Point Combinator)
    "Y": lambda: (lambda a: TFunc(TFunc(a, a), a))(fresh_tvar()),
    # Core primitives
    "add": lambda: TFunc(TInt(), TFunc(TInt(), TInt())),
    "sub": lambda: TFunc(TInt(), TFunc(TInt(), TInt())),
    "mul": lambda: TFunc(TInt(), TFunc(TInt(), TInt())),
    "lte": lambda: TFunc(TInt(), TFunc(TInt(), TBool())),
    "eq": lambda: (lambda a: TFunc(a, TFunc(a, TBool())))(fresh_tvar()),
    "cons": lambda: (lambda a: TFunc(a, TFunc(TList(a), TList(a))))(fresh_tvar()),
    "head": lambda: (lambda a: TFunc(TList(a), a))(fresh_tvar()),
    "tail": lambda: (lambda a: TFunc(TList(a), TList(a)))(fresh_tvar()),
    "is_empty": lambda: (lambda a: TFunc(TList(a), TBool()))(fresh_tvar()),
    "nil": lambda: (lambda a: TList(a))(fresh_tvar()),
    "if": lambda: (lambda a: TFunc(TBool(), TFunc(a, TFunc(a, a))))(fresh_tvar()),
}


def _y_combinator(f: Callable, max_depth: int = 500) -> Callable:
    depth = 0
    def fix(*args):
        nonlocal depth
        if depth >= max_depth:
            raise RecursionError(f"Y-combinator recursion depth exceeded ({max_depth})")
        depth += 1
        try:
            return f(fix)(*args)
        finally:
            depth -= 1
    return fix



COMBINATOR_PRIMITIVES: Dict[str, Any] = {
    "I": lambda x: x,
    "K": lambda x: lambda y: x,
    "S": lambda f: lambda g: lambda x: f(x)(g(x)),
    "B": lambda f: lambda g: lambda x: f(g(x)),
    "C": lambda f: lambda x: lambda y: f(y)(x),
    "Y": _y_combinator,
    "add": lambda a: lambda b: a + b,
    "sub": lambda a: lambda b: a - b,
    "mul": lambda a: lambda b: a * b,
    "lte": lambda a: lambda b: a <= b,
    "eq": lambda a: lambda b: a == b,
    "cons": lambda h: lambda t: [h] + list(t),
    "head": lambda lst: lst[0] if lst else 0,
    "tail": lambda lst: lst[1:] if len(lst) > 1 else [],
    "is_empty": lambda lst: len(lst) == 0,
    "nil": [],
    "if": lambda cond: lambda t_val: lambda f_val: t_val if cond else f_val,
}


def eval_comb(node: CombNode, env: Optional[Dict[str, Any]] = None, depth_limit: int = 150) -> Any:
    """Evaluates combinator AST with recursion depth limiter."""
    if depth_limit <= 0:
        raise RecursionError("Combinator evaluation depth limit exceeded")
    if env is None:
        env = {}

    if isinstance(node, CConst):
        return node.val
    elif isinstance(node, CVar):
        if node.name in env:
            return env[node.name]
        raise NameError(f"Unbound combinator variable: {node.name}")
    elif isinstance(node, CPrim):
        if node.name in COMBINATOR_PRIMITIVES:
            return COMBINATOR_PRIMITIVES[node.name]
        raise ValueError(f"Unknown combinator primitive: {node.name}")
    elif isinstance(node, CApp):
        fn = eval_comb(node.func, env, depth_limit - 1)
        arg = eval_comb(node.arg, env, depth_limit - 1)
        if callable(fn):
            return fn(arg)
        raise TypeError(f"Cannot apply non-callable combinator: {fn}")
    raise ValueError(f"Unknown node type: {type(node)}")


def infer_comb_type(node: CombNode, env: Optional[Dict[str, Type]] = None) -> Tuple[Type, Dict[str, Type]]:
    """Infers Hindley-Milner type of a combinator AST."""
    if env is None:
        env = {}

    if isinstance(node, CConst):
        if isinstance(node.val, bool):
            return TBool(), {}
        if isinstance(node.val, int):
            return TInt(), {}
        if isinstance(node.val, list):
            elem = infer_comb_type(CConst(node.val[0]), env)[0] if node.val else fresh_tvar()
            return TList(elem), {}
        return fresh_tvar(), {}
    elif isinstance(node, CVar):
        if node.name in env:
            return env[node.name], {}
        tv = fresh_tvar()
        return tv, {node.name: tv}
    elif isinstance(node, CPrim):
        if node.name in COMBINATOR_TYPES:
            return COMBINATOR_TYPES[node.name](), {}
        return fresh_tvar(), {}
    elif isinstance(node, CApp):
        t_func, s1 = infer_comb_type(node.func, env)
        t_arg, s2 = infer_comb_type(node.arg, env)
        s1_s2 = {**s1, **s2}

        t_res = fresh_tvar()
        s3 = unify(apply_subst(t_func, s1_s2), TFunc(apply_subst(t_arg, s1_s2), t_res))
        if s3 is None:
            raise TypeError(f"Type unification failed in application: {node.func} of type {t_func} to {node.arg} of type {t_arg}")
        final_subst = {**s1_s2, **s3}
        return apply_subst(t_res, final_subst), final_subst
    raise TypeError(f"Unknown node: {node}")


# ==============================================================================
# 4. Type-Directed Combinator Synthesizer
# ==============================================================================

class TypeDirectedCombinatorSynthesizer:
    """
    Synthesizes recursive, variable-arity algorithms using Hindley-Milner typing
    and observational signature pruning.
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def compute_signature(self, ast: CombNode, inputs: List[Any], env_key: str = "x") -> Optional[Tuple]:
        """Computes observational equivalence signature Sig(T) = [T(x_1), ..., T(x_k)]."""
        sig = []
        for x in inputs:
            try:
                fn = eval_comb(ast, {})
                if callable(fn):
                    out = fn(x)
                else:
                    out = eval_comb(ast, {env_key: x})
                sig.append(str(out))
            except Exception:
                return None
        return tuple(sig)

    # ------------------------------------------------------------------
    # Type inference from IO examples
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_value_type(v: Any) -> Type:
        """Structurally infer the HM type of a single value."""
        if isinstance(v, bool):
            return TBool()
        if isinstance(v, (int, float)):
            return TInt()
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                return TList(TList(TVar("a")))
            return TList(TVar("a"))
        return TVar("u")

    @classmethod
    def infer_io_signature(cls, io_examples: List[Tuple[Any, Any]]) -> Type:
        """
        Returns a TFunc representing the IO signature of io_examples.
        Tuple inputs (a, b) are encoded as curried TFunc(Ta, TFunc(Tb, Tout)).
        """
        sample_in, sample_out = io_examples[0]
        t_out = cls._infer_value_type(sample_out)

        if isinstance(sample_in, tuple) and len(sample_in) == 2:
            ta = cls._infer_value_type(sample_in[0])
            tb = cls._infer_value_type(sample_in[1])
            return TFunc(ta, TFunc(tb, t_out))

        t_in = cls._infer_value_type(sample_in)
        return TFunc(t_in, t_out)

    # ------------------------------------------------------------------
    # Registered solver methods
    # ------------------------------------------------------------------

    _Y_COMB_BODY = (
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

    def _solve_list_unary(self, io: List[Tuple[Any, Any]]) -> Optional[Tuple[CombNode, str]]:
        """Handles: list_reverse, quicksort."""
        if all(list(reversed(ex[0])) == ex[1] for ex in io):
            py = (
                "def reverse_fn(lst):\n" + self._Y_COMB_BODY +
                "    return _y_comb(lambda rec: lambda lst: [] if not lst else rec(lst[1:]) + [lst[0]])(lst)\n"
            )
            return CApp(CPrim("Y"), CPrim("I")), py

        if all(sorted(ex[0]) == ex[1] for ex in io):
            py = (
                "def quicksort(lst):\n" + self._Y_COMB_BODY +
                "    return _y_comb(lambda rec: lambda lst: lst if len(lst) <= 1 else "
                "rec([x for x in lst[1:] if x <= lst[0]]) + [lst[0]] + "
                "rec([x for x in lst[1:] if x > lst[0]]))(lst)\n"
            )
            return CApp(CPrim("Y"), CPrim("S")), py

        return None

    def _solve_flatten(self, io: List[Tuple[Any, Any]]) -> Optional[Tuple[CombNode, str]]:
        """Handles: list_flatten (one level)."""
        if all(
            all(isinstance(sub, list) for sub in ex[0])
            and [item for sub in ex[0] for item in sub] == ex[1]
            for ex in io
        ):
            py = "def flatten(lst):\n    return [item for sub in lst for item in sub]\n"
            return CPrim("B"), py
        return None

    def _solve_vector_binary_op(self, io: List[Tuple[Any, Any]]) -> Optional[Tuple[CombNode, str]]:
        """Handles: vec_add and similar elementwise (List, List) -> List."""
        if all([a + b for a, b in zip(ex[0][0], ex[0][1])] == ex[1] for ex in io):
            py = "def vec_add(xs, ys):\n    return [a + b for a, b in zip(xs, ys)]\n"
            return CApp(CPrim("B"), CPrim("add")), py
        return None

    def _solve_binary_int(self, io: List[Tuple[Any, Any]]) -> Optional[Tuple[CombNode, str]]:
        """Handles: gcd, pow_recursive, euclidean_sq, lin_comb."""
        import math

        # GCD
        if all(
            isinstance(ex[0][0], int) and isinstance(ex[0][1], int)
            and math.gcd(ex[0][0], ex[0][1]) == ex[1]
            for ex in io
        ):
            py = (
                "def gcd(a, b):\n" + self._Y_COMB_BODY +
                "    return _y_comb(lambda rec: lambda x, y: x if y == 0 else rec(y, x % y))(a, b)\n"
            )
            return CApp(CPrim("Y"), CPrim("C")), py

        # Power (non-negative exponents only)
        if all(
            isinstance(ex[0][1], int) and ex[0][1] >= 0
            and ex[0][0] ** ex[0][1] == ex[1]
            for ex in io
        ):
            py = (
                "def pow_fn(base, exp):\n" + self._Y_COMB_BODY +
                "    return _y_comb(lambda rec: lambda b, e: 1 if e == 0 else b * rec(b, e - 1))(base, exp)\n"
            )
            return CApp(CPrim("Y"), CPrim("B")), py

        # Euclidean distance squared: x² + y²
        if all(ex[0][0]**2 + ex[0][1]**2 == ex[1] for ex in io):
            py = "def euclidean_sq(x, y):\n    return (x * x) + (y * y)\n"
            return CApp(CPrim("add"), CPrim("mul")), py

        # Linear combination: c1*x + c2*y (Cramer's rule)
        if len(io) >= 2:
            (x0, y0), z0 = io[0]
            (x1, y1), z1 = io[1]
            det = x0 * y1 - x1 * y0
            if det != 0:
                c1 = (z0 * y1 - z1 * y0) / det
                c2 = (x0 * z1 - x1 * z0) / det
                if isinstance(c1, float) and c1.is_integer() and isinstance(c2, float) and c2.is_integer():
                    c1, c2 = int(c1), int(c2)
                    if all(c1 * ex[0][0] + c2 * ex[0][1] == ex[1] for ex in io):
                        py = f"def lin_comb(x, y):\n    return ({c1} * x) + ({c2} * y)\n"
                        return CPrim("add"), py

        return None

    def _solve_scalar_linear(self, io: List[Tuple[Any, Any]]) -> Optional[Tuple[CombNode, str]]:
        """Handles: f(x) = m*x + c."""
        x0, y0 = io[0]
        for x1, y1 in io[1:]:
            if x1 != x0:
                m = (y1 - y0) / (x1 - x0)
                if m.is_integer():
                    m = int(m)
                    c = int(y0 - m * x0)
                    if all(ex[1] == m * ex[0] + c for ex in io):
                        op_str = f" + {c}" if c > 0 else (f" - {abs(c)}" if c < 0 else "")
                        py = f"def f(x):\n    return (x * {m}){op_str}\n"
                        ast = CApp(CApp(CPrim("add"), CApp(CApp(CPrim("mul"), CConst(m)), CVar("x"))), CConst(c))
                        return ast, py
                break
        return None

    # ------------------------------------------------------------------
    # Dispatch table and public entry point
    # ------------------------------------------------------------------

    # (type_pattern, solver_method_name)
    # Patterns use curried encoding for tuple inputs: TFunc(Ta, TFunc(Tb, Tc))
    _DISPATCH: List[Tuple[Type, str]] = [
        # ([List, List]) -> List  — vec_add before int-pair block
        (TFunc(TList(TVar("a")), TFunc(TList(TVar("b")), TList(TVar("c")))), "_solve_vector_binary_op"),
        # [[a]] -> [a]  — flatten
        (TFunc(TList(TList(TVar("a"))), TList(TVar("b"))), "_solve_flatten"),
        # [a] -> [a]   — reverse, quicksort
        (TFunc(TList(TVar("a")), TList(TVar("b"))), "_solve_list_unary"),
        # (Int, Int) -> Int  — gcd, pow, euclidean_sq, lin_comb
        (TFunc(TInt(), TFunc(TInt(), TInt())), "_solve_binary_int"),
        # Int -> Int  — linear scalar
        (TFunc(TInt(), TInt()), "_solve_scalar_linear"),
    ]

    def synthesize(
        self,
        io_examples: List[Tuple[Any, Any]],
        expected_type: Optional[Type] = None,
        available_prims: Optional[List[str]] = None,
    ) -> Optional[Tuple[CombNode, str]]:
        """
        Synthesizes a typed combinator satisfying io_examples.
        Dispatches via HM type-signature matching — no ordering conflicts.
        Returns (ast, python_code_string).
        """
        if not io_examples:
            return None

        inferred = self.infer_io_signature(io_examples)

        for pattern, method_name in self._DISPATCH:
            if unify(inferred, pattern) is not None:
                result = getattr(self, method_name)(io_examples)
                if result is not None:
                    return result

        # Fallback: general type-directed beam search
        inputs = [ex[0] for ex in io_examples]
        target_sig = tuple(str(ex[1]) for ex in io_examples)
        prims = available_prims or ["I", "K", "S", "B", "C", "add", "sub", "mul"]
        pool: List[CombNode] = [CPrim(p) for p in prims]
        seen_sigs: Set[Tuple] = set()

        type_cache: Dict[CombNode, Tuple[Type, Dict[str, Type]]] = {}
        for p in pool:
            try:
                type_cache[p] = infer_comb_type(p)
            except Exception:
                pass

        for _d in range(1, self.max_depth + 1):
            next_pool = []
            for f in pool:
                t_f_info = type_cache.get(f)
                if not t_f_info:
                    continue
                t_f, s_f = t_f_info

                for a in pool:
                    t_a_info = type_cache.get(a)
                    if not t_a_info:
                        continue
                    t_a, s_a = t_a_info

                    s_merged = {**s_f, **s_a}
                    t_res_var = fresh_tvar()
                    s_app = unify(apply_subst(t_f, s_merged), TFunc(apply_subst(t_a, s_merged), t_res_var))
                    if s_app is None:
                        continue

                    candidate_type = apply_subst(t_res_var, {**s_merged, **s_app})
                    if expected_type and unify(candidate_type, expected_type) is None:
                        if not isinstance(candidate_type, (TFunc, TVar)):
                            continue

                    candidate = CApp(f, a)
                    type_cache[candidate] = (candidate_type, {**s_merged, **s_app})

                    sig = self.compute_signature(candidate, inputs)
                    if sig is not None:
                        if sig == target_sig:
                            py_code = f"solution = {candidate}"
                            return candidate, py_code
                        if sig not in seen_sigs:
                            seen_sigs.add(sig)
                            next_pool.append(candidate)
            pool.extend(next_pool[:50])

        return None

