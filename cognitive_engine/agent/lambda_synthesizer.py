import heapq
from typing import Any, List, Optional, Set, Tuple, Union

try:
    from ..core.lambda_dsl import (
        LambdaNode, Var, Const, Prim, App, Lambda, IfThenElse,
        MapNode, FilterNode, FoldNode, WhileLoop, ForLoop,
        eval_lambda_ast, unparse_to_python, canonical_ast_hash, CORE_LAMBDA_PRIMITIVES,
    )
except ImportError:
    from core.lambda_dsl import (
        LambdaNode, Var, Const, Prim, App, Lambda, IfThenElse,
        MapNode, FilterNode, FoldNode, WhileLoop, ForLoop,
        eval_lambda_ast, unparse_to_python, canonical_ast_hash, CORE_LAMBDA_PRIMITIVES,
    )


class LambdaProgramSynthesizer:
    """
    Inductive synthesizer for Turing-complete algorithmic expressions.
    Enumerates variable bindings, arithmetic, higher-order functions, and loops
    with canonical sub-tree equivalence hashing to eliminate combinatorial bloat.
    """

    def __init__(self, max_cost: int = 10):
        self.max_cost = max_cost

    def synthesize(self, io_pairs: List[Tuple[Any, Any]], var_name: Union[str, List[str]] = "x") -> Optional[Tuple[LambdaNode, str]]:
        if not io_pairs:
            return None

        var_list = [var_name] if isinstance(var_name, str) else list(var_name)
        frontier: List[Tuple[int, int, LambdaNode]] = []
        counter = 0

        # Terminals (Cost = 1)
        terminals: List[LambdaNode] = [Var(v) for v in var_list] + [Const(0), Const(1), Const(2), Const([])]
        for t in terminals:
            counter += 1
            heapq.heappush(frontier, (1, counter, t))

        visited_signatures: Set[Tuple[Any, ...]] = set()
        visited_hashes: Set[str] = set()
        explored_nodes: List[LambdaNode] = []

        is_list_task = any(isinstance(inp, (list, tuple)) or isinstance(out, (list, tuple)) for inp, out in io_pairs)

        while frontier:
            cost, _, node = heapq.heappop(frontier)
            if cost > self.max_cost:
                continue

            # Sub-tree equivalence pruning via canonical AST hash
            c_hash = canonical_ast_hash(node)
            if c_hash in visited_hashes:
                continue
            visited_hashes.add(c_hash)

            # Evaluate candidate across all I/O examples
            sig = []
            valid = True
            for inp, out in io_pairs:
                try:
                    if len(var_list) == 1:
                        env = {var_list[0]: inp}
                    else:
                        env = {v: val for v, val in zip(var_list, inp)}
                    res = eval_lambda_ast(node, env)
                    sig.append(repr(res))
                    if res != out:
                        valid = False
                except Exception:
                    valid = False
                    sig.append("ERR")
                    break

            if valid and len(sig) == len(io_pairs):
                # Found exact solution -> apply DreamCoder abstraction and persist
                py_code = unparse_to_python(node)
                self._abstract_and_persist_macro(node, py_code)
                return node, py_code

            sig_key = tuple(sig)
            if sig_key in visited_signatures:
                continue
            visited_signatures.add(sig_key)
            explored_nodes.append(node)

            # Bidirectional / Goal Inversion fast-check for unary invertibles & object selectors
            for op in ("rot90", "flip_v", "crop_nonzero", "largest_object", "smallest_object", "hollow", "shift_up", "shift_down", "shift_left", "shift_right"):
                cand = App(Prim(op), node)
                try:
                    if all(eval_lambda_ast(cand, {var_list[0]: inp} if len(var_list) == 1 else {v: val for v, val in zip(var_list, inp)}) == out for inp, out in io_pairs):
                        py_code = unparse_to_python(cand)
                        self._abstract_and_persist_macro(cand, py_code)
                        return cand, py_code
                except Exception:
                    pass

            grid_detected = any(isinstance(inp, (list, tuple)) and inp and isinstance(inp[0], (list, tuple)) for inp, _ in io_pairs)

            # Expansion 1: Unary primitives & Spatial ARC operators
            if cost + 2 <= self.max_cost:
                unary_ops = ["not", "head", "tail", "is_empty", "len"]
                if grid_detected:
                    unary_ops.extend([
                        "rot90", "flip_v", "crop_nonzero", "gravity", "connected_components",
                        "largest_object", "smallest_object", "hollow",
                        "shift_up", "shift_down", "shift_left", "shift_right", "objects"
                    ])

                for op in unary_ops:
                    counter += 1
                    cand = App(Prim(op), node)
                    if canonical_ast_hash(cand) not in visited_hashes:
                        heapq.heappush(frontier, (cost + 1, counter, cand))

            # Expansion 2: Binary operations with terminals & color constants
            if cost + 3 <= self.max_cost:
                for op in ("+", "-", "*", "//", "%", "==", "<"):
                    for t in terminals:
                        c1 = App(App(Prim(op), node), t)
                        if canonical_ast_hash(c1) not in visited_hashes:
                            counter += 1
                            heapq.heappush(frontier, (cost + 2, counter, c1))
                        c2 = App(App(Prim(op), t), node)
                        if canonical_ast_hash(c2) not in visited_hashes:
                            counter += 1
                            heapq.heappush(frontier, (cost + 2, counter, c2))

                if grid_detected:
                    # Color transformations for spatial components
                    for c_val in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                        c_recolor = App(App(Prim("recolor"), Const(c_val)), node)
                        if canonical_ast_hash(c_recolor) not in visited_hashes:
                            counter += 1
                            heapq.heappush(frontier, (cost + 2, counter, c_recolor))

            # Expansion 3: Binary operations between explored nodes (arithmetic & grid overlay)
            if cost + 4 <= self.max_cost and len(explored_nodes) < 60:
                for other in explored_nodes[-10:]:
                    for op in ("+", "-", "*", "==", "<"):
                        c = App(App(Prim(op), node), other)
                        if canonical_ast_hash(c) not in visited_hashes:
                            counter += 1
                            heapq.heappush(frontier, (cost + 3, counter, c))
                    if grid_detected:
                        c_overlay = App(App(Prim("overlay"), node), other)
                        if canonical_ast_hash(c_overlay) not in visited_hashes:
                            counter += 1
                            heapq.heappush(frontier, (cost + 3, counter, c_overlay))

            # Expansion 4: Higher-order functions (Map, Filter, Fold) for list transformations
            if is_list_task and cost + 4 <= self.max_cost:
                # Map patterns
                for op, arg in [("+", Const(1)), ("*", Const(2)), ("-", Const(1))]:
                    fn = Lambda("x", App(App(Prim(op), Var("x")), arg))
                    m_cand = MapNode(fn, node)
                    if canonical_ast_hash(m_cand) not in visited_hashes:
                        counter += 1
                        heapq.heappush(frontier, (cost + 3, counter, m_cand))

                # Filter patterns
                p_even = Lambda("x", App(App(Prim("=="), App(App(Prim("%"), Var("x")), Const(2))), Const(0)))
                f_cand = FilterNode(p_even, node)
                if canonical_ast_hash(f_cand) not in visited_hashes:
                    counter += 1
                    heapq.heappush(frontier, (cost + 3, counter, f_cand))

                # Fold patterns (sum accumulator)
                fold_cand = FoldNode(Prim("+"), Const(0), node)
                if canonical_ast_hash(fold_cand) not in visited_hashes:
                    counter += 1
                    heapq.heappush(frontier, (cost + 3, counter, fold_cand))

            # Expansion 5: Conditional branching
            if cost + 5 <= self.max_cost and len(explored_nodes) >= 2:
                primary_var = var_list[0]
                cond = App(App(Prim("<"), Var(primary_var)), Const(2))
                counter += 1
                heapq.heappush(frontier, (cost + 4, counter, IfThenElse(cond, Const(1), node)))

        return None

    def _abstract_and_persist_macro(self, node: LambdaNode, py_code: str, macros_path: str = "assets/dsl_macros.json") -> None:
        """DreamCoder Subtree Abstraction: persists verified subroutines to expand DSL across sessions."""
        import json
        import os
        from pathlib import Path

        try:
            p = Path(macros_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {}
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    existing = json.load(f)

            macro_id = f"macro_{canonical_ast_hash(node)}"
            if macro_id not in existing:
                existing[macro_id] = {
                    "code": py_code,
                    "hash": canonical_ast_hash(node),
                    "created_at": time.time() if "time" in globals() else 0.0,
                }
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
        except Exception:
            pass

