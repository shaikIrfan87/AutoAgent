from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from .dsl import Grid, UNARY_PRIMITIVES
from ..agent.program_synthesizer import SynthesizedProgram


@dataclass
class MacroPrimitive:
    name: str
    operations: Tuple[Tuple[str, Tuple], ...]
    fn: Callable[[Grid], Grid]
    utility_score: float


class ProgramCompressor:
    """Pillar 3: AST Subtree Mining and Library Compression.

    Mines verified synthesis traces from autotelic experiments, extracts
    frequently co-occurring operation subtrees, and abstracts them into reusable
    DSL macro-primitives using Minimum Description Length (MDL) heuristics.
    """

    def __init__(self, min_frequency: int = 2, max_macro_len: int = 3):
        self.min_frequency = min_frequency
        self.max_macro_len = max_macro_len
        self.invented_library: Dict[str, MacroPrimitive] = {}

    def extract_subsequences(
        self, steps: List[Tuple[str, Tuple]]
    ) -> List[Tuple[Tuple[str, Tuple], ...]]:
        """Extracts all contiguous step n-grams up to max_macro_len."""
        subseqs = []
        n = len(steps)
        for length in range(2, min(n + 1, self.max_macro_len + 1)):
            for i in range(n - length + 1):
                subseqs.append(tuple(steps[i : i + length]))
        return subseqs

    def mine_abstractions(
        self, verified_programs: List[SynthesizedProgram]
    ) -> List[MacroPrimitive]:
        """Identifies reusable macro primitives across a corpus of synthesized programs."""
        if not verified_programs:
            return []

        counter: Counter = Counter()
        for prog in verified_programs:
            if not prog.steps or len(prog.steps) < 2:
                continue
            subseqs = set(self.extract_subsequences(prog.steps))
            for subseq in subseqs:
                counter[subseq] += 1

        new_macros: List[MacroPrimitive] = []
        macro_idx = len(self.invented_library) + 1

        for subseq, freq in counter.items():
            if freq < self.min_frequency:
                continue

            base_name = "_".join(step[0].split("(")[0] for step in subseq)
            while f"macro_{macro_idx}_{base_name}" in UNARY_PRIMITIVES or f"macro_{macro_idx}_{base_name}" in self.invented_library:
                macro_idx += 1
            macro_name = f"macro_{macro_idx}_{base_name}"

            def build_composed_fn(chain):
                def composed(g: Grid) -> Grid:
                    cur = g
                    for name, args in chain:
                        if name in UNARY_PRIMITIVES:
                            cur = UNARY_PRIMITIVES[name](cur)
                    return cur
                return composed

            utility = freq * (len(subseq) - 1.0)
            macro_obj = MacroPrimitive(
                name=macro_name,
                operations=subseq,
                fn=build_composed_fn(subseq),
                utility_score=utility,
            )
            new_macros.append(macro_obj)
            macro_idx += 1

        new_macros.sort(key=lambda m: m.utility_score, reverse=True)
        return new_macros

    def consolidate_into_dsl(self, macros: List[MacroPrimitive]) -> int:
        """Injects mined macros directly into the active DSL primitive table."""
        added = 0
        for macro in macros:
            UNARY_PRIMITIVES[macro.name] = macro.fn
            self.invented_library[macro.name] = macro
            added += 1
        return added


@dataclass
class LambdaMacroPrimitive:
    name: str
    ast_node: Any
    fn: Callable
    mdl_gain: float
    frequency: int
    size: int


class LambdaSubtreeMiner:
    """DreamCoder-style anti-unification and MDL-driven macro mining for LambdaNode ASTs."""

    def __init__(self, min_frequency: int = 2, min_size: int = 2):
        self.min_frequency = min_frequency
        self.min_size = min_size
        self.promoted_macros: Dict[str, LambdaMacroPrimitive] = {}

    def extract_subtrees(self, node: Any) -> List[Any]:
        """Recursively extracts all candidate non-trivial subtrees from a LambdaNode AST."""
        from .lambda_dsl import (
            LambdaNode, Var, Const, Prim, Lambda, App, IfThenElse, MapNode, FilterNode, FoldNode, ast_size
        )
        subtrees: List[LambdaNode] = []

        def _traverse(cur: Any):
            if cur is None or not isinstance(cur, LambdaNode):
                return
            if ast_size(cur) >= self.min_size:
                subtrees.append(cur)
            if isinstance(cur, Lambda):
                _traverse(cur.body)
            elif isinstance(cur, App):
                _traverse(cur.func)
                _traverse(cur.arg)
            elif isinstance(cur, IfThenElse):
                _traverse(cur.cond)
                _traverse(cur.true_br)
                _traverse(cur.false_br)
            elif isinstance(cur, MapNode):
                _traverse(cur.func)
                _traverse(cur.lst)
            elif isinstance(cur, FilterNode):
                _traverse(cur.pred)
                _traverse(cur.lst)
            elif isinstance(cur, FoldNode):
                _traverse(cur.func)
                _traverse(cur.init)
                _traverse(cur.lst)

        _traverse(node)
        return subtrees

    def compute_mdl_gain(self, subtree: Any, occurrences: int) -> float:
        """
        MDL Compression Gain:
        Gain(M) = occurrences * (size(M) - 1) - size(M) (overhead of defining M)
        """
        from .lambda_dsl import ast_size
        sz = ast_size(subtree)
        gain = occurrences * (sz - 1.0) - sz
        return float(gain)

    def mine_and_promote(
        self,
        verified_asts: List[Any],
        env: Optional[Dict[str, Any]] = None,
    ) -> List[LambdaMacroPrimitive]:
        """
        Mines recurring subtrees from verified ASTs (Delta_S = +1.0), computes MDL gain,
        and automatically registers winning macros into CORE_LAMBDA_PRIMITIVES.
        """
        from .lambda_dsl import canonical_ast_hash, eval_lambda_ast, register_lambda_primitive, ast_size

        if not verified_asts:
            return []

        counts: Dict[str, Tuple[Any, int]] = {}
        for ast_tree in verified_asts:
            subtrees = self.extract_subtrees(ast_tree)
            seen_hashes = set()
            for sub in subtrees:
                h = canonical_ast_hash(sub)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                if h not in counts:
                    counts[h] = (sub, 1)
                else:
                    existing_sub, freq = counts[h]
                    counts[h] = (existing_sub, freq + 1)

        promoted: List[LambdaMacroPrimitive] = []
        base_env = env or {}

        macro_idx = len(self.promoted_macros) + 1
        for h, (sub, freq) in counts.items():
            if freq < self.min_frequency:
                continue

            gain = self.compute_mdl_gain(sub, freq)
            if gain <= 0:
                continue

            macro_name = f"lambda_macro_{macro_idx}"
            sz = ast_size(sub)

            # Construct executable function wrapper for evaluation
            def make_fn(bound_sub):
                return lambda *args: eval_lambda_ast(
                    bound_sub,
                    {**base_env, **{f"arg{i}": a for i, a in enumerate(args)}},
                )

            macro_fn = make_fn(sub)
            register_lambda_primitive(macro_name, macro_fn, ast_node=sub)

            macro_obj = LambdaMacroPrimitive(
                name=macro_name,
                ast_node=sub,
                fn=macro_fn,
                mdl_gain=gain,
                frequency=freq,
                size=sz,
            )
            self.promoted_macros[macro_name] = macro_obj
            promoted.append(macro_obj)
            macro_idx += 1

        promoted.sort(key=lambda m: m.mdl_gain, reverse=True)
        return promoted

