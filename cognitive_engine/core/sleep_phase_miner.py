import ast
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .dsl import (
    Grid,
    UNARY_PRIMITIVES,
    crop_nonzero,
    find_connected_components,
    flip_h,
    flip_v,
    gravity,
    keep_largest_object,
    keep_smallest_object,
    replace_color,
    rot90,
    rot180,
    rot270,
    shift_left,
    shift_right,
    shift_up,
    shift_down,
    to_grid,
    transpose,
)
from .compression import MacroPrimitive, ProgramCompressor
from .macro_store import PersistentMacroStore
from .sharded_replay_buffer import ShardedReplayBuffer, ContrastiveTrajectory


class SleepPhaseMacroMiner:
    """DreamCoder Induction & Sleep-Phase Macro Mining Engine.

    Mines recurrent AST subtrees and call chains from verified contrastive
    trajectories in the ShardedReplayBuffer, abstracts them into parameterized
    DSL MacroPrimitives, and appends them to dsl_macros.json to collapse search depth.
    """

    def __init__(
        self,
        macro_store: Optional[PersistentMacroStore] = None,
        store_path: str = "assets/dsl_macros.json",
    ):
        self.macro_store = macro_store or PersistentMacroStore(store_path=store_path)
        self.compressor = ProgramCompressor(min_frequency=1, max_macro_len=3)
        self.mined_macros: Dict[str, MacroPrimitive] = {}

    def extract_ast_subchains(self, code: str) -> List[Tuple[Tuple[str, Tuple], ...]]:
        """Extracts compositional call chains from program code strings using AST."""
        chains: List[Tuple[Tuple[str, Tuple], ...]] = []
        if not code or not isinstance(code, str):
            return chains

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return chains

        # Traverse AST Call nodes to extract function composition chains
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                chain = self._resolve_call_chain(node)
                if len(chain) >= 2:
                    chains.append(tuple(chain))

        return chains

    def _resolve_call_chain(self, call_node: ast.Call) -> List[Tuple[str, Tuple]]:
        """Unrolls nested Call(f, [Call(g, ...)]) into a sequential execution pipeline."""
        steps = []
        curr = call_node
        while isinstance(curr, ast.Call):
            fn_name = ""
            args = ()
            if isinstance(curr.func, ast.Name):
                fn_name = curr.func.id
                args = tuple(
                    arg.value for arg in curr.args if isinstance(arg, ast.Constant)
                )
            elif isinstance(curr.func, ast.Attribute):
                fn_name = curr.func.attr
            elif isinstance(curr.func, ast.Call):
                # e.g. replace_color(0, 5)(g)
                if isinstance(curr.func.func, ast.Name):
                    fn_name = curr.func.func.id
                    args = tuple(
                        arg.value for arg in curr.func.args if isinstance(arg, ast.Constant)
                    )

            if fn_name and fn_name != "if_then_else":
                steps.append((fn_name, args))

            # Step down into the primary argument
            if curr.args:
                curr = curr.args[0]
            else:
                break

        # Return in forward execution order (innermost executed first)
        steps.reverse()
        return steps

    def mine_macros_from_trajectories(
        self,
        trajectories: List[ContrastiveTrajectory],
    ) -> List[MacroPrimitive]:
        """Identifies reusable macro primitives across a collection of contrastive trajectories."""
        discovered: List[MacroPrimitive] = []
        seen_signatures: Set[Tuple] = set()

        for traj in trajectories:
            code = traj.pos_code
            chains = self.extract_ast_subchains(code)

            for chain in chains:
                sig = tuple((name, args) for name, args in chain)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                macro = self._build_macro_primitive(chain)
                if macro:
                    discovered.append(macro)
                    self.mined_macros[macro.name] = macro

        return discovered

    def _build_macro_primitive(self, chain: Tuple[Tuple[str, Tuple], ...]) -> Optional[MacroPrimitive]:
        """Constructs an executable MacroPrimitive from a verified call chain."""
        base_name = "_".join(step[0] for step in chain)
        macro_name = f"macro_{base_name}"
        idx = 1
        while macro_name in UNARY_PRIMITIVES or macro_name in self.mined_macros:
            macro_name = f"macro_{base_name}_{idx}"
            idx += 1

        def build_runner(op_chain):
            def runner(g: Grid) -> Grid:
                cur = g
                for name, args in op_chain:
                    if name in UNARY_PRIMITIVES:
                        cur = UNARY_PRIMITIVES[name](cur)
                    elif name == "replace_color" and len(args) == 2:
                        cur = replace_color(cur, args[0], args[1])
                    elif name == "crop_nonzero":
                        cur = crop_nonzero(cur)
                    elif name == "keep_largest_object":
                        cur = keep_largest_object(cur)
                    elif name == "keep_smallest_object":
                        cur = keep_smallest_object(cur)
                    elif name == "shift_left":
                        cur = shift_left(cur)
                    elif name == "rot90":
                        cur = rot90(cur)
                    elif name == "flip_h":
                        cur = flip_h(cur)
                return cur
            return runner

        fn = build_runner(chain)
        utility = len(chain) * 2.0
        return MacroPrimitive(
            name=macro_name,
            operations=chain,
            fn=fn,
            utility_score=utility,
        )

    def run_sleep_cycle(
        self,
        replay_buffer: Optional[ShardedReplayBuffer] = None,
        candidate_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Executes a full sleep-phase consolidation pass:

        1. Gathers positive trajectories from the replay buffer.
        2. Mines recurring AST subtrees.
        3. Appends parameterized macros to dsl_macros.json.
        4. Injects them into UNARY_PRIMITIVES for depth collapsing.
        """
        trajectories: List[ContrastiveTrajectory] = []

        if replay_buffer is not None:
            # Sample all active shards
            trajectories.extend(replay_buffer.sample_batch(batch_size=50, allow_cold_hydration=True))

        if candidate_codes:
            import torch
            for c in candidate_codes:
                trajectories.append(
                    ContrastiveTrajectory(
                        pos_code=c,
                        pos_log_probs=torch.tensor([1.0]),
                        neg_code="identity(g)",
                        neg_log_probs=torch.tensor([0.0]),
                        margin=2.0,
                        domain="arc_visual_induction",
                    )
                )

        # Mine new macro abstractions
        new_macros = self.mine_macros_from_trajectories(trajectories)

        # Persist to dsl_macros.json
        if new_macros:
            self.macro_store.save(self.mined_macros)
            # Inject into active DSL table
            for m in new_macros:
                UNARY_PRIMITIVES[m.name] = m.fn

        return {
            "trajectories_evaluated": len(trajectories),
            "new_macros_discovered": len(new_macros),
            "macro_names": [m.name for m in new_macros],
            "total_persisted_macros": len(self.mined_macros),
            "active_dsl_primitives": len(UNARY_PRIMITIVES),
        }
