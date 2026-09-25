import json
import os
from typing import Any, Dict, List, Tuple
from .dsl import UNARY_PRIMITIVES, Grid
from .compression import MacroPrimitive


class PersistentMacroStore:
    """Manages cold-reboot persistence for mined DSL macros."""

    def __init__(self, store_path: str = "assets/dsl_macros.json"):
        self.store_path = store_path

    def save(self, macros: Dict[str, MacroPrimitive]):
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        serialized = {}
        for name, macro in macros.items():
            serialized[name] = {
                "name": macro.name,
                "operations": macro.operations,
                "utility_score": macro.utility_score,
            }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load_into_dsl(self) -> int:
        """Restores serialized macros into active UNARY_PRIMITIVES."""
        if not os.path.exists(self.store_path):
            return 0

        with open(self.store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_count = 0
        for name, entry in data.items():
            if name in UNARY_PRIMITIVES:
                continue

            if not isinstance(entry, dict) or "operations" not in entry:
                continue

            ops = [tuple(step) for step in entry["operations"]]

            def build_fn(chain):
                def runner(g: Grid) -> Grid:
                    cur = g
                    for op_name, _ in chain:
                        if op_name in UNARY_PRIMITIVES:
                            cur = UNARY_PRIMITIVES[op_name](cur)
                    return cur
                return runner

            UNARY_PRIMITIVES[name] = build_fn(ops)
            loaded_count += 1

        return loaded_count

    def export_macro_to_skill_code(self, name: str, operations: List[Tuple[str, Any]]) -> str:
        """Translates composite MacroPrimitive pipeline into standalone executable Python code."""
        code_lines = [
            f'"""Standalone ARC DSL Macro: {name}"""',
            "from typing import List, Tuple, Union",
            "",
            "Grid = Tuple[Tuple[int, ...], ...]",
            "",
            "def _rot90(g): return tuple(zip(*g[::-1]))",
            "def _rot180(g): return tuple(tuple(row[::-1]) for row in g[::-1])",
            "def _rot270(g): return tuple(zip(*g))[::-1]",
            "def _flip_v(g): return g[::-1]",
            "def _flip_h(g): return tuple(row[::-1] for row in g)",
            "def _transpose(g): return tuple(zip(*g))",
            "",
            "def run(grid: Union[List[List[int]], Grid]) -> List[List[int]]:",
            "    cur = tuple(tuple(int(c) for c in r) for r in grid)",
        ]
        for op_name, _ in operations:
            if op_name in ("rot90", "rotate_90"):
                code_lines.append("    cur = _rot90(cur)")
            elif op_name in ("rot180", "rotate_180"):
                code_lines.append("    cur = _rot180(cur)")
            elif op_name in ("rot270", "rotate_270"):
                code_lines.append("    cur = _rot270(cur)")
            elif op_name in ("flip_v", "vmirror"):
                code_lines.append("    cur = _flip_v(cur)")
            elif op_name in ("flip_h", "hmirror"):
                code_lines.append("    cur = _flip_h(cur)")
            elif op_name == "transpose":
                code_lines.append("    cur = _transpose(cur)")
            else:
                code_lines.append(f"    # Operation: {op_name}")
        code_lines.append("    return [list(row) for row in cur]")
        code_lines.append("")
        code_lines.append("def solution():")
        code_lines.append("    sample = [[1, 2], [3, 4]]")
        code_lines.append("    return run(sample)")
        return "\n".join(code_lines) + "\n"

    def export_to_skills(self, skill_library: Any, tenant_id: str = "default_tenant") -> List[str]:
        """Exports all saved DSL macros into standalone executable skills in the library."""
        if not os.path.exists(self.store_path):
            return []
        with open(self.store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        exported = []
        for name, entry in data.items():
            ops = [tuple(step) for step in entry.get("operations", [])]
            code = self.export_macro_to_skill_code(name, ops)
            doc = f"Exported ARC DSL macro '{name}' with operations: {ops}"
            skill_library.register_skill(name=f"arc_{name}", code=code, doc=doc, tenant_id=tenant_id)
            exported.append(f"arc_{name}")
        return exported
