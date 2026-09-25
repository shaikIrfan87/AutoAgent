import functools
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass(frozen=True)
class LambdaNode:
    pass


@dataclass(frozen=True)
class Var(LambdaNode):
    name: str


@dataclass(frozen=True)
class Const(LambdaNode):
    val: Any


@dataclass(frozen=True)
class Prim(LambdaNode):
    op: str


@dataclass(frozen=True)
class App(LambdaNode):
    func: LambdaNode
    arg: LambdaNode


@dataclass(frozen=True)
class Lambda(LambdaNode):
    param: str
    body: LambdaNode


@dataclass(frozen=True)
class IfThenElse(LambdaNode):
    cond: LambdaNode
    true_br: LambdaNode
    false_br: LambdaNode


@dataclass(frozen=True)
class MapNode(LambdaNode):
    func: LambdaNode
    lst: LambdaNode


@dataclass(frozen=True)
class FilterNode(LambdaNode):
    pred: LambdaNode
    lst: LambdaNode


@dataclass(frozen=True)
class FoldNode(LambdaNode):
    func: LambdaNode
    init: LambdaNode
    lst: LambdaNode


@dataclass(frozen=True)
class WhileLoop(LambdaNode):
    cond_fn: LambdaNode
    step_fn: LambdaNode
    init_val: LambdaNode


@dataclass(frozen=True)
class ForLoop(LambdaNode):
    limit: LambdaNode
    step_fn: LambdaNode
    init_val: LambdaNode


def _get_connected_components(g: Any) -> List[Any]:
    """Returns list of masked grids, each containing exactly one 4-connected monochromatic object."""
    if not isinstance(g, (list, tuple)) or not g or not isinstance(g[0], (list, tuple)):
        return []
    h, w = len(g), len(g[0])
    visited = set()
    components = []
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0 and (r, c) not in visited:
                color = g[r][c]
                comp = set()
                q = [(r, c)]
                visited.add((r, c))
                while q:
                    cr, cc = q.pop(0)
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and g[nr][nc] != 0:
                            if g[nr][nc] == color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                mask_grid = [[g[row][col] if (row, col) in comp else 0 for col in range(w)] for row in range(h)]
                components.append(mask_grid)
    return components


def _largest_object(g: Any) -> Any:
    objs = _get_connected_components(g)
    if not objs:
        return g
    return max(objs, key=lambda o: sum(1 for row in o for x in row if x != 0))


def _smallest_object(g: Any) -> Any:
    objs = _get_connected_components(g)
    if not objs:
        return g
    return min(objs, key=lambda o: sum(1 for row in o for x in row if x != 0))


def _overlay(g1: Any) -> Callable[[Any], Any]:
    def _inner(g2: Any) -> Any:
        if not isinstance(g1, (list, tuple)) or not isinstance(g2, (list, tuple)):
            return g2 if g2 else g1
        h = min(len(g1), len(g2))
        if h == 0:
            return g2 if g2 else g1
        w = min(len(g1[0]), len(g2[0]))
        return [
            [g2[r][c] if g2[r][c] != 0 else g1[r][c] for c in range(w)]
            for r in range(h)
        ]
    return _inner


def _recolor(color: int) -> Callable[[Any], Any]:
    def _inner(g: Any) -> Any:
        if not isinstance(g, (list, tuple)) or not g or not isinstance(g[0], (list, tuple)):
            return g
        return [[color if x != 0 else 0 for x in row] for row in g]
    return _inner


def _hollow(g: Any) -> Any:
    if not isinstance(g, (list, tuple)) or not g or not isinstance(g[0], (list, tuple)):
        return g
    h, w = len(g), len(g[0])
    out = [[g[r][c] for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                if all(0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0 for nr, nc in neighbors):
                    out[r][c] = 0
    return out


def _fill_holes(fill_color: int) -> Callable[[Any], Any]:
    def _inner(g: Any) -> Any:
        if not isinstance(g, (list, tuple)) or not g or not isinstance(g[0], (list, tuple)):
            return g
        h, w = len(g), len(g[0])
        reachable = set()
        q = []
        for r in range(h):
            for c in (0, w - 1):
                if g[r][c] == 0 and (r, c) not in reachable:
                    reachable.add((r, c))
                    q.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if g[r][c] == 0 and (r, c) not in reachable:
                    reachable.add((r, c))
                    q.append((r, c))
        while q:
            cr, cc = q.pop(0)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and g[nr][nc] == 0:
                    reachable.add((nr, nc))
                    q.append((nr, nc))
        out = [[fill_color if (r, c) not in reachable and g[r][c] == 0 else g[r][c] for c in range(w)] for r in range(h)]
        return out
    return _inner


def _shift(dr: int, dc: int) -> Callable[[Any], Any]:
    def _inner(g: Any) -> Any:
        if not isinstance(g, (list, tuple)) or not g or not isinstance(g[0], (list, tuple)):
            return g
        h, w = len(g), len(g[0])
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = g[r][c]
        return out
    return _inner


CORE_LAMBDA_PRIMITIVES: Dict[str, Callable] = {
    "+": lambda a: lambda b: a + b,
    "-": lambda a: lambda b: a - b,
    "*": lambda a: lambda b: a * b,
    "add": lambda a: lambda b: a + b,
    "sub": lambda a: lambda b: a - b,
    "mul": lambda a: lambda b: a * b,
    "branch": lambda cond: lambda t_val: lambda f_val: t_val if cond else f_val,
    "//": lambda a: lambda b: a // b if b != 0 else 0,
    "%": lambda a: lambda b: a % b if b != 0 else 0,
    "==": lambda a: lambda b: a == b,
    "<": lambda a: lambda b: a < b,
    "<=": lambda a: lambda b: a <= b,
    ">": lambda a: lambda b: a > b,
    ">=": lambda a: lambda b: a >= b,
    "not": lambda a: not a,
    "cons": lambda h: lambda t: [h] + list(t),
    "head": lambda lst: lst[0] if lst else 0,
    "tail": lambda lst: lst[1:] if len(lst) > 1 else [],
    "is_empty": lambda lst: len(lst) == 0,
    "len": lambda lst: len(lst),
    "map": lambda f: lambda lst: [f(x) for x in lst],
    "filter": lambda f: lambda lst: [x for x in lst if f(x)],
    "fold": lambda f: lambda acc: lambda lst: functools.reduce(lambda a, b: f(a)(b) if callable(f(a)) else f(a, b), lst, acc),
    "range": lambda n: list(range(int(n))),
    "rot90": lambda g: [list(r) for r in zip(*g[::-1])] if isinstance(g, (list, tuple)) and g and isinstance(g[0], (list, tuple)) else g,
    "flip_v": lambda g: list(g[::-1]) if isinstance(g, (list, tuple)) else g,
    "crop_nonzero": lambda g: [row for row in g if any(x != 0 for x in row)] if isinstance(g, (list, tuple)) else g,
    "gravity": lambda g: [[x for x in col if x != 0] for col in zip(*g)] if isinstance(g, (list, tuple)) else g,
    "connected_components": lambda g: len(set(x for row in g for x in row if x != 0)) if isinstance(g, (list, tuple)) and g and isinstance(g[0], (list, tuple)) else 0,
    "objects": _get_connected_components,
    "largest_object": _largest_object,
    "smallest_object": _smallest_object,
    "overlay": _overlay,
    "recolor": _recolor,
    "hollow": _hollow,
    "fill_holes": _fill_holes,
    "shift_up": _shift(-1, 0),
    "shift_down": _shift(1, 0),
    "shift_left": _shift(0, -1),
    "shift_right": _shift(0, 1),
}


def eval_lambda_ast(node: LambdaNode, env: Dict[str, Any], depth_limit: int = 64) -> Any:
    """Safely evaluates lambda expressions with recursion guard limits."""
    if depth_limit <= 0:
        raise RecursionError("Evaluation budget exceeded")

    if isinstance(node, Const):
        return node.val
    elif isinstance(node, Var):
        if node.name in env:
            return env[node.name]
        raise NameError(f"Unbound variable: {node.name}")
    elif isinstance(node, Prim):
        if node.op in CORE_LAMBDA_PRIMITIVES:
            return CORE_LAMBDA_PRIMITIVES[node.op]
        raise ValueError(f"Unknown primitive: {node.op}")
    elif isinstance(node, Lambda):
        return lambda arg: eval_lambda_ast(node.body, {**env, node.param: arg}, depth_limit - 1)
    elif isinstance(node, App):
        fn = eval_lambda_ast(node.func, env, depth_limit - 1)
        arg = eval_lambda_ast(node.arg, env, depth_limit - 1)
        if callable(fn):
            return fn(arg)
        raise TypeError(f"Cannot call non-function {fn}")
    elif isinstance(node, IfThenElse):
        c = eval_lambda_ast(node.cond, env, depth_limit - 1)
        branch = node.true_br if c else node.false_br
        return eval_lambda_ast(branch, env, depth_limit - 1)
    elif isinstance(node, MapNode):
        fn = eval_lambda_ast(node.func, env, depth_limit - 1)
        items = eval_lambda_ast(node.lst, env, depth_limit - 1)
        return [fn(x) for x in (items if isinstance(items, (list, tuple)) else [items])]
    elif isinstance(node, FilterNode):
        pred = eval_lambda_ast(node.pred, env, depth_limit - 1)
        items = eval_lambda_ast(node.lst, env, depth_limit - 1)
        return [x for x in (items if isinstance(items, (list, tuple)) else [items]) if pred(x)]
    elif isinstance(node, FoldNode):
        fn = eval_lambda_ast(node.func, env, depth_limit - 1)
        acc = eval_lambda_ast(node.init, env, depth_limit - 1)
        items = eval_lambda_ast(node.lst, env, depth_limit - 1)
        for x in (items if isinstance(items, (list, tuple)) else [items]):
            acc = fn(acc)(x) if callable(fn(acc)) else fn(acc, x)
        return acc
    elif isinstance(node, WhileLoop):
        cond = eval_lambda_ast(node.cond_fn, env, depth_limit - 1)
        step = eval_lambda_ast(node.step_fn, env, depth_limit - 1)
        val = eval_lambda_ast(node.init_val, env, depth_limit - 1)
        budget = min(depth_limit, 200)
        while budget > 0 and cond(val):
            val = step(val)
            budget -= 1
        return val
    elif isinstance(node, ForLoop):
        limit = int(eval_lambda_ast(node.limit, env, depth_limit - 1))
        step = eval_lambda_ast(node.step_fn, env, depth_limit - 1)
        val = eval_lambda_ast(node.init_val, env, depth_limit - 1)
        for i in range(min(limit, 200)):
            val = step(val)(i) if callable(step(val)) else step(val)
        return val
    raise NotImplementedError(f"Unsupported AST node: {type(node)}")


def canonical_ast_hash(node: LambdaNode) -> str:
    """Computes a semantic sub-tree equivalence hash invariant to commutative argument order."""
    if isinstance(node, Const):
        return f"C({repr(node.val)})"
    elif isinstance(node, Var):
        return f"V({node.name})"
    elif isinstance(node, Prim):
        return f"P({node.op})"
    elif isinstance(node, Lambda):
        return f"L({node.param},{canonical_ast_hash(node.body)})"
    elif isinstance(node, App):
        # Handle commutative primitives: (+) (*) (==) overlay
        if isinstance(node.func, App) and isinstance(node.func.func, Prim):
            op = node.func.func.op
            if op in ("+", "*", "==", "overlay", "add", "mul"):
                h1 = canonical_ast_hash(node.func.arg)
                h2 = canonical_ast_hash(node.arg)
                ordered = sorted([h1, h2])
                return f"App2({op},{ordered[0]},{ordered[1]})"
        return f"App({canonical_ast_hash(node.func)},{canonical_ast_hash(node.arg)})"
    elif isinstance(node, IfThenElse):
        return f"If({canonical_ast_hash(node.cond)},{canonical_ast_hash(node.true_br)},{canonical_ast_hash(node.false_br)})"
    elif isinstance(node, MapNode):
        return f"Map({canonical_ast_hash(node.func)},{canonical_ast_hash(node.lst)})"
    elif isinstance(node, FilterNode):
        return f"Filter({canonical_ast_hash(node.pred)},{canonical_ast_hash(node.lst)})"
    elif isinstance(node, FoldNode):
        return f"Fold({canonical_ast_hash(node.func)},{canonical_ast_hash(node.init)},{canonical_ast_hash(node.lst)})"
    elif isinstance(node, WhileLoop):
        return f"While({canonical_ast_hash(node.cond_fn)},{canonical_ast_hash(node.step_fn)},{canonical_ast_hash(node.init_val)})"
    elif isinstance(node, ForLoop):
        return f"For({canonical_ast_hash(node.limit)},{canonical_ast_hash(node.step_fn)},{canonical_ast_hash(node.init_val)})"
    return hashlib.md5(repr(node).encode()).hexdigest()[:12]


def unparse_to_python(node: LambdaNode) -> str:
    """Translates typed lambda ASTs into clean executable Python syntax."""
    if isinstance(node, Const):
        return repr(node.val)
    elif isinstance(node, Var):
        return node.name
    elif isinstance(node, Prim):
        return f"_PRIMS[{repr(node.op)}]"
    elif isinstance(node, Lambda):
        return f"(lambda {node.param}: {unparse_to_python(node.body)})"
    elif isinstance(node, App):
        # Pretty-print standard curried arithmetic and binary primitives
        if isinstance(node.func, App) and isinstance(node.func.func, Prim):
            op = node.func.func.op
            if op in ("+", "-", "*", "//", "%", "==", "<", "<=", ">", ">="):
                arg1 = unparse_to_python(node.func.arg)
                arg2 = unparse_to_python(node.arg)
                return f"({arg1} {op} {arg2})"
            if op == "add":
                return f"({unparse_to_python(node.func.arg)} + {unparse_to_python(node.arg)})"
            if op == "sub":
                return f"({unparse_to_python(node.func.arg)} - {unparse_to_python(node.arg)})"
            if op == "mul":
                return f"({unparse_to_python(node.func.arg)} * {unparse_to_python(node.arg)})"
            if op in ("overlay", "recolor", "fill_holes"):
                arg1 = unparse_to_python(node.func.arg)
                arg2 = unparse_to_python(node.arg)
                return f"(_PRIMS[{repr(op)}]({arg1})({arg2}))"
        if isinstance(node.func, Prim):
            op = node.func.op
            if op == "not":
                return f"(int(not ({unparse_to_python(node.arg)})))"
            elif op == "len":
                return f"len({unparse_to_python(node.arg)})"
            elif op == "sum":
                return f"sum({unparse_to_python(node.arg)})"
            elif op == "head":
                return f"({unparse_to_python(node.arg)}[0] if {unparse_to_python(node.arg)} else 0)"
            elif op == "tail":
                return f"{unparse_to_python(node.arg)}[1:]"
            elif op == "is_empty":
                return f"(len({unparse_to_python(node.arg)}) == 0)"
            elif op in CORE_LAMBDA_PRIMITIVES:
                return f"(_PRIMS[{repr(op)}]({unparse_to_python(node.arg)}))"
        return f"({unparse_to_python(node.func)})({unparse_to_python(node.arg)})"
    elif isinstance(node, IfThenElse):
        return f"({unparse_to_python(node.true_br)} if {unparse_to_python(node.cond)} else {unparse_to_python(node.false_br)})"
    elif isinstance(node, MapNode):
        fn = unparse_to_python(node.func)
        lst = unparse_to_python(node.lst)
        return f"[{fn}(_x) for _x in {lst}]"
    elif isinstance(node, FilterNode):
        pred = unparse_to_python(node.pred)
        lst = unparse_to_python(node.lst)
        return f"[_x for _x in {lst} if {pred}(_x)]"
    elif isinstance(node, FoldNode):
        fn = unparse_to_python(node.func)
        init = unparse_to_python(node.init)
        lst = unparse_to_python(node.lst)
        return f"__import__('functools').reduce(lambda _a, _b: ({fn})(_a)(_b) if callable(({fn})(_a)) else ({fn})(_a, _b), {lst}, {init})"
    elif isinstance(node, WhileLoop):
        cond = unparse_to_python(node.cond_fn)
        step = unparse_to_python(node.step_fn)
        init = unparse_to_python(node.init_val)
        return f"(lambda _v, _c=({cond}), _s=({step}): [None for _ in iter(lambda: _c(_v), False) if (_v := _s(_v)) or True] and _v)({init})"
    elif isinstance(node, ForLoop):
        lim = unparse_to_python(node.limit)
        step = unparse_to_python(node.step_fn)
        init = unparse_to_python(node.init_val)
        return f"__import__('functools').reduce(lambda _v, _i: ({step})(_v)(_i) if callable(({step})(_v)) else ({step})(_v), range({lim}), {init})"
    return "None"


def ast_size(node: LambdaNode) -> int:
    """Computes structural tree size for Minimum Description Length (MDL) evaluations."""
    if isinstance(node, (Var, Const, Prim)):
        return 1
    elif isinstance(node, Lambda):
        return 1 + ast_size(node.body)
    elif isinstance(node, App):
        return 1 + ast_size(node.func) + ast_size(node.arg)
    elif isinstance(node, IfThenElse):
        return 1 + ast_size(node.cond) + ast_size(node.true_br) + ast_size(node.false_br)
    elif isinstance(node, MapNode):
        return 1 + ast_size(node.func) + ast_size(node.lst)
    elif isinstance(node, FilterNode):
        return 1 + ast_size(node.pred) + ast_size(node.lst)
    elif isinstance(node, FoldNode):
        return 1 + ast_size(node.func) + ast_size(node.init) + ast_size(node.lst)
    elif isinstance(node, WhileLoop):
        return 1 + ast_size(node.cond_fn) + ast_size(node.step_fn) + ast_size(node.init_val)
    elif isinstance(node, ForLoop):
        return 1 + ast_size(node.limit) + ast_size(node.step_fn) + ast_size(node.init_val)
    return 1


def register_lambda_primitive(name: str, fn: Callable, ast_node: Optional[LambdaNode] = None) -> None:
    """Registers a discovered macro primitive dynamically into CORE_LAMBDA_PRIMITIVES."""
    CORE_LAMBDA_PRIMITIVES[name] = fn

