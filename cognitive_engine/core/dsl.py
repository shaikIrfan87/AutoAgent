"""
Domain-Specific Language (DSL) of pure transformations for abstract grid reasoning (ARC-AGI style).
"""
from typing import Callable, Dict, List, Set, Tuple, Union

Grid = Tuple[Tuple[int, ...], ...]


def to_grid(data: Union[List[List[int]], Grid]) -> Grid:
    """Normalize input into an immutable hashable 2D Grid."""
    return tuple(tuple(int(c) for c in row) for row in data)


def rot90(g: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return tuple(zip(*g[::-1]))


def rot180(g: Grid) -> Grid:
    """Rotate 180 degrees."""
    return tuple(tuple(row[::-1]) for row in g[::-1])


def rot270(g: Grid) -> Grid:
    """Rotate 270 degrees clockwise."""
    return tuple(zip(*g))[::-1]


def flip_v(g: Grid) -> Grid:
    """Vertical flip (reflection across horizontal axis)."""
    return g[::-1]


def flip_h(g: Grid) -> Grid:
    """Horizontal flip (reflection across vertical axis)."""
    return tuple(row[::-1] for row in g)


def transpose(g: Grid) -> Grid:
    """Transpose rows and columns."""
    return tuple(zip(*g))


def crop_nonzero(g: Grid) -> Grid:
    """Crop grid to bounding box of non-zero elements."""
    rows = [r for r, row in enumerate(g) if any(row)]
    cols = [c for c in range(len(g[0])) if any(g[r][c] for r in range(len(g)))]
    if not rows or not cols:
        return ((0,),)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    return tuple(g[r][min_c : max_c + 1] for r in range(min_r, max_r + 1))


def upscale(g: Grid, factor: int) -> Grid:
    """Integer nearest-neighbor upscaling for dimension expansion tasks."""
    if factor <= 1:
        return g
    return tuple(
        tuple(c for c in row for _ in range(factor))
        for row in g for _ in range(factor)
    )


def tile_grid(g: Grid, r_reps: int, c_reps: int) -> Grid:
    """Kronecker/tiling self-similarity expansion."""
    if r_reps <= 1 and c_reps <= 1:
        return g
    return tuple(
        tuple(list(row) * c_reps)
        for _ in range(r_reps)
        for row in g
    )


def crop_bbox(g: Grid, bbox: Tuple[int, int, int, int]) -> Grid:
    """Crops grid to exact (r_min, r_max, c_min, c_max) bounds."""
    r_min, r_max, c_min, c_max = bbox
    h, w = len(g), len(g[0])
    r_min = max(0, min(r_min, h - 1))
    r_max = max(0, min(r_max, h - 1))
    c_min = max(0, min(c_min, w - 1))
    c_max = max(0, min(c_max, w - 1))
    if r_min > r_max or c_min > c_max:
        return ((0,),)
    return tuple(tuple(g[r][c_min : c_max + 1]) for r in range(r_min, r_max + 1))


def overlay_binary(g1: Grid, g2: Grid, op: str = "xor") -> Grid:
    """Binary multi-grid composition for quadrant folding tasks."""
    h = min(len(g1), len(g2))
    w = min(len(g1[0]), len(g2[0]))
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            v1, v2 = g1[r][c], g2[r][c]
            if op == "xor":
                row.append(v1 if (v1 != 0 and v2 == 0) else (v2 if (v2 != 0 and v1 == 0) else 0))
            elif op == "or":
                row.append(v1 if v1 != 0 else v2)
            else:
                row.append(v1 if v1 == v2 else 0)
        out.append(tuple(row))
    return tuple(out)


def replace_color(g: Grid, old_c: int, new_c: int) -> Grid:
    """Replace all instances of old_c with new_c."""
    return tuple(tuple(new_c if cell == old_c else cell for cell in row) for row in g)


def gravity(g: Grid) -> Grid:
    """Drop non-zero cells downward to bottom of each column."""
    h, w = len(g), len(g[0])
    cols = []
    for c in range(w):
        vals = [g[r][c] for r in range(h) if g[r][c] != 0]
        col = [0] * (h - len(vals)) + vals
        cols.append(col)
    return tuple(tuple(cols[c][r] for c in range(w)) for r in range(h))


def flood_fill(g: Grid, r: int, c: int, new_c: int) -> Grid:
    """Standard 4-way flood fill from seed coordinate."""
    h, w = len(g), len(g[0])
    if not (0 <= r < h and 0 <= c < w):
        return g
    target = g[r][c]
    if target == new_c:
        return g
    grid_mat = [list(row) for row in g]
    q = [(r, c)]
    visited = {(r, c)}
    while q:
        cr, cc = q.pop(0)
        grid_mat[cr][cc] = new_c
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid_mat[nr][nc] == target:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return tuple(tuple(row) for row in grid_mat)


def invert(g: Grid) -> Grid:
    """Invert non-zero values (9 - val)."""
    return tuple(tuple(9 - c if c != 0 else 0 for c in row) for row in g)


def pad(g: Grid, pad_val: int = 0) -> Grid:
    """Surround grid with a 1-pixel border of pad_val."""
    w = len(g[0]) + 2
    top = (tuple(pad_val for _ in range(w)),)
    body = tuple((pad_val,) + row + (pad_val,) for row in g)
    return top + body + top


def unpad(g: Grid) -> Grid:
    """Strip 1-pixel outer border if dimensions allow."""
    if len(g) <= 2 or len(g[0]) <= 2:
        return g
    return tuple(row[1:-1] for row in g[1:-1])


def reflect_sym_h(g: Grid) -> Grid:
    """Mirror grid horizontally: [g | flip_h(g)]."""
    return tuple(row + row[::-1] for row in g)


def reflect_sym_v(g: Grid) -> Grid:
    """Mirror grid vertically: [g / flip_v(g)]."""
    return g + g[::-1]


def outline(g: Grid) -> Grid:
    """Keep only boundary pixels of connected non-zero regions."""
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                is_interior = all(
                    0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0 for nr, nc in neighbors
                )
                if is_interior:
                    out[r][c] = 0
    return tuple(tuple(row) for row in out)


def find_connected_components(g: Grid, monochromatic: bool = False) -> List[Set[Tuple[int, int]]]:
    """Find 4-connected components of non-zero elements (optionally monochromatic)."""
    h, w = len(g), len(g[0])
    visited: Set[Tuple[int, int]] = set()
    components: List[Set[Tuple[int, int]]] = []
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
                            if not monochromatic or g[nr][nc] == color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                components.append(comp)
    return components



def label_components(g: Grid) -> Grid:
    """Label each connected non-zero component with a distinct integer color (1..9)."""
    components = find_connected_components(g)
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for idx, comp in enumerate(components):
        color = (idx % 9) + 1
        for r, c in comp:
            out[r][c] = color
    return tuple(tuple(row) for row in out)


def keep_largest_object(g: Grid) -> Grid:
    """Retain only the connected component with the largest pixel count."""
    components = find_connected_components(g)
    if not components:
        return g
    largest = max(components, key=len)
    h, w = len(g), len(g[0])
    return tuple(tuple(g[r][c] if (r, c) in largest else 0 for c in range(w)) for r in range(h))


def keep_smallest_object(g: Grid) -> Grid:
    """Retain only the connected component with the smallest pixel count."""
    components = find_connected_components(g)
    if not components:
        return g
    smallest = min(components, key=len)
    h, w = len(g), len(g[0])
    return tuple(tuple(g[r][c] if (r, c) in smallest else 0 for c in range(w)) for r in range(h))


def shift(g: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    """Translate grid coordinates by (dr, dc)."""
    h, w = len(g), len(g[0])
    out = [[fill] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g[r][c]
    return tuple(tuple(row) for row in out)


def shift_up(g: Grid) -> Grid:
    return shift(g, -1, 0)


def shift_down(g: Grid) -> Grid:
    return shift(g, 1, 0)


def shift_left(g: Grid) -> Grid:
    return shift(g, 0, -1)


def shift_right(g: Grid) -> Grid:
    return shift(g, 0, 1)


def remove_isolated_objects(g: Grid) -> Grid:
    """Remove connected components that do not touch any other component."""
    components = find_connected_components(g, monochromatic=True)
    if len(components) <= 1:
        return g
    h, w = len(g), len(g[0]) if g else 0
    touching_pixels = set()
    for c1 in components:
        c1_set = set(c1)
        touches = False
        for r, c in c1:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != 0 and (nr, nc) not in c1_set:
                    touches = True
                    break
            if touches:
                break
        if touches:
            touching_pixels.update(c1)
    return tuple(tuple(g[r][c] if (r, c) in touching_pixels else 0 for c in range(w)) for r in range(h))


# Primitive registry with callable factories
UNARY_PRIMITIVES: Dict[str, Callable[[Grid], Grid]] = {
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_v": flip_v,
    "flip_h": flip_h,
    "transpose": transpose,
    "crop_nonzero": crop_nonzero,
    "gravity": gravity,
    "invert": invert,
    "pad": pad,
    "unpad": unpad,
    "reflect_sym_h": reflect_sym_h,
    "reflect_sym_v": reflect_sym_v,
    "outline": outline,
    "label_components": label_components,
    "keep_largest_object": keep_largest_object,
    "keep_smallest_object": keep_smallest_object,
    "shift_up": shift_up,
    "shift_down": shift_down,
    "shift_left": shift_left,
    "shift_right": shift_right,
    "remove_isolated_objects": remove_isolated_objects,
    "upscale_2": lambda g: upscale(g, 2),
    "upscale_3": lambda g: upscale(g, 3),
    "tile_2x2": lambda g: tile_grid(g, 2, 2),
    "tile_3x3": lambda g: tile_grid(g, 3, 3),
    "tile_1x2": lambda g: tile_grid(g, 1, 2),
    "tile_2x1": lambda g: tile_grid(g, 2, 1),
}


# General Python AST Lambda Primitives (Arithmetic, String Slicing, Dict Lookups)
PYTHON_LAMBDA_PRIMITIVES: Dict[str, Callable] = {
    # Arithmetic
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else 0.0,
    "floordiv": lambda a, b: a // b if b != 0 else 0,
    "mod": lambda a, b: a % b if b != 0 else 0,
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
    # String Slicing & Operations
    "str_slice": lambda s, start=0, end=None, step=1: str(s)[start:end:step],
    "str_concat": lambda a, b: str(a) + str(b),
    "str_len": lambda s: len(str(s)),
    "str_lower": lambda s: str(s).lower(),
    "str_upper": lambda s: str(s).upper(),
    "str_strip": lambda s: str(s).strip(),
    "str_split": lambda s, sep=None: str(s).split(sep),
    # Dict Lookups & Updates
    "dict_get": lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default,
    "dict_set": lambda d, k, v: {**d, k: v} if isinstance(d, dict) else {k: v},
    "dict_keys": lambda d: list(d.keys()) if isinstance(d, dict) else [],
    "dict_has": lambda d, k: k in d if isinstance(d, dict) else False,
    # Sequence / List Operations
    "list_get": lambda lst, idx, default=None: lst[idx] if 0 <= idx < len(lst) else default,
    "list_append": lambda lst, x: list(lst) + [x],
    "list_len": lambda lst: len(lst),
}



# Predicates and Higher-Order Lambda AST Combinators
def is_symmetric_h(g: Grid) -> bool:
    """Check if grid has horizontal reflection symmetry."""
    return all(row == row[::-1] for row in g)


def is_symmetric_v(g: Grid) -> bool:
    """Check if grid has vertical reflection symmetry."""
    return g == g[::-1]


def has_color(c: int) -> Callable[[Grid], bool]:
    """Predicate testing if color c is present in grid."""
    return lambda g: any(c in row for row in g)


def has_multiple_components(g: Grid) -> bool:
    """Predicate testing if more than one disconnected non-zero component exists."""
    return len(find_connected_components(g)) > 1


def if_then_else(
    cond_fn: Callable[[Grid], bool],
    true_fn: Callable[[Grid], Grid],
    false_fn: Callable[[Grid], Grid],
) -> Callable[[Grid], Grid]:
    """Higher-order conditional branching operator."""
    return lambda g: true_fn(g) if cond_fn(g) else false_fn(g)


def repeat_until(
    step_fn: Callable[[Grid], Grid],
    cond_fn: Callable[[Grid], bool],
    max_steps: int = 4,
) -> Callable[[Grid], Grid]:
    """Iterative fixed-point loop operator."""
    def runner(g: Grid) -> Grid:
        cur = g
        for _ in range(max_steps):
            if cond_fn(cur):
                break
            nxt = step_fn(cur)
            if nxt == cur:
                break
            cur = nxt
        return cur
    return runner


# Compound Relational Predicates & Inferred Grammar
def count_color(g: Grid, c: int) -> int:
    """Return number of cells matching color c."""
    return sum(cell == c for row in g for cell in row)


def color_count_greater(c1: int, c2: int) -> Callable[[Grid], bool]:
    """Relational predicate: Count(c1) > Count(c2)."""
    return lambda g: count_color(g, c1) > count_color(g, c2)


def has_even_parity(c: int) -> Callable[[Grid], bool]:
    """Parity predicate: Count(c) is even and non-zero."""
    return lambda g: (cnt := count_color(g, c)) > 0 and cnt % 2 == 0


def area_greater_than(c: int, threshold: int) -> Callable[[Grid], bool]:
    """Area comparison predicate: Count(c) > threshold."""
    return lambda g: count_color(g, c) > threshold


def is_taller_than_wide(g: Grid) -> bool:
    """Aspect ratio predicate: height > width."""
    return len(g) > len(g[0])


def infer_relational_predicates(examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Callable[[Grid], bool]]]:
    """Generate minimal active relational predicates distinguishing training examples."""
    predicates: List[Tuple[str, Callable[[Grid], bool]]] = [
        ("is_symmetric_h", is_symmetric_h),
        ("is_symmetric_v", is_symmetric_v),
        ("is_taller_than_wide", is_taller_than_wide),
        ("has_multiple_components", has_multiple_components),
    ]
    colors = {c for inp, _ in examples for row in inp for c in row if c != 0}
    for c in colors:
        predicates.append((f"has_color({c})", has_color(c)))
        predicates.append((f"has_even_parity({c})", has_even_parity(c)))
    for c1 in colors:
        for c2 in colors:
            if c1 != c2:
                predicates.append((f"count({c1}) > count({c2})", color_count_greater(c1, c2)))
    return predicates


