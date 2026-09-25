"""Exported ARC DSL macro 'macro_4_flip_v_replace_color' with operations: [('flip_v', []), ('replace_color(1, 2)', [1, 2])]"""
"""Standalone ARC DSL Macro: macro_4_flip_v_replace_color"""
from typing import List, Tuple, Union

Grid = Tuple[Tuple[int, ...], ...]

def _rot90(g): return tuple(zip(*g[::-1]))
def _rot180(g): return tuple(tuple(row[::-1]) for row in g[::-1])
def _rot270(g): return tuple(zip(*g))[::-1]
def _flip_v(g): return g[::-1]
def _flip_h(g): return tuple(row[::-1] for row in g)
def _transpose(g): return tuple(zip(*g))

def run(grid: Union[List[List[int]], Grid]) -> List[List[int]]:
    cur = tuple(tuple(int(c) for c in r) for r in grid)
    cur = _flip_v(cur)
    # Operation: replace_color(1, 2)
    return [list(row) for row in cur]

def solution():
    sample = [[1, 2], [3, 4]]
    return run(sample)
