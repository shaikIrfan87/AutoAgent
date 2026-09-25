"""
Inductive Program Synthesizer for Abstract Reasoning & Few-Shot Generalization.
"""
import heapq
import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ..core.dsl import Grid, UNARY_PRIMITIVES, replace_color, to_grid, if_then_else, infer_relational_predicates
    from ..core.world_model import LatentWorldModel
except ImportError:
    from core.dsl import Grid, UNARY_PRIMITIVES, replace_color, to_grid, if_then_else, infer_relational_predicates
    from core.world_model import LatentWorldModel

class NeuralPolicyValueNetwork(nn.Module):
    """Dual-headed neural policy and value guidance network for program synthesis."""

    def __init__(self, num_ops: int=22, feature_dim: int=64):
        super().__init__()
        self.num_ops = num_ops
        self.encoder = nn.Sequential(nn.Linear(14, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.policy_head = nn.Linear(feature_dim, num_ops)
        self.value_head = nn.Linear(feature_dim, 1)

    @staticmethod
    def extract_features(grids: List[Grid]) -> torch.Tensor:
        all_feats = []
        for g in grids:
            h, w = (len(g), len(g[0]) if g else 0)
            counts = [0.0] * 10
            total = max(h * w, 1)
            for row in g:
                for c in row:
                    if 0 <= c < 10:
                        counts[c] += 1.0 / total
            dim_feats = [float(h) / 30.0, float(w) / 30.0, float(h) / max(float(w), 1.0), float(total) / 900.0]
            all_feats.append(counts + dim_feats)
        avg_feat = [sum((f[i] for f in all_feats)) / max(len(all_feats), 1) for i in range(14)]
        return torch.tensor([avg_feat], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return (logits, value)

    def train_step(self, states: List[List[Grid]], target_action_indices: List[int], target_values: List[float], lr: float=0.005) -> float:
        """Perform gradient optimization step on policy cross-entropy and value MSE."""
        if not states:
            return 0.0
        feats = torch.cat([self.extract_features(st) for st in states], dim=0)
        target_actions = torch.tensor(target_action_indices, dtype=torch.long)
        target_v = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        logits, values = self.forward(feats)
        loss_p = F.cross_entropy(logits, target_actions)
        loss_v = F.mse_loss(values, target_v)
        loss = loss_p + 0.5 * loss_v
        loss.backward()
        optimizer.step()
        return float(loss.item())

@dataclass
class SynthesizedProgram:
    steps: List[Tuple[str, Tuple]]
    fn: Callable[[Grid], Grid]
    code: str

    def __call__(self, g: Union[List[List[int]], Grid]) -> Grid:
        return self.fn(to_grid(g))

@dataclass
class MCTSProgramNode:
    """Node in PUCT-guided Monte Carlo program synthesis search tree."""
    grids: List[Grid]
    op_chain: List[Tuple[str, Tuple, Callable[[Grid], Grid]]]
    code_expr: str
    parent: Optional['MCTSProgramNode'] = None
    children: List['MCTSProgramNode'] = field(default_factory=list)
    op_taken: Optional[Tuple[str, Tuple, Callable[[Grid], Grid]]] = None
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0
    depth: int = 0

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def puct_score(self, c_puct: float=1.414) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else self.visits
        u = c_puct * self.prior * (math.sqrt(parent_visits) / (1 + self.visits))
        return self.q_value + u

class ProgramSynthesizer:
    """Synthesizes verified programs from few-shot input/output grid pairs using guided search."""

    def __init__(self, max_depth: int=3, max_expansions: int=5000):
        self.max_depth = max_depth
        self.max_expansions = max_expansions

    @staticmethod
    def _heuristic_distance(grids: List[Grid], targets: List[Grid]) -> float:
        """Heuristic cell and dimension mismatch distance (lower is closer)."""
        dist = 0.0
        for g, t in zip(grids, targets):
            if len(g) != len(t) or len(g[0]) != len(t[0]):
                dist += 10.0 + abs(len(g) - len(t)) + abs(len(g[0]) - len(t[0]))
            else:
                dist += sum((c1 != c2 for r1, r2 in zip(g, t) for c1, c2 in zip(r1, r2)))
        return dist

    def _solve_partition_subset(self, subset: List[Tuple[Grid, Grid]]) -> Optional[SynthesizedProgram]:
        """Solve a partitioned subset using identity or 1-to-2 step composition."""
        if all((i == o for i, o in subset)):
            return SynthesizedProgram(steps=[], fn=lambda g: g, code='lambda g: g')
        for name, fn in UNARY_PRIMITIVES.items():
            try:
                if all((fn(i) == o for i, o in subset)):
                    return SynthesizedProgram(steps=[(name, ())], fn=fn, code=name)
            except Exception:
                continue
        for n1, f1 in UNARY_PRIMITIVES.items():
            try:
                t1 = [f1(i) for i, _ in subset]
            except Exception:
                continue
            for n2, f2 in UNARY_PRIMITIVES.items():
                try:
                    if all((f2(mid) == o for mid, (_, o) in zip(t1, subset))):
                        composed_fn = (lambda f_a, f_b: lambda g: f_b(f_a(g)))(f1, f2)
                        return SynthesizedProgram(steps=[(n1, ()), (n2, ())], fn=composed_fn, code=f'{n2}({n1}(g))')
                except Exception:
                    continue
        return None

    def _infer_candidate_ops(self, examples: List[Tuple[Grid, Grid]]) -> List[Tuple[str, Tuple, Callable[[Grid], Grid]]]:
        """Extract candidate unary, parameter-bound, and composite branch operators from example pairs."""
        ops: List[Tuple[str, Tuple, Callable[[Grid], Grid]]] = []
        for name, fn in UNARY_PRIMITIVES.items():
            ops.append((name, (), fn))
        all_in_colors: Set[int] = set()
        all_out_colors: Set[int] = set()
        for inp, outp in examples:
            all_in_colors.update((c for row in inp for c in row))
            all_out_colors.update((c for row in outp for c in row))
        for c1 in all_in_colors:
            for c2 in all_out_colors:
                if c1 != c2:
                    op_fn = (lambda old_c, new_c: lambda g: replace_color(g, old_c, new_c))(c1, c2)
                    ops.append((f'replace_color({c1}, {c2})', (c1, c2), op_fn))
        if len(examples) >= 2:
            preds = infer_relational_predicates(examples)
            for p_name, p_fn in preds:
                try:
                    eval_p = [p_fn(i) for i, _ in examples]
                except Exception:
                    continue
                if any(eval_p) and (not all(eval_p)):
                    true_subset = [(i, o) for (i, o), flag in zip(examples, eval_p) if flag]
                    false_subset = [(i, o) for (i, o), flag in zip(examples, eval_p) if not flag]
                    if true_subset and false_subset:
                        prog_t = self._solve_partition_subset(true_subset)
                        prog_f = self._solve_partition_subset(false_subset)
                        if prog_t and prog_f:
                            branch_fn = if_then_else(p_fn, prog_t.fn, prog_f.fn)
                            ops.append((f'if_then_else({p_name}, {prog_t.code}, {prog_f.code})', (p_name, prog_t.code, prog_f.code), branch_fn))
                            break
        if examples:
            first_in, first_out = examples[0]
            h_in, w_in = (len(first_in), len(first_in[0]))
            h_out, w_out = (len(first_out), len(first_out[0]))
            if h_out < h_in or w_out < w_in:
                try:
                    from ..core.dsl import find_connected_components, crop_bbox
                except ImportError:
                    from core.dsl import find_connected_components, crop_bbox
                for inp, _ in examples[:2]:
                    comps = find_connected_components(inp)
                    for comp in comps:
                        rs = [r for r, _ in comp]
                        cs = [c for _, c in comp]
                        rmin, rmax = (min(rs), max(rs))
                        cmin, cmax = (min(cs), max(cs))
                        if rmax - rmin + 1 == h_out and cmax - cmin + 1 == w_out:
                            bbox = (rmin, rmax, cmin, cmax)
                            bbox_name = f'crop_bbox_{rmin}_{cmin}'
                            if not any((o[0] == bbox_name for o in ops)):
                                ops.append((bbox_name, bbox, (lambda b=bbox: lambda g: crop_bbox(g, b))()))
        return ops

    def synthesize(self, examples: List[Tuple[Union[List[List[int]], Grid], Union[List[List[int]], Grid]]]) -> Optional[SynthesizedProgram]:
        """Search shortest composition of DSL primitives satisfying all training demonstrations."""
        norm_examples = [(to_grid(i), to_grid(o)) for i, o in examples]
        if not norm_examples:
            return None
        if all((i == o for i, o in norm_examples)):
            return SynthesizedProgram(steps=[], fn=lambda g: g, code='lambda g: g')
        candidate_ops = self._infer_candidate_ops(norm_examples)
        targets = [o for _, o in norm_examples]
        counter = itertools.count()
        initial_inputs = [i for i, _ in norm_examples]
        init_h = self._heuristic_distance(initial_inputs, targets)
        queue: List[Tuple[float, int, int, List[Grid], List[Tuple[str, Tuple, Callable[[Grid], Grid]]], str]] = [(init_h, 0, next(counter), initial_inputs, [], 'g')]
        visited_states: Set[Tuple[Grid, ...]] = {tuple(initial_inputs)}
        expansions = 0
        while queue and expansions < self.max_expansions:
            _, depth, _, current_grids, op_chain, code_expr = heapq.heappop(queue)
            expansions += 1
            if depth >= self.max_depth:
                continue
            for op_name, args, op_fn in candidate_ops:
                try:
                    transformed = [op_fn(g) for g in current_grids]
                except Exception:
                    continue
                if all((t == o for t, o in zip(transformed, targets))):
                    new_chain = op_chain + [(op_name, args, op_fn)]
                    final_code = f'{op_name}({code_expr})' if code_expr != 'g' else f'{op_name}(g)'

                    def composed(grid: Grid, chain=new_chain) -> Grid:
                        curr = grid
                        for _, _, fn in chain:
                            curr = fn(curr)
                        return curr
                    return SynthesizedProgram(steps=[(name, a) for name, a, _ in new_chain], fn=composed, code=final_code)
                state_key = tuple(transformed)
                if state_key not in visited_states and depth + 1 < self.max_depth:
                    visited_states.add(state_key)
                    h = self._heuristic_distance(transformed, targets)
                    new_code = f'{op_name}({code_expr})' if code_expr != 'g' else f'{op_name}(g)'
                    heapq.heappush(queue, (float(depth + 1) + h, depth + 1, next(counter), transformed, op_chain + [(op_name, args, op_fn)], new_code))
        return None

    def benchmark(self, eval_suite: Optional[List[Tuple[str, List[Tuple[Grid, Grid]]]]]=None) -> Dict[str, Any]:
        """Benchmark synthesis accuracy and latency across canonical grid tasks."""
        if eval_suite is None:
            eval_suite = [('identity', [(((1, 2), (3, 4)), ((1, 2), (3, 4)))]), ('rot90', [(((1, 2), (3, 4)), ((3, 1), (4, 2)))]), ('flip_h', [(((1, 2), (3, 4)), ((2, 1), (4, 3)))]), ('replace_color', [(((1, 2), (3, 4)), ((9, 2), (3, 4)))]), ('rot90_replace', [(((1, 0), (0, 0)), ((0, 7), (0, 0)))])]
        results = []
        start = time.perf_counter()
        for name, pairs in eval_suite:
            prog = self.synthesize(pairs)
            results.append({'name': name, 'solved': prog is not None, 'code': prog.code if prog else None})
        elapsed = time.perf_counter() - start
        solved_count = sum((1 for r in results if r['solved']))
        return {'total': len(results), 'solved': solved_count, 'accuracy': solved_count / len(results) if results else 0.0, 'elapsed_seconds': round(elapsed, 4), 'details': results}

class MCTSProgramSynthesizer(ProgramSynthesizer):
    """Monte Carlo Tree Search program synthesizer with PUCT exploration and neural policy/value guidance."""

    def __init__(self, max_depth: int=5, max_expansions: int=1500, c_puct: float=1.414, use_neural_guidance: bool=True):
        super().__init__(max_depth=max_depth, max_expansions=max_expansions)
        self.c_puct = c_puct
        self.use_neural_guidance = use_neural_guidance
        self.neural_net = NeuralPolicyValueNetwork(num_ops=len(UNARY_PRIMITIVES))
        self.world_model = LatentWorldModel(action_dim=len(UNARY_PRIMITIVES))

    def train_policy_value(self, states: List[List[Grid]], action_indices: List[int], values: List[float], lr: float=0.005) -> float:
        """Train the neural policy/value network directly on search trajectories."""
        return self.neural_net.train_step(states, action_indices, values, lr=lr)

    def train_world_model_step(self, s_t: torch.Tensor, a_idx: int, s_tp1: torch.Tensor, reward_target: float, risk_target: float, lr: float=0.005) -> float:
        """Train latent transition dynamics on real observed state transitions."""
        return self.world_model.train_transition_step(s_t, a_idx, s_tp1, reward_target, risk_target, lr=lr)

    def _compute_op_prior(self, op_name: str, examples: List[Tuple[Grid, Grid]]) -> float:
        """Assign prior probability P(a) based on domain invariants."""
        dim_transposed = any((len(i) == len(o[0]) and len(i[0]) == len(o) and (len(i) != len(i[0])) for i, o in examples))
        in_colors = {c for i, _ in examples for row in i for c in row}
        out_colors = {c for _, o in examples for row in o for c in row}
        colors_differ = in_colors != out_colors
        prior = 1.0
        if 'replace_color' in op_name or 'label_components' in op_name:
            prior *= 3.0 if colors_differ else 0.3
        if op_name in ('rot90', 'rot270', 'transpose'):
            if dim_transposed:
                prior *= 4.0
        if 'shift' in op_name or 'crop' in op_name:
            prior *= 1.5
        return prior

    def _evaluate_reward(self, grids: List[Grid], targets: List[Grid], depth: int) -> float:
        """Continuous reward combining pixel match ratio, color histogram, and MDL penalty."""
        total_sim = 0.0
        for g, t in zip(grids, targets):
            h_g, w_g = (len(g), len(g[0]))
            h_t, w_t = (len(t), len(t[0]))
            if h_g == h_t and w_g == w_t:
                pixel_match = sum((c1 == c2 for r1, r2 in zip(g, t) for c1, c2 in zip(r1, r2))) / (h_g * w_g)
            else:
                pixel_match = 0.0
            c_g = Counter((c for row in g for c in row))
            c_t = Counter((c for row in t for c in row))
            all_cols = set(c_g.keys()) | set(c_t.keys())
            total_cells = max(h_g * w_g, h_t * w_t)
            hist_overlap = sum((min(c_g[c], c_t[c]) for c in all_cols)) / total_cells if total_cells > 0 else 0.0
            total_sim += 0.7 * pixel_match + 0.3 * hist_overlap
        avg_sim = total_sim / len(grids)
        mdl_penalty = 0.015 * depth
        return max(0.0, avg_sim - mdl_penalty)

    def synthesize(self, examples: List[Tuple[Union[List[List[int]], Grid], Union[List[List[int]], Grid]]]) -> Optional[SynthesizedProgram]:
        """Search composition using PUCT Monte Carlo Tree Search."""
        norm_examples = [(to_grid(i), to_grid(o)) for i, o in examples]
        if not norm_examples:
            return None
        if all((i == o for i, o in norm_examples)):
            return SynthesizedProgram(steps=[], fn=lambda g: g, code='lambda g: g')
        fast_prog = super().synthesize(examples)
        if fast_prog is not None:
            return fast_prog
        candidate_ops = self._infer_candidate_ops(norm_examples)
        targets = [o for _, o in norm_examples]
        initial_inputs = [i for i, _ in norm_examples]
        if self.use_neural_guidance:
            with torch.no_grad():
                feats = self.neural_net.extract_features(initial_inputs)
                logits, _ = self.neural_net(feats)
                probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
            raw_priors = [probs[idx % len(probs)] * 0.6 + self._compute_op_prior(name, norm_examples) * 0.4 for idx, (name, _, _) in enumerate(candidate_ops)]
        else:
            raw_priors = [self._compute_op_prior(name, norm_examples) for name, _, _ in candidate_ops]
        total_p = sum(raw_priors) or 1.0
        norm_priors = [p / total_p for p in raw_priors]
        root = MCTSProgramNode(grids=initial_inputs, op_chain=[], code_expr='g', depth=0)
        seen_signatures: Set[Tuple[Tuple[Tuple[int, ...], ...], ...]] = {tuple((tuple((tuple(r) for r in g)) for g in initial_inputs))}
        for _ in range(self.max_expansions):
            node = root
            while node.children and node.depth < self.max_depth:
                node = max(node.children, key=lambda ch: ch.puct_score(self.c_puct))
            if not node.children and node.depth < self.max_depth:
                op_keys = list(UNARY_PRIMITIVES.keys())
                s_feats = self.neural_net.extract_features(node.grids) if self.use_neural_guidance else None
                for idx, ((op_name, args, op_fn), prior) in enumerate(zip(candidate_ops, norm_priors)):
                    op_idx = op_keys.index(op_name) if op_name in op_keys else idx % len(op_keys)
                    if self.use_neural_guidance and s_feats is not None:
                        action_seq = [op_keys.index(n) if n in op_keys else 0 for n, _, _ in node.op_chain] + [op_idx]
                        _, max_risk = self.world_model.imagine_rollout(s_feats, action_seq)
                        if max_risk > 0.85:
                            continue
                    try:
                        transformed = [op_fn(g) for g in node.grids]
                    except Exception:
                        continue
                    state_sig = tuple((tuple((tuple(r) for r in g)) for g in transformed))
                    if state_sig in seen_signatures:
                        continue
                    seen_signatures.add(state_sig)
                    new_code = f'{op_name}({node.code_expr})' if node.code_expr != 'g' else f'{op_name}(g)'
                    child = MCTSProgramNode(grids=transformed, op_chain=node.op_chain + [(op_name, args, op_fn)], code_expr=new_code, parent=node, op_taken=(op_name, args, op_fn), prior=prior, depth=node.depth + 1)
                    node.children.append(child)
                    if all((t == o for t, o in zip(transformed, targets))):
                        chain = child.op_chain

                        def composed(grid: Grid, c=chain) -> Grid:
                            curr = grid
                            for _, _, fn in c:
                                curr = fn(curr)
                            return curr
                        return SynthesizedProgram(steps=[(n, a) for n, a, _ in chain], fn=composed, code=child.code_expr)
            eval_node = node.children[0] if node.children else node
            reward = self._evaluate_reward(eval_node.grids, targets, eval_node.depth)
            if self.use_neural_guidance:
                with torch.no_grad():
                    val_feat = self.neural_net.extract_features(eval_node.grids)
                    _, v_pred = self.neural_net(val_feat)
                    v_val = max(0.0, float(v_pred.item()))
                    op_keys = list(UNARY_PRIMITIVES.keys())
                    path_actions = [op_keys.index(n) if n in op_keys else 0 for n, _, _ in eval_node.op_chain]
                    imagined_rew, _ = self.world_model.imagine_rollout(val_feat, path_actions)
                    reward = 0.6 * reward + 0.2 * v_val + 0.2 * max(0.0, imagined_rew)
            curr: Optional[MCTSProgramNode] = eval_node
            while curr is not None:
                curr.visits += 1
                curr.total_value += reward
                curr = curr.parent
        return None

def get_search_depth(self) -> int:
    return 6