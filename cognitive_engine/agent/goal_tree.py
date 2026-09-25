"""
Hierarchical Task Network (HTN): Long-horizon goal decomposition, dependency DAG, and state suspension.
"""

from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class GoalNode:
    id: str
    description: str
    parent_id: Optional[str] = None
    subgoal_ids: List[str] = field(default_factory=list)
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    result: Optional[str] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    code: str = ""


class GoalTree:
    """Hierarchical Task Network managing long-horizon goal execution across interruptions."""

    def __init__(self, root_id: str = "root", root_description: str = "Root Goal"):
        self.nodes: Dict[str, GoalNode] = {}
        self.root_id = root_id
        self.nodes[root_id] = GoalNode(id=root_id, description=root_description)

    def decompose(self, parent_id: str, subgoals: List[tuple[str, str]]) -> List[str]:
        """Decomposes a parent goal into an ordered sequence of subgoals (id, description)."""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent goal '{parent_id}' not found")

        created_ids = []
        for sg_id, desc in subgoals:
            self.nodes[sg_id] = GoalNode(id=sg_id, description=desc, parent_id=parent_id)
            self.nodes[parent_id].subgoal_ids.append(sg_id)
            created_ids.append(sg_id)
        return created_ids

    def get_next_actionable_goal(self) -> Optional[GoalNode]:
        """Returns the next pending leaf node with no uncompleted subgoals."""
        for node in self.nodes.values():
            if node.status == "pending":
                # Only actionable if it has no children or all children are complete
                if not node.subgoal_ids:
                    return node
                child_nodes = [self.nodes[cid] for cid in node.subgoal_ids]
                if all(c.status == "completed" for c in child_nodes):
                    return node
        return None

    def mark_completed(self, goal_id: str, result: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        if goal_id in self.nodes:
            self.nodes[goal_id].status = "completed"
            self.nodes[goal_id].result = result
            if context:
                self.nodes[goal_id].context.update(context)
            # Propagate completion up if all siblings are complete
            parent_id = self.nodes[goal_id].parent_id
            if parent_id and parent_id in self.nodes:
                parent = self.nodes[parent_id]
                if all(self.nodes[cid].status == "completed" for cid in parent.subgoal_ids):
                    self.mark_completed(parent_id, "All subgoals achieved")

    def mark_failed(self, goal_id: str, error: str = "") -> None:
        if goal_id in self.nodes:
            self.nodes[goal_id].status = "failed"
            self.nodes[goal_id].error = error

    def reset_subgoal(self, goal_id: str) -> None:
        """Resets a failed or in-progress subgoal for local re-execution without discarding prior progress."""
        if goal_id in self.nodes:
            self.nodes[goal_id].status = "pending"
            self.nodes[goal_id].error = None

    def get_execution_progress(self) -> Dict[str, Any]:
        """Returns summary progress metrics for all goals in the HTN DAG."""
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for node in self.nodes.values():
            if node.id != self.root_id or not node.subgoal_ids:
                counts[node.status] = counts.get(node.status, 0) + 1
        total = sum(counts.values())
        return {
            "total": total,
            "completed": counts.get("completed", 0),
            "pending": counts.get("pending", 0),
            "failed": counts.get("failed", 0),
            "in_progress": counts.get("in_progress", 0),
            "pct_complete": round((counts.get("completed", 0) / total) * 100.0, 1) if total > 0 else 0.0,
        }

    def serialize(self) -> str:
        data = {k: v.__dict__ for k, v in self.nodes.items()}
        return json.dumps({"root_id": self.root_id, "nodes": data})

    @classmethod
    def deserialize(cls, json_str: str) -> "GoalTree":
        raw = json.loads(json_str)
        tree = cls(root_id=raw["root_id"], root_description="")
        tree.nodes = {k: GoalNode(**v) for k, v in raw["nodes"].items()}
        return tree


class UnsupervisedGoalInducer:
    """
    Autonomous Goal Induction & Problem Framing:
    Extracts implicit subgoals, state invariants, and executable test specifications
    from messy environmental states (S_0, S_target) or observational demonstration pairs,
    without requiring human assertion strings or regex template hints.
    """

    @staticmethod
    def extract_state_invariants(initial_state: Any, target_state: Any) -> Dict[str, Any]:
        """
        Analyzes discrepancy between initial and target states to infer invariant properties:
        - length/cardinality changes (filtering vs expansion vs permutation)
        - monotonicity / sorting
        - symmetry / spatial geometry
        - object preservation
        """
        invariants: Dict[str, Any] = {
            "type_match": type(initial_state) == type(target_state),
            "is_permutation": False,
            "is_filter": False,
            "is_monotonic_increase": False,
            "is_monotonic_decrease": False,
            "spatial_transformation": None,
            "value_delta": None,
        }

        # List / sequence analysis
        if isinstance(initial_state, (list, tuple)) and isinstance(target_state, (list, tuple)):
            n_in = len(initial_state)
            n_out = len(target_state)

            if n_in == n_out and sorted(str(x) for x in initial_state) == sorted(str(x) for x in target_state):
                invariants["is_permutation"] = True
                try:
                    if list(target_state) == sorted(target_state):
                        invariants["is_monotonic_increase"] = True
                    elif list(target_state) == sorted(target_state, reverse=True):
                        invariants["is_monotonic_decrease"] = True
                except Exception:
                    pass
            elif n_out < n_in and all(x in initial_state for x in target_state):
                invariants["is_filter"] = True

            # 2D Grid analysis
            if n_in > 0 and isinstance(initial_state[0], (list, tuple)) and n_out > 0 and isinstance(target_state[0], (list, tuple)):
                h_in, w_in = len(initial_state), len(initial_state[0])
                h_out, w_out = len(target_state), len(target_state[0])
                if (h_in, w_in) == (w_out, h_out):
                    invariants["spatial_transformation"] = "transposition_or_rotation"
                elif (h_in, w_in) == (h_out, w_out):
                    invariants["spatial_transformation"] = "in_place_projection"

        elif isinstance(initial_state, (int, float)) and isinstance(target_state, (int, float)):
            invariants["value_delta"] = float(target_state - initial_state)

        return invariants

    @classmethod
    def cross_validate_invariants(
        cls,
        transition_pairs: List[Tuple[Any, Any]],
        min_confidence: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Cross-validates candidate invariants across multiple independent state transitions.
        Computes Laplace-smoothed Bayesian posterior confidence:
            C = (k + 1) / (N + 2)
        Prunes coincidental correlations to prevent premature over-constraining.
        """
        if not transition_pairs:
            return {}

        N = len(transition_pairs)
        if N == 1:
            inv = cls.extract_state_invariants(transition_pairs[0][0], transition_pairs[0][1])
            inv["confidence_scores"] = {k: 0.67 for k, v in inv.items() if v}
            return inv

        all_invs = [cls.extract_state_invariants(s0, s1) for s0, s1 in transition_pairs]
        keys = ["is_permutation", "is_filter", "is_monotonic_increase", "is_monotonic_decrease", "spatial_transformation"]

        consensus_invariants: Dict[str, Any] = {
            "type_match": all(inv.get("type_match") for inv in all_invs),
            "is_permutation": False,
            "is_filter": False,
            "is_monotonic_increase": False,
            "is_monotonic_decrease": False,
            "spatial_transformation": None,
            "confidence_scores": {},
        }

        for key in keys:
            if key == "spatial_transformation":
                transforms = [inv.get(key) for inv in all_invs if inv.get(key)]
                if transforms:
                    most_common = max(set(transforms), key=transforms.count)
                    k = transforms.count(most_common)
                    conf = (k + 1.0) / (N + 2.0)
                    consensus_invariants["confidence_scores"][key] = conf
                    if conf >= min_confidence and k >= (N // 2):
                        consensus_invariants[key] = most_common
            else:
                k = sum(1 for inv in all_invs if inv.get(key) is True)
                conf = (k + 1.0) / (N + 2.0)
                consensus_invariants["confidence_scores"][key] = conf
                if conf >= min_confidence and k == N:
                    consensus_invariants[key] = True

        return consensus_invariants

    @classmethod
    def frame_problem_to_htn(
        cls,
        initial_state: Any,
        target_state: Optional[Any] = None,
        goal_name: str = "Induced Problem",
        transition_history: Optional[List[Tuple[Any, Any]]] = None,
    ) -> GoalTree:
        """
        Synthesizes a GoalTree (HTN) containing decomposing subgoals derived from invariant discovery.
        Supports single (initial_state, target_state) or cross-validated transition_history.
        """
        tree = GoalTree(root_id="root", root_description=goal_name)
        if transition_history:
            invariants = cls.cross_validate_invariants(transition_history)
        elif target_state is not None:
            invariants = cls.extract_state_invariants(initial_state, target_state)
        else:
            invariants = {}

        subgoals: List[tuple[str, str]] = []
        if invariants.get("is_filter"):
            subgoals.append(("sg_1_filter", "Extract and retain only elements satisfying target retention predicate"))
            subgoals.append(("sg_2_reconstruct", "Reassemble filtered collection in consistent sequence"))
        elif invariants.get("is_monotonic_increase"):
            subgoals.append(("sg_1_order_metric", "Compute monotonic ordering key for elements"))
            subgoals.append(("sg_2_sort_partition", "Permute sequence into monotonically ascending order"))
        elif invariants.get("is_permutation"):
            subgoals.append(("sg_1_permute", "Reorder elements to match target permutation alignment"))
        elif invariants.get("spatial_transformation"):
            subgoals.append(("sg_1_spatial_orient", "Rotate, transpose, or reflect grid axes"))
            subgoals.append(("sg_2_object_mask", "Project object component masks onto target canvas"))
        else:
            subgoals.append(("sg_1_synthesize_delta", "Apply functional state transformation: S_0 -> S_*"))

        tree.decompose("root", subgoals)
        return tree


    @classmethod
    def frame_unsupervised_io_pairs(cls, observation_history: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        """
        Cleans and formats raw observational state transitions into deterministic I/O pairs for the synthesizer.
        """
        io_pairs = []
        for s_before, s_after in observation_history:
            if s_before is not None and s_after is not None:
                io_pairs.append((s_before, s_after))
        return io_pairs

