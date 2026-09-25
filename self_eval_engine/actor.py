import uuid
from typing import Any, List, Optional
from .schemas import GraphTrajectory


class GraphActor:
    def __init__(self, client: Any = None):
        self.client = client

    def generate_candidate(
        self,
        task: str,
        feedback: Optional[List[str]] = None,
        greedy: bool = False,
    ) -> GraphTrajectory:
        if hasattr(self.client, "generate_trace_and_answer"):
            trace, out, nodes, edges, entropy = self.client.generate_trace_and_answer(
                task, feedback, greedy=greedy
            )
        else:
            # ponytail: mock generator for standalone execution
            trace = f"Executed reasoning trace for: {task}"
            out = f"def solution(): return '{task}'"
            nodes = ["node_start", "node_exec"]
            edges = [("node_start", "node_exec")]
            entropy = 0.2 if greedy else 0.8

        return GraphTrajectory(
            id=str(uuid.uuid4()),
            reasoning_trace=trace,
            terminal_output=out,
            token_entropy=entropy,
            active_node_ids=nodes,
            traversed_edges=edges,
            dynamic_depth_reached=len(nodes),
        )

    def generate_candidates(
        self,
        task: str,
        feedback: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[GraphTrajectory]:
        return [self.generate_candidate(task, feedback, greedy=False) for _ in range(k)]
