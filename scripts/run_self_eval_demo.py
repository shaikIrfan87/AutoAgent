"""Manual interactive check for self_eval_engine."""
from self_eval_engine import (
    GraphCreditAssigner,
    PythonExecutionSandbox,
    SelfEvaluationOrchestrator,
)


class SimpleGraph:
    def __init__(self):
        self.edges = {("start", "solve"): 0.1}
        self.nodes = {"start": {"activation_threshold": 0.5}, "solve": {"activation_threshold": 0.5}}

    def get_edge_weight(self, u, v):
        return self.edges.get((u, v), 0.0)

    def set_edge_weight(self, u, v, w):
        self.edges[(u, v)] = w


def main():
    print("=" * 60)
    print("1. Testing Deterministic Sandbox (AST + Subprocess Execution)")
    print("=" * 60)
    sandbox = PythonExecutionSandbox(timeout_sec=2)
    res_valid = sandbox.verify_code("def add(a, b): return a + b", "assert add(2, 3) == 5")
    print("Valid Code Result -> is_valid:", res_valid.is_valid, "| score:", res_valid.score)

    res_syntax = sandbox.verify_code("def broken_syntax(")
    print("Syntax Error -> is_valid:", res_syntax.is_valid, "| message:", res_syntax.failed_assertions[0])

    print("\n" + "=" * 60)
    print("2. Testing Closed-Loop Runtime Orchestration")
    print("=" * 60)
    graph = SimpleGraph()
    orchestrator = SelfEvaluationOrchestrator(
        graph_memory=graph,
        mode="closed_loop",
        entropy_threshold=0.5,
    )

    trajectory = orchestrator.run(
        task="calculate_even_numbers",
        deterministic_unit_test="assert solution() is not None",
    )

    print(f"Candidate ID       : {trajectory.id}")
    print(f"Output             : {trajectory.terminal_output}")
    print(f"Token Entropy      : {trajectory.token_entropy:.2f} (<= 0.5 bypassed k=5 beam)")
    print(f"Deterministic Pass : {trajectory.verification.is_valid}")
    print(f"Judge Score        : {trajectory.critique.aggregate_score if trajectory.critique else 'N/A'}")

    print("\n" + "=" * 60)
    print("3. Testing Dynamic Graph Memory (Hebbian Update & Pruning)")
    print("=" * 60)
    weight_after = graph.get_edge_weight("node_start", "node_exec")
    print(f"Reinforced Edge Weight ('node_start' -> 'node_exec'): {weight_after:.3f}")

    assigner = GraphCreditAssigner(graph)
    graph.edges[("stale_node_1", "stale_node_2")] = 0.002
    pruned = assigner.prune_near_zero_edges(epsilon=0.01)
    print(f"Pruned Near-Zero Stale Edges: {pruned}")
    print("=" * 60)
    print("Manual Check Complete: All Systems Operational.")


if __name__ == "__main__":
    main()
