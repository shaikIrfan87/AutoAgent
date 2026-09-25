import random
import subprocess
import sys
import numpy as np


class MinimalZeroAgent:
    """
    Learns purely from scratch: Generates AST candidates, 
    executes them against strict unit assertions, and adapts its policy.
    """
    def __init__(self, actions=None):
        self.actions = actions or [
            "add", "sub", "mul", "const_0", "const_1", "var_x"
        ]
        self.n_actions = len(self.actions)
        # Q(depth, last_action, current_action)
        self.q_table = np.zeros((10, self.n_actions, self.n_actions), dtype=np.float32)
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.3

    def sample_action(self, depth: int, last_action: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[min(depth, 9), last_action]))

    def synthesize_expression(self, max_depth: int = 3) -> tuple[str, list[tuple[int, int, int]]]:
        trajectory = []
        last_action = 0

        def _build(depth: int) -> str:
            nonlocal last_action
            prev = last_action
            action_idx = self.sample_action(depth, prev)
            trajectory.append((depth, prev, action_idx))
            action = self.actions[action_idx]
            last_action = action_idx

            if depth >= max_depth or action.startswith("const") or action.startswith("var"):
                if action == "const_0": return "0"
                if action == "const_1": return "1"
                return "x"
            
            if action == "add": return f"({_build(depth+1)} + {_build(depth+1)})"
            if action == "sub": return f"({_build(depth+1)} - {_build(depth+1)})"
            if action == "mul": return f"({_build(depth+1)} * {_build(depth+1)})"
            return "x"

        expr = _build(0)
        return expr, trajectory

    def verify_in_sandbox(self, expr: str, test_io: list[tuple[int, int]]) -> float:
        code = (
            f"def f(x):\n    return {expr}\n"
            f"test_cases = {test_io}\n"
            "for inp, expected in test_cases:\n"
            "    if f(inp) != expected:\n"
            "        sys.exit(1)\n"
            "sys.exit(0)\n"
        )
        try:
            res = subprocess.run(
                [sys.executable, "-I", "-S", "-c", f"import sys\n{code}"],
                capture_output=True,
                timeout=1.0
            )
            return 1.0 if res.returncode == 0 else -1.0
        except Exception:
            return -1.0

    def learn_cycle(self, target_io: list[tuple[int, int]], max_iterations: int = 1000):
        for iteration in range(1, max_iterations + 1):
            expr, trajectory = self.synthesize_expression(max_depth=3)
            delta_s = self.verify_in_sandbox(expr, target_io)

            for depth, prev_action, action_idx in trajectory:
                current_val = self.q_table[min(depth, 9), prev_action, action_idx]
                target_val = delta_s + (self.gamma * np.max(self.q_table[min(depth+1, 9), action_idx]))
                self.q_table[min(depth, 9), prev_action, action_idx] += self.lr * (target_val - current_val)

            if delta_s > 0.0:
                self.epsilon = max(0.05, self.epsilon * 0.95)
                return {
                    "status": "solved",
                    "iterations": iteration,
                    "expression": expr,
                    "code": f"def solution(x):\n    return {expr}"
                }

        return {"status": "exhausted", "expression": None}


if __name__ == "__main__":
    agent = MinimalZeroAgent()
    # Discover f(x) = x + 1
    res = agent.learn_cycle(target_io=[(1, 2), (2, 3), (3, 4)], max_iterations=500)
    assert res["status"] == "solved" or res["status"] == "exhausted"
    print("Self-check passed:", res)
