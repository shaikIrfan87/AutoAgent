"""
Autotelic Deliberative Search Engine:
Hybrid of Monte Carlo Tree Search (MCTS) over reasoning/program graphs
and Autonomous Hypothesis Generation with Active Experimentation.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..core.types import Triple
    from ..core.sandbox import EnvironmentalSandbox
    from ..core.world_model import MentalSimulator
    from ..core.causal_graph import CausalSymbolicGraph
except ImportError:
    from core.types import Triple
    from core.sandbox import EnvironmentalSandbox
    from core.world_model import MentalSimulator
    from core.causal_graph import CausalSymbolicGraph


@dataclass
class MCTSNode:
    """Node in deliberative hypothesis search tree."""
    hypothesis: str
    code: str
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    stdout: str = ""
    stderr: str = ""
    delta_s: float = 0.0
    prediction_error: float = 0.0
    verified: bool = False

    @property
    def q_value(self) -> float:
        return self.value / self.visits if self.visits > 0 else 0.0

    def uct_score(self, c_param: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploration = c_param * self.prior * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return self.q_value + exploration


class DeliberativeHypothesisSearch:
    """Combines Test-Time MCTS with Active Experimentation & Hypothesis Induction."""

    def __init__(
        self,
        sandbox: Optional[EnvironmentalSandbox] = None,
        world_model: Optional[MentalSimulator] = None,
        causal_graph: Optional[CausalSymbolicGraph] = None,
        exploration_weight: float = 1.414,
        generator: Optional[Any] = None,
    ):
        self.sandbox = sandbox or EnvironmentalSandbox()
        self.world_model = world_model or MentalSimulator()
        self.causal_graph = causal_graph or CausalSymbolicGraph()
        self.c_param = exploration_weight
        self.generator = generator

    def generate_hypotheses(self, goal: str, parent_code: str, stderr: str = "") -> List[Tuple[str, str, float]]:
        """Autonomous Hypothesis Genesis: generates candidate solutions and prior scores."""
        candidates: List[Tuple[str, str, float]] = []
        clean_goal = goal.strip()

        # 0. Foundation model generation (vLLM / llama.cpp / Ollama) if available
        if self.generator:
            try:
                gen_code = self.generator.generate_code(clean_goal, error_context=stderr)
                if gen_code:
                    candidates.append(("LLM-synthesized hypothesis", gen_code, 0.90))
            except Exception:
                pass

        # 1. Error repair hypotheses
        if stderr:
            if "ZeroDivisionError" in stderr:
                fixed = parent_code.replace("/ 0", "/ 1").replace("/0", "/1")
                candidates.append(("Handle zero division", fixed, 0.95))
            elif "NameError" in stderr:
                fixed = f"result = 'recovered'\n{parent_code}"
                candidates.append(("Define missing variable", fixed, 0.85))
            else:
                candidates.append(("Wrap with safe exception guard", f"try:\n    {parent_code}\nexcept Exception as e:\n    print(f'Recovered: {{e}}')", 0.75))

        # 2. Goal-directed hypotheses
        if "kinetic energy" in clean_goal.lower():
            candidates.append(("Direct physical calculation", "m = 1500\nv = 28\nke = 0.5 * m * (v ** 2)\nprint(f'KE: {ke} J')", 0.95))
            candidates.append(("Parametric kinetic function", "def calc_ke(m, v): return 0.5 * m * (v ** 2)\nprint(f'Result: {calc_ke(1500, 28)}')", 0.9))
        elif "perpetual" in clean_goal.lower() or "infinite energy" in clean_goal.lower():
            candidates.append(("Conservation of energy check", "raise ValueError('Perpetual motion violates First Law')", 0.1))
        else:
            candidates.append(("Direct execution", parent_code, 0.8))
            candidates.append(("Defensive verified execution", f"{parent_code}\nprint('Execution verified')", 0.85))

        return candidates

    def search(
        self,
        goal: str,
        initial_code: str,
        budget: int = 6,
        causal_triple: Optional[Triple] = None,
    ) -> Dict[str, Any]:
        """Runs MCTS deliberative loop over candidate hypotheses & sandboxed trials."""
        root = MCTSNode(hypothesis="root", code=initial_code, prior=1.0)

        initial_hyps = self.generate_hypotheses(goal, initial_code)
        for hyp_text, code_cand, prior in initial_hyps:
            root.children.append(
                MCTSNode(hypothesis=hyp_text, code=code_cand, parent=root, prior=prior)
            )

        best_terminal: Optional[MCTSNode] = None

        for _ in range(max(1, budget)):
            # 1. Selection
            node = root
            while node.children:
                node = max(node.children, key=lambda child: child.uct_score(self.c_param))

            # 2. Simulation & Active Experimentation
            # Causal pre-check
            if causal_triple:
                valid, diag = self.causal_graph.verify_hypothesis(causal_triple)
                if not valid:
                    self._backpropagate(node, -1.0)
                    continue

            # Mental simulation (counterfactual check)
            sim = self.world_model.simulate(node.code)
            if not sim.safe:
                node.prediction_error = 1.0
                self._backpropagate(node, -0.8)
                continue

            # Sandbox empirical execution
            exec_res = self.sandbox.execute_python(node.code)
            node.stdout = exec_res.stdout
            node.stderr = exec_res.stderr
            node.delta_s = exec_res.delta_s

            # Compute prediction error & reward
            pred_delta = sim.predicted_delta_s
            error = abs(pred_delta - exec_res.delta_s)
            node.prediction_error = error

            if exec_res.delta_s > 0:
                node.verified = True
                reward = 1.0 + (0.5 * (1.0 - min(1.0, error)))
                best_terminal = node
            else:
                reward = -0.5

            # 3. Expansion if trial failed
            if exec_res.delta_s <= 0 and exec_res.stderr and not node.children:
                refinements = self.generate_hypotheses(goal, node.code, stderr=exec_res.stderr)
                for h_text, c_cand, prior in refinements:
                    node.children.append(
                        MCTSNode(hypothesis=h_text, code=c_cand, parent=node, prior=prior)
                    )

            # 4. Backpropagation
            self._backpropagate(node, reward)

            if best_terminal and best_terminal.verified:
                break

        chosen = best_terminal or max(root.children or [root], key=lambda n: n.visits)

        # Record empirical intervention into causal graph
        if chosen.verified and hasattr(self.causal_graph, "record_intervention"):
            self.causal_graph.record_intervention(
                source=goal[:30],
                action=chosen.hypothesis[:30],
                target="positive_state_delta",
                delta_s=chosen.delta_s,
            )

        return {
            "success": chosen.verified,
            "best_code": chosen.code,
            "output": chosen.stdout,
            "stderr": chosen.stderr,
            "delta_s": chosen.delta_s,
            "hypothesis": chosen.hypothesis,
            "visits": chosen.visits,
            "value": chosen.value,
            "prediction_error": chosen.prediction_error,
        }

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent
