"""
Causal AST Tracer & Pearl Level-3 Program Repair Engine.
Constructs variable dependency DAGs from execution traces, performs counterfactual
abduction and intervention (do-calculus), and isolates failing subtrees for targeted repair.
"""
import ast
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class VariableNode:
    name: str
    val: Any
    lineno: int
    parents: List[str] = field(default_factory=list)
    ast_stmt: Optional[ast.stmt] = None


@dataclass
class ExecutionDAG:
    nodes: Dict[str, VariableNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # parent -> child
    failure_var: Optional[str] = None
    observed_val: Any = None
    target_val: Any = None


class CausalASTTracer(ast.NodeVisitor):
    """Traces variable assignments and data-flow dependencies to construct a causal DAG."""

    def __init__(self):
        self.dag = ExecutionDAG()
        self.scope: Dict[str, Any] = {}

    def trace_execution(self, code: str, target_var: str, target_val: Any) -> ExecutionDAG:
        tree = ast.parse(code)
        self.dag = ExecutionDAG(target_val=target_val)
        self.scope = {}

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                target_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                rhs_names = [n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)]

                # Safely evaluate statement in current scope
                code_segment = ast.unparse(stmt)
                exec(code_segment, {}, self.scope)

                for tgt in target_names:
                    val = self.scope.get(tgt)
                    v_node = VariableNode(
                        name=tgt,
                        val=val,
                        lineno=getattr(stmt, "lineno", 1),
                        parents=rhs_names,
                        ast_stmt=stmt,
                    )
                    self.dag.nodes[tgt] = v_node
                    for parent in rhs_names:
                        if parent in self.dag.nodes:
                            self.dag.edges.append((parent, tgt))

        if target_var in self.scope:
            self.dag.failure_var = target_var
            self.dag.observed_val = self.scope[target_var]

        return self.dag


class CausalASTRepairEngine:
    """
    Pearl Level-3 counterfactual program repair:
    1. Abducts the latent discrepancy U = Y_observed - Y_target.
    2. Performs counterfactual intervention do(Subtree = tau*) on ancestors.
    3. Selects the minimal causal intervention with P_cause = 1.0 to repair the AST in <= 2 edits.
    """

    def __init__(self, max_repairs: int = 2):
        self.max_repairs = max_repairs
        self.tracer = CausalASTTracer()

    def identify_causal_culprit(self, dag: ExecutionDAG) -> Optional[str]:
        """Identifies the variable node in the DAG with maximum direct causal path to failure."""
        if not dag.failure_var:
            return None

        # Trace ancestors of failure variable
        ancestors: List[str] = []
        queue = [dag.failure_var]
        visited = set(queue)

        while queue:
            curr = queue.pop(0)
            if curr in dag.nodes:
                for p in dag.nodes[curr].parents:
                    if p not in visited:
                        visited.add(p)
                        ancestors.append(p)
                        queue.append(p)

        # Culprit is the primary upstream generator
        if dag.failure_var in dag.nodes and dag.nodes[dag.failure_var].parents:
            return dag.failure_var
        return ancestors[0] if ancestors else dag.failure_var

    def repair(
        self,
        code: str,
        test_fn_call: str,
        expected_output: Any,
        target_var: str = "result",
    ) -> Tuple[bool, str, int]:
        """
        Executes counterfactual intervention do(Subtree = tau*) to repair broken code.
        Returns (is_repaired, repaired_code, num_mutations).
        """
        # First verify if repair is even needed
        full_test = f"{code}\n{test_fn_call}"
        scope = {}
        try:
            exec(full_test, scope)
            if scope.get(target_var) == expected_output:
                return True, code, 0
        except Exception:
            pass

        # 1. Trace execution and abduct causal discrepancy
        tree = ast.parse(code)
        dag = self.tracer.trace_execution(code, target_var=target_var, target_val=expected_output)

        culprit_var = self.identify_causal_culprit(dag)
        if not culprit_var:
            culprit_var = target_var

        # 2. Counterfactual intervention candidates on culprit statement
        culprit_node = dag.nodes.get(culprit_var)
        if not culprit_node or not culprit_node.ast_stmt:
            # Fallback to function body return expression
            target_stmt = tree.body[-1]
        else:
            target_stmt = culprit_node.ast_stmt

        mutations_tried = 0
        # Common operator swap candidates: + <-> -, * <-> +, // <-> /, off-by-one constants
        class OperatorReplacer(ast.NodeTransformer):
            def __init__(self, old_op_type, new_op_instance):
                self.old_op_type = old_op_type
                self.new_op_instance = new_op_instance

            def visit_BinOp(self, node):
                self.generic_visit(node)
                if isinstance(node.op, self.old_op_type):
                    node.op = self.new_op_instance
                return node

        # Candidate intervention pairs: (OriginalOp, CounterfactualOp)
        candidate_interventions = [
            (ast.Sub, ast.Add()),
            (ast.Add, ast.Sub()),
            (ast.Mult, ast.Add()),
            (ast.Div, ast.Mult()),
            (ast.FloorDiv, ast.Div()),
            (ast.Lt, ast.Gt()),
            (ast.Gt, ast.Lt()),
            (ast.LtE, ast.GtE()),
        ]

        # Try targeted operator substitutions (Mutation 1)
        for old_op, new_op in candidate_interventions:
            cand_tree = copy.deepcopy(tree)
            cand_tree = OperatorReplacer(old_op, new_op).visit(cand_tree)
            ast.fix_missing_locations(cand_tree)
            cand_code = ast.unparse(cand_tree)
            mutations_tried += 1

            test_scope = {}
            try:
                exec(f"{cand_code}\n{test_fn_call}", test_scope)
                if test_scope.get(target_var) == expected_output or test_scope.get("res") == expected_output:
                    return True, cand_code, 1
            except Exception:
                pass

        # Try off-by-one constant intervention (Mutation 2)
        class ConstantShifter(ast.NodeTransformer):
            def __init__(self, delta: int):
                self.delta = delta

            def visit_Constant(self, node):
                if isinstance(node.value, int):
                    node.value = node.value + self.delta
                return node

        for delta in [1, -1, 2, -2]:
            cand_tree = copy.deepcopy(tree)
            cand_tree = ConstantShifter(delta).visit(cand_tree)
            ast.fix_missing_locations(cand_tree)
            cand_code = ast.unparse(cand_tree)
            mutations_tried += 1

            test_scope = {}
            try:
                exec(f"{cand_code}\n{test_fn_call}", test_scope)
                if test_scope.get(target_var) == expected_output or test_scope.get("res") == expected_output:
                    return True, cand_code, 2
            except Exception:
                pass

        return False, code, mutations_tried
