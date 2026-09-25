from enum import Enum
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .types import Triple, InterventionRecord
    from .scm_engine import ContinuousSCMEngine
except ImportError:
    from types import Triple, InterventionRecord
    try:
        from scm_engine import ContinuousSCMEngine
    except ImportError:
        ContinuousSCMEngine = None



class EdgeType(str, Enum):
    STRUCTURAL_TRANSITIVE = "structural_transitive"
    CONDITIONAL_CAUSAL = "conditional_causal"


class CausalSymbolicGraph:
    """Invariant 2: Neuro-Symbolic Relational Engine.
    Enforces directional logic, interventional causal discovery, and Pearl Level 3 counterfactuals.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.interventions: List[InterventionRecord] = []
        self.scm: Optional["StructuralEquationModel"] = None
        self._load_default_axioms()

    def add_axiom(self, triple: Triple) -> None:
        self.graph.add_edge(
            triple.subject.lower(),
            triple.target.lower(),
            relation=triple.relation,
            polarity=triple.polarity,
        )

    def add_edge(
        self,
        sub: str,
        rel: str,
        tgt: str,
        edge_type: Optional[EdgeType] = None,
        is_valid: bool = True,
    ) -> None:
        """Add typed directional edge to causal graph."""
        e_type = edge_type or (
            EdgeType.STRUCTURAL_TRANSITIVE
            if rel in ("is_a", "part_of", "leads_to")
            else EdgeType.CONDITIONAL_CAUSAL
        )
        self.graph.add_edge(
            sub.lower(),
            tgt.lower(),
            relation=rel,
            polarity=is_valid,
            edge_type=e_type,
            weight=1.0 if is_valid else 0.0,
        )

    def get_edge_weight(self, u: str, v: str) -> float:
        u_l, v_l = u.lower(), v.lower()
        if self.graph.has_edge(u_l, v_l):
            edge_data = next(iter(self.graph[u_l][v_l].values()))
            return float(edge_data.get("weight", 0.5))
        return 0.0

    def set_edge_weight(self, u: str, v: str, weight: float) -> None:
        u_l, v_l = u.lower(), v.lower()
        if self.graph.has_edge(u_l, v_l):
            for edge_data in self.graph[u_l][v_l].values():
                edge_data["weight"] = float(weight)
        else:
            self.graph.add_edge(u_l, v_l, weight=float(weight), relation="associated_with", polarity=True)

    def induce_edge(
        self,
        source: str,
        target: str,
        polarity: bool = True,
        weight: float = 1.0,
        relation: str = "causes",
    ) -> bool:
        """Dynamically induce causal transitions observed from sandbox execution (ΔS).
        Rejects induction if it creates a direct logical contradiction or cyclic violation.
        """
        sub_l = source.strip().lower()
        tgt_l = target.strip().lower()
        if not sub_l or not tgt_l or sub_l == tgt_l:
            return False

        valid, _ = self.verify_hypothesis(sub_l, relation, tgt_l, is_valid=polarity)
        if not valid:
            return False

        self.add_edge(sub_l, relation, tgt_l, is_valid=polarity)
        self.set_edge_weight(sub_l, tgt_l, weight if polarity else -weight)
        return True

    def record_intervention(
        self,
        source: str,
        action: str,
        target: str,
        delta_s: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> InterventionRecord:
        """Record an empirical interventional observation do(action) on state transitions."""
        rec = InterventionRecord(
            source=source.strip().lower(),
            action=action.strip().lower(),
            target=target.strip().lower(),
            delta_s=float(delta_s),
            metrics=metrics or {},
            timestamp=time.time(),
        )
        self.interventions.append(rec)
        return rec

    def induce_interventional_edge(
        self,
        source: str,
        action: str,
        target: str,
        min_samples: int = 3,
        threshold: float = 0.6,
    ) -> Tuple[bool, float, str]:
        """Interventional Causal Discovery:
        Evaluates empirical trials do(action) using Laplace-smoothed Bayesian posterior score.
        Deduces directional edges and verifies consistency against causal axioms.
        """
        sub_l = source.strip().lower()
        act_l = action.strip().lower()
        tgt_l = target.strip().lower()

        matched = [
            r for r in self.interventions
            if (r.source == sub_l or sub_l in r.source)
            and (r.action == act_l or act_l in r.action)
            and (r.target == tgt_l or tgt_l in r.target)
        ]

        n = len(matched)
        if n < min_samples:
            return False, 0.0, f"Insufficient empirical observations ({n}/{min_samples})"

        k = sum(1 for r in matched if r.delta_s > 0)
        score = (k + 1.0) / (n + 2.0)
        pos_ratio = k / n

        if pos_ratio < threshold or score < threshold:
            return False, score, f"Interventional confidence ({score:.2f}) below threshold ({threshold:.2f})"

        valid, diag = self.verify_hypothesis(sub_l, act_l, tgt_l, is_valid=True)
        if not valid:
            return False, score, f"Axiom contradiction blocks induction: {diag}"
        valid_c, diag_c = self.verify_hypothesis(sub_l, "causes", tgt_l, is_valid=True)
        if not valid_c:
            return False, score, f"Axiom contradiction blocks induction: {diag_c}"

        self.add_edge(sub_l, act_l, tgt_l, is_valid=True)
        self.set_edge_weight(sub_l, tgt_l, score)
        return True, score, f"Empirical edge ({sub_l}, {act_l}, {tgt_l}) induced with score {score:.2f}"

    def conduct_interventional_experiment(
        self,
        sandbox_or_sim: Any,
        var_names: List[str],
        experiment_code_template: Optional[str] = None,
        intervention_trials: Optional[List[Dict[str, float]]] = None,
        exogenous_vars: Optional[List[str]] = None,
        max_iter: int = 150,
        lr: float = 0.01,
        non_linear: bool = False,
    ) -> Tuple[np.ndarray, Optional["StructuralEquationModel"]]:

        """
        Active Counterfactual Experimentation (do(X) Calculus in Sandbox):
        Executes physical or simulated interventions do(X_j = c) in the isolated sandbox,
        gathers empirical state trajectory data, and learns the directed acyclic graph W
        via continuous NOTEARS augmented Lagrangian optimization (linear or non-linear MLP).
        """
        if ContinuousSCMEngine is None:
            return np.zeros((len(var_names), len(var_names)), dtype=np.float32), None

        dim = len(var_names)
        data_rows: List[List[float]] = []

        # If custom trials not provided, generate diverse interventional values
        if not intervention_trials:
            trials = []
            for i, v in enumerate(var_names):
                for val in [-3.0, -1.5, 0.0, 1.5, 3.0]:
                    trials.append({v: val})
            intervention_trials = trials

        for trial in intervention_trials:
            row: Optional[List[float]] = None
            if callable(sandbox_or_sim):
                res = sandbox_or_sim(trial)
                if isinstance(res, dict):
                    row = [float(res.get(v, 0.0)) for v in var_names]
                elif isinstance(res, (list, tuple, np.ndarray)) and len(res) == dim:
                    row = [float(x) for x in res]
            elif hasattr(sandbox_or_sim, "execute") and experiment_code_template:
                code = experiment_code_template
                for k, v in trial.items():
                    code = code.replace(f"{{{k}}}", str(v))
                exec_res = sandbox_or_sim.execute(code)
                if exec_res.get("success") and exec_res.get("stdout"):
                    try:
                        out_dict = json.loads(exec_res["stdout"].strip())
                        row = [float(out_dict.get(v, 0.0)) for v in var_names]
                    except Exception:
                        pass
            if row is not None:
                data_rows.append(row)

        if len(data_rows) < 5:
            return np.zeros((dim, dim), dtype=np.float32), None

        X_data = np.array(data_rows, dtype=np.float32)
        scm_engine = ContinuousSCMEngine(dim=dim)
        if exogenous_vars is not None:
            intervened_indices = [var_names.index(k) for k in exogenous_vars if k in var_names]
        elif intervention_trials:
            common_keys = set(intervention_trials[0].keys())
            for t in intervention_trials[1:]:
                common_keys &= set(t.keys())
            intervened_indices = [var_names.index(k) for k in common_keys if k in var_names]
        else:
            intervened_indices = []
        W = scm_engine.fit(
            X_data,
            max_iter=max_iter,
            lr=lr,
            exogenous_indices=intervened_indices,
            non_linear=non_linear,
        )



        # Wire discovered edges into causal graph
        for i in range(dim):
            for j in range(dim):
                weight = float(W[i, j])
                if abs(weight) >= 0.05:
                    u = var_names[i].lower()
                    v = var_names[j].lower()
                    rel = "causes" if weight > 0 else "inhibits"
                    self.add_edge(u, rel, v, is_valid=True)
                    self.set_edge_weight(u, v, weight)

        self.scm = StructuralEquationModel(torch.tensor(W, dtype=torch.float32), var_names=var_names)
        return W, self.scm


    def evaluate_counterfactual(
        self,
        observed_state: str,
        failed_action: str,
        candidate_action: str,
        target_invariant: str,
    ) -> Tuple[bool, float, str]:
        """Pearl Level 3 Counterfactual Evaluator:
        'Given that failed_action in observed_state caused invariant violation,
        would candidate_action have satisfied target_invariant without contradiction?'
        """
        obs_l = observed_state.strip().lower()
        cand_l = candidate_action.strip().lower()
        tgt_l = target_invariant.strip().lower()

        # 1. Structural check: candidate action must not violate target invariant or axioms
        valid, diag = self.verify_hypothesis(cand_l, "causes", tgt_l, is_valid=True)
        if not valid:
            return False, 0.0, f"Counterfactual violation: {diag}"

        # 2. Candidate must not be prohibited by observed state
        if self.graph.has_edge(obs_l, cand_l):
            for edge_data in self.graph[obs_l][cand_l].values():
                if edge_data.get("relation") in ("cannot_be", "mutually_exclusive") and edge_data.get("polarity", True):
                    return False, 0.0, f"Candidate action prohibited by observed state ({obs_l})"

        # 3. Assess empirical confidence from intervention history or graph connectivity
        matched_cand = [r for r in self.interventions if r.action == cand_l or cand_l in r.action]
        if matched_cand:
            k = sum(1 for r in matched_cand if r.delta_s > 0)
            empirical_conf = (k + 1.0) / (len(matched_cand) + 2.0)
        else:
            empirical_conf = self.get_edge_weight(cand_l, tgt_l) or 0.65

        return True, empirical_conf, f"Counterfactual candidate '{cand_l}' is causally consistent with target '{tgt_l}'"



    @property
    def edges(self):
        return {(u, v): d.get("weight", 0.5) for u, v, d in self.graph.edges(data=True)}

    @property
    def nodes(self):
        return self.graph.nodes

    def _load_default_axioms(self) -> None:
        defaults = [
            Triple(subject="vacuum", relation="cannot_be", target="air", polarity=True),
            Triple(subject="vacuum", relation="causes", target="pressure", polarity=False),
            Triple(subject="biologicalorganism", relation="mutually_exclusive", target="syntheticmachine", polarity=True),
            Triple(subject="human", relation="is_a", target="biologicalorganism", polarity=True),
            Triple(subject="android", relation="is_a", target="syntheticmachine", polarity=True),
            Triple(subject="water", relation="cannot_be", target="dry", polarity=True),
            Triple(subject="perpetual motion machine", relation="causes", target="infinite energy", polarity=False),
            Triple(subject="perpetual motion machine", relation="cannot_be", target="infinite energy", polarity=True),
            Triple(subject="perpetual motion machine", relation="cannot_generate", target="infinite energy", polarity=True),
            Triple(subject="entropy decrease", relation="cannot_be", target="without energy dissipation", polarity=True),
            Triple(subject="decrease system heat entropy", relation="cannot_be", target="without energy dissipation", polarity=True),
        ]
        for t in defaults:
            self.add_axiom(t)

    def verify_hypothesis(
        self,
        sub_or_triple: Union[Triple, str],
        rel: Optional[str] = None,
        tgt: Optional[str] = None,
        is_valid: bool = True,
    ) -> Tuple[bool, str]:
        """Verify hypothesis against 1-hop and 2-hop constraints. Returns (valid, diagnostic)."""
        if isinstance(sub_or_triple, Triple):
            sub = sub_or_triple.subject.lower()
            tgt = sub_or_triple.target.lower()
            rel_val = sub_or_triple.relation
            pol = sub_or_triple.polarity
        else:
            sub = str(sub_or_triple).lower()
            rel_val = str(rel)
            tgt = str(tgt).lower()
            pol = is_valid

        rel = rel_val

        # 1-hop direct check
        if self.graph.has_edge(sub, tgt):
            for edge_data in self.graph[sub][tgt].values():
                edge_rel = edge_data.get("relation")
                edge_pol = edge_data.get("polarity", True)

                # Direct polarity contradiction on same relation
                if edge_rel == rel and edge_pol != pol:
                    return False, f"Direct contradiction: ({sub}, {rel}, {tgt}) contradicts existing polarity={edge_pol}"

                # Direct 'cannot_be' or 'mutually_exclusive' or negative polarity prohibiting positive relation
                if pol and (edge_rel in ("cannot_be", "mutually_exclusive", "cannot_generate") or not edge_pol):
                    return False, f"Direct negative constraint: ({sub}, {edge_rel}, {tgt}) blocks hypothesis"

        # Check reverse for mutually_exclusive
        if pol and rel == "is_a" and self.graph.has_edge(tgt, sub):
            for edge_data in self.graph[tgt][sub].values():
                if edge_data.get("relation") == "mutually_exclusive" and edge_data.get("polarity", True):
                    return False, f"Mutual exclusion: {tgt} is mutually exclusive with {sub}"

        # Define relational categories according to Pearl's causal calculus
        STRUCTURAL_TRANSITIVE = {"is_a", "part_of"}
        CONDITIONAL_CAUSAL = {"causes"}

        # 2-hop & multi-hop transitive contradiction with typed composition
        structural_subgraph = nx.DiGraph()
        causal_subgraph = nx.DiGraph()

        for u, v, data in self.graph.edges(data=True):
            if data.get("polarity", True):
                r = data.get("relation")
                if r in STRUCTURAL_TRANSITIVE:
                    structural_subgraph.add_edge(u, v)
                elif r in CONDITIONAL_CAUSAL:
                    causal_subgraph.add_edge(u, v)

        # If hypothesis is (sub, causes, tgt):
        if pol and rel == "causes":
            # Check direct causal prohibition
            if self.graph.has_edge(sub, tgt):
                for edge_data in self.graph[sub][tgt].values():
                    if edge_data.get("relation") == "causes" and not edge_data.get("polarity", True):
                        return False, f"Causal prohibition: {sub} cannot cause {tgt}"

            # Check if tgt transitively cannot_be sub (or vice versa)
            test_subgraph = causal_subgraph.copy()
            test_subgraph.add_edge(sub, tgt)
            try:
                cycles = list(nx.simple_cycles(test_subgraph))
                for cycle in cycles:
                    if sub in cycle and tgt in cycle:
                        for i in range(len(cycle)):
                            c_u = cycle[i]
                            c_v = cycle[(i + 1) % len(cycle)]
                            if self.graph.has_edge(c_u, c_v):
                                for d in self.graph[c_u][c_v].values():
                                    if d.get("relation") in ("cannot_be", "mutually_exclusive"):
                                        return False, f"Cyclic contradiction detected across path: {' -> '.join(cycle)}"
            except Exception:
                pass

            if self.graph.has_edge(tgt, sub):
                for edge_data in self.graph[tgt][sub].values():
                    if edge_data.get("relation") in ("cannot_be", "mutually_exclusive"):
                        return False, f"Cyclic contradiction: {sub} causes {tgt}, but {tgt} cannot_be {sub}"

        # Helper for safe path check
        def _safe_has_path(g: nx.DiGraph, s: str, t: str) -> bool:
            return s in g and t in g and nx.has_path(g, s, t)

        # If hypothesis is (sub, cannot_be, tgt):
        if pol and rel == "cannot_be":
            # Directional transitivity: check structural or causal paths
            if _safe_has_path(causal_subgraph, sub, tgt) or _safe_has_path(structural_subgraph, sub, tgt):
                return False, f"Cyclic contradiction: {sub} already reaches {tgt} in graph"
            if _safe_has_path(causal_subgraph, tgt, sub) or _safe_has_path(structural_subgraph, tgt, sub):
                return False, f"Cyclic contradiction: {tgt} already reaches {sub} in graph"



        # Transitive is_a contradiction
        if pol and rel == "is_a":
            if sub in self.graph:
                for parent in self.graph.successors(sub):
                    for edge_data in self.graph[sub][parent].values():
                        if edge_data.get("relation") == "is_a" and edge_data.get("polarity", True):
                            if self.graph.has_edge(parent, tgt):
                                for p_edge in self.graph[parent][tgt].values():
                                    if p_edge.get("relation") in ("cannot_be", "mutually_exclusive") and p_edge.get("polarity", True):
                                        return False, f"Transitive contradiction: {sub} is_a {parent}, but {parent} {p_edge['relation']} {tgt}"
                            if self.graph.has_edge(tgt, parent):
                                for p_edge in self.graph[tgt][parent].values():
                                    if p_edge.get("relation") == "mutually_exclusive" and p_edge.get("polarity", True):
                                        return False, f"Transitive contradiction: {sub} is_a {parent}, which is mutually exclusive with {tgt}"

        return True, "Hypothesis consistent with causal axioms"

    def fit_continuous_scm(
        self,
        X: torch.Tensor,
        var_names: Optional[List[str]] = None,
        epochs: int = 150,
        lr: float = 0.01,
        threshold: float = 0.1,
    ) -> "StructuralEquationModel":
        """Fits a differentiable DAG via NOTEARS and instantiates a continuous SCM."""
        discovery = DifferentiableCausalDiscovery(dim=X.shape[1])
        W = discovery.fit(X, epochs=epochs, lr=lr, threshold=threshold)
        self.scm = StructuralEquationModel(W, var_names=var_names)
        return self.scm

    def evaluate_counterfactual_scm(
        self,
        factual_obs: torch.Tensor,
        interventions: Dict[Union[str, int], float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pearl Level 3 Counterfactual on continuous SCM:
        Returns (X_counterfactual, U_abducted).
        """
        if self.scm is None:
            raise ValueError("SCM has not been fit. Call fit_continuous_scm() first.")
        return self.scm.intervene(factual_obs, interventions)


class DifferentiableCausalDiscovery(nn.Module):
    """Learns continuous Directed Acyclic Graphs (DAGs) from observational latent vectors
    using the continuous NOTEARS acyclicity constraint: h(W) = Tr(exp(W * W)) - d = 0.
    """

    def __init__(self, dim: int, l1_reg: float = 0.01):
        super().__init__()
        self.dim = dim
        self.l1_reg = l1_reg
        self.W = nn.Parameter(torch.zeros(dim, dim))

    def _notears_constraint(self, W: torch.Tensor) -> torch.Tensor:
        M = W * W
        return torch.trace(torch.matrix_exp(M)) - self.dim

    def fit(
        self,
        X: torch.Tensor,
        epochs: int = 150,
        lr: float = 0.01,
        rho: float = 1.0,
        alpha: float = 0.0,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Optimizes W via Augmented Lagrangian under NOTEARS acyclicity."""
        optimizer = optim.Adam([self.W], lr=lr)
        n = X.shape[0]

        for _ in range(epochs):
            optimizer.zero_grad()
            W_effective = self.W * (1.0 - torch.eye(self.dim, device=self.W.device))
            pred = torch.matmul(X, W_effective)
            recon_loss = 0.5 / n * torch.sum((X - pred) ** 2)
            l1_loss = self.l1_reg * torch.sum(torch.abs(W_effective))
            h = self._notears_constraint(W_effective)
            lagrangian = recon_loss + l1_loss + alpha * h + 0.5 * rho * (h ** 2)
            lagrangian.backward()
            optimizer.step()

        with torch.no_grad():
            W_final = (self.W * (1.0 - torch.eye(self.dim, device=self.W.device))).clone()
            W_final[torch.abs(W_final) < threshold] = 0.0

            # Break 2-cycles by retaining the dominant directed edge
            for i in range(self.dim):
                for j in range(i + 1, self.dim):
                    if abs(W_final[i, j]) > 0 and abs(W_final[j, i]) > 0:
                        if abs(W_final[i, j]) >= abs(W_final[j, i]):
                            W_final[j, i] = 0.0
                        else:
                            W_final[i, j] = 0.0

            # Guarantee acyclicity: prune weakest cycle edges if h(W) remains non-zero
            while float(self._notears_constraint(W_final).item()) > 1e-3:
                nonzero_mask = torch.abs(W_final) > 0
                if not nonzero_mask.any():
                    break
                min_val = torch.min(torch.abs(W_final)[nonzero_mask])
                W_final[torch.abs(W_final) == min_val] = 0.0

            self.W.copy_(W_final)
            return W_final


class StructuralEquationModel:
    """Judea Pearl Structural Causal Model (SCM): L_SCM = <U, V, F, P(u)>.
    Computes Level 3 Counterfactuals:
      1. Abduction:  U = X_obs - X_obs * W
      2. Action:     do(X_k = x_val) modifies structural equations by zeroing W[:, k]
      3. Prediction: X_cf = U_do * (I - W_do)^(-1)
    """

    def __init__(self, W: torch.Tensor, var_names: Optional[List[str]] = None):
        self.W = W.detach().float()
        self.dim = self.W.shape[0]
        self.var_names = var_names or [f"v{i}" for i in range(self.dim)]
        self._name_to_idx = {name: i for i, name in enumerate(self.var_names)}

    def abduct_noise(self, X_obs: torch.Tensor) -> torch.Tensor:
        """Step 1 (Abduction): infer background exogenous noise U from factual observation."""
        return X_obs - torch.matmul(X_obs, self.W)

    def intervene(
        self,
        X_obs: torch.Tensor,
        interventions: Dict[Union[str, int], float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pearl Level 3 Counterfactual (Abduction -> Action -> Prediction).
        interventions: dict mapping variable name or index to intervention value x*.
        Returns (X_counterfactual, U_abducted).
        """
        if X_obs.ndim == 1:
            X_obs = X_obs.unsqueeze(0)

        # 1. Abduction
        U = self.abduct_noise(X_obs).clone()

        # 2. Action: sever incoming arrows to intervened variables
        W_do = self.W.clone()
        U_do = U.clone()

        for var, val in interventions.items():
            idx = self._name_to_idx.get(var, var) if isinstance(var, str) else var
            W_do[:, idx] = 0.0
            U_do[:, idx] = float(val)

        # 3. Prediction: solve X_cf * (I - W_do) = U_do
        I = torch.eye(self.dim, device=self.W.device)
        M = I - W_do
        X_cf = torch.linalg.solve(M.T, U_do.T).T
        return X_cf, U

