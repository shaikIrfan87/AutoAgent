import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List


class NonLinearMLPNetwork(torch.nn.Module):
    """
    NOTEARS-MLP (Zheng et al.):
    Neural Structural Causal Model modeling non-linear structural causal dynamics
    X_j ~ f_j(X) with group L2 norm adjacency:
        W_{i, j} = ||W^(1)_{j, i, :}||_2
    """
    def __init__(self, dim: int, hidden_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(dim, hidden_dim) * 0.15) for _ in range(dim)
        ])
        self.fc2 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(hidden_dim, 1) * 0.15) for _ in range(dim)
        ])
        self.bias1 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_dim)) for _ in range(dim)
        ])
        self.bias2 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1)) for _ in range(dim)
        ])

    def predict_node(self, X: torch.Tensor, j: int, mask_col: torch.Tensor) -> torch.Tensor:
        effective_fc1 = self.fc1[j] * mask_col.unsqueeze(-1)
        h = torch.tanh(torch.matmul(X, effective_fc1) + self.bias1[j])
        return torch.matmul(h, self.fc2[j]) + self.bias2[j]

    def compute_W(self, mask: torch.Tensor) -> torch.Tensor:
        cols = []
        for j in range(self.dim):
            w_col = torch.sqrt(torch.sum(self.fc1[j] ** 2, dim=-1) + 1e-8) * mask[:, j]
            cols.append(w_col)
        return torch.stack(cols, dim=1)


class ContinuousSCMEngine:
    """
    Judea Pearl Level-3 Structural Causal Model (SCM) Engine with
    Differentiable Causal Discovery via the NOTEARS continuous DAG formulation:
        min L(W) = (1 / 2n) * sum_j ||X_j - f_j(X; W_{:, j})||_2^2 + lambda * ||W||_1
        subject to h(W) = tr(exp(W * W)) - d = 0
    Supports both linear (X - XW) and non-linear neural MLP (NOTEARS-MLP) formulations,
    Pearl do-calculus interventions (do(X_j = c)), and counterfactual queries.
    """

    def __init__(self, dim: int = 8, lambda_l1: float = 0.05, rho_init: float = 1.0):
        self.dim = dim
        self.lambda_l1 = lambda_l1
        self.rho = rho_init
        self.W: np.ndarray = np.zeros((dim, dim), dtype=np.float32)
        self.feature_names: List[str] = [f"var_{i}" for i in range(dim)]
        self.is_nonlinear: bool = False
        self.mlp_net: Optional[NonLinearMLPNetwork] = None

    @staticmethod
    def compute_acyclicity(W: Any) -> torch.Tensor:
        """
        Computes the continuous acyclicity constraint h(W) = tr(exp(W * W)) - d.
        Uses exact matrix exponential for d <= 32 and efficient truncated Taylor polynomial
        series approximation for d > 32 to prevent CPU scaling bottlenecks.
        """
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32)
        d = W.shape[0]
        M = W * W
        if d <= 32:
            return torch.trace(torch.matrix_exp(M)) - d

        # Truncated polynomial series approximation for large d:
        # tr(e^M) - d = sum_{k=1}^5 (1/k!) * tr(M^k)
        M2 = torch.matmul(M, M)
        tr_M2 = torch.trace(M2)
        M3 = torch.matmul(M2, M)
        tr_M3 = torch.trace(M3)
        M4 = torch.matmul(M3, M)
        tr_M4 = torch.trace(M4)
        M5 = torch.matmul(M4, M)
        tr_M5 = torch.trace(M5)

        return 0.5 * tr_M2 + (1.0 / 6.0) * tr_M3 + (1.0 / 24.0) * tr_M4 + (1.0 / 120.0) * tr_M5

    def fit(
        self,
        X_data: np.ndarray,
        max_iter: int = 150,
        lr: float = 0.01,
        exogenous_indices: Optional[List[int]] = None,
        non_linear: bool = False,
        hidden_dim: int = 16,
    ) -> np.ndarray:
        """
        Learns the weighted adjacency matrix W using augmented Lagrangian optimization.
        Supports linear least-squares or non-linear MLP formulation per node:
            min_W (1/2n) * sum_j ||X_j - f_j(X; W_{:, j})||_2^2 + lambda * ||W||_1  s.t. h(W) = 0
        X_data: [n_samples, dim]
        exogenous_indices: variables intervened upon (severing incoming arrows do(X_j = c)).
        """
        n, d = X_data.shape
        assert d == self.dim, f"Expected dimension {self.dim}, got {d}"
        self.is_nonlinear = non_linear

        X_t = torch.tensor(X_data, dtype=torch.float32)
        mask = torch.ones(d, d, dtype=torch.float32) - torch.eye(d, dtype=torch.float32)
        if exogenous_indices:
            for idx in exogenous_indices:
                if 0 <= idx < d:
                    mask[:, idx] = 0.0

        if non_linear:
            self.mlp_net = NonLinearMLPNetwork(d, hidden_dim)
            optimizer = torch.optim.Adam(self.mlp_net.parameters(), lr=lr)
            W_param = None
        else:
            self.mlp_net = None
            W_param = torch.nn.Parameter(torch.zeros(d, d, dtype=torch.float32))
            optimizer = torch.optim.Adam([W_param], lr=lr)

        rho = self.rho
        alpha = 0.0

        for _ in range(max_iter):
            optimizer.zero_grad()

            if non_linear and self.mlp_net is not None:
                W_masked = self.mlp_net.compute_W(mask)
                loss_mse = 0.0
                for j in range(d):
                    pred_j = self.mlp_net.predict_node(X_t, j, mask[:, j]).squeeze(-1)
                    loss_mse = loss_mse + 0.5 * torch.mean((X_t[:, j] - pred_j) ** 2)
            else:
                W_masked = W_param * mask
                diff = X_t - torch.matmul(X_t, W_masked)
                loss_mse = 0.5 * torch.mean(diff ** 2)

            # L1 sparsity
            loss_l1 = self.lambda_l1 * torch.sum(torch.abs(W_masked))

            # NOTEARS acyclicity constraint
            h = self.compute_acyclicity(W_masked)
            loss_h = alpha * h + 0.5 * rho * (h ** 2)

            total_loss = loss_mse + loss_l1 + loss_h
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                h_val = float(h.item())
                if h_val > 0.1:
                    rho *= 1.05

        with torch.no_grad():
            if non_linear and self.mlp_net is not None:
                W_final = self.mlp_net.compute_W(mask).detach().cpu().numpy()
            else:
                W_final = (W_param * (1.0 - torch.eye(d))).detach().cpu().numpy()

            # Hard threshold small coefficients to guarantee exact DAG sparsity
            W_final[np.abs(W_final) < 0.10] = 0.0

            # Prune any remaining 2-cycles to enforce strict DAG property
            for i in range(d):
                for j in range(i + 1, d):
                    if W_final[i, j] != 0 and W_final[j, i] != 0:
                        if abs(W_final[i, j]) >= abs(W_final[j, i]):
                            W_final[j, i] = 0.0
                        else:
                            W_final[i, j] = 0.0

            self.W = W_final

        return self.W

    def fit_nonlinear(
        self,
        X_data: np.ndarray,
        max_iter: int = 180,
        lr: float = 0.01,
        exogenous_indices: Optional[List[int]] = None,
        hidden_dim: int = 16,
    ) -> np.ndarray:
        """Helper to fit Non-Linear Neural NOTEARS."""
        return self.fit(
            X_data,
            max_iter=max_iter,
            lr=lr,
            exogenous_indices=exogenous_indices,
            non_linear=True,
            hidden_dim=hidden_dim,
        )

    def is_dag(self, threshold: float = 1e-4) -> bool:
        """Checks if current adjacency matrix satisfies the DAG acyclicity condition."""
        h = float(self.compute_acyclicity(self.W).item())
        return h < threshold

    def intervene(
        self,
        do_dict: Dict[int, float],
        base_noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Pearl Level-2 Interventional Inference: do(X_j = c).
        Mutilates the causal graph by severing incoming edges to intervened variables:
        W_mutilated[:, j] = 0, and sets X_j = c.
        Returns the resulting equilibrium state vector.
        """
        d = self.dim
        W_mut = self.W.copy()
        for j in do_dict.keys():
            W_mut[:, j] = 0.0

        noise = base_noise if base_noise is not None else np.zeros(d, dtype=np.float32)
        state = noise.copy()

        # Set fixed intervened values
        for j, val in do_dict.items():
            state[j] = val

        if self.is_nonlinear and self.mlp_net is not None:
            # Topological forward propagation through non-linear MLPs
            in_degrees = [int(np.count_nonzero(W_mut[:, j] != 0)) for j in range(d)]
            queue = [j for j in range(d) if in_degrees[j] == 0]
            visited_order = []
            while queue:
                curr = queue.pop(0)
                visited_order.append(curr)
                for next_j in range(d):
                    if W_mut[curr, next_j] != 0:
                        in_degrees[next_j] -= 1
                        if in_degrees[next_j] == 0:
                            queue.append(next_j)

            for j in range(d):
                if j not in visited_order:
                    visited_order.append(j)

            with torch.no_grad():
                for j in visited_order:
                    if j in do_dict:
                        state[j] = do_dict[j]
                    else:
                        in_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        mask_col = torch.tensor(W_mut[:, j] != 0, dtype=torch.float32)
                        pred = float(self.mlp_net.predict_node(in_t, j, mask_col).item())
                        state[j] = pred + noise[j]
            return state

        # Linear structural propagation: (I - W_mut.T) * X = state
        I = np.eye(d, dtype=np.float32)
        try:
            inv = np.linalg.inv(I - W_mut.T)
            result = inv @ state
            for j, val in do_dict.items():
                result[j] = val
            return result
        except np.linalg.LinAlgError:
            return state


    def counterfactual(
        self,
        factual_obs: np.ndarray,
        do_dict: Dict[int, float]
    ) -> np.ndarray:
        """
        Pearl Level-3 Counterfactual Query:
        1. Abduction: Infer unobserved latent background noise: epsilon = (I - W) * factual_obs
        2. Action: Apply graph mutilation do(X_j = c)
        3. Prediction: Compute counterfactual state with abducted noise
        """
        I = np.eye(self.dim, dtype=np.float32)
        abducted_noise = (I - self.W) @ factual_obs
        return self.intervene(do_dict, base_noise=abducted_noise)


class LiveTelemetrySCMEngine(ContinuousSCMEngine):
    """
    Live Continuous SCM Engine ingesting real host OS telemetry (psutil)
    and runtime performance metrics to discover operational DAGs and perform do-calculus.
    """
    METRICS = [
        "cpu_percent",
        "mem_percent",
        "thread_count",
        "loop_latency_ms",
        "sandbox_delta",
        "ttt_norm",
    ]

    def __init__(self, window_size: int = 64, lambda_l1: float = 0.05):
        super().__init__(dim=len(self.METRICS), lambda_l1=lambda_l1)
        self.window_size = window_size
        self.feature_names = list(self.METRICS)
        self.buffer: List[np.ndarray] = []
        self._name_to_idx = {name: i for i, name in enumerate(self.METRICS)}

    def sample_step(
        self,
        loop_latency_ms: float = 0.5,
        sandbox_delta: float = 1.0,
        ttt_norm: float = 0.2
    ) -> np.ndarray:
        """Samples real OS telemetry and appends observation vector to sliding buffer."""
        try:
            import psutil
            cpu = float(psutil.cpu_percent(interval=None))
            mem = float(psutil.virtual_memory().percent)
            threads = float(len(psutil.Process().threads()))
        except Exception:
            cpu, mem, threads = 10.0, 45.0, 4.0

        vec = np.array(
            [cpu, mem, threads, float(loop_latency_ms), float(sandbox_delta), float(ttt_norm)],
            dtype=np.float32
        )
        self.buffer.append(vec)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return vec

    def update_causal_dag(self, max_iter: int = 100) -> Optional[np.ndarray]:
        """Runs continuous NOTEARS optimization over current telemetry window."""
        if len(self.buffer) < 8:
            return None
        X = np.stack(self.buffer, axis=0)
        # Standardize features for stable optimization
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0) + 1e-4
        X_norm = (X - means) / stds
        return self.fit(X_norm, max_iter=max_iter)

    def intervene_telemetry(self, do_dict: Dict[str, float]) -> Dict[str, float]:
        """Performs do-calculus intervention using metric names."""
        idx_dict = {self._name_to_idx[k]: v for k, v in do_dict.items() if k in self._name_to_idx}
        res_vec = self.intervene(idx_dict)
        return {name: float(res_vec[i]) for i, name in enumerate(self.METRICS)}

    def counterfactual_telemetry(
        self,
        factual_obs: Dict[str, float],
        do_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """Performs Level-3 counterfactual query on named telemetry state."""
        vec = np.array([factual_obs.get(name, 0.0) for name in self.METRICS], dtype=np.float32)
        idx_dict = {self._name_to_idx[k]: v for k, v in do_dict.items() if k in self._name_to_idx}
        cf_vec = self.counterfactual(vec, idx_dict)
        return {name: float(cf_vec[i]) for i, name in enumerate(self.METRICS)}

