"""
Predictive Latent State Machine & Counterfactual Mental Simulator.
Projects state transitions ŝ_{t+1} = T(s_t, a_t), estimates memory/file side-effects,
and evaluates an energy-based compatibility metric prior to sandbox execution.
"""

import ast
from dataclasses import dataclass, field
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class EnvironmentState:
    """Latent state representation of the execution environment."""
    memory_mb: float = 0.0
    open_file_descriptors: int = 0
    modified_variables: List[str] = field(default_factory=list)
    active_operations: List[str] = field(default_factory=list)
    projected_duration_ms: float = 0.0
    entropy_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_mb": self.memory_mb,
            "open_file_descriptors": self.open_file_descriptors,
            "modified_variables": list(self.modified_variables),
            "active_operations": list(self.active_operations),
            "projected_duration_ms": self.projected_duration_ms,
            "entropy_delta": self.entropy_delta,
        }


@dataclass
class StateTransition:
    """Projected state delta ŝ_{t+1} = T(s_t, a_t)."""
    prior_state: EnvironmentState
    projected_state: EnvironmentState
    delta_s: float
    energy_score: float
    divergent: bool = False
    rejection_reason: str = ""


@dataclass
class SimulationPrediction:
    safe: bool
    risk_score: float  # 0.0 to 1.0
    predicted_delta_s: float
    blast_radius: List[str]
    rejection_reason: str = ""
    transition: Optional[StateTransition] = None
    energy_score: float = 0.0


class MentalSimulator:
    """Predictive state machine and counterfactual simulator with latent neural transitions."""

    def __init__(self, risk_threshold: float = 0.70, energy_threshold: float = 1.0):
        self.risk_threshold = risk_threshold
        self.energy_threshold = energy_threshold
        self.latent_model = LatentWorldModel()
        self.env_encoder = StructuredEnvironmentEncoder()
        self._tuner: Optional[Any] = None

    @property
    def tuner(self) -> "BackgroundWorldModelTuner":
        if self._tuner is None:
            self._tuner = BackgroundWorldModelTuner(self.latent_model)
        return self._tuner

    def record_sandbox_transition(
        self,
        code: str,
        delta_s: float,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        latency_ms: float = 0.5,
        memory_mb: float = 15.0,
    ) -> None:
        """Translates real sandbox execution into latent transition training targets via StructuredEnvironmentEncoder."""
        # Baseline state features s_t
        s_t = self.env_encoder(
            stdout_len=0,
            stderr_len=0,
            exit_code=0,
            files_modified=0,
            latency_ms=0.5,
            memory_mb=15.0,
            semantic_error_hash=0.0,
            ast_diff_count=len(code),
        )
        # Outcome state features s_{t+1}
        err_hash = float(hash(stderr) % 100) / 100.0 if stderr else 0.0
        s_tp1 = self.env_encoder(
            stdout_len=len(stdout),
            stderr_len=len(stderr),
            exit_code=exit_code,
            files_modified=1 if ("open(" in code and ("'w'" in code or '"w"' in code)) else 0,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            semantic_error_hash=err_hash,
            ast_diff_count=len(code),
        )
        reward = 1.0 if delta_s > 0.0 else -1.0
        risk = 0.05 if delta_s > 0.0 else 0.95
        a_idx = 0
        if any(w in code for w in ("os.", "shutil.", "subprocess.", "rmdir", "remove", "unlink")):
            a_idx = 1
        elif "open(" in code:
            a_idx = 2
        elif any(w in code for w in ("for ", "while ")):
            a_idx = 3

        self.tuner.enqueue_transition(s_t, a_idx=a_idx, s_tp1=s_tp1, reward=reward, risk=risk)

    def prospective_veto(
        self,
        code: str,
        horizon: int = 5,
        risk_threshold: float = 0.80,
    ) -> Tuple[bool, float, str]:
        """
        Runs an H-step prospective mental rollout in latent space to veto dangerous code before sandbox execution.
        Returns (is_vetoed, risk_score, reason).
        """
        sim = self.simulate(code)
        if not sim.safe or sim.risk_score >= risk_threshold:
            return True, sim.risk_score, sim.rejection_reason or f"Static blast-radius risk {sim.risk_score:.2f} >= {risk_threshold:.2f}"

        # Prospective multi-step rollout in latent space
        s_0 = self.env_encoder(ast_diff_count=len(code), memory_mb=sim.transition.projected_state.memory_mb if sim.transition else 15.0)
        action_seq = [0] * max(1, horizon)
        if any(w in code for w in ("os.", "shutil.", "subprocess.", "remove", "unlink", "rmtree")):
            action_seq = [1] * max(1, horizon)
        elif "while " in code:
            action_seq = [3] * max(1, horizon)

        rollout = self.latent_model.rollout_trajectory(s_0, action_seq)
        max_latent_risk = float(rollout.get("max_risk", 0.0))
        fused_risk = 0.6 * sim.risk_score + 0.4 * max_latent_risk

        if fused_risk >= risk_threshold:
            return True, fused_risk, f"Prospective mental rollout rejected action: latent_risk={fused_risk:.2f} >= {risk_threshold:.2f}"
        return False, fused_risk, ""



    def snapshot_environment(self) -> EnvironmentState:
        """Captures baseline environment profile before execution."""
        return EnvironmentState(memory_mb=15.0)

    def _eval_ast_number(self, node: ast.AST) -> Optional[float]:
        """Safely evaluates deterministic arithmetic constants in AST for buffer estimation."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_number(node.left)
            right = self._eval_ast_number(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Pow) and right <= 8:
                    return left ** right
        return None

    def predict_transition(
        self,
        code: str,
        current_state: Optional[EnvironmentState] = None,
    ) -> StateTransition:
        """Projects latent state transition ŝ_{t+1} = T(s_t, a_t)."""
        prior = current_state or self.snapshot_environment()
        projected = EnvironmentState(
            memory_mb=prior.memory_mb,
            open_file_descriptors=prior.open_file_descriptors,
            modified_variables=[],
            active_operations=[],
            projected_duration_ms=2.0,
            entropy_delta=0.0,
        )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return StateTransition(
                prior_state=prior,
                projected_state=projected,
                delta_s=-1.0,
                energy_score=2.0,
                divergent=True,
                rejection_reason=f"Syntax error: {e}",
            )

        assigned_vars: Set[str] = set()
        active_ops: List[str] = []
        est_alloc_mb = 0.0
        est_duration_ms = 2.0
        risk_score = 0.0

        for node in ast.walk(tree):
            # Track variable mutations
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                assigned_vars.add(node.target.id)

            # Estimate large memory buffer allocations
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("bytearray", "bytes") and node.args:
                    arg_val = self._eval_ast_number(node.args[0])
                    if arg_val is not None:
                        est_alloc_mb += arg_val / (1024 * 1024)

                if func_name in ("open", "write"):
                    active_ops.append(f"file_io:{func_name}")
                    projected.open_file_descriptors += 1

                if func_name in ("remove", "rmdir", "unlink", "rmtree", "system"):
                    active_ops.append(f"destructive_filesystem:{func_name}")
                    risk_score += 0.85

                elif func_name in ("socket", "connect", "urlopen", "requests"):
                    active_ops.append(f"unbounded_network:{func_name}")
                    risk_score += 0.50

            elif isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    active_ops.append("runaway_infinite_loop")
                    est_duration_ms += 5000.0
                    risk_score += 0.40

            elif isinstance(node, (ast.For, ast.While)):
                est_duration_ms += 5.0

        projected.memory_mb += round(est_alloc_mb, 2)
        projected.modified_variables = sorted(list(assigned_vars))
        projected.active_operations = active_ops
        projected.projected_duration_ms = round(est_duration_ms, 2)

        # Predictive latent transition network evaluation: z_hat_{t+1} = T_phi(z_t, a_t)
        feat = torch.zeros(14, dtype=torch.float32)
        feat[0] = float(prior.memory_mb) / 1024.0
        feat[1] = float(prior.open_file_descriptors) / 10.0
        feat[2] = float(est_alloc_mb) / 1024.0
        feat[3] = float(est_duration_ms) / 1000.0
        feat[4] = float(risk_score)
        feat[5] = float(len(assigned_vars)) / 10.0
        with torch.no_grad():
            z_t = self.latent_model.encode(feat)
            act_idx = min(len(active_ops), self.latent_model.action_dim - 1)
            _, _, latent_risk = self.latent_model(z_t, act_idx)
            latent_energy = float(latent_risk.item())

        # Compatibility Energy: fused AST blast-radius + latent neural risk
        mem_penalty = max(0.0, (est_alloc_mb - 256.0) / 100.0) if est_alloc_mb > 256.0 else 0.0
        energy_score = 0.5 * risk_score + 0.5 * latent_energy + mem_penalty
        divergent = energy_score > self.energy_threshold


        rejection = ""
        if divergent:
            rejection = f"Energy compatibility failure: energy={energy_score:.2f} > {self.energy_threshold:.2f}"

        delta_s = -1.0 if divergent else 1.0
        return StateTransition(
            prior_state=prior,
            projected_state=projected,
            delta_s=delta_s,
            energy_score=energy_score,
            divergent=divergent,
            rejection_reason=rejection,
        )

    def simulate(
        self,
        code: str,
        current_state: Optional[EnvironmentState] = None,
        goal_state: Optional[EnvironmentState] = None,
    ) -> SimulationPrediction:
        """Projects collateral damage, state mutations, and safety blast-radius."""
        transition = self.predict_transition(code, current_state)
        blast_radius = list(transition.projected_state.active_operations)

        if "syntax" in transition.rejection_reason.lower():
            return SimulationPrediction(
                safe=False,
                risk_score=1.0,
                predicted_delta_s=-1.0,
                blast_radius=["syntax"],
                rejection_reason=transition.rejection_reason,
                transition=transition,
                energy_score=transition.energy_score,
            )

        # Calculate base risk score from blast radius items
        risk_score = 0.0
        for op in blast_radius:
            if "destructive_filesystem" in op:
                risk_score += 0.85
            elif "unbounded_network" in op:
                risk_score += 0.50
            elif "runaway_infinite_loop" in op:
                risk_score += 0.40

        risk_score = min(1.0, risk_score)
        safe = (risk_score < self.risk_threshold) and not transition.divergent

        rejection_reason = ""
        if not safe:
            if risk_score >= self.risk_threshold:
                rejection_reason = (
                    f"Mental simulation rejected action: risk_score={risk_score:.2f} "
                    f"exceeds threshold={self.risk_threshold:.2f} [{', '.join(blast_radius)}]"
                )
            else:
                rejection_reason = transition.rejection_reason

        predicted_delta_s = 1.0 if safe else -1.0
        return SimulationPrediction(
            safe=safe,
            risk_score=risk_score,
            predicted_delta_s=predicted_delta_s,
            blast_radius=blast_radius,
            rejection_reason=rejection_reason,
            transition=transition,
            energy_score=transition.energy_score,
        )


class LatentWorldModel(nn.Module):
    """
    Differentiable latent transition model:
    z_{t+1} = z_t + T_phi(z_t, a_t)
    Predicts state dynamics, imagined reward, and safety risk.
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Encoder: maps 14D grid feature vector to latent space z
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Latent transition network
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Prediction heads
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def encode(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        return self.encoder(state_features).squeeze(0)

    def forward(self, z: torch.Tensor, a_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_onehot = F.one_hot(torch.tensor([a_idx], device=z.device), num_classes=self.action_dim).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        za = torch.cat([z, a_onehot], dim=-1)
        z_next = z + self.transition(za)  # Residual latent transition
        reward = self.reward_head(z_next)
        risk = self.risk_head(z_next)
        return z_next.squeeze(0), reward.squeeze(), risk.squeeze()

    def train_transition_step(
        self,
        s_t: torch.Tensor,
        a_idx: int,
        s_tp1: torch.Tensor,
        reward_target: float,
        risk_target: float,
        lr: float = 0.005,
    ) -> float:
        """Trains latent dynamics directly on real observed transitions."""
        if not hasattr(self, "_optimizer") or self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = self._optimizer
        z_t = self.encoder(s_t if s_t.dim() == 2 else s_t.unsqueeze(0))
        z_target = self.encoder(s_tp1 if s_tp1.dim() == 2 else s_tp1.unsqueeze(0)).detach()

        a_onehot = F.one_hot(torch.tensor([a_idx], device=s_t.device), num_classes=self.action_dim).float()
        if a_onehot.dim() == 1:
            a_onehot = a_onehot.unsqueeze(0)
        za = torch.cat([z_t, a_onehot], dim=-1)
        z_pred = z_t + self.transition(za)
        r_pred = self.reward_head(z_pred)
        c_pred = self.risk_head(z_pred)

        loss_dyn = F.mse_loss(z_pred, z_target)
        loss_rew = F.mse_loss(r_pred, torch.tensor([[reward_target]], device=s_t.device))
        loss_risk = F.binary_cross_entropy(c_pred, torch.tensor([[risk_target]], device=s_t.device))
        total_loss = loss_dyn + 0.5 * loss_rew + 0.5 * loss_risk

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return float(total_loss.item())

    def imagine_rollout(self, s_0: torch.Tensor, action_sequence: List[int]) -> Tuple[float, float]:
        """Rolls out multi-step imagination entirely in latent space."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(s_0 if s_0.dim() == 2 else s_0.unsqueeze(0)).squeeze(0)
            total_reward = 0.0
            max_risk = 0.0
            for a in action_sequence:
                z, r, c = self.forward(z, a)
                total_reward += float(r.item()) if r.dim() == 0 else float(r[0].item())
                max_risk = max(max_risk, float(c.item()) if c.dim() == 0 else float(c[0].item()))
        return total_reward, max_risk

    def rollout_trajectory(
        self,
        s_0: torch.Tensor,
        action_sequence: List[int],
        gamma: float = 0.99
    ) -> Dict[str, Any]:
        """
        Deep multi-step prospective planning over horizon H = len(action_sequence).
        Returns full latent trajectory, per-step reward, and cumulative discounted returns.
        """
        self.eval()
        with torch.no_grad():
            z = self.encoder(s_0 if s_0.dim() == 2 else s_0.unsqueeze(0)).squeeze(0)
            latent_states = [z]
            rewards: List[float] = []
            risks: List[float] = []
            discounted_reward = 0.0

            for t, a in enumerate(action_sequence):
                z, r, c = self.forward(z, a)
                r_val = float(r.item()) if r.dim() == 0 else float(r[0].item())
                c_val = float(c.item()) if c.dim() == 0 else float(c[0].item())
                latent_states.append(z)
                rewards.append(r_val)
                risks.append(c_val)
                discounted_reward += (gamma ** t) * r_val

            max_risk = max(risks) if risks else 0.0
            return {
                "latent_trajectory": latent_states,
                "step_rewards": rewards,
                "step_risks": risks,
                "discounted_reward": discounted_reward,
                "max_risk": max_risk,
                "safe": max_risk < 0.85,
            }


class StructuredEnvironmentEncoder(nn.Module):
    """
    Encodes unstructured OS environment feedback (stdout, stderr, exit codes,
    file modifications, latency, memory, semantic error hash, ast diff count)
    into 14D latent state vector.
    """
    def __init__(self, in_features: int = 8, out_features: int = 14):
        super().__init__()
        self.in_features = in_features
        self.proj = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Linear(32, out_features),
        )

    def forward(
        self,
        stdout_len: int = 0,
        stderr_len: int = 0,
        exit_code: int = 0,
        files_modified: int = 0,
        latency_ms: float = 0.5,
        memory_mb: float = 15.0,
        semantic_error_hash: float = 0.0,
        ast_diff_count: int = 0,
    ) -> torch.Tensor:
        raw = torch.tensor(
            [
                float(stdout_len) / 500.0,
                float(stderr_len) / 500.0,
                float(exit_code),
                float(files_modified),
                float(latency_ms) / 10.0,
                float(memory_mb) / 100.0,
                float(semantic_error_hash),
                float(ast_diff_count),
            ],
            dtype=torch.float32
        )
        if raw.shape[0] > self.in_features:
            raw = raw[:self.in_features]
        elif raw.shape[0] < self.in_features:
            raw = F.pad(raw, (0, self.in_features - raw.shape[0]))
        return self.proj(raw)



class TransitionRingBuffer:
    """Thread-safe circular ring buffer for transition quadruples (s_t, a_t, s_{t+1}, reward, risk)."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._buffer: List[Tuple[torch.Tensor, int, torch.Tensor, float, float]] = []
        self._idx = 0
        self._lock = threading.Lock()

    def append(self, s_t: torch.Tensor, a_idx: int, s_tp1: torch.Tensor, reward: float, risk: float) -> None:
        with self._lock:
            item = (s_t.detach().clone(), a_idx, s_tp1.detach().clone(), float(reward), float(risk))
            if len(self._buffer) < self.capacity:
                self._buffer.append(item)
            else:
                self._buffer[self._idx] = item
                self._idx = (self._idx + 1) % self.capacity

    def sample_batch(self, batch_size: int = 16) -> List[Tuple[torch.Tensor, int, torch.Tensor, float, float]]:
        with self._lock:
            if not self._buffer:
                return []
            import random
            k = min(batch_size, len(self._buffer))
            return random.sample(self._buffer, k)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class BackgroundWorldModelTuner:
    """Asynchronous background fine-tuner for LatentWorldModel based on physical sandbox outcomes."""

    def __init__(self, latent_model: LatentWorldModel, buffer_capacity: int = 1000):
        self.latent_model = latent_model
        self.ring_buffer = TransitionRingBuffer(capacity=buffer_capacity)
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self.total_training_steps = 0
        self.last_loss: float = 0.0
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def enqueue_transition(
        self,
        s_t: torch.Tensor,
        a_idx: int,
        s_tp1: torch.Tensor,
        reward: float,
        risk: float,
    ) -> None:
        self.ring_buffer.append(s_t, a_idx, s_tp1, reward, risk)
        try:
            self._queue.put_nowait((s_t, a_idx, s_tp1, reward, risk))
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                # Idle step: perform mini-batch replay if buffer has enough samples
                if len(self.ring_buffer) >= 4:
                    batch = self.ring_buffer.sample_batch(batch_size=4)
                    for s_t, a_idx, s_tp1, reward, risk in batch:
                        try:
                            loss = self.latent_model.train_transition_step(
                                s_t=s_t,
                                a_idx=a_idx,
                                s_tp1=s_tp1,
                                reward_target=reward,
                                risk_target=risk,
                            )
                            self.last_loss = loss
                            self.total_training_steps += 1
                        except Exception:
                            pass
                continue

            s_t, a_idx, s_tp1, reward, risk = item
            try:
                loss = self.latent_model.train_transition_step(
                    s_t=s_t,
                    a_idx=a_idx,
                    s_tp1=s_tp1,
                    reward_target=reward,
                    risk_target=risk,
                )
                self.last_loss = loss
                self.total_training_steps += 1
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_training_steps": self.total_training_steps,
            "last_loss": self.last_loss,
            "buffer_size": len(self.ring_buffer),
            "queue_size": self._queue.qsize(),
        }

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)



