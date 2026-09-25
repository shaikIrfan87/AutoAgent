import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

try:
    from ..agent.zero_neural_synthesizer import ZeroNeuralSynthesizer
    from .virtual_workspace import EphemeralVirtualSystem
    from .host_commit_gate import HostCommitGate
except ImportError:
    from cognitive_engine.agent.zero_neural_synthesizer import ZeroNeuralSynthesizer
    from cognitive_engine.core.virtual_workspace import EphemeralVirtualSystem
    from cognitive_engine.core.host_commit_gate import HostCommitGate


@dataclass
class MacroActionCandidate:
    action_id: str
    action_type: str            # "induct_code", "web_search", "system_telemetry"
    target: str
    payload: str
    complexity: float           # 0.0 to 1.0 prior
    reversibility: float        # 1.0 = ephemeral sandbox, 0.0 = host filesystem write


@dataclass
class UnifiedRLCDResult:
    action: MacroActionCandidate
    calibrated_confidence: float
    net_utility: float
    delta_s: float
    committed_to_host: bool
    output_payload: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ExecutiveDecisionNet(nn.Module):
    """Macro-tier: Dual-headed utility and calibrated confidence estimator."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.utility_head.bias, 0.5)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.confidence_head[0].bias, 0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.utility_head(h), self.confidence_head(h)


try:
    from .sharded_replay_buffer import ShardedReplayBuffer, ContrastiveTrajectory
except ImportError:
    from cognitive_engine.core.sharded_replay_buffer import ShardedReplayBuffer, ContrastiveTrajectory


class UnifiedRLCDEngine:
    """Combines Macro Decision Calibration with Micro AST Trajectory Distillation."""

    def __init__(
        self,
        synthesizer: Optional[ZeroNeuralSynthesizer] = None,
        approval_threshold: float = 0.60,
        risk_weight: float = 1.0,
        gamma_occam: float = 0.01,
        num_shards: int = 4,
        max_shard_capacity: int = 250,
    ):
        self.synth = synthesizer or ZeroNeuralSynthesizer()
        self.macro_net = ExecutiveDecisionNet()
        self.macro_opt = torch.optim.Adam(self.macro_net.parameters(), lr=0.002)
        self.synth_opt = torch.optim.Adam(self.synth.net.parameters(), lr=0.001)
        self.approval_threshold = approval_threshold
        self.risk_weight = risk_weight
        self.gamma_occam = gamma_occam
        self.replay_buffer = ShardedReplayBuffer(num_shards=num_shards, max_shard_capacity=max_shard_capacity)


    def _extract_macro_features(self, cand: MacroActionCandidate, entropy: float) -> torch.Tensor:
        feat = torch.zeros(32, dtype=torch.float32)
        feat[0] = cand.complexity
        feat[1] = cand.reversibility
        feat[2] = 1.0 - cand.reversibility
        feat[3] = entropy
        feat[4] = min(len(cand.payload) / 500.0, 1.0)
        type_idx = {"induct_code": 10, "induct_package": 10, "web_search": 11, "system_telemetry": 12}.get(cand.action_type, 13)
        feat[type_idx] = 1.0
        return feat.unsqueeze(0)

    def select_macro_action(
        self,
        candidates: List[MacroActionCandidate],
        entropy: float = 1.0,
        contracts_map: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[MacroActionCandidate, float, float, bool]:
        """Macro-tier selection: picks the optimal path with calibrated risk."""
        if not candidates:
            raise ValueError("No macro action candidates provided")

        evaluated = []
        for c in candidates:
            x = self._extract_macro_features(c, entropy)
            with torch.no_grad():
                u, p = self.macro_net(x)
            p_val = p.item()
            u_val = u.item()
            contract_bonus = 0.5 if (contracts_map and contracts_map.get(c.action_id)) else 0.0
            risk = (1.0 - p_val) * (1.0 + (1.0 - c.reversibility))
            net_score = u_val + contract_bonus - (self.risk_weight * risk)
            evaluated.append((c, p_val, net_score, x))

        evaluated.sort(key=lambda item: item[2], reverse=True)
        best_cand, best_p, best_net, _ = evaluated[0]
        approved = (best_p >= self.approval_threshold) or (best_net > 0.0)
        return best_cand, best_p, best_net, approved

    def run_micro_trajectory_distillation(
        self,
        assertions: List[str],
        num_rollouts: int = 4,
        beta: float = 0.1,
    ) -> Tuple[str, float, float]:
        """Micro-tier distillation: pairs best vs worst candidate ASTs via DPO margin."""
        # 1. First test if bottom-up inductive synthesis discovers an immediate verified contract
        if assertions:
            synth_res = self.synth.learn_task(assertions, max_episodes=5)
            if synth_res.get("status") == "discovered":
                return synth_res.get("code", ""), float(synth_res.get("delta_s", 1.0)), 0.0

        # 2. Contrastive Trajectory Distillation across rollouts
        rollouts = []
        for _ in range(num_rollouts):
            code, seq, log_probs, _ = self.synth.generate_candidate_code()
            delta_s = self.synth.verify_in_sandbox(code, assertions)
            score = delta_s - (self.synth.beta * len(seq))
            rollouts.append({"code": code, "log_probs": log_probs, "delta_s": delta_s, "score": score})

        rollouts.sort(key=lambda r: r["score"], reverse=True)
        best = rollouts[0]
        worst = rollouts[-1]

        micro_loss = 0.0
        if best["score"] > worst["score"] and best["log_probs"].numel() > 0 and worst["log_probs"].numel() > 0:
            margin = float(best["score"] - worst["score"])
            self.replay_buffer.push(
                pos_code=best["code"],
                pos_log_probs=best["log_probs"],
                neg_code=worst["code"],
                neg_log_probs=worst["log_probs"],
                margin=margin,
            )
            pos_score = best["log_probs"].sum()
            neg_score = worst["log_probs"].sum()
            occam_penalty = self.gamma_occam * len(best["code"].splitlines())
            loss = -F.logsigmoid(beta * (pos_score - neg_score)) + occam_penalty
            self.synth_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.synth.net.parameters(), 1.0)
            self.synth_opt.step()
            micro_loss = float(loss.item())

        return best["code"], best["delta_s"], micro_loss

    def replay_and_distill(self, batch_size: int = 4, beta: float = 0.1) -> float:
        """Asynchronously replays prioritized contrastive trajectories across shards."""
        batch = self.replay_buffer.sample_batch(batch_size=batch_size)
        if not batch:
            return 0.0

        total_loss = torch.tensor(0.0, requires_grad=True)
        count = 0
        self.synth_opt.zero_grad()
        for item in batch:
            if item.pos_log_probs.numel() > 0 and item.neg_log_probs.numel() > 0:
                pos_score = item.pos_log_probs.sum()
                neg_score = item.neg_log_probs.sum()
                loss = -F.logsigmoid(beta * (pos_score - neg_score))
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            mean_loss = total_loss / count
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.synth.net.parameters(), 1.0)
            self.synth_opt.step()
            return float(mean_loss.item())
        return 0.0

    def update_macro_calibration(
        self,
        chosen: MacroActionCandidate,
        alt: MacroActionCandidate,
        delta_s: float,
        entropy: float = 1.0,
        tau: float = 1.0,
        alpha: float = 1.0,
    ) -> float:
        """Calibrates executive confidence based on verified sandbox ground truth."""
        x_chosen = self._extract_macro_features(chosen, entropy)
        x_alt = self._extract_macro_features(alt, entropy)

        u_c, p_c = self.macro_net(x_chosen)
        u_a, _ = self.macro_net(x_alt)

        contrast_loss = -F.logsigmoid((u_c - u_a) / tau) if chosen.action_id != alt.action_id else torch.tensor(0.0, device=u_c.device)
        target_brier = 1.0 if delta_s > 0.0 else 0.0
        brier_loss = (p_c - torch.tensor([[target_brier]], dtype=torch.float32, device=p_c.device)) ** 2

        total_macro_loss = contrast_loss + alpha * brier_loss.mean()
        self.macro_opt.zero_grad()
        total_macro_loss.backward()
        self.macro_opt.step()
        return float(total_macro_loss.item())

    def execute_unified_pipeline(
        self,
        candidates: List[MacroActionCandidate],
        assertions_map: Dict[str, List[str]],
        host_gate: HostCommitGate,
        entropy: float = 1.0,
    ) -> UnifiedRLCDResult:
        """Coordinates Macro-Selection -> Virtual Micro-Synthesis -> Attestation Commit."""
        # 1. Macro-Level Action Selection
        chosen, conf, net_u, approved = self.select_macro_action(candidates, entropy, contracts_map=assertions_map)
        alt = [c for c in candidates if c != chosen][0] if len(candidates) > 1 else chosen

        if not approved:
            return UnifiedRLCDResult(
                action=chosen,
                calibrated_confidence=conf,
                net_utility=net_u,
                delta_s=-1.0,
                committed_to_host=False,
                output_payload="",
                diagnostics={"status": "rejected_by_macro_gate", "confidence": conf, "net_utility": net_u},
            )

        # 2. Virtual Workspace Scratchpad
        with EphemeralVirtualSystem(source_workspace=".") as v_sys:
            delta_s = -1.0
            payload = ""
            micro_loss = 0.0

            if chosen.action_type == "induct_code":
                assertions = assertions_map.get(chosen.action_id, [])
                payload, delta_s, micro_loss = self.run_micro_trajectory_distillation(assertions)
            elif chosen.action_type == "induct_package":
                import json
                try:
                    pkg_files = json.loads(chosen.payload) if isinstance(chosen.payload, str) else chosen.payload
                    # Verify syntax of all python files in package
                    syntax_ok = True
                    for fname, code in pkg_files.items():
                        if fname.endswith(".py"):
                            try:
                                compile(code, fname, "exec")
                            except Exception:
                                syntax_ok = False
                                break
                    delta_s = 1.0 if syntax_ok else -1.0
                except Exception:
                    pkg_files = {}
                    delta_s = -1.0
                payload = chosen.payload
            else:
                res = self.synth.verify_in_sandbox(chosen.payload, assertions_map.get(chosen.action_id, []))
                delta_s = res
                payload = chosen.payload

            # 3. Host Commit Gate Attestation
            committed = False
            if chosen.action_type == "induct_package":
                import json
                pkg_files = json.loads(payload) if isinstance(payload, str) else payload
                is_safe, reason = host_gate.inspect_and_verify_package(pkg_files, delta_s)
                if is_safe:
                    host_gate.apply_package_to_real_host(chosen.target, pkg_files)
                    committed = True
            else:
                is_safe, reason = host_gate.inspect_and_verify(payload, delta_s)
                if is_safe:
                    host_gate.apply_to_real_host(chosen.target, payload)
                    committed = True

            # 4. Joint Feedback Loop (Calibrate Macro-Tier with Micro ground truth)
            macro_loss = self.update_macro_calibration(chosen, alt, delta_s, entropy)

            return UnifiedRLCDResult(
                action=chosen,
                calibrated_confidence=conf,
                net_utility=net_u,
                delta_s=delta_s,
                committed_to_host=committed,
                output_payload=payload,
                diagnostics={
                    "gate_verdict": reason,
                    "micro_dpo_loss": micro_loss,
                    "macro_calibration_loss": macro_loss,
                },
            )

    def evaluate_macro_action(
        self, state_vec: torch.Tensor, action_idx: int = 0, irreversibility: float = 0.5
    ) -> Dict[str, Any]:
        if not hasattr(self, "_complete_rlcd"):
            from .unified_rlcd_engine import CompleteRLCDEngine
            self._complete_rlcd = CompleteRLCDEngine()
        return self._complete_rlcd.evaluate_macro_action(state_vec, action_idx, irreversibility)

    def train_macro_step(self, state_vec: torch.Tensor, action_idx: int, delta_s: float):
        if not hasattr(self, "_complete_rlcd"):
            from .unified_rlcd_engine import CompleteRLCDEngine
            self._complete_rlcd = CompleteRLCDEngine()
        return self._complete_rlcd.train_macro_step(state_vec, action_idx, delta_s)

    def contrastive_micro_update(self, tau_pos: torch.Tensor, tau_neg: torch.Tensor) -> float:
        if not hasattr(self, "_complete_rlcd"):
            from .unified_rlcd_engine import CompleteRLCDEngine
            self._complete_rlcd = CompleteRLCDEngine()
        return self._complete_rlcd.contrastive_micro_update(tau_pos, tau_neg)
