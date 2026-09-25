import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Optional
from collections import deque


class MacroDecisionNet(nn.Module):
    """
    Tier 1: Evaluates discrete action choices, outputs Q(s, a) and calibrated P(success).
    Enforces dynamic blast-radius risk gating without token generation.
    """
    def __init__(self, input_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.q_head = nn.Linear(64, num_actions)
        self.calibration_head = nn.Linear(64, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        q_vals = self.q_head(feat)
        p_success = torch.sigmoid(self.calibration_head(feat))
        return q_vals, p_success


class MicroASTPolicyValueNet(nn.Module):
    """
    Tier 2: Dual-head neural prior for MCTS program expansion over typed Lambda DSL.
    Initialized with zero prior weights and optimized via contrastive self-play.
    """
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_seq)
        out, h = self.gru(emb)
        last_hidden = out[:, -1, :]
        logits = self.policy_head(out)
        val = torch.tanh(self.value_head(last_hidden))
        return logits, val


class CompleteRLCDEngine:
    def __init__(self, state_dim: int = 64, vocab_size: int = 32, lr: float = 1e-3, beta: float = 0.5):
        self.beta = beta
        self.macro_net = MacroDecisionNet(input_dim=state_dim, num_actions=4)
        self.macro_opt = optim.Adam(self.macro_net.parameters(), lr=lr)
        
        self.micro_net = MicroASTPolicyValueNet(vocab_size=vocab_size, hidden_dim=64)
        self.micro_opt = optim.Adam(self.micro_net.parameters(), lr=lr)
        
        self.brier_criterion = nn.MSELoss()
        self.replay_pairs = deque(maxlen=2000)

    def evaluate_macro_action(
        self, state_vec: torch.Tensor, action_idx: int = 0, irreversibility: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculates Net Score = Q(s, a) - Risk.
        Risk = (1.0 - P(success)) * (1.0 + Irreversibility)
        """
        self.macro_net.eval()
        with torch.no_grad():
            if state_vec.dim() == 1:
                state_vec = state_vec.unsqueeze(0)
            if state_vec.size(-1) != self.macro_net.shared[0].in_features:
                # Pad or slice to state_dim
                dim = self.macro_net.shared[0].in_features
                if state_vec.size(-1) < dim:
                    state_vec = torch.nn.functional.pad(state_vec, (0, dim - state_vec.size(-1)))
                else:
                    state_vec = state_vec[:, :dim]

            q_vals, p_success = self.macro_net(state_vec)
            q = q_vals[0, action_idx].item()
            p = p_success[0, action_idx].item()
            
        risk = (1.0 - p) * (1.0 + irreversibility)
        net_score = q - risk
        # Calibrated approval check
        approved = (p >= 0.40) or (risk < 0.90)
        return {
            "action_idx": action_idx,
            "q_value": q,
            "p_success": p,
            "risk": risk,
            "net_score": net_score,
            "approved": approved
        }

    def train_macro_step(self, state_vec: torch.Tensor, action_idx: int, delta_s: float):
        """
        Aligns internal confidence with empirical ground truth using Brier calibration loss.
        """
        self.macro_net.train()
        self.macro_opt.zero_grad()
        
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        dim = self.macro_net.shared[0].in_features
        if state_vec.size(-1) < dim:
            state_vec = torch.nn.functional.pad(state_vec, (0, dim - state_vec.size(-1)))
        else:
            state_vec = state_vec[:, :dim]

        q_vals, p_success = self.macro_net(state_vec)
        target_y = 1.0 if delta_s > 0 else 0.0
        
        target_tensor = torch.tensor([target_y], dtype=torch.float32)
        cal_loss = self.brier_criterion(p_success[0, action_idx:action_idx+1], target_tensor)
        q_loss = (q_vals[0, action_idx] - delta_s) ** 2
        
        loss = q_loss + 1.5 * cal_loss
        loss.backward()
        self.macro_opt.step()

    def contrastive_micro_update(
        self, tau_pos: torch.Tensor, tau_neg: torch.Tensor
    ) -> float:
        """
        Direct Preference Optimization (DPO) contrastive trajectory margin update:
        L_micro = -log σ(β * [Σ log π(τ⁺) - Σ log π(τ⁻)])
        """
        self.micro_net.train()
        self.micro_opt.zero_grad()

        # Compute log-probs for positive trajectory
        logits_pos, _ = self.micro_net(tau_pos.unsqueeze(0))
        log_probs_pos = torch.log_softmax(logits_pos, dim=-1)
        # Gather token probabilities
        lp_pos = log_probs_pos[0, torch.arange(tau_pos.size(0)), tau_pos].sum()

        # Compute log-probs for negative trajectory
        logits_neg, _ = self.micro_net(tau_neg.unsqueeze(0))
        log_probs_neg = torch.log_softmax(logits_neg, dim=-1)
        lp_neg = log_probs_neg[0, torch.arange(tau_neg.size(0)), tau_neg].sum()

        # Contrastive Preference Margin
        margin = self.beta * (lp_pos - lp_neg)
        loss = -torch.log(torch.sigmoid(margin) + 1e-8) + 0.01 * float(tau_pos.size(0))

        loss.backward()
        self.micro_opt.step()
        return loss.item()
