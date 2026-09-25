import math
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


class TTTAttentionLayer(nn.Module):
    """
    Attention-integrated Test-Time Training (TTT) layer.
    Online fast-weight matrices (S_t) directly modulate multi-head self-attention
    projections via an Oja-bounded delta rule (S_t = gamma * S_{t-1} - eta * grad_loss),
    allowing session context to actively reshape attention states during inference.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        gamma: float = 0.98,
        eta: float = 0.05,
        frobenius_limit: float = 2.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.gamma = gamma
        self.eta = eta
        self.frobenius_limit = frobenius_limit

        # Static query, key, value and output projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dynamic fast-weights buffer: [num_heads, head_dim, head_dim]
        self.register_buffer(
            "fast_weights",
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )

    def reset_state(self) -> None:
        """Resets dynamic fast weights to zero."""
        self.fast_weights.zero_()

    @property
    def frobenius_norm(self) -> float:
        """Returns the maximum Frobenius norm across all head fast-weight matrices."""
        return float(torch.linalg.norm(self.fast_weights, ord="fro", dim=(-2, -1)).max().item())


    @torch.no_grad()
    def adapt_step(self, k: torch.Tensor, v: torch.Tensor) -> float:
        """
        Online step adapting fast weights using Oja's normalized Hebbian delta rule.
        k, v: [num_heads, head_dim] or [d_model]
        """
        if k.dim() == 1 and k.shape[0] == self.d_model:
            k = k.view(self.num_heads, self.head_dim)
        if v.dim() == 1 and v.shape[0] == self.d_model:
            v = v.view(self.num_heads, self.head_dim)

        # RMSNorm on input features per head
        k_norm = k / (torch.sqrt(torch.mean(k**2, dim=-1, keepdim=True)) + 1e-6)
        v_norm = v / (torch.sqrt(torch.mean(v**2, dim=-1, keepdim=True)) + 1e-6)

        for h in range(self.num_heads):
            kh = k_norm[h]
            vh = v_norm[h]
            sh = self.fast_weights[h]

            # Fast weight associative reconstruction error
            pred_v = sh @ kh
            err = vh - pred_v
            denom = 1.0 + torch.dot(kh, kh).item()

            # Normalized Oja-style delta update: S_t = gamma * S_{t-1} + eta * (err x k) / denom
            delta_s = torch.outer(err, kh) / denom
            sh.mul_(self.gamma).add_(delta_s, alpha=self.eta)

            # Enforce Frobenius norm invariant
            norm = torch.linalg.norm(sh, ord="fro")
            if norm > self.frobenius_limit:
                sh.mul_(self.frobenius_limit / (norm + 1e-9))

        return self.frobenius_norm

    def forward(
        self,
        x: torch.Tensor,
        adapt_online: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with attention modulated by active TTT fast weights.
        x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Online TTT adaptation on incoming sequence tokens if enabled
        if adapt_online and not self.training:
            with torch.no_grad():
                for t in range(seq_len):
                    k_t = k[:, :, t, :].mean(dim=0)  # [num_heads, head_dim]
                    v_t = v[:, :, t, :].mean(dim=0)  # [num_heads, head_dim]
                    self.adapt_step(k_t, v_t)

        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Modulate representation using associative fast-weight readout: S_t @ q_t
        q_trans = q.unsqueeze(-1)
        s_expanded = self.fast_weights.unsqueeze(0).unsqueeze(2)
        associative_mod = torch.matmul(s_expanded, q_trans).squeeze(-1)

        fused_out = attn_out + 0.1 * associative_mod
        fused_out = fused_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(fused_out)

        if squeeze_batch:
            out = out.squeeze(0)

        return out, self.frobenius_norm

    def modulate_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        alpha: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Directly modulates active KV-cache tensors using current fast-weight parameters (S_t).
        Ensures long-horizon in-context retention is dynamically shaped by the TTT associative state.
        k_cache, v_cache: [batch_size, num_heads, seq_len, head_dim] or [batch_size, seq_len, d_model]
        """
        if k_cache.dim() == 4:
            B, H, S, D = k_cache.shape
            if H == self.num_heads and D == self.head_dim:
                s_expanded = self.fast_weights.unsqueeze(0).unsqueeze(2)
                k_trans = k_cache.unsqueeze(-1)
                v_trans = v_cache.unsqueeze(-1)

                k_mod = torch.matmul(s_expanded, k_trans).squeeze(-1)
                v_mod = torch.matmul(s_expanded, v_trans).squeeze(-1)

                return k_cache + alpha * k_mod, v_cache + alpha * v_mod
            else:
                min_h = min(H, self.num_heads)
                min_d = min(D, self.head_dim)
                k_out = k_cache.clone()
                v_out = v_cache.clone()
                for h in range(min_h):
                    sh = self.fast_weights[h, :min_d, :min_d]
                    k_out[:, h, :, :min_d] += alpha * torch.matmul(k_cache[:, h, :, :min_d], sh.t())
                    v_out[:, h, :, :min_d] += alpha * torch.matmul(v_cache[:, h, :, :min_d], sh.t())
                return k_out, v_out
        elif k_cache.dim() == 3:
            B, S, D = k_cache.shape
            k_out = k_cache.clone()
            v_out = v_cache.clone()
            min_d = min(D, self.d_model)
            fused = torch.block_diag(*[self.fast_weights[h] for h in range(self.num_heads)])
            fused_sub = fused[:min_d, :min_d]
            k_out[:, :, :min_d] += alpha * torch.matmul(k_cache[:, :, :min_d], fused_sub.t())
            v_out[:, :, :min_d] += alpha * torch.matmul(v_cache[:, :, :min_d], fused_sub.t())
            return k_out, v_out
        return k_cache, v_cache

    def sync_from_plastic(self, plastic: Any) -> float:
        """Directly synchronizes multi-head fast weights from a FastPlasticLinear layer."""
        if hasattr(plastic, "get_head_fast_weights"):
            hw = plastic.get_head_fast_weights(num_heads=self.num_heads, head_dim=self.head_dim)
            self.fast_weights.copy_(hw)
        elif hasattr(plastic, "A_fast"):
            src = plastic.A_fast
            min_d = min(self.head_dim, src.shape[0] // max(1, self.num_heads))
            for h in range(self.num_heads):
                idx = h * min_d
                self.fast_weights[h, :min_d, :min_d].copy_(src[idx : idx + min_d, idx : idx + min_d])
        return self.frobenius_norm

