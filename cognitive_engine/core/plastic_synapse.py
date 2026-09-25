import torch
import torch.nn as nn
from typing import Tuple

class BoundedPlasticTTTLayer(nn.Module):
    """
    Invariant 4: Fast synaptic plasticity operating via an Oja-bounded associative rule.
    Dynamic state remains strictly clamped: ||A_fast||_F <= cap.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4, frobenius_cap: float = 2.0, eta: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.frobenius_cap = frobenius_cap
        self.eta = eta
        
        # Slow backbone weights (Structural Reasoning)
        self.w_slow = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        
        # Ephemeral fast associative state: (num_heads, head_dim, head_dim)
        self.register_buffer("A_fast", torch.zeros(num_heads, self.head_dim, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Static projection through structural backbone
        slow_out = self.w_slow(x)
        
        # 2. Dynamic associative modulation
        b_sz, seq_len, _ = x.shape
        x_norm = self.norm(x)
        x_heads = x_norm.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Matrix multiplication per head subspace: (H, D_h, D_h) @ (B, H, D_h, S)
        fast_out = torch.einsum("hde,bhte->bhtd", self.A_fast, x_heads)
        fast_out = fast_out.transpose(1, 2).contiguous().view(b_sz, seq_len, self.dim)
        
        return slow_out + fast_out

    def adapt_online(self, k: torch.Tensor, v: torch.Tensor, delta_s: float) -> Tuple[float, float]:
        """
        Adapts A_fast using empirical grounding feedback:
        ΔS = +1.0 -> Normalized Oja associative write.
        ΔS <= 0.0 -> Anti-Hebbian unlearning / synaptic decay.
        """
        with torch.no_grad():
            if delta_s <= 0.0:
                # Suppress ungrounded/failed reasoning branch
                self.A_fast.mul_(0.85)
                return float(torch.norm(self.A_fast, p="fro")), 0.0

            # Compute normalized associative update across head projections
            k_flat = k.reshape(self.dim)
            v_flat = v.reshape(self.dim)
            k_proj = self.norm(k_flat).view(self.num_heads, self.head_dim)
            v_proj = v_flat.view(self.num_heads, self.head_dim)
            
            delta_magnitudes = []
            for h in range(self.num_heads):
                kh = k_proj[h]
                vh = v_proj[h]
                
                # Associative error signal: e = v - A @ k
                pred_error = vh - torch.matmul(self.A_fast[h], kh)
                # Bounded Delta Rule
                delta_Ah = torch.outer(pred_error, kh) / (1.0 + torch.dot(kh, kh))
                
                self.A_fast[h].add_(self.eta * delta_Ah)
                delta_magnitudes.append(float(torch.norm(delta_Ah, p="fro")))

            # Enforce Invariant: Frobenius norm clamp (||A_fast||_F <= cap)
            total_frobenius = float(torch.norm(self.A_fast, p="fro"))
            if total_frobenius > self.frobenius_cap:
                self.A_fast.mul_(self.frobenius_cap / (total_frobenius + 1e-8))
                total_frobenius = self.frobenius_cap

            max_delta = max(delta_magnitudes)
            return total_frobenius, max_delta

    def modulate_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        alpha: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically shifts multi-head key/value attention contexts using learned fast weights.
        k_cache, v_cache shape: (batch, num_heads, seq_len, head_dim) or (batch, seq_len, dim)
        """
        orig_device = k_cache.device
        orig_dtype = k_cache.dtype
        A = self.A_fast.to(device=orig_device, dtype=orig_dtype)

        if k_cache.dim() == 4:
            k_mod = k_cache + alpha * torch.einsum("hde,bhse->bhsd", A, k_cache)
            v_mod = v_cache + alpha * torch.einsum("hde,bhse->bhsd", A, v_cache)
            return k_mod, v_mod
        elif k_cache.dim() == 3:
            b_sz, seq_len, _ = k_cache.shape
            k_heads = k_cache.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_heads = v_cache.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_mod = k_heads + alpha * torch.einsum("hde,bhse->bhsd", A, k_heads)
            v_mod = v_heads + alpha * torch.einsum("hde,bhse->bhsd", A, v_heads)
            k_out = k_mod.transpose(1, 2).contiguous().view(b_sz, seq_len, self.dim)
            v_out = v_mod.transpose(1, 2).contiguous().view(b_sz, seq_len, self.dim)
            return k_out, v_out
        return k_cache, v_cache

