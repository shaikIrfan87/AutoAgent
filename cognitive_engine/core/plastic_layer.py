import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple


class FastPlasticLinear(nn.Module):
    """Invariant 4: Bounded Fast-Weight / Test-Time Training (TTT) Cell.
    Online memory layer that absorbs session tokens directly into dynamic parameters
    via an Oja-bounded delta rule without backpropagation.
    """

    def __init__(
        self,
        in_features: int = 384,
        out_features: int = 384,
        gamma: float = 0.95,
        eta: float = 0.05,
        frobenius_limit: float = 2.0,
        flush_threshold: float = 0.35,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.eta = eta
        self.frobenius_limit = frobenius_limit
        self.flush_threshold = flush_threshold
        self.num_heads = num_heads if (in_features % num_heads == 0 and out_features % num_heads == 0) else 1
        self.head_dim = in_features // self.num_heads
        self.last_trace_magnitude: float = 0.0
        self.last_flush_required: bool = False

        # Slow weights: static/frozen representation
        self.W_slow = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )

        # Fast weights: dynamic session memory buffer
        self.register_buffer("A_fast", torch.zeros(out_features, in_features))
        self._session_states: dict[tuple[str, str], torch.Tensor] = {}

        # Intermediate-rate TTT adapter: multi-turn session persistence
        self.gamma_mid = 0.999
        self.eta_mid = 0.01
        self._mid_states: dict[tuple[str, str], torch.Tensor] = {}

    def _get_session_state(self, tenant_id: str = "default_tenant", session_id: str = "default_session") -> torch.Tensor:
        key = (tenant_id, session_id)
        if key not in self._session_states:
            self._session_states[key] = torch.zeros(self.out_features, self.in_features, device=self.W_slow.device)
        return self._session_states[key]

    def _get_mid_state(self, tenant_id: str = "default_tenant", session_id: str = "default_session") -> torch.Tensor:
        key = (tenant_id, session_id)
        if key not in self._mid_states:
            self._mid_states[key] = torch.zeros(self.out_features, self.in_features, device=self.W_slow.device)
        return self._mid_states[key]

    def reset_session(
        self, tenant_id: str = "default_tenant", session_id: str = "default_session", reset_mid: bool = False
    ) -> None:
        """Clear dynamic fast weights between user sessions (optionally keeping mid-term adapter)."""
        buf = self._get_session_state(tenant_id, session_id)
        buf.zero_()
        self.A_fast.copy_(buf)
        if reset_mid:
            buf_mid = self._get_mid_state(tenant_id, session_id)
            buf_mid.zero_()
        self.last_trace_magnitude = 0.0
        self.last_flush_required = False

    def dump_state(self, tenant_id: str = "default_tenant", session_id: str = "default_session") -> bytes:
        """Serialize fast weights tensor for ACID persistence."""
        buf = self._get_session_state(tenant_id, session_id)
        return buf.cpu().numpy().astype(np.float32).tobytes()

    def load_state(self, raw_bytes: bytes, tenant_id: str = "default_tenant", session_id: str = "default_session") -> None:
        """Restore fast weights tensor from bytes."""
        arr = np.frombuffer(raw_bytes, dtype=np.float32).copy().reshape(self.out_features, self.in_features)
        buf = self._get_session_state(tenant_id, session_id)
        buf.copy_(torch.from_numpy(arr))
    @torch.no_grad()
    def adapt_online(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        delta_s: float,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> Tuple[float, float]:
        """
        Adapts fast weights with true grounding feedback:
        Delta S <= 0.0 -> Anti-Hebbian exponential unlearning (0.85 decay factor).
        Delta S > 0.0  -> Normalized Oja associative absorption with RMSNorm preconditioning.
        """
        buf = self._get_session_state(tenant_id, session_id)
        self.A_fast.copy_(buf)
        if delta_s <= 0.0:
            self.A_fast.mul_(0.85)
            buf.copy_(self.A_fast)
            norm = float(torch.linalg.norm(self.A_fast, ord="fro").item())
            return norm, 0.0

        norm = self.absorb(k, v, steps=1, tenant_id=tenant_id, session_id=session_id)
        return norm, self.last_trace_magnitude

    def modulate_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        alpha: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically shifts multi-head key/value attention contexts using learned fast weights.
        Supports:
        - 4D: (batch, num_heads, seq_len, head_dim)
        - 3D: (batch, seq_len, features)
        - 2D: (seq_len, features)
        """
        orig_device = k_cache.device
        orig_dtype = k_cache.dtype
        A = self.A_fast.to(device=orig_device, dtype=orig_dtype)

        if k_cache.dim() == 4:
            b, h, s, d = k_cache.shape
            if self.num_heads == h and self.head_dim == d:
                blocks = [A[i * d : (i + 1) * d, i * d : (i + 1) * d] for i in range(h)]
                A_heads = torch.stack(blocks, dim=0)
                k_mod = k_cache + alpha * torch.einsum("hde,bhse->bhsd", A_heads, k_cache)
                v_mod = v_cache + alpha * torch.einsum("hde,bhse->bhsd", A_heads, v_cache)
            else:
                A_sub = A[:d, :d] if (d <= self.out_features and d <= self.in_features) else A
                k_mod = k_cache + alpha * (k_cache @ A_sub.T)
                v_mod = v_cache + alpha * (v_cache @ A_sub.T)
            return k_mod, v_mod
        elif k_cache.dim() == 3:
            k_mod = k_cache + alpha * (k_cache @ A.T)
            v_mod = v_cache + alpha * (v_cache @ A.T)
            return k_mod, v_mod
        elif k_cache.dim() == 2:
            k_mod = k_cache + alpha * (k_cache @ A.T)
            v_mod = v_cache + alpha * (v_cache @ A.T)
            return k_mod, v_mod
        return k_cache, v_cache

    @torch.no_grad()
    def absorb(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        steps: int = 1,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> float:
        """Associative Delta Rule Update with Decoupled Multi-Head Projection:
        Splits key/value into H orthogonal subspace heads to eliminate channel cross-talk.
        """
        buf = self._get_session_state(tenant_id, session_id)
        self.A_fast.copy_(buf)

        k = key.view(-1).float()
        v = value.view(-1).float()

        # RMSNorm prior to head projection: prevents subspace alignment drift across heads
        k = k / (torch.sqrt(torch.mean(k**2)) + 1e-6)
        v = v / (torch.sqrt(torch.mean(v**2)) + 1e-6)

        denom = 1.0 + torch.dot(k, k).item()
        pred_init = self.A_fast @ k

        error_init = v - pred_init
        self.last_trace_magnitude = float(
            (torch.linalg.norm(error_init) * torch.linalg.norm(k) / denom).item()
        )
        self.last_flush_required = self.last_trace_magnitude >= self.flush_threshold

        for _ in range(steps):
            if self.num_heads > 1:
                # Multi-head decoupled block update: zero cross-talk across distinct heads
                for h in range(self.num_heads):
                    idx_start = h * self.head_dim
                    idx_end = (h + 1) * self.head_dim
                    k_h = k[idx_start:idx_end]
                    v_h = v[idx_start:idx_end]
                    pred_h = self.A_fast[idx_start:idx_end, idx_start:idx_end] @ k_h
                    err_h = v_h - pred_h
                    d_denom = 1.0 + torch.dot(k_h, k_h).item()
                    d_a_h = torch.outer(err_h, k_h) / d_denom
                    self.A_fast[idx_start:idx_end, idx_start:idx_end].mul_(self.gamma).add_(d_a_h, alpha=self.eta)
            else:
                pred = self.A_fast @ k
                error = v - pred
                delta_a = torch.outer(error, k) / denom
                self.A_fast.mul_(self.gamma).add_(delta_a, alpha=self.eta)

            # Strict Frobenius norm bound
            norm = torch.linalg.norm(self.A_fast, ord="fro")
            if norm > self.frobenius_limit:
                self.A_fast.mul_(self.frobenius_limit / (norm + 1e-9))

        buf.copy_(self.A_fast)
        # Multi-speed intermediate adapter absorption
        buf_mid = self._get_mid_state(tenant_id, session_id)
        delta_mid = torch.outer(v, k) / denom
        buf_mid.mul_(self.gamma_mid).add_(delta_mid, alpha=self.eta_mid)
        norm_mid = torch.linalg.norm(buf_mid, ord="fro")
        if norm_mid > self.frobenius_limit:
            buf_mid.mul_(self.frobenius_limit / (norm_mid + 1e-9))

        return float(torch.linalg.norm(self.A_fast, ord="fro").item())

    @torch.no_grad()
    def penalize(
        self,
        key: Optional[Any] = None,
        value: Optional[Any] = None,
        penalty: float = 0.1,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> float:
        """Anti-Hebbian unlearning / edge penalization for failed trajectories."""
        buf = self._get_session_state(tenant_id, session_id)
        self.A_fast.copy_(buf)

        if key is None or value is None:
            self.A_fast.mul_(1.0 - penalty)
            buf.copy_(self.A_fast)
            return float(torch.linalg.norm(self.A_fast, ord="fro").item())

        if isinstance(key, str):
            toks = [ord(c) % 64 for c in key[:self.in_features]]
            toks += [0] * (self.in_features - len(toks))
            k = torch.tensor(toks, dtype=torch.float32)
        elif isinstance(key, np.ndarray):
            k = torch.from_numpy(key).float()
        else:
            k = key.view(-1).float()

        if isinstance(value, str):
            toks = [ord(c) % 64 for c in value[:self.out_features]]
            toks += [0] * (self.out_features - len(toks))
            v = torch.tensor(toks, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            v = torch.from_numpy(value).float()
        else:
            v = value.view(-1).float()

        k = k / (torch.sqrt(torch.mean(k**2)) + 1e-6)
        v = v / (torch.sqrt(torch.mean(v**2)) + 1e-6)

        if self.num_heads > 1:
            for h in range(self.num_heads):
                idx_start = h * self.head_dim
                idx_end = (h + 1) * self.head_dim
                k_h = k[idx_start:idx_end]
                v_h = v[idx_start:idx_end]
                d_denom = 1.0 + torch.dot(k_h, k_h).item()
                d_a_h = torch.outer(v_h, k_h) / d_denom
                self.A_fast[idx_start:idx_end, idx_start:idx_end].sub_(d_a_h, alpha=penalty)
        else:
            delta = torch.outer(v, k) / (1.0 + torch.dot(k, k).item())
            self.A_fast.sub_(delta, alpha=penalty)

        norm = torch.linalg.norm(self.A_fast, ord="fro")
        if norm > self.frobenius_limit:
            self.A_fast.mul_(self.frobenius_limit / (norm + 1e-9))

        buf.copy_(self.A_fast)
        return float(torch.linalg.norm(self.A_fast, ord="fro").item())

    @torch.no_grad()
    def absorb_with_flush_check(
        self,
        key: Any,
        value: Any,
        steps: int = 1,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> tuple[float, bool, float]:
        if isinstance(key, str):
            toks = [ord(c) % 64 for c in key[:self.in_features]]
            toks += [0] * (self.in_features - len(toks))
            key = torch.tensor(toks, dtype=torch.float32)
        elif isinstance(key, np.ndarray):
            key = torch.from_numpy(key).float()

        if isinstance(value, str):
            toks = [ord(c) % 64 for c in value[:self.out_features]]
            toks += [0] * (self.out_features - len(toks))
            value = torch.tensor(toks, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()

        norm = self.absorb(key, value, steps=steps, tenant_id=tenant_id, session_id=session_id)
        return norm, self.last_flush_required, self.last_trace_magnitude

    def forward(self, x: torch.Tensor, tenant_id: str = "default_tenant", session_id: str = "default_session") -> torch.Tensor:
        """y = x @ (W_slow + A_mid + A_fast)^T"""
        buf = self._get_session_state(tenant_id, session_id)
        buf_mid = self._get_mid_state(tenant_id, session_id)
        W_eff = self.W_slow + buf_mid + buf
        return x @ W_eff.t()

    def recall_fast(
        self,
        key: torch.Tensor,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> torch.Tensor:
        """Fast recall query from session A_fast directly: y = A_fast @ k"""
        buf = self._get_session_state(tenant_id, session_id)
        k = key.view(-1).float()
        return buf @ k

    def get_head_fast_weights(
        self,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> torch.Tensor:
        """Extract multi-head fast weights tensor of shape [num_heads, head_dim, head_dim]."""
        buf = self._get_session_state(tenant_id, session_id)
        n_heads = num_heads or self.num_heads
        h_dim = head_dim or (self.in_features // n_heads)

        if n_heads == self.num_heads and h_dim == self.head_dim:
            heads = []
            for h in range(n_heads):
                idx_s = h * h_dim
                idx_e = (h + 1) * h_dim
                heads.append(buf[idx_s:idx_e, idx_s:idx_e])
            return torch.stack(heads, dim=0)

        out = torch.zeros(n_heads, h_dim, h_dim, device=buf.device, dtype=buf.dtype)
        min_dim = min(h_dim, self.head_dim)
        for h in range(min(n_heads, self.num_heads)):
            src_s = h * self.head_dim
            out[h, :min_dim, :min_dim] = buf[src_s : src_s + min_dim, src_s : src_s + min_dim]
        return out

    def modulate_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        alpha: float = 0.15,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Directly modulates multi-head KV cache tensors with active empirical fast weights.
        k_cache, v_cache: [batch, num_heads, seq_len, head_dim] or [batch, seq_len, dim]
        """
        if k_cache.dim() == 4:
            B, H, S, D = k_cache.shape
            head_weights = self.get_head_fast_weights(num_heads=H, head_dim=D, tenant_id=tenant_id, session_id=session_id)
            s_expanded = head_weights.unsqueeze(0).unsqueeze(2)  # [1, H, 1, D, D]
            k_trans = k_cache.unsqueeze(-1)
            v_trans = v_cache.unsqueeze(-1)

            k_mod = torch.matmul(s_expanded, k_trans).squeeze(-1)
            v_mod = torch.matmul(s_expanded, v_trans).squeeze(-1)

            return k_cache + alpha * k_mod, v_cache + alpha * v_mod
        elif k_cache.dim() == 3:
            B, S, D = k_cache.shape
            buf = self._get_session_state(tenant_id, session_id)
            min_d = min(D, self.in_features)
            W_sub = buf[:min_d, :min_d]
            k_out = k_cache.clone()
            v_out = v_cache.clone()
            k_out[:, :, :min_d] += alpha * torch.matmul(k_cache[:, :, :min_d], W_sub.t())
            v_out[:, :, :min_d] += alpha * torch.matmul(v_cache[:, :, :min_d], W_sub.t())
            return k_out, v_out
        return k_cache, v_cache


class MultiHeadPlasticLayer(FastPlasticLinear):
    """Invariant 4: Multi-Head Plastic Layer with decoupled block updates and Oja bounding."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 4,
        decay_rate: float = 0.95,
        eta: float = 0.05,
        frobenius_limit: float = 2.0,
        flush_threshold: float = 0.35,
    ):
        super().__init__(
            in_features=dim,
            out_features=dim,
            gamma=decay_rate,
            eta=eta,
            frobenius_limit=frobenius_limit,
            flush_threshold=flush_threshold,
            num_heads=num_heads,
        )
        self._keys: list[torch.Tensor] = []
        self._vals: list[torch.Tensor] = []
        self._weights: list[float] = []

    def reset_session(self) -> None:
        super().reset_session()
        self._keys.clear()
        self._vals.clear()
        self._weights.clear()

    @torch.no_grad()
    def adapt(self, key: torch.Tensor, value: torch.Tensor, steps: int = 1) -> float:
        k = key.view(1, -1).float()
        v = value.view(1, -1).float()
        k_norm = k / (torch.linalg.norm(k) + 1e-6)
        v_norm = v / (torch.linalg.norm(v) + 1e-6)

        # Decay fast matrix and stored association weights
        self.A_fast.mul_(self.gamma)
        for i in range(len(self._weights)):
            self._weights[i] *= self.gamma

        # Store association in modern Hopfield continuous memory buffer
        self._keys.append(k_norm)
        self._vals.append(v_norm)
        self._weights.append(1.0)
        if len(self._keys) > 512:
            self._keys.pop(0)
            self._vals.pop(0)
            self._weights.pop(0)

        # Update fast weight matrix with Oja's bounded rule
        outer = torch.mm(v_norm.t(), k_norm)
        self.A_fast.add_(outer, alpha=self.eta)
        norm = torch.linalg.norm(self.A_fast, ord="fro")
        if norm > self.frobenius_limit:
            self.A_fast.mul_(self.frobenius_limit / (norm + 1e-9))

        return float(torch.linalg.norm(self.A_fast, ord="fro").item())

    def recall(self, key: torch.Tensor) -> torch.Tensor:
        k = key.view(1, -1).float()
        k_norm = k / (torch.linalg.norm(k) + 1e-6)

        if len(self._keys) > 0:
            all_keys = torch.cat(self._keys, dim=0)
            all_vals = torch.cat(self._vals, dim=0)
            weights = torch.tensor(self._weights, device=k.device).unsqueeze(1)

            # Cosine affinities with kernel sharpening
            sims = torch.mm(all_keys, k_norm.t())
            weights_eff = weights * torch.exp(sims * 10.0)
            weights_norm = weights_eff / (torch.sum(weights_eff) + 1e-9)
            recalled = torch.mm(weights_norm.t(), all_vals)
            return recalled.view(key.shape)

        return (self.A_fast @ k_norm.t()).t().view(key.shape)


class TTTAttentionLayer(nn.Module):
    """Integrated Test-Time Training (TTT) Attention Layer.
    Combines multi-head scaled dot-product attention with online test-time parameter adaptation.
    During inference, fast weights W_ttt adapt online to sequence tokens via gradient descent
    on a self-supervised reconstruction objective, modulating primary hidden representations.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        eta: float = 0.05,
        gamma: float = 0.98,
        frobenius_limit: float = 3.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eta = eta
        self.gamma = gamma
        self.frobenius_limit = frobenius_limit

        # Base attention projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dynamic TTT fast-weight parameters (adapted at test-time)
        self.register_buffer("W_ttt", torch.zeros(embed_dim, embed_dim))

    def reset_ttt(self) -> None:
        """Clear dynamic test-time fast weights."""
        self.W_ttt.zero_()

    def forward(
        self,
        x: torch.Tensor,
        adapt_online: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass with online test-time parameter adaptation.
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            adapt_online: Whether to perform online TTT updates during forward pass
        Returns:
            output: Output tensor (batch, seq_len, embed_dim)
            ttt_norm: Current Frobenius norm of test-time fast weights
        """
        B, T, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 1. Base multi-head scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)

        # 2. Test-Time Training (TTT) online adaptation across sequence tokens
        if adapt_online:
            with torch.no_grad():
                for t in range(T):
                    k_t = K[:, t, :]  # (B, D)
                    v_t = V[:, t, :]  # (B, D)

                    # Online prediction error: e_t = v_t - k_t @ W_ttt
                    pred_v = torch.matmul(k_t, self.W_ttt)
                    err = v_t - pred_v  # (B, D)

                    # Test-time gradient step on reconstruction loss 0.5 * ||k W - v||^2
                    delta_W = self.eta * torch.matmul(k_t.t(), err) / B

                    # Decay and update
                    self.W_ttt.mul_(self.gamma)
                    self.W_ttt.add_(delta_W)

                    # Invariant 4: Frobenius norm bounding
                    norm = torch.linalg.norm(self.W_ttt, ord="fro")
                    if norm > self.frobenius_limit:
                        self.W_ttt.mul_(self.frobenius_limit / (norm + 1e-9))

        # 3. Modulate output representation with TTT fast projection
        ttt_modulation = torch.matmul(Q, self.W_ttt)
        combined = attn_out + ttt_modulation
        output = self.out_proj(combined)
        current_norm = float(torch.linalg.norm(self.W_ttt, ord="fro").item())
        return output, current_norm

    def modulate_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        alpha: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modulate key and value cache tensors using test-time fast weights W_ttt."""
        if k_cache.dim() == 4:
            B, H, S, D = k_cache.shape
            min_d = min(D, self.head_dim)
            k_mod = torch.zeros_like(k_cache)
            v_mod = torch.zeros_like(v_cache)
            for h in range(min(H, self.num_heads)):
                h_s = h * self.head_dim
                head_w = self.W_ttt[h_s : h_s + min_d, h_s : h_s + min_d]
                k_mod[:, h, :, :min_d] = torch.matmul(k_cache[:, h, :, :min_d], head_w.t())
                v_mod[:, h, :, :min_d] = torch.matmul(v_cache[:, h, :, :min_d], head_w.t())
            return k_cache + alpha * k_mod, v_cache + alpha * v_mod
        elif k_cache.dim() == 3:
            B, S, D = k_cache.shape
            min_d = min(D, self.embed_dim)
            W_sub = self.W_ttt[:min_d, :min_d]
            k_out = k_cache.clone()
            v_out = v_cache.clone()
            k_out[:, :, :min_d] += alpha * torch.matmul(k_cache[:, :, :min_d], W_sub.t())
            v_out[:, :, :min_d] += alpha * torch.matmul(v_cache[:, :, :min_d], W_sub.t())
            return k_out, v_out
        return k_cache, v_cache

    def sync_from_plastic(self, plastic: Any) -> float:
        """Synchronizes test-time fast weights directly from a FastPlasticLinear instance."""
        if hasattr(plastic, "A_fast"):
            src = plastic.A_fast
            min_in = min(self.embed_dim, src.shape[1])
            min_out = min(self.embed_dim, src.shape[0])
            self.W_ttt.zero_()
            self.W_ttt[:min_out, :min_in].copy_(src[:min_out, :min_in])
        return float(torch.linalg.norm(self.W_ttt, ord="fro").item())



