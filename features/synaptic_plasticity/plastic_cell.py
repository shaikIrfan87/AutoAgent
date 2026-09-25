import numpy as np

class PlasticFastWeightCell:
    """
    Invariant 4: Plastic Fast-Weights (TTT Cell / Oja Delta Rule).
    Fast synaptic weight adaptation with strict Frobenius norm clamping.
    """
    def __init__(
        self,
        dim: int = 64,
        frobenius_cap: float = 2.0,
        decay_gamma: float = 0.95,
        plasticity_eta: float = 0.05,
    ):
        self.dim = dim
        self.frobenius_cap = frobenius_cap
        self.decay_gamma = decay_gamma
        self.plasticity_eta = plasticity_eta

        self.W_slow = np.eye(dim, dtype=np.float32)
        self.A_fast = np.zeros((dim, dim), dtype=np.float32)

    def adapt(self, k: np.ndarray, v: np.ndarray, delta_s: float):
        """
        Oja-bounded fast-weight adaptation: ΔA = ((v - A_fast·k) ⊗ kᵀ) / (1 + ||k||²)
        Clamped to ||A_fast||_F <= frobenius_cap.
        Anti-Hebbian decay penalty on delta_s <= 0.0.
        """
        if delta_s > 0.0:
            pred_v = np.dot(self.A_fast, k)
            err = v - pred_v
            k_norm_sq = float(np.dot(k, k))
            delta_A = np.outer(err, k) / (1.0 + k_norm_sq)
            self.A_fast = (self.decay_gamma * self.A_fast) + (self.plasticity_eta * delta_A)
        else:
            self.A_fast *= 0.85

        f_norm = float(np.linalg.norm(self.A_fast, "fro"))
        if f_norm > self.frobenius_cap:
            self.A_fast = (self.A_fast / f_norm) * self.frobenius_cap

    @property
    def frobenius_norm(self) -> float:
        return float(np.linalg.norm(self.A_fast, "fro"))
