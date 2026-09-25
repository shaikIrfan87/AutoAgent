"""
Dense semantic embedding provider using in-process quantized ONNX fastembed.
Falls back to topological hashing if fastembed is unavailable.
"""

from typing import Callable, Optional
import numpy as np

_embedder = None


def safe_normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """Normalizes vector to unit sphere; protects against NaN / zero-norm division."""
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def get_semantic_embedding(text: str, fallback_fn: Optional[Callable[[str], np.ndarray]] = None) -> np.ndarray:
    """Lazy-loaded in-process BGE-small embedding (384-d, <10ms CPU)."""
    global _embedder
    try:
        if _embedder is None:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        emb = list(_embedder.embed([text]))[0]
        vec = emb.astype(np.float32)
    except Exception:
        if fallback_fn:
            vec = fallback_fn(text)
        else:
            vec = np.zeros(384, dtype=np.float32)

    return safe_normalize_embedding(vec)
