import math
import re
from collections import Counter
from typing import Tuple

class EpistemicGatingEvaluator:
    """
    Invariant 1: Epistemic Gating & Sensory Saliency.
    Computes Shannon entropy H(X), hex burst anomaly ratios, and epistemic uncertainty.
    """
    def __init__(self, min_entropy: float = 0.8, max_hex_ratio: float = 0.30, entropy_scale: float = 4.5):
        self.min_entropy = min_entropy
        self.max_hex_ratio = max_hex_ratio
        self.entropy_scale = entropy_scale
        self._hex_pattern = re.compile(r"(?:[0-9a-fA-F]{2}){4,}")

    def evaluate_saliency(self, text: str, novelty_prior: float = 0.8) -> Tuple[bool, float, str]:
        """
        Calculates Shannon Entropy and Out-of-Vocabulary / Hex Anomaly Ratios.
        Returns: (passed: bool, uncertainty: float, reason: str)
        """
        raw_len = len(text)
        if raw_len == 0:
            return False, 1.0, "Empty payload"

        cleaned = text.strip()
        words = cleaned.split()

        counts = Counter(text)
        probs = (count / raw_len for count in counts.values())
        shannon_h = -sum(p * math.log2(p) for p in probs)

        # Catch pure repetitive mono-token or single-character noise
        is_repetitive = len(set(cleaned.replace(" ", ""))) <= 2 and raw_len > 4

        # Short valid natural language queries pass directly unless pure repetitive noise
        is_short_valid_query = bool(1 <= len(words) <= 8 and any(c.isalpha() for c in cleaned) and not is_repetitive)

        if not is_short_valid_query and (shannon_h < self.min_entropy or is_repetitive):
            return False, 0.0, f"Dropped: Low-entropy spam (H={shannon_h:.2f} < {self.min_entropy})"

        oov_fragments = len(self._hex_pattern.findall(text))
        if (oov_fragments * 8) / raw_len > self.max_hex_ratio:
            return False, 1.0, "Dropped: Masked hex/cryptic blob detected"

        h_norm = min(1.0, shannon_h / self.entropy_scale)
        uncertainty = (0.6 * novelty_prior) + (0.4 * h_norm)
        return True, uncertainty, "Accepted"
