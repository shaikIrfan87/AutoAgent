"""
Schmidhuber-Style Compression Progress & Algorithmic Information Gain Evaluator.
Prevents curiosity daemons from falling into epistemic noise traps by evaluating:
R_intrinsic(tau) = MDL(M_t) - MDL(M_{t+1})
where MDL is estimated via normalized Kolmogorov complexity (deflate/zlib compression).
"""
import math
import zlib
from typing import Any, List, Optional, Tuple, Union


class CompressionProgressEvaluator:
    """
    Evaluates whether an inquiry, task, or memory candidate provides genuine
    compression progress over existing episodic memory, rejecting uncompressible noise.
    """

    def __init__(self, min_progress_threshold: float = 0.05):
        self.min_progress_threshold = min_progress_threshold

    @staticmethod
    def estimate_mdl(data: Union[str, bytes, List[str]]) -> int:
        """Estimates Kolmogorov complexity / Minimum Description Length in bits via zlib."""
        if isinstance(data, list):
            joined = "\n".join(str(x) for x in data)
            raw = joined.encode("utf-8")
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = bytes(data)
        if not raw:
            return 0
        return len(zlib.compress(raw, level=9)) * 8

    @staticmethod
    def shannon_entropy(s: str) -> float:
        """Computes Shannon entropy H(X) in bits per character."""
        if not s:
            return 0.0
        counts = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        return -sum((cnt / n) * math.log2(cnt / n) for cnt in counts.values())

    def is_noise_trap(self, candidate_text: str) -> bool:
        """
        Detects uncompressible stochastic noise (e.g., random hex, hashes, random character streams)
        which lacks natural word structure, contains pseudo-random character distributions, or is high-entropy hex.
        """
        import re
        cleaned = candidate_text.strip()
        if len(cleaned) < 16:
            return False

        # 1. Pure hex strings >= 16 chars (hashes, random crypto tokens)
        if re.fullmatch(r"[0-9a-fA-F]{16,}", cleaned):
            return True

        # 2. Long uninterrupted tokens without spaces
        words = cleaned.split()
        if not words:
            return True
        for w in words:
            if len(w) >= 25 and len(set(w)) >= 10:
                return True

        # 3. Low word count relative to high string length
        if len(cleaned) >= 40 and len(words) <= 2 and len(set(cleaned)) >= 12:
            return True

        return False

    def evaluate_progress(
        self,
        baseline_corpus: List[str],
        new_candidate: str,
        structured_target_examples: Optional[List[str]] = None,
    ) -> Tuple[float, bool]:
        """
        Computes R_intrinsic = MDL(M_t) - MDL(M_{t+1}).
        Returns (progress_score, is_viable).
        If is_noise_trap is True, progress_score <= 0 and is_viable = False.
        """
        if self.is_noise_trap(new_candidate):
            return -1.0, False

        # If evaluating a structured rule against targets:
        if structured_target_examples:
            # Measure if new_candidate compresses target examples (e.g. an inductive rule explains them)
            uncompressed_bits = self.estimate_mdl(structured_target_examples)
            combined = [new_candidate] + structured_target_examples
            compressed_bits = self.estimate_mdl(combined)

            # Gain represents compression efficiency over raw concatenation
            gain = (uncompressed_bits - compressed_bits) / max(uncompressed_bits, 1.0)
            is_viable = gain > self.min_progress_threshold
            return float(gain), is_viable

        # Global episodic memory MDL comparison
        m_t_bits = self.estimate_mdl(baseline_corpus)
        m_tp1_bits = self.estimate_mdl(baseline_corpus + [new_candidate])

        # A good lemma adds fewer bits than its raw entropy by compressing patterns
        raw_bits = len(new_candidate.encode("utf-8")) * 8
        added_bits = m_tp1_bits - m_t_bits

        # Progress is positive when added_bits is significantly less than raw_bits (i.e. high cross-entropy overlap)
        savings = (raw_bits - added_bits) / max(raw_bits, 1.0)
        is_viable = savings > self.min_progress_threshold
        return float(savings), is_viable
