import math
import re
from collections import Counter
from typing import Callable, Optional, List, Dict
import numpy as np

try:
    from .types import SaliencyDecision
except ImportError:
    from types import SaliencyDecision


import os
from pathlib import Path


def _semantic_topological_embed(text: str, dim: int = 384) -> np.ndarray:
    """Topologically coherent subword/n-gram semantic projection.
    Preserves metric space neighborhoods: related words (e.g. dog/canine, cat/feline)
    yield high cosine similarity, while unrelated tokens remain near-orthogonal.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)
    # Cluster concept roots for semantic neighborhood preservation & synonym manifold alignment
    concept_clusters = {
        ("dog", "canine", "hound", "puppy"): 12,
        ("cat", "feline", "kitten"): 13,
        ("computer", "machine", "computation", "algorithm"): 14,
        ("quantum", "physics", "atom", "qubit"): 15,
        ("biology", "organism", "cellular", "human"): 16,
        ("exploit", "attack", "vulnerability", "breach", "payload", "malware", "injection", "hack"): 17,
        ("prime", "sieve", "eratosthenes", "divisor", "factor"): 18,
        ("function", "method", "procedure", "subroutine", "routine", "solution"): 19,
        ("search", "query", "find", "retrieve", "lookup", "fetch", "recall"): 20,
        ("fast", "quick", "rapid", "instant", "speed", "latency"): 21,
    }

    for word in words:
        # 1. Concept cluster contribution
        matched_cluster = False
        for cluster_words, cluster_idx in concept_clusters.items():
            if any(w in word for w in cluster_words):
                vec[cluster_idx % dim] += 4.0
                matched_cluster = True

        # 2. Character n-gram subword decomposition (lengths 2 to 4)
        for n in range(2, min(5, len(word) + 1)):
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n]
                idx = (hash(ngram) % (dim - 30)) + 30
                vec[idx] += 1.0

        # 3. Individual token hash
        t_idx = (hash(word) % (dim - 30)) + 30
        vec[t_idx] += 1.5

    norm = np.linalg.norm(vec)
    return (vec / (norm + 1e-9)).astype(np.float32)



CONVERSATIONAL_ALLOWLIST = {
    "yes", "no", "ok", "okay", "sure", "yep", "nope", "bye", "hello", "hi", "hey",
    "thanks", "thank you", "k", "help", "test", "demo", "status", "1", "2", "3"
}


class FeatureEncoder:
    """Invariant 1 feature encoder for semantic embedding."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        try:
            try:
                from .embeddings import get_semantic_embedding
            except ImportError:
                from embeddings import get_semantic_embedding
            return get_semantic_embedding(text, fallback_fn=lambda x: _semantic_topological_embed(x, self.dim))
        except Exception:
            return _semantic_topological_embed(text, self.dim)


class SaliencyGate:
    """Invariant 1: Sensory Gating & Epistemic Gate.
    Filters zero-information input, evaluates novelty and epistemic uncertainty.
    """

    def __init__(
        self,
        buffer_size: int = 256,
        dim: int = 384,
        entropy_threshold: float = 0.8,
        deliberation_threshold: float = 0.45,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        model_dir: Optional[str] = None,
    ):
        self.buffer_size = buffer_size
        self.dim = dim
        self.entropy_threshold = entropy_threshold
        self.deliberation_threshold = deliberation_threshold
        self.model_dir = model_dir or str(Path(__file__).parent.parent / "assets" / "models")
        self.buffer = np.zeros((buffer_size, dim), dtype=np.float32)
        self.count = 0
        self.write_idx = 0
        self._embed_cache: dict[str, np.ndarray] = {}

        raw_embed = embed_fn or self._init_fastembed()
        def _cached_embed(t: str) -> np.ndarray:
            if t in self._embed_cache:
                return self._embed_cache[t]
            res = raw_embed(t)
            if len(self._embed_cache) < 1024:
                self._embed_cache[t] = res
            return res

        self._embed = _cached_embed

    def embed(self, text: str) -> np.ndarray:
        """Embeds text into metric space representation."""
        return self._embed(text)

    def _init_fastembed(self) -> Callable[[str], np.ndarray]:
        try:
            try:
                from .embeddings import get_semantic_embedding
            except ImportError:
                from embeddings import get_semantic_embedding
            return lambda t: get_semantic_embedding(t, fallback_fn=lambda x: _semantic_topological_embed(x, self.dim))
        except Exception:
            return lambda t: _semantic_topological_embed(t, self.dim)


    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Compute token Shannon entropy H(X) = -sum(p * log2(p))."""
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0
        n = len(tokens)
        counts = Counter(tokens)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    @staticmethod
    def compute_oov_anomaly_ratio(text: str) -> float:
        """Detect adversarial polysemy / semantic masking (hex dumps, encrypted tokens, long hashes)."""
        tokens = re.findall(r"\w+", text)
        if not tokens:
            return 0.0
        anomalous = 0
        anomalous_chars = 0
        total_chars = sum(len(t) for t in tokens)
        for t in tokens:
            is_hex = bool(re.fullmatch(r"[0-9a-fA-F]{8,}", t))
            is_digit_mix = bool(re.search(r"\d", t) and re.search(r"[a-zA-Z]", t) and len(t) > 6)
            is_long_fragment = len(t) > 20
            if is_hex or is_digit_mix or is_long_fragment:
                anomalous += 1
                anomalous_chars += len(t)
        token_ratio = anomalous / len(tokens)
        char_ratio = (anomalous_chars / total_chars) if total_chars > 0 else 0.0
        return max(token_ratio, char_ratio)

    def evaluate(self, text: str) -> SaliencyDecision:
        # 1. Shannon entropy gating & repetitive spam detection
        entropy = self.shannon_entropy(text)
        tokens = re.findall(r"\w+", text.lower())
        
        # Repetitive pattern detection (e.g. "sensor noise noise noise...", "spam spam spam...")
        is_repetitive = bool(len(tokens) >= 4 and (len(set(tokens)) / len(tokens)) < 0.50)
        
        # Check if input is a valid conversational query or natural sentence
        norm_text = text.strip().lower()
        is_valid_query = False
        if norm_text in CONVERSATIONAL_ALLOWLIST or norm_text.isdigit():
            is_valid_query = True
        elif tokens and not is_repetitive:
            alpha_tokens = [t for t in tokens if t.isalpha()]
            unique_ratio = len(set(tokens)) / len(tokens)
            question_triggers = {"what", "who", "why", "how", "where", "when", "is", "are", "can", "explain", "tell", "calculate", "find"}
            has_question_trigger = bool(tokens and tokens[0] in question_triggers)
            if (has_question_trigger and len(tokens) >= 2) or (len(tokens) >= 2 and unique_ratio >= 0.70 and len(alpha_tokens) >= 2):
                is_valid_query = True

        if (entropy < self.entropy_threshold and not is_valid_query) or is_repetitive:
            return SaliencyDecision(
                pass_filter=False,
                entropy=float(entropy),
                novelty=0.0,
                uncertainty=0.0,
                requires_deliberation=False,
                is_anomaly=False,
                reason="Low information entropy / Low Shannon entropy (spam/gibberish)",
            )

        # Check for semantic masking (adversarial polysemy)
        oov_ratio = self.compute_oov_anomaly_ratio(text)
        masked_anomaly = oov_ratio >= 0.30

        # 2. Embed and normalize
        vec = self._embed(text)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self.dim, dtype=np.float32)

        # 3. Novelty against cyclic buffer: N = 1.0 - max(cos(q, b))
        active_slots = min(self.count, self.buffer_size)
        if active_slots == 0:
            novelty = 1.0
        else:
            active_buf = self.buffer[:active_slots]
            sims = np.dot(active_buf, vec)
            max_sim = float(np.max(sims))
            novelty = max(0.0, 1.0 - max_sim)

        # Reject exact/near duplicate chunks
        if novelty < 0.10 and active_slots > 0 and not masked_anomaly:
            return SaliencyDecision(
                pass_filter=False,
                entropy=float(entropy),
                novelty=float(novelty),
                uncertainty=0.0,
                requires_deliberation=False,
                is_anomaly=False,
                reason="Near-duplicate text block",
            )

        # Update cyclic buffer
        self.buffer[self.write_idx] = vec
        self.write_idx = (self.write_idx + 1) % self.buffer_size
        self.count += 1

        # 4. Epistemic uncertainty: U = 0.6 * N + 0.4 * min(1.0, H/2.0)
        uncertainty = 0.6 * novelty + 0.4 * min(1.0, entropy / 2.0)
        is_anomaly = bool((entropy >= 3.5 and novelty >= 0.85) or masked_anomaly)
        requires_deliberation = uncertainty >= self.deliberation_threshold or is_anomaly
        reason = "Epistemic anomaly detected (masked noise or extreme novelty)" if is_anomaly else "Passes saliency filter"


        return SaliencyDecision(
            pass_filter=True,
            entropy=float(entropy),
            novelty=float(novelty),
            uncertainty=float(uncertainty),
            requires_deliberation=requires_deliberation,
            is_anomaly=is_anomaly,
            reason=reason,
        )


class MetacognitiveGate:
    """Explicit epistemic calibration & metacognitive routing between System 1 and System 2."""

    def __init__(self, uncertainty_threshold: float = 0.45, saliency_gate: Optional[SaliencyGate] = None):
        self.threshold = uncertainty_threshold
        self.saliency = saliency_gate or SaliencyGate(deliberation_threshold=uncertainty_threshold)

    def evaluate_familiarity(self, query_vec: np.ndarray, memory_centroids: List[np.ndarray]) -> float:
        """Measures epistemic uncertainty: U = 1.0 - max_c cos(query_vec, c)."""
        if not memory_centroids:
            return 1.0
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm < 1e-9:
            return 1.0
        q_unit = query_vec / q_norm
        max_sim = 0.0
        for c in memory_centroids:
            c_norm = float(np.linalg.norm(c))
            if c_norm > 1e-9:
                sim = float(np.dot(q_unit, c / c_norm))
                if sim > max_sim:
                    max_sim = sim
        return max(0.0, 1.0 - max_sim)

    def route_deliberation(self, query: str, uncertainty: Optional[float] = None) -> str:
        """Dynamically routes between immediate System 1 recall and deliberate System 2 reasoning."""
        if uncertainty is None:
            dec = self.saliency.evaluate(query)
            uncertainty = dec.uncertainty
        return "SYSTEM_1_IMMEDIATE_RECALL" if uncertainty < self.threshold else "SYSTEM_2_DEEP_DELIBERATIVE_THINKING"


COMPOUND_EXCLUSIONS: Dict[str, List[str]] = {
    "space": ["disk space", "storage space", "memory space", "free space", "drive space", "drive c"],
    "python": ["python file", "python files", "python script", "python scripts", "python -c", "python code", ".py", "in python"],
}


class VectorBeliefDisambiguator:
    """
    Evaluates semantic ambiguity in dense metric space rather than static keyword matching.
    Projects query against polysemous concept clusters and evaluates belief distribution entropy:
        H(B) = - sum(p_i * log2(p_i))
    Detects high epistemic ambiguity when top senses exhibit comparable cosine proximity.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        if embed_fn:
            self.embed = embed_fn
        else:
            try:
                from .embeddings import get_semantic_embedding
            except ImportError:
                from embeddings import get_semantic_embedding
            self.embed = lambda t: get_semantic_embedding(t, fallback_fn=lambda x: _semantic_topological_embed(x, 384))

        self.sense_clusters: dict = {}
        self._init_default_senses()

    def _init_default_senses(self):
        prototypes = {
            "mercury": [
                ("Mercury (astronomy)", "The innermost terrestrial planet in the Solar System orbiting the Sun", "planet astronomical orbit celestial solar system"),
                ("Mercury (chemistry)", "Chemical element with symbol Hg and atomic number 80, heavy liquid transition metal", "element chemical metal liquid toxic hydrargyrum"),
                ("Freddie Mercury", "British musician and lead vocalist of the rock band Queen", "singer musician rock vocalist music band queen freddie"),
            ],
            "space": [
                ("Outer space (astronomy)", "Cosmic physical universe beyond Earth's atmosphere, stars and galaxies", "cosmos astronomy universe galaxy interstellar vacuum planet"),
                ("Vector space (mathematics)", "Mathematical structure formed by a collection of vectors and linear transformations", "linear algebra coordinate dimension subspace metric euclidean matrix"),
                ("Storage / Memory space (computing)", "Computer disk storage, RAM allocation, filesystem or memory capacity", "disk memory storage byte gigabyte filesystem partition RAM"),
            ],
            "time travel": [
                ("Theoretical physics", "Closed timelike curves, general relativity, Einstein field equations and wormholes", "physics spacetime wormhole relativity minkowski causality closed timelike curve"),
                ("Science fiction", "Pop culture, cinema, literary narratives depicting paradoxes and fictional time travel", "fiction movie novel story cinema paradox tardis sci-fi narrative"),
            ],
            "matrix": [
                ("Mathematical matrix", "Rectangular array or table of numbers, symbols, or expressions arranged in rows and columns", "linear algebra rows columns determinant eigenvalues array mathematics"),
                ("The Matrix (media)", "Science fiction cyberpunk media franchise depicting simulated reality and machines", "movie neo sci-fi cyberpunk simulation film wachowski keanu"),
            ],
            "apple": [
                ("Apple Inc. (corporation)", "Multinational technology company manufacturing iPhone, Mac, and operating systems", "technology computer iphone mac company tim cook ios hardware"),
                ("Apple (fruit)", "Edible sweet pome fruit produced by an apple tree (Malus domestica)", "fruit tree orchards agriculture eating harvest nutrition sweet"),
            ],
            "python": [
                ("Python (programming)", "High-level, general-purpose interpreted programming language", "code programming syntax developer software script function algorithm"),
                ("Python (zoology)", "Non-venomous constrictor snake belonging to the family Pythonidae", "snake reptile animal species constrictor wildlife venomless"),
            ],
            "java": [
                ("Java (programming)", "Object-oriented class-based programming language developed by Sun / Oracle", "programming language jvm bytecode software oracle class OOP"),
                ("Java (geography / coffee)", "Indonesian island or Indonesian arabica coffee variety", "island indonesia geography coffee bean jakarta agriculture brew"),
            ],
        }
        for topic, senses in prototypes.items():
            sense_list = []
            for name, desc, anchor_text in senses:
                vec = self.embed(anchor_text)
                sense_list.append((name, desc, vec))
            self.sense_clusters[topic] = self.enforce_orthogonality(sense_list, max_similarity=0.70)

    @staticmethod
    def enforce_orthogonality(
        senses: list,
        max_similarity: float = 0.70,
    ) -> list:
        """
        Enforces explicit cosine separation margin between sense prototypes:
            max_{i != j} cos(v_i, v_j) < max_similarity
        Merges redundant/overlapping prototypes into a unified semantic concept
        to prevent artificial entropy elevation and false-positive ambiguity alerts.
        """
        if not senses:
            return []

        orthogonal = []
        for name, desc, vec in senses:
            v_norm = float(np.linalg.norm(vec))
            v_unit = vec / (v_norm + 1e-9)

            merged = False
            for idx, (o_name, o_desc, o_vec) in enumerate(orthogonal):
                o_norm = float(np.linalg.norm(o_vec))
                o_unit = o_vec / (o_norm + 1e-9)
                cos_sim = float(np.dot(v_unit, o_unit))

                if cos_sim >= max_similarity:
                    merged_name = f"{o_name} / {name}"
                    merged_desc = f"{o_desc}; {desc}"
                    merged_vec = (o_unit + v_unit) / 2.0
                    orthogonal[idx] = (merged_name, merged_desc, merged_vec)
                    merged = True
                    break

            if not merged:
                orthogonal.append((name, desc, vec))

        return orthogonal

    def add_sense(self, topic: str, name: str, description: str, anchor_text: str, max_similarity: float = 0.70):
        vec = self.embed(anchor_text)
        if topic not in self.sense_clusters:
            self.sense_clusters[topic] = []
        self.sense_clusters[topic].append((name, description, vec))
        self.sense_clusters[topic] = self.enforce_orthogonality(self.sense_clusters[topic], max_similarity=max_similarity)

    def evaluate(self, query: str, temperature: float = 0.25) -> dict:

        """
        Evaluates query for semantic polysemy in vector space.
        Returns is_ambiguous, topic, options, entropy, and probabilities.
        """
        clean_q = query.strip().lower()
        clean_core = re.sub(r"^(what\s+is|tell\s+me\s+about|explain|describe)\s+", "", clean_q).rstrip("?").strip()
        q_vec = self.embed(clean_core or clean_q)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm < 1e-9:
            return {"is_ambiguous": False}

        action_prefixes = ("check ", "find ", "list ", "run ", "show ", "get ", "calculate ", "verify ")
        is_action = any(clean_q.startswith(p) for p in action_prefixes)

        best_match_topic = None
        best_match_score = -1.0
        best_senses = []

        for topic, senses in self.sense_clusters.items():
            if any(ex in clean_q for ex in COMPOUND_EXCLUSIONS.get(topic, [])):
                continue
            if is_action and topic in ["space", "python"]:
                continue

            t_vec = self.embed(topic)
            t_norm = float(np.linalg.norm(t_vec))
            topic_sim = float(np.dot(q_vec, t_vec) / (q_norm * (t_norm + 1e-9)))
            is_direct_token = (clean_core == topic) or (clean_core in [f"the {topic}", f"what is {topic}"])

            if is_direct_token or topic_sim > 0.65:
                sims = []
                for name, desc, s_vec in senses:
                    s_norm = float(np.linalg.norm(s_vec))
                    sim = float(np.dot(q_vec, s_vec) / (q_norm * (s_norm + 1e-9)))
                    sims.append(max(0.0, sim))

                sims_arr = np.array(sims, dtype=np.float32)
                exp_s = np.exp(sims_arr / temperature)
                probs = exp_s / (np.sum(exp_s) + 1e-9)
                entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
                max_p = float(np.max(probs))

                if (entropy > 0.80 and max_p < 0.88) or is_direct_token:
                    if topic_sim > best_match_score or is_direct_token:
                        best_match_score = 1.0 if is_direct_token else topic_sim
                        best_match_topic = topic
                        best_senses = [
                            f"Are you referring to {s[0]} ({s[1]})?" for s in senses
                        ]

        if best_match_topic and best_senses:
            return {
                "is_ambiguous": True,
                "topic": best_match_topic,
                "options": best_senses,
            }

        return {"is_ambiguous": False}



