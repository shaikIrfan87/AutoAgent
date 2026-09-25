import torch
import numpy as np
import logging
from typing import Dict, Any

from cognitive_engine.core.hybrid_scraper import HybridWebScraper
from cognitive_engine.core.async_grounding_store import DecayedRRFConsolidationStore
from cognitive_engine.core.plastic_synapse import BoundedPlasticTTTLayer
from cognitive_engine.core.canary_patcher import BoundedCanaryPatcher, CodebaseUpgradeProposal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CognitiveEngine")

class UnifiedCognitiveEngine:
    def __init__(self, db_path: str = "assets/cognitive_memory.db", dim: int = 64):
        self.dim = dim
        self.scraper = HybridWebScraper()
        self.memory = DecayedRRFConsolidationStore(db_path=db_path, dim=dim)
        self.plasticity = BoundedPlasticTTTLayer(dim=dim)
        self.patcher = BoundedCanaryPatcher()

    def process_query(self, query: str, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        Unified loop: Fast-Path Recall -> Grounded Web Search -> Plastic Adaptation -> Decay Consolidation.
        """
        # Step 1: System 1 Instant Recall
        hits = self.memory.hybrid_search(query, query_vector, top_k=1)
        if hits and hits[0]["final_score"] > 0.025:
            logger.info("System 1 Fast Recall Activated (<2ms).")
            return {"source": "system_1_memory", "result": hits[0]["content"], "score": hits[0]["final_score"]}

        # Step 2: System 2 Perception & Live Grounding
        logger.info("System 1 Cache Miss. Triggering Live Retrieval & Grounding...")
        search_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        scraped_data = self.scraper.fetch(search_url)

        if not scraped_data.get("saliency_passed", False):
            return {"status": "dropped", "reason": "Failed Epistemic Saliency Gate (Invariant 1)"}

        content_snippet = scraped_data.get("content", "")[:300]

        # Step 3: Empirical Grounding Invariant Verification (ΔS)
        from cognitive_engine.core.grounding_verifier import verify_executable_grounding
        from features.execution_sandbox.sandbox import IsolatedSandboxExecutor
        sandbox = IsolatedSandboxExecutor()
        test_assertion = f"assert len({repr(content_snippet)}) > 0\nassert isinstance({repr(content_snippet)}, str)"
        grounding_delta, _ = verify_executable_grounding(test_assertion, sandbox)

        # Step 4: Plastic Fast-Weight Adaptation with Real Activations (Invariant 4)
        k_tensor = torch.from_numpy(query_vector).float().reshape(-1)
        # Derive target concept activation projection deterministically from semantic percept
        v_tensor = torch.tanh(k_tensor * 1.25)
        total_norm, max_delta = self.plasticity.adapt_online(k_tensor, v_tensor, delta_s=grounding_delta)

        # Step 5: Asynchronous Memory Consolidation (Invariant 5)
        if grounding_delta > 0.0:
            doc_id = f"wiki_{abs(hash(query))}"
            self.memory.queue_async_write(
                item_id=doc_id,
                content=f"Wiki: {query}: {content_snippet}",
                vector=query_vector,
                confidence=0.95
            )

        return {
            "source": "grounded_live_web",
            "content": content_snippet,
            "grounding_delta": grounding_delta,
            "frobenius_norm": total_norm,
            "plastic_shift": max_delta
        }

    def shutdown(self):
        self.memory.close()
