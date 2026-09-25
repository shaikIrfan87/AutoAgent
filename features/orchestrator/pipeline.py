from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from features.protocol_guard import TypeSafeProtocolGuard, IngressDecision
from features.epistemic_gating import EpistemicGatingEvaluator
from features.causal_reasoning import CausalAxiomVerifier
from features.execution_sandbox import IsolatedSandboxExecutor
from features.synaptic_plasticity import PlasticFastWeightCell
from features.memory_store import DualTierWalMemoryStore
from features.canary_evolver import SelfEvolvingKernelTCB

@dataclass
class CognitiveCycleResult:
    ingress_decision: IngressDecision
    saliency_passed: bool
    uncertainty: float
    causal_safe: bool
    execution_delta_s: float
    execution_message: str
    plastic_frobenius_norm: float
    memory_consolidated: bool
    egress_verified_payload: Optional[Dict[str, Any]] = None
    halted_stage: Optional[str] = None
    ttt_frobenius_norm: float = 0.0

class UnifiedCognitiveEngine:
    """
    Unified Orchestrator combining all feature modules into a single, cohesive
    runtime pipeline across the Hardware TEE / seL4 Microkernel boundary.
    """
    def __init__(
        self,
        db_path: str = ":memory:",
        dim: int = 64,
        risk_cutoff: float = 35.0,
        risk_aversion: float = 1.6,
    ):
        self.guard = TypeSafeProtocolGuard(risk_cutoff=risk_cutoff)
        self.epistemic = EpistemicGatingEvaluator()
        self.causal = CausalAxiomVerifier()
        self.sandbox = IsolatedSandboxExecutor()
        self.plasticity = PlasticFastWeightCell(dim=dim)
        self.memory = DualTierWalMemoryStore(db_path=db_path)
        self.evolver = SelfEvolvingKernelTCB(risk_aversion=risk_aversion)

        # Attention-Coupled TTT & Continuous SCM Modules
        try:
            from cognitive_engine.core.ttt_attention import TTTAttentionLayer
            from cognitive_engine.core.scm_engine import ContinuousSCMEngine
            self.ttt_attention: Optional[TTTAttentionLayer] = TTTAttentionLayer(d_model=dim, num_heads=4)
            self.scm: Optional[ContinuousSCMEngine] = ContinuousSCMEngine(dim=min(dim, 8))
        except Exception:
            self.ttt_attention = None
            self.scm = None


    def process_cycle(
        self,
        query: str,
        code: Optional[str] = None,
        causal_check: Optional[Tuple[str, str, str]] = None,
        k_vec: Optional[np.ndarray] = None,
        v_vec: Optional[np.ndarray] = None,
        egress_schema: Optional[str] = None,
        egress_payload: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> CognitiveCycleResult:
        """
        Executes an end-to-end cognitive cycle through all feature modules in sequence.
        """
        # 1. Ingress Diode Protocol Guard
        ingress = self.guard.evaluate_ingress(query)
        if not ingress.is_compliant_noul:
            return CognitiveCycleResult(
                ingress_decision=ingress,
                saliency_passed=False,
                uncertainty=1.0,
                causal_safe=False,
                execution_delta_s=0.0,
                execution_message="Halted: Non-compliant ingress",
                plastic_frobenius_norm=self.plasticity.frobenius_norm,
                memory_consolidated=False,
                halted_stage="protocol_guard_ingress"
            )

        # 2. Epistemic Gating & Saliency (Invariant 1)
        passed_saliency, uncertainty, saliency_msg = self.epistemic.evaluate_saliency(query)
        if not passed_saliency:
            return CognitiveCycleResult(
                ingress_decision=ingress,
                saliency_passed=False,
                uncertainty=uncertainty,
                causal_safe=False,
                execution_delta_s=0.0,
                execution_message=f"Halted: Epistemic gating failure ({saliency_msg})",
                plastic_frobenius_norm=self.plasticity.frobenius_norm,
                memory_consolidated=False,
                halted_stage="epistemic_gating"
            )

        # 3. Causal & Counterfactual Verification (Invariant 2)
        causal_safe = True
        if causal_check:
            subj, rel, tgt = causal_check
            causal_safe = self.causal.verify_causal_safety(subj, rel, tgt)
            if not causal_safe:
                return CognitiveCycleResult(
                    ingress_decision=ingress,
                    saliency_passed=True,
                    uncertainty=uncertainty,
                    causal_safe=False,
                    execution_delta_s=0.0,
                    execution_message="Halted: Causal axiom contradiction detected",
                    plastic_frobenius_norm=self.plasticity.frobenius_norm,
                    memory_consolidated=False,
                    halted_stage="causal_verification"
                )

        # 4. Grounded Sandboxed Execution (Invariant 3)
        delta_s = 0.0
        exec_msg = "No code executed"
        if code:
            delta_s, exec_msg = self.sandbox.execute(code)

        # 5. Fast Plastic Weights & Attention-Coupled TTT Adaptation (Invariant 4)
        ttt_norm = 0.0
        if k_vec is not None and v_vec is not None:
            self.plasticity.adapt(k_vec, v_vec, delta_s=delta_s)
            if self.ttt_attention is not None:
                import torch
                k_t = torch.from_numpy(k_vec).float()
                v_t = torch.from_numpy(v_vec).float()
                ttt_norm = self.ttt_attention.adapt_step(k_t, v_t)

        # 6. Memory Store Consolidation (Invariant 5)
        mem_consolidated = False
        if delta_s > 0.0 and memory_id and egress_payload:
            self.memory.commit_longterm(memory_id, str(egress_payload.get("parameters", {})))
            mem_consolidated = True

        # 7. Egress Diode Contract Enforcement
        verified_egress = None
        if egress_payload and egress_schema:
            verified_egress = self.guard.enforce_egress_contract(egress_payload, egress_schema)

        return CognitiveCycleResult(
            ingress_decision=ingress,
            saliency_passed=True,
            uncertainty=uncertainty,
            causal_safe=causal_safe,
            execution_delta_s=delta_s,
            execution_message=exec_msg,
            plastic_frobenius_norm=self.plasticity.frobenius_norm,
            memory_consolidated=mem_consolidated,
            egress_verified_payload=verified_egress,
            halted_stage=None,
            ttt_frobenius_norm=ttt_norm
        )

