"""
Features Root Package: Domain-Driven Cognitive & Security Invariant Features.
"""
from features.protocol_guard import TypeSafeProtocolGuard, IngressDecision
from features.epistemic_gating import EpistemicGatingEvaluator
from features.causal_reasoning import CausalAxiomVerifier
from features.execution_sandbox import IsolatedSandboxExecutor
from features.synaptic_plasticity import PlasticFastWeightCell
from features.memory_store import DualTierWalMemoryStore
from features.canary_evolver import SelfEvolvingKernelTCB, KernelPatchAction
from features.orchestrator import UnifiedCognitiveEngine, CognitiveCycleResult

__all__ = [
    "TypeSafeProtocolGuard",
    "IngressDecision",
    "EpistemicGatingEvaluator",
    "CausalAxiomVerifier",
    "IsolatedSandboxExecutor",
    "PlasticFastWeightCell",
    "DualTierWalMemoryStore",
    "SelfEvolvingKernelTCB",
    "KernelPatchAction",
    "UnifiedCognitiveEngine",
    "CognitiveCycleResult",
]
