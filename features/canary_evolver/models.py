from dataclasses import dataclass

@dataclass
class KernelPatchAction:
    action_id: str
    target_filepath: str
    proposed_ast_str: str
    impact_weight: float       # Expected Utility ΔP
    risk_variance: float       # Operational Risk
    irreversibility: float     # System blast radius
