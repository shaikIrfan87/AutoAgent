from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass
class MutationProposal:
    proposal_id: str
    target_subsystem: str  # e.g., "synthesizer", "causal_graph", "saliency"
    description: str
    patch_code: Optional[str]  # Executable patch or None for heuristic parameter updates
    parameters_delta: Dict[str, Any]

    # Risk and utility priors (0.0 to 1.0)
    expected_speedup: float = 0.0
    expected_accuracy_gain: float = 0.0
    failure_probability: float = 0.1
    reversibility_score: float = 1.0  # 1.0 = fully reversible memory update; 0.1 = risky code change


@dataclass
class DecisionEvaluation:
    proposal_id: str
    net_score: float
    risk_score: float
    utility_score: float
    approved: bool
    rationale: str


class RiskScoreGovernor:
    """Calculates risk-adjusted utility for autonomous actions."""

    def __init__(self, risk_aversion: float = 1.5, approval_threshold: float = 0.25):
        self.risk_aversion = risk_aversion
        self.approval_threshold = approval_threshold

    def evaluate(self, proposal: MutationProposal) -> DecisionEvaluation:
        # 1. Compute expected utility
        utility = (0.6 * proposal.expected_accuracy_gain) + (0.4 * proposal.expected_speedup)

        # 2. Compute risk: failure rate scaled by irreversibility
        irreversibility = 1.0 - proposal.reversibility_score
        risk = proposal.failure_probability * (1.0 + irreversibility)

        # 3. Compute net risk-adjusted decision score
        net_score = utility - (self.risk_aversion * risk)
        approved = net_score >= self.approval_threshold

        rationale = (
            f"Net score {net_score:.3f} >= {self.approval_threshold} threshold. Proceed to canary."
            if approved
            else f"Rejected: Risk ({risk:.3f}) exceeds utility ({utility:.3f})."
        )

        return DecisionEvaluation(
            proposal_id=proposal.proposal_id,
            net_score=net_score,
            risk_score=risk,
            utility_score=utility,
            approved=approved,
            rationale=rationale,
        )
