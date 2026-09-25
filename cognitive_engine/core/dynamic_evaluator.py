from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CandidateAction:
    action_id: str
    target_code_file: str
    proposed_patch: str  # Python code to inject or alter
    variables: Dict[str, Any]  # Environmental inputs & parameters
    impact_weight: float  # Potential gain (0.0 to 1.0)
    risk_variance: float  # Risk uncertainty (0.0 to 1.0)
    irreversibility: float  # Cost of failure (0.0 to 1.0)
    target_function: str = "execute_task"


class FastRiskAnalyzer:
    """Multi-variable risk-adjusted utility analyzer for N candidate options."""

    def __init__(self, risk_aversion: float = 1.6, approval_threshold: float = 0.20):
        self.risk_aversion = risk_aversion
        self.approval_threshold = approval_threshold
        self.outcome_history: List[bool] = []

    def adapt_risk_aversion(self, success: bool, alpha: float = 0.05, target_margin: float = 0.70) -> float:
        """Dynamically scales lambda based on empirical regression/success drift."""
        self.outcome_history.append(success)
        if len(self.outcome_history) > 50:
            self.outcome_history.pop(0)
        recent_failures = sum(1 for s in self.outcome_history if not s) / len(self.outcome_history)
        drift = recent_failures - (1.0 - target_margin)
        self.risk_aversion = round(max(0.5, min(4.0, self.risk_aversion * (1.0 + alpha * drift))), 4)
        return self.risk_aversion

    def evaluate_and_rank(self, options: List[CandidateAction]) -> List[CandidateAction]:
        """Calculates expected score across all N options and selects optimal paths."""
        scored = []
        for opt in options:
            utility = opt.impact_weight
            risk = opt.risk_variance * (1.0 + opt.irreversibility)
            net_score = utility - (self.risk_aversion * risk)
            if net_score >= self.approval_threshold:
                scored.append((net_score, opt))

        # Sort descending: highest analytical value first
        scored.sort(key=lambda x: x[0], reverse=True)
        return [opt for _, opt in scored]
