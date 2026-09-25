from dataclasses import dataclass
from typing import Literal, Dict, Any

@dataclass(frozen=True)
class IngressDecision:
    choice: Literal["safe_query", "adversarial_exploit", "compute_heavy", "tool_dispatch"]
    risk_score: float  # Calibrated [0.0 - 100.0]
    is_compliant_noul: bool  # Calibrated boolean probability >= 0.95
    sanitized_payload: Dict[str, Any]
    noul_score: float = 1.0  # Calibrated non-generative compliance score [0.0, 1.0]

    @property
    def noul_compliance(self) -> bool:
        return self.is_compliant_noul

