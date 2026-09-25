from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict


class SaliencyDecision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pass_filter: bool
    entropy: float
    novelty: float
    uncertainty: float
    requires_deliberation: bool
    is_anomaly: bool = False
    reason: str



class Triple(BaseModel):
    subject: str
    relation: str  # e.g. "is_a", "causes", "part_of", "cannot_be", "mutually_exclusive", "leads_to"
    target: str
    polarity: bool = True  # True = Affirmative rule; False = Prohibited state


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    delta_s: float  # +1.0 (success), -1.0 (error/timeout), 0.0 (no effect)
    duration_sec: float = 0.0


class ExecutionContext(BaseModel):
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"


class MemoryRecord(BaseModel):
    id: str
    content: str
    vector: Optional[list[float]] = None
    confidence: float = 0.8
    access_count: int = 1
    last_accessed: float = 0.0
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"


class UnresolvedGroundingFailure(BaseModel):
    goal: str
    error_trace: str
    iterations: int
    diagnostic: str


class InterventionRecord(BaseModel):
    source: str
    action: str
    target: str
    delta_s: float  # Outcome observed under interventional do(action)
    metrics: dict[str, float] = {}
    timestamp: float = 0.0


