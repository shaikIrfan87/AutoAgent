from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class RubricCriteria(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str


class VerificationResult(BaseModel):
    is_valid: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    failed_assertions: List[str] = Field(default_factory=list)
    execution_logs: Optional[str] = None


class CritiquePayload(BaseModel):
    criterion_scores: Dict[str, float] = Field(default_factory=dict)
    aggregate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    detected_flaws: List[str] = Field(default_factory=list)
    remediation_hints: List[str] = Field(default_factory=list)


class Trajectory(BaseModel):
    id: str
    reasoning_trace: str
    terminal_output: str
    verification: Optional[VerificationResult] = None
    critique: Optional[CritiquePayload] = None
    token_entropy: float = 0.0


class GraphTrajectory(Trajectory):
    active_node_ids: List[str] = Field(default_factory=list)
    traversed_edges: List[Tuple[str, str]] = Field(default_factory=list)
    dynamic_depth_reached: int = 0
    activation_energies: Dict[str, float] = Field(default_factory=dict)
