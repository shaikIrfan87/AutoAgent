from typing import Literal, Union, Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError


class PhysicsGoalPayload(BaseModel):
    intent_type: Literal["physics_calculation"] = "physics_calculation"
    quantity: str = Field(description="Target quantity, e.g., kinetic_energy")
    parameters: Dict[str, float] = Field(description="Normalized SI units: mass_kg, velocity_mps, etc.")


class AlgorithmicGoalPayload(BaseModel):
    intent_type: Literal["algorithmic_synthesis"] = "algorithmic_synthesis"
    target_fn: str
    io_examples: List[Dict[str, Any]]


class AmbiguityResolutionPayload(BaseModel):
    intent_type: Literal["ambiguity_resolution"] = "ambiguity_resolution"
    clarification_target: str
    candidate_interpretations: List[str]


AgentGoal = Union[PhysicsGoalPayload, AlgorithmicGoalPayload, AmbiguityResolutionPayload]


def route_intent(query: str, episodic_store: Optional[Any] = None) -> AgentGoal:
    """
    Schema-constrained intent router.
    Routes queries into typed Pydantic payloads (AgentGoal) without brittle regex matching.
    If parsing fails validation or intent is ambiguous, routes directly to AmbiguityResolutionPayload.
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # 1. Physics Calculation Ingress
    if any(k in q_lower for k in ("kinetic energy", "velocity", "mass", "joules", "acceleration", "force", "momentum")):
        # Semantic extraction of parameters
        params: Dict[str, float] = {}
        # Parse tokens for numeric values and associated units without hardcoded regex traps
        words = q_lower.replace(",", " ").split()
        for idx, w in enumerate(words):
            try:
                num = float(w)
                if idx + 1 < len(words):
                    next_w = words[idx + 1]
                    if "kg" in next_w or "kilo" in next_w or "mass" in next_w:
                        params["mass_kg"] = num
                    elif "m/s" in next_w or "mps" in next_w or "speed" in next_w or "vel" in next_w:
                        params["velocity_mps"] = num
                    elif "n" in next_w or "newton" in next_w or "force" in next_w:
                        params["force_n"] = num
            except ValueError:
                continue

        quantity = "kinetic_energy" if "kinetic" in q_lower or "energy" in q_lower else "physical_quantity"
        try:
            return PhysicsGoalPayload(
                intent_type="physics_calculation",
                quantity=quantity,
                parameters=params
            )
        except ValidationError:
            pass

    # 2. Algorithmic Synthesis Ingress
    if any(k in q_lower for k in ("synthesize", "implement", "algorithm", "function", "io_examples", "i/o")):
        target_fn = "target_func"
        for word in q_clean.split():
            if "(" in word or word.endswith("_fn"):
                target_fn = word.split("(")[0]
                break
        try:
            return AlgorithmicGoalPayload(
                intent_type="algorithmic_synthesis",
                target_fn=target_fn,
                io_examples=[]
            )
        except ValidationError:
            pass

    # 3. Episodic Disambiguation Query & Ambiguity Resolution
    candidates = []
    if episodic_store is not None:
        try:
            # Query historical disambiguation traces from episodic memory
            traces = episodic_store.hybrid_search(f"disambiguation: {q_clean}", top_k=3)
            for t in traces:
                if "content" in t and t["content"] not in candidates:
                    candidates.append(t["content"])
        except Exception:
            pass

    if not candidates:
        candidates = [
            f"Interpret '{q_clean}' as domain-specific scientific calculation",
            f"Interpret '{q_clean}' as general factual research inquiry"
        ]

    return AmbiguityResolutionPayload(
        intent_type="ambiguity_resolution",
        clarification_target=q_clean,
        candidate_interpretations=candidates
    )
