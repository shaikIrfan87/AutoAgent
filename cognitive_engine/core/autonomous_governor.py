import copy
import time
from typing import Any, Callable, Dict, Optional
from .autonomous_decision import DecisionEvaluation, MutationProposal, RiskScoreGovernor
from .meta_optimizer import MetaSelfOptimizer


class AutonomousResult(dict):
    """Dictionary supporting attribute access for backwards compatibility."""
    def __getattr__(self, name: str) -> Any:
        return self.get(name)


class AutonomousGovernor:
    """Autonomous executive: Evaluates proposals, runs canary validations, and applies self-updates."""

    def __init__(
        self,
        target_engine: Any,
        governor: Optional[RiskScoreGovernor] = None,
        optimizer: Optional[MetaSelfOptimizer] = None,
    ):
        self.engine = target_engine
        self.governor = governor or RiskScoreGovernor()
        self.optimizer = optimizer or getattr(target_engine, "meta_optimizer", None) or MetaSelfOptimizer()
        self.audit_log = []

    def process_and_execute(
        self,
        proposal: MutationProposal,
        canary_validator: Callable[[Any], bool],
    ) -> AutonomousResult:
        # Step 1: Autonomous decision based on risk/score calculation
        decision: DecisionEvaluation = self.governor.evaluate(proposal)

        if not decision.approved:
            record = AutonomousResult({
                "status": "rejected",
                "applied": False,
                "decision": decision,
                "timestamp": time.time(),
            })
            self.audit_log.append(record)
            return record

        # Step 2: Attempt canary run in isolation before modifying live system
        target_obj = getattr(self.engine, proposal.target_subsystem, None)
        if target_obj is None and hasattr(self.engine, "autotelic"):
            target_obj = getattr(self.engine.autotelic, proposal.target_subsystem, None)

        if target_obj is None:
            return AutonomousResult({"status": "error", "applied": False, "message": f"Subsystem {proposal.target_subsystem} not found"})

        # Step 3: Run isolated hot-swap or parameter update
        if proposal.patch_code:
            res = self.optimizer.test_and_hotswap(
                target_obj=target_obj,
                attr_name="update_fn",
                new_source=proposal.patch_code,
                fn_name="get_search_depth" if "get_search_depth" in proposal.patch_code else "run_update",
                custom_canary=canary_validator,
            )
            applied = res.applied
        else:
            # Atomic parameter update with rollback state saved
            original_state = {k: getattr(target_obj, k) for k in proposal.parameters_delta}
            try:
                for k, v in proposal.parameters_delta.items():
                    setattr(target_obj, k, v)

                # Verify stability under the new parameter
                applied = canary_validator(target_obj)
                if not applied:
                    # Rollback if canary test fails
                    for k, v in original_state.items():
                        setattr(target_obj, k, v)
            except Exception:
                for k, v in original_state.items():
                    setattr(target_obj, k, v)
                applied = False

        record = AutonomousResult({
            "status": "applied" if applied else "canary_failed_rolled_back",
            "applied": applied,
            "decision": decision,
            "timestamp": time.time(),
        })
        self.audit_log.append(record)
        return record
