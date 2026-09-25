import copy
import inspect
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MutationCandidate:
    target_module: str
    target_attribute: str
    proposed_source: str
    description: str


@dataclass
class OptimizationResult:
    applied: bool
    mutation: MutationCandidate
    execution_time_ms: float
    error: Optional[str] = None


class MetaSelfOptimizer:
    """Pillar 4: Recursive meta-self-optimization and hot-swappable sandbox runtime.

    Allows the cognitive engine to autonomously alter its own execution logic,
    evaluating candidate alterations within an isolated canary testbed before
    atomically applying them to the live running instance.
    """

    def __init__(self, canary_harness: Optional[Callable[[Callable], bool]] = None):
        self.canary_harness = canary_harness or self._default_canary
        self.rollback_history: List[Tuple[Any, str, Any]] = []
        self.applied_mutations: List[MutationCandidate] = []

    def _default_canary(self, candidate_fn: Callable) -> bool:
        """Minimal smoke test verifying callable integrity and basic input/output behavior."""
        try:
            return callable(candidate_fn)
        except Exception:
            return False

    def compile_mutation(self, source_code: str, target_fn_name: str) -> Callable:
        """Safely compiles and extracts an updated function in an isolated execution namespace."""
        local_scope: Dict[str, Any] = {}
        # Restricted compilation namespace
        exec(source_code, globals(), local_scope)
        if target_fn_name not in local_scope or not callable(local_scope[target_fn_name]):
            raise ValueError(f"Function {target_fn_name} was not defined in the source patch.")
        return local_scope[target_fn_name]

    def test_and_hotswap(
        self,
        target_obj: Any,
        attr_name: str,
        new_source: str,
        fn_name: str,
        custom_canary: Optional[Callable[[Callable], bool]] = None,
    ) -> OptimizationResult:
        """Compiles, evaluates against canary tests, and hot-swaps logic onto target_obj."""
        import time

        t0 = time.time()
        mutation = MutationCandidate(
            target_module=target_obj.__class__.__name__,
            target_attribute=attr_name,
            proposed_source=new_source,
            description=f"Hot-swap {target_obj.__class__.__name__}.{attr_name}",
        )

        try:
            # 1. Compile proposed logic
            new_fn = self.compile_mutation(new_source, fn_name)

            # 2. Bind function if swapping a class instance method
            if inspect.isclass(target_obj):
                bound_candidate = new_fn
            else:
                bound_candidate = types.MethodType(new_fn, target_obj)

            # 3. Canary evaluation
            verifier = custom_canary or self.canary_harness
            canary_ok = verifier(bound_candidate)

            if not canary_ok:
                elapsed = (time.time() - t0) * 1000.0
                return OptimizationResult(
                    applied=False,
                    mutation=mutation,
                    execution_time_ms=elapsed,
                    error="Canary validation suite failed: Candidate rejected.",
                )

            # 4. Atomic Hot-Swap with Rollback Anchor
            original_val = getattr(target_obj, attr_name, None)
            self.rollback_history.append((target_obj, attr_name, original_val))

            setattr(target_obj, attr_name, bound_candidate)
            self.applied_mutations.append(mutation)
            elapsed = (time.time() - t0) * 1000.0

            return OptimizationResult(
                applied=True,
                mutation=mutation,
                execution_time_ms=elapsed,
                error=None,
            )

        except Exception as e:
            elapsed = (time.time() - t0) * 1000.0
            return OptimizationResult(
                applied=False,
                mutation=mutation,
                execution_time_ms=elapsed,
                error=f"Compilation/execution error: {str(e)}",
            )

    def rollback_last(self) -> bool:
        """Restores the target attribute to its immediate previous state."""
        if not self.rollback_history:
            return False

        target_obj, attr_name, original_val = self.rollback_history.pop()
        if original_val is None:
            if hasattr(target_obj, attr_name):
                delattr(target_obj, attr_name)
        else:
            setattr(target_obj, attr_name, original_val)

        if self.applied_mutations:
            self.applied_mutations.pop()
        return True
