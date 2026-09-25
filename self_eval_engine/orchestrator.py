import asyncio
from typing import Any, List, Literal, Optional
from .actor import GraphActor
from .consensus import ConsensusEngine
from .credit import GraphCreditAssigner
from .schemas import CritiquePayload, GraphTrajectory, RubricCriteria
from .verifiers.deterministic import PythonExecutionSandbox
from .verifiers.neural_judge import DisentangledJudge


class SelfEvaluationOrchestrator:
    def __init__(
        self,
        actor_client: Any = None,
        judge_client: Any = None,
        rubrics: Optional[List[RubricCriteria]] = None,
        graph_memory: Any = None,
        mode: Literal["shadow", "read_only", "closed_loop"] = "closed_loop",
        max_iterations: int = 3,
        k_samples: int = 5,
        entropy_threshold: float = 0.5,
        confidence_threshold: float = 0.70,
        db_path: Optional[str] = None,
    ):
        self.actor = GraphActor(actor_client)
        self.judge = DisentangledJudge(judge_client, rubrics or [])
        self.sandbox = PythonExecutionSandbox()
        self.credit_assigner = (
            GraphCreditAssigner(graph_memory, db_path=db_path) if graph_memory else None
        )
        self.mode = mode
        self.max_iterations = max_iterations
        self.k_samples = k_samples
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold

    async def run_async(
        self,
        task: str,
        deterministic_unit_test: str = "",
        ctx: Optional[Any] = None,
    ) -> GraphTrajectory:
        tenant_id = getattr(ctx, "tenant_id", "default_tenant") if ctx else "default_tenant"
        feedback: List[str] = []
        best_candidate: Optional[GraphTrajectory] = None

        for _ in range(self.max_iterations):
            # 1. Speculative Execution: start greedy (k=1)
            greedy_cand = self.actor.generate_candidate(task, feedback, greedy=True)
            v_res = self.sandbox.verify_code(
                greedy_cand.terminal_output, deterministic_unit_test
            )
            greedy_cand.verification = v_res

            # 2. Asynchronous Short-Circuiting:
            # If deterministic checks fail, bypass expensive judge call immediately
            if not v_res.is_valid:
                feedback = [
                    f"Deterministic Error: {err}" for err in v_res.failed_assertions
                ]
                best_candidate = greedy_cand
                # Apply negative feedback if failing
                self._apply_credit(
                    best_candidate,
                    CritiquePayload(aggregate_score=0.0, detected_flaws=v_res.failed_assertions),
                    passed=False,
                    tenant_id=tenant_id,
                )
                continue

            # 3. Entropy-Gated Speculative Execution
            if greedy_cand.token_entropy <= self.entropy_threshold:
                best_candidate = greedy_cand
                critique = await self.judge.evaluate_async(task, best_candidate)
                best_candidate.critique = critique
                passed = critique.aggregate_score >= 0.85
                self._apply_credit(best_candidate, critique, passed, tenant_id=tenant_id)
                if passed or self.mode == "shadow":
                    return best_candidate
            else:
                # Spawn k stochastic beam
                candidates = [greedy_cand] + self.actor.generate_candidates(
                    task, feedback, k=self.k_samples - 1
                )
                valid_candidates = []
                for c in candidates:
                    check = (
                        c.verification
                        or self.sandbox.verify_code(
                            c.terminal_output, deterministic_unit_test
                        )
                    )
                    c.verification = check
                    if check.is_valid:
                        valid_candidates.append(c)

                pool = valid_candidates if valid_candidates else candidates
                best_candidate, consensus_ratio = ConsensusEngine.resolve_majority(pool)

                critique = await self.judge.evaluate_async(task, best_candidate)
                best_candidate.critique = critique
                passed = (consensus_ratio >= self.confidence_threshold) and (
                    critique.aggregate_score >= 0.85
                )
                self._apply_credit(best_candidate, critique, passed, tenant_id=tenant_id)
                if passed or self.mode == "shadow":
                    return best_candidate

            # 4. Zimmerman Attribution Feedback for next iteration
            feedback = self._build_attribution_feedback(
                best_candidate.critique, best_candidate
            )

        return best_candidate or greedy_cand

    def run(
        self,
        task: str,
        deterministic_unit_test: str = "",
        ctx: Optional[Any] = None,
    ) -> GraphTrajectory:
        return asyncio.run(self.run_async(task, deterministic_unit_test, ctx=ctx))

    def _apply_credit(
        self,
        candidate: GraphTrajectory,
        critique: CritiquePayload,
        passed: bool,
        tenant_id: str = "default_tenant",
    ) -> None:
        if self.mode == "closed_loop" and self.credit_assigner:
            self.credit_assigner.apply_feedback(candidate, critique, passed, tenant_id=tenant_id)

    def _build_attribution_feedback(
        self, critique: Optional[CritiquePayload], candidate: GraphTrajectory
    ) -> List[str]:
        if not critique:
            return []
        hints = [
            f"Flaw: {f} -> Hint: {h}"
            for f, h in zip(critique.detected_flaws, critique.remediation_hints)
        ]
        if candidate.verification and not candidate.verification.is_valid:
            hints.extend(candidate.verification.failed_assertions)
        return hints
