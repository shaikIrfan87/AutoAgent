from typing import Any, List
from ..schemas import CritiquePayload, RubricCriteria, Trajectory


class DisentangledJudge:
    def __init__(self, model_client: Any = None, rubrics: List[RubricCriteria] = None):
        self.client = model_client
        self.rubrics = rubrics or []

    async def evaluate_async(self, task: str, candidate: Trajectory) -> CritiquePayload:
        prompt = self._construct_judge_prompt(task, candidate)
        if hasattr(self.client, "generate_structured_async"):
            return await self.client.generate_structured_async(
                prompt, response_model=CritiquePayload
            )
        return self.evaluate(task, candidate)

    def evaluate(self, task: str, candidate: Trajectory) -> CritiquePayload:
        prompt = self._construct_judge_prompt(task, candidate)
        if hasattr(self.client, "generate_structured"):
            return self.client.generate_structured(prompt, response_model=CritiquePayload)
        # ponytail: deterministic additive heuristic when standalone model client is omitted
        score = 1.0 if (candidate.terminal_output and not candidate.verification or (candidate.verification and candidate.verification.is_valid)) else 0.0
        return CritiquePayload(
            criterion_scores={"logic": score},
            aggregate_score=score,
            detected_flaws=[] if score > 0.5 else ["Unresolved runtime failure"],
            remediation_hints=[] if score > 0.5 else ["Check test assertions and bounds"],
        )

    def _strip_formatting(self, text: str) -> str:
        import re
        return re.sub(r"```[a-zA-Z]*\n?|\n?```", "", text).strip()

    def _construct_judge_prompt(self, task: str, candidate: Trajectory) -> str:
        rubric_str = "\n".join(
            [f"- {r.name} (weight: {r.weight}): {r.description}" for r in self.rubrics]
        )
        clean_output = self._strip_formatting(candidate.terminal_output)
        return (
            f"You are a strict, objective verification engine.\n"
            f"TASK: {task}\n"
            f"PROPOSED REASONING TRACE: {candidate.reasoning_trace}\n"
            f"PROPOSED ANSWER: {clean_output}\n\n"
            f"Evaluate against these criteria:\n{rubric_str}\n"
            f"Score each item independently [0.0 - 1.0]. Focus exclusively on logic."
        )
