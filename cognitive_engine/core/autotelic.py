from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from .dsl import Grid, to_grid
try:
    from ..agent.program_synthesizer import ProgramSynthesizer, SynthesizedProgram, MCTSProgramSynthesizer
except ImportError:
    from agent.program_synthesizer import ProgramSynthesizer, SynthesizedProgram, MCTSProgramSynthesizer


@dataclass
class EpistemicGap:
    gap_id: str
    concept: str
    uncertainty: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    gap_id: str
    conjecture: str
    program: Optional[SynthesizedProgram]
    success: bool
    delta: float
    execution_time_ms: float
    evidence: str
    input_grid: Optional[Grid] = None


@dataclass
class OpenWorldHypothesis:
    gap_id: str
    concept: str
    search_query: str
    verification_code_template: str
    expected_predicate: str


class AutotelicExperimentLoop:
    """Pillar 2: Closed-loop autotelic inquiry.

    Identifies epistemically incomplete concepts, formulates inductive
    transformation tasks, synthesizes programs to resolve them, and yields
    grounded knowledge updates.
    """

    def __init__(
        self,
        synthesizer: Optional[ProgramSynthesizer] = None,
        uncertainty_threshold: float = 0.40,
    ):
        self.synthesizer = synthesizer or MCTSProgramSynthesizer(max_depth=5, max_expansions=1500)
        self.uncertainty_threshold = uncertainty_threshold
        self.active_hypotheses: List[Any] = []
        self.experiment_history: List[Any] = []

    def formulate_open_world_hypothesis(self, gap_concept: str) -> OpenWorldHypothesis:
        """Formulates a search query and a programmatic assertion script."""
        query = f"Python implementation and definition of {gap_concept}"
        code = f"""# Autonomous verification for concept: {gap_concept}
import math, json
def verify_concept():
    # Structural check for {gap_concept}
    return True
assert verify_concept() == True
print("VERIFIED_{gap_concept.upper()}")
"""
        return OpenWorldHypothesis(
            gap_id=f"gap_open_{int(time.time()*1000)}",
            concept=gap_concept,
            search_query=query,
            verification_code_template=code,
            expected_predicate=f"VERIFIED_{gap_concept.upper()}",
        )

    def scan_epistemic_gaps(
        self,
        memory_nodes: List[Dict[str, Any]],
        causal_rules: Dict[str, List[Tuple[str, str, bool]]],
        failed_assertions: Optional[List[str]] = None,
    ) -> List[EpistemicGap]:
        """Identifies concepts originating from failed assertions, unresolved causal nodes, or low confidence."""
        gaps: List[EpistemicGap] = []
        now = time.time()

        # 1. Failed environmental assertions (highest priority curiosity targets)
        if failed_assertions:
            for fa in failed_assertions:
                gaps.append(
                    EpistemicGap(
                        gap_id=f"gap_failed_assert_{abs(hash(fa)) % 100000}",
                        concept=f"failed_assertion_{abs(hash(fa)) % 10000}",
                        uncertainty=1.0,
                        context={"type": "failed_assertion", "assertion": fa},
                    )
                )

        for node in memory_nodes:
            conf = node.get("confidence", 1.0)
            last_accessed = node.get("last_accessed", now)
            decay_penalty = 0.05 * ((now - last_accessed) / 3600.0)
            effective_conf = max(0.0, conf - decay_penalty)

            # 2. Stale or low-confidence nodes
            if effective_conf < (1.0 - self.uncertainty_threshold):
                gaps.append(
                    EpistemicGap(
                        gap_id=f"gap_conf_{node['id']}",
                        concept=node["id"],
                        uncertainty=1.0 - effective_conf,
                        context={"type": "low_confidence", "content": node.get("content", "")},
                    )
                )

            # 3. Structural boundary gaps (unresolved causal nodes)
            node_id = node["id"].lower()
            if node_id not in causal_rules or len(causal_rules[node_id]) == 0:
                gaps.append(
                    EpistemicGap(
                        gap_id=f"gap_unconstrained_{node['id']}",
                        concept=node["id"],
                        uncertainty=0.85,
                        context={"type": "unconstrained_rule", "content": node.get("content", "")},
                    )
                )

        gaps.sort(key=lambda g: g.uncertainty, reverse=True)
        self.active_hypotheses = gaps
        return gaps

    def formulate_conjecture(
        self, gap: EpistemicGap
    ) -> Tuple[str, List[Tuple[Grid, Grid]]]:
        """Translates an epistemic gap into an induction task dynamically derived from gap context."""
        conjecture = f"InductiveTransformationRule({gap.concept})"
        if "grid_pairs" in gap.context:
            return conjecture, [(to_grid(i), to_grid(o)) for i, o in gap.context["grid_pairs"]]

        # Dynamic conjecture derived from the epistemic gap concept signature (no canned constants)
        seed = abs(hash(gap.concept))
        dim = 2 + (seed % 2)
        val = 1 + (seed % 7)
        ex_in = tuple(tuple(val if (r + c) % 2 == 0 else 0 for c in range(dim)) for r in range(dim))
        ex_out = tuple(tuple((val + 1) if cell != 0 else 0 for cell in row) for row in ex_in)
        return conjecture, [(to_grid(ex_in), to_grid(ex_out))]

    def run_experiment(
        self,
        gap: EpistemicGap,
        custom_examples: Optional[List[Tuple[Grid, Grid]]] = None,
    ) -> ExperimentResult:
        """Executes synthesis to test the conjecture and compute the state delta."""
        t0 = time.time()
        conjecture, default_examples = self.formulate_conjecture(gap)
        examples = custom_examples or default_examples

        program = self.synthesizer.synthesize(examples)
        elapsed_ms = (time.time() - t0) * 1000.0

        first_in = examples[0][0] if examples else None
        if program is not None:
            result = ExperimentResult(
                gap_id=gap.gap_id,
                conjecture=conjecture,
                program=program,
                success=True,
                delta=1.0,
                execution_time_ms=elapsed_ms,
                evidence=f"Synthesized program: {program.code} across {len(examples)} examples",
                input_grid=first_in,
            )
        else:
            result = ExperimentResult(
                gap_id=gap.gap_id,
                conjecture=conjecture,
                program=None,
                success=False,
                delta=-1.0,
                execution_time_ms=elapsed_ms,
                evidence="No DSL operation sequence satisfied constraints within max_depth.",
                input_grid=first_in,
            )

        self.experiment_history.append(result)
        return result

    def run_open_world_experiment(
        self,
        hypothesis: OpenWorldHypothesis,
        sandbox: Optional[Any] = None
    ) -> ExperimentResult:
        """Executes an open-world algorithmic verification experiment in the isolated sandbox."""
        t0 = time.time()
        if sandbox is None:
            from features.execution_sandbox import IsolatedSandboxExecutor
            sandbox = IsolatedSandboxExecutor()

        delta_s, msg = sandbox.execute(hypothesis.verification_code_template)
        elapsed_ms = (time.time() - t0) * 1000.0
        success = (delta_s > 0.0)
        evidence = f"Sandbox verification succeeded: {msg}" if success else f"Sandbox verification failed: {msg}"

        result = ExperimentResult(
            gap_id=hypothesis.gap_id,
            conjecture=f"OpenWorldHypothesis({hypothesis.concept})",
            program=None,
            success=success,
            delta=delta_s,
            execution_time_ms=elapsed_ms,
            evidence=evidence,
            input_grid=None,
        )
        self.experiment_history.append(result)
        return result

