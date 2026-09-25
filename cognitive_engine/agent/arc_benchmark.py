"""
Empirical ARC-AGI Benchmark Harness for Fluid Synthesizer Evaluation.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..core.dsl import Grid, to_grid
    from .program_synthesizer import MCTSProgramSynthesizer, ProgramSynthesizer, SynthesizedProgram
except ImportError:
    from core.dsl import Grid, to_grid
    from agent.program_synthesizer import MCTSProgramSynthesizer, ProgramSynthesizer, SynthesizedProgram


@dataclass
class ARCTask:
    name: str
    category: str
    train: List[Tuple[Grid, Grid]]
    test_in: Grid
    expected_test_out: Grid


def get_canonical_arc_tasks() -> List[ARCTask]:
    """Canonical ARC evaluation tasks spanning geometric, topological, and composite domains."""
    return [
        ARCTask(
            name="horizontal_reflection_symmetry",
            category="geometric_symmetry",
            train=[
                (to_grid([[1, 2], [3, 4]]), to_grid([[1, 2, 2, 1], [3, 4, 4, 3]])),
                (to_grid([[5, 0], [0, 5]]), to_grid([[5, 0, 0, 5], [0, 5, 5, 0]])),
            ],
            test_in=to_grid([[7, 8], [9, 1]]),
            expected_test_out=to_grid([[7, 8, 8, 7], [9, 1, 1, 9]]),
        ),
        ARCTask(
            name="gravitational_dropping",
            category="dynamics",
            train=[
                (to_grid([[1, 0], [0, 2], [0, 0]]), to_grid([[0, 0], [0, 0], [1, 2]])),
                (to_grid([[0, 3], [4, 0], [0, 0]]), to_grid([[0, 0], [0, 0], [4, 3]])),
            ],
            test_in=to_grid([[5, 0], [0, 6], [0, 0]]),
            expected_test_out=to_grid([[0, 0], [0, 0], [5, 6]]),
        ),
        ARCTask(
            name="topological_largest_component",
            category="topology",
            train=[
                (to_grid([[3, 3, 0], [3, 0, 0], [0, 0, 8]]), to_grid([[3, 3, 0], [3, 0, 0], [0, 0, 0]])),
                (to_grid([[0, 0, 2], [4, 4, 0], [4, 4, 0]]), to_grid([[0, 0, 0], [4, 4, 0], [4, 4, 0]])),
            ],
            test_in=to_grid([[1, 1, 0], [1, 0, 0], [0, 0, 9]]),
            expected_test_out=to_grid([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ),
        ARCTask(
            name="color_remapping",
            category="color_logic",
            train=[
                (to_grid([[1, 2], [1, 0]]), to_grid([[8, 2], [8, 0]])),
                (to_grid([[0, 1], [1, 3]]), to_grid([[0, 8], [8, 3]])),
            ],
            test_in=to_grid([[1, 1], [0, 2]]),
            expected_test_out=to_grid([[8, 8], [0, 2]]),
        ),
        ARCTask(
            name="deep_composite_rotation_and_recolor",
            category="multi_step_composition",
            train=[
                (to_grid([[1, 0], [0, 0]]), to_grid([[0, 7], [0, 0]])),
                (to_grid([[1, 1], [0, 0]]), to_grid([[0, 7], [0, 7]])),
            ],
            test_in=to_grid([[0, 1], [0, 0]]),
            expected_test_out=to_grid([[0, 0], [0, 7]]),
        ),
    ]


class ARCBenchmarkHarness:
    """Evaluates program synthesizers against held-out ARC generalization tasks."""

    def __init__(self, tasks: Optional[List[ARCTask]] = None):
        self.tasks = tasks or get_canonical_arc_tasks()

    def run_benchmark(self, synthesizer: Optional[ProgramSynthesizer] = None) -> Dict[str, Any]:
        synth = synthesizer or MCTSProgramSynthesizer(max_depth=5, max_expansions=1500)
        results = []
        start_time = time.perf_counter()

        total_count = len(self.tasks)
        for idx, task in enumerate(self.tasks):
            t0 = time.perf_counter()
            prog: Optional[SynthesizedProgram] = synth.synthesize(task.train)
            task_time_ms = (time.perf_counter() - t0) * 1000.0

            solved_train = prog is not None
            generalized = False
            pred_test_out = None

            if prog is not None:
                try:
                    pred_test_out = prog(task.test_in)
                    generalized = (pred_test_out == task.expected_test_out)
                except Exception:
                    generalized = False

            print(f"[{idx + 1}/{total_count}] {task.name}: train={solved_train}, test={generalized} ({task_time_ms:.1f}ms)", flush=True)

            results.append({
                "task": task.name,
                "category": task.category,
                "solved_train": solved_train,
                "generalized": generalized,
                "latency_ms": round(task_time_ms, 2),
                "code": prog.code if prog else None,
            })

        total_elapsed_sec = time.perf_counter() - start_time
        total_tasks = len(self.tasks)
        solved_train_count = sum(1 for r in results if r["solved_train"])
        generalized_count = sum(1 for r in results if r["generalized"])

        return {
            "total_tasks": total_tasks,
            "solved_train": solved_train_count,
            "generalized_test": generalized_count,
            "generalization_rate": round(generalized_count / total_tasks, 4) if total_tasks else 0.0,
            "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / total_tasks, 2) if total_tasks else 0.0,
            "total_elapsed_sec": round(total_elapsed_sec, 3),
            "results": results,
        }

    @classmethod
    def from_json_file(cls, json_path: str, task_name: Optional[str] = None) -> "ARCBenchmarkHarness":
        """Load ARC task(s) from a standard ARC JSON file or directory."""
        import json
        import os
        tasks = []
        if os.path.isdir(json_path):
            for fname in sorted(os.listdir(json_path)):
                if fname.endswith(".json"):
                    tasks.extend(cls._parse_arc_file(os.path.join(json_path, fname), fname[:-5]))
        elif os.path.isfile(json_path):
            name = task_name or os.path.splitext(os.path.basename(json_path))[0]
            tasks.extend(cls._parse_arc_file(json_path, name))
        return cls(tasks=tasks)

    @staticmethod
    def _parse_arc_file(path: str, name: str) -> List[ARCTask]:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = []
        train_pairs = [(to_grid(p["input"]), to_grid(p["output"])) for p in data.get("train", [])]
        for idx, test_p in enumerate(data.get("test", [])):
            t_in = to_grid(test_p["input"])
            t_out = to_grid(test_p["output"]) if "output" in test_p else to_grid([[0]])
            tasks.append(ARCTask(
                name=f"{name}_test{idx}",
                category="streamed_eval",
                train=train_pairs,
                test_in=t_in,
                expected_test_out=t_out,
            ))
        return tasks

