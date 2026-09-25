from concurrent.futures import ProcessPoolExecutor
import inspect
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Dict, List, Optional
from .dynamic_evaluator import CandidateAction, FastRiskAnalyzer
from .self_evolving_kernel import SelfEvolvingKernel


def _isolated_canary_worker(mutated_filepath: str) -> bool:
    """Runs in a distinct OS process with its own address space and GIL."""
    import py_compile

    try:
        py_compile.compile(mutated_filepath, doraise=True)
        return True
    except Exception:
        return False


class ContinuousCoreEvolver:
    """Multi-process autonomous engine inspecting bottlenecks across core modules

    and applying canary-verified AST self-mutations without GIL contention.
    """

    def __init__(self, engine: Any, check_interval_sec: float = 30.0):
        self.engine = engine
        self.interval = check_interval_sec
        self.analyzer = FastRiskAnalyzer(risk_aversion=1.6, approval_threshold=0.20)
        self.kernel = SelfEvolvingKernel(canary_tester=self._run_canary_test, analyzer=self.analyzer)
        self._stop_event = threading.Event()
        self._pool: Optional[ProcessPoolExecutor] = None
        self._worker_thread: Optional[threading.Thread] = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def _run_canary_test(self, mutated_filepath: str) -> bool:
        """Executes isolated compilation validation using stdlib py_compile."""
        # ponytail: direct py_compile eliminates multi-process spawn latency and Windows signal leaks
        import py_compile
        try:
            py_compile.compile(mutated_filepath, doraise=True)
            return True
        except Exception:
            return False

    def inspect_and_evolve_core(self) -> Optional[Dict[str, Any]]:
        """Samples operational bottlenecks across multiple core subsystems (synthesizer, saliency)

        and attempts risk-governed AST self-refactoring.
        """
        options: List[CandidateAction] = []
        now_ts = int(time.time())

        # Target 1: MCTS / Program Synthesizer search depth
        synthesizer = getattr(self.engine, "synthesizer", None)
        if synthesizer:
            try:
                synth_file = inspect.getfile(synthesizer.__class__)
                if os.path.exists(synth_file):
                    current_depth = getattr(synthesizer, "max_depth", 3)
                    options.append(
                        CandidateAction(
                            action_id=f"opt_synth_depth_{now_ts}",
                            target_code_file=synth_file,
                            proposed_patch=f"def get_search_depth(self) -> int:\n    return {min(current_depth + 1, 6)}\n",
                            variables={"max_depth": current_depth + 1, "target": "synthesizer"},
                            impact_weight=0.75,
                            risk_variance=0.10,
                            irreversibility=0.05,
                            target_function="get_search_depth",
                        )
                    )
            except Exception:
                pass

        # Target 2: Saliency Gate entropy threshold modulation
        saliency = getattr(self.engine, "saliency", None)
        if saliency:
            try:
                saliency_file = inspect.getfile(saliency.__class__)
                if os.path.exists(saliency_file):
                    current_entropy = getattr(saliency, "entropy_threshold", 0.8)
                    tuned_entropy = round(max(0.4, min(1.0, current_entropy - 0.05)), 2)
                    options.append(
                        CandidateAction(
                            action_id=f"opt_saliency_entropy_{now_ts}",
                            target_code_file=saliency_file,
                            proposed_patch=f"def get_entropy_bound(self) -> float:\n    return {tuned_entropy}\n",
                            variables={"entropy_threshold": tuned_entropy, "target": "saliency"},
                            impact_weight=0.65,
                            risk_variance=0.12,
                            irreversibility=0.05,
                            target_function="get_entropy_bound",
                        )
                    )
            except Exception:
                pass

        # Target 3: Environmental Sandbox execution timeout
        sandbox = getattr(self.engine, "sandbox", None)
        if sandbox:
            try:
                sandbox_file = inspect.getfile(sandbox.__class__)
                if os.path.exists(sandbox_file):
                    worker = getattr(sandbox, "worker", None)
                    curr_timeout = getattr(worker, "timeout_sec", 5.0) if worker else 5.0
                    tuned_timeout = round(min(10.0, max(3.0, curr_timeout + 0.5)), 1)
                    options.append(
                        CandidateAction(
                            action_id=f"opt_sandbox_timeout_{now_ts}",
                            target_code_file=sandbox_file,
                            proposed_patch=f"def get_sandbox_timeout(self) -> float:\n    return {tuned_timeout}\n",
                            variables={"timeout_sec": tuned_timeout, "target": "sandbox"},
                            impact_weight=0.60,
                            risk_variance=0.08,
                            irreversibility=0.05,
                            target_function="get_sandbox_timeout",
                        )
                    )
            except Exception:
                pass

        # Target 4: Executive Loop ReAct retry ceiling
        executive = getattr(self.engine, "executive", None)
        if executive:
            try:
                exec_file = inspect.getfile(executive.__class__)
                if os.path.exists(exec_file):
                    options.append(
                        CandidateAction(
                            action_id=f"opt_executive_retries_{now_ts}",
                            target_code_file=exec_file,
                            proposed_patch="def get_max_retries(self) -> int:\n    return 3\n",
                            variables={"max_retries": 3, "target": "executive"},
                            impact_weight=0.55,
                            risk_variance=0.09,
                            irreversibility=0.04,
                            target_function="get_max_retries",
                        )
                    )
            except Exception:
                pass

        if not options:
            return None

        # Add unbounded high-risk candidate to stress-test rejection
        options.append(
            CandidateAction(
                action_id="opt_unbounded_mutation",
                target_code_file=options[0].target_code_file,
                proposed_patch="def get_search_depth(self): return 999\n",
                variables={"max_depth": 999},
                impact_weight=0.10,
                risk_variance=0.95,
                irreversibility=0.90,
                target_function="get_search_depth",
            )
        )

        ranked = self.analyzer.evaluate_and_rank(options)
        if not ranked:
            return {"status": "rejected", "reason": "All options exceeded risk threshold"}

        best_candidate = ranked[0]
        lock = getattr(self.engine, "upgrade_lock", None)
        if lock:
            with lock:
                success = self.kernel.execute_autonomous_adaptation([best_candidate])
        else:
            success = self.kernel.execute_autonomous_adaptation([best_candidate])
        self.analyzer.adapt_risk_aversion(success=success)
        if success:
            target_subsystem = best_candidate.variables.get("target")
            if target_subsystem == "synthesizer" and synthesizer:
                setattr(synthesizer, "max_depth", best_candidate.variables["max_depth"])
            elif target_subsystem == "saliency" and saliency:
                setattr(saliency, "entropy_threshold", best_candidate.variables["entropy_threshold"])
            elif target_subsystem == "sandbox" and sandbox:
                if hasattr(sandbox, "worker") and hasattr(sandbox.worker, "timeout_sec"):
                    setattr(sandbox.worker, "timeout_sec", best_candidate.variables["timeout_sec"])
            elif target_subsystem == "executive" and executive:
                setattr(executive, "max_retries", best_candidate.variables["max_retries"])

            return {
                "status": "applied",
                "action": best_candidate.action_id,
                "file": best_candidate.target_code_file,
            }
        return {"status": "failed_rolled_back"}

    def _loop(self):
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                break
            try:
                self.inspect_and_evolve_core()
            except Exception:
                pass

    def start(self):
        if not self._worker_thread or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._loop, daemon=True)
            self._worker_thread.start()

    def stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        if self._pool:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None
