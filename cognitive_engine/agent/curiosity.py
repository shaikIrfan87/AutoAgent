"""
Curiosity Daemon: Autonomous inquiry, compression, and risk-governed self-updating daemon.
"""
from collections import deque
import inspect
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

try:
    from ..core.autotelic import AutotelicExperimentLoop, ExperimentResult
    from ..core.compression import ProgramCompressor
    from ..core.meta_optimizer import MetaSelfOptimizer, OptimizationResult
    from ..core.autonomous_decision import MutationProposal, RiskScoreGovernor
    from ..core.autonomous_governor import AutonomousGovernor, AutonomousResult
    from ..core.graph_epistemic_scanner import GraphEpistemicScanner, EpistemicTarget
    from ..core.self_evolving_kernel import SelfEvolvingKernel, make_default_canary_harness
    from ..core.dynamic_evaluator import CandidateAction
    from .curiosity_explorer import AutonomousCuriosityExplorer
    from ..core.hybrid_scraper import HybridWebScraper
except ImportError:
    from core.autotelic import AutotelicExperimentLoop, ExperimentResult
    from core.compression import ProgramCompressor
    from core.meta_optimizer import MetaSelfOptimizer, OptimizationResult
    from core.autonomous_decision import MutationProposal, RiskScoreGovernor
    from core.autonomous_governor import AutonomousGovernor, AutonomousResult
    from core.graph_epistemic_scanner import GraphEpistemicScanner, EpistemicTarget
    from core.self_evolving_kernel import SelfEvolvingKernel, make_default_canary_harness
    from core.dynamic_evaluator import CandidateAction
    from agent.curiosity_explorer import AutonomousCuriosityExplorer
    try:
        from cognitive_engine.core.hybrid_scraper import HybridWebScraper
    except ImportError:
        HybridWebScraper = None


class CuriosityDaemon:
    """Autonomous inquiry, compression, and risk-governed self-updating daemon."""

    def __init__(
        self,
        engine: Any,
        interval_sec: float = 60.0,
        autotelic: Optional[AutotelicExperimentLoop] = None,
        compressor: Optional[ProgramCompressor] = None,
        compression_batch_size: int = 5,
        optimizer: Optional[MetaSelfOptimizer] = None,
        governor: Optional[AutonomousGovernor] = None,
        scanner: Optional[GraphEpistemicScanner] = None,
        explorer: Optional[AutonomousCuriosityExplorer] = None,
        kernel: Optional[SelfEvolvingKernel] = None,
        telemetry_window_size: int = 10,
        failure_threshold_ratio: float = 0.60,
        eval_window_size: int = 3,
        degradation_tolerance: float = 0.10,
    ):
        self.engine = engine
        self.interval_sec = interval_sec
        self.autotelic = autotelic
        self.compressor = compressor
        self.compression_batch_size = compression_batch_size
        self.optimizer = optimizer
        self.governor = governor or AutonomousGovernor(target_engine=engine, optimizer=self.optimizer)
        self.scanner = scanner
        if self.scanner is None and hasattr(self.engine, "consolidation"):
            db_conn = getattr(self.engine.consolidation, "conn", None)
            causal_g = getattr(self.engine, "causal", None)
            if db_conn:
                self.scanner = GraphEpistemicScanner(db_conn=db_conn, causal_graph=causal_g)

        self.explorer = explorer
        if self.explorer is None and hasattr(self.engine, "executive_loop") and hasattr(self.engine, "consolidation"):
            self.explorer = AutonomousCuriosityExplorer(
                executive_loop=self.engine.executive_loop,
                store=self.engine.consolidation,
                scanner=self.scanner,
                autotelic=self.autotelic,
            )

        self.kernel = kernel

        self.failure_threshold_ratio = failure_threshold_ratio
        self.eval_window_size = eval_window_size
        self.degradation_tolerance = degradation_tolerance

        self._cycle_counter = 0

        self._last_compressed_count = 0
        self.telemetry_window: Deque[bool] = deque(maxlen=telemetry_window_size)
        self.post_opt_window: Deque[bool] = deque(maxlen=eval_window_size)

        self.last_optimization: Optional[Any] = None
        self.is_evaluating_optimization: bool = False
        self.pre_opt_failure_rate: float = 0.0
        self._pre_opt_depth: int = 3
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._paused: bool = False
        self.idle_delay_sec: float = 5.0
        self._last_active_time: float = 0.0
        self.scraper = HybridWebScraper() if HybridWebScraper else None

    # Prefixes written by curiosity/ingestor that should not be used as scrape targets
    _POLLUTION_PREFIXES = (
        "Ground truth", "Fact:", "Empirical gap", "Verify empirical",
        "Query:", "Trace:", "RLCD Verified", "Verified Goal:",
    )

    def scan_decayed_memories(self, threshold: float = 0.60) -> List[Dict[str, Any]]:
        """Extracts decaying/stale epistemic concepts from consolidation database (C_eff < threshold)."""
        import sqlite3
        import math
        db_path = getattr(getattr(self.engine, "consolidation", None), "db_path", "assets/cognitive_memory.db")
        if not os.path.exists(db_path):
            return []
        decayed = []
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT id, content, confidence, last_accessed FROM memories").fetchall()
                now = time.time()
                for m_id, content, conf, last_acc in rows:
                    # Skip sentinel/pollution records — they must not be used as curiosity targets
                    if any(content.startswith(p) for p in self._POLLUTION_PREFIXES):
                        continue
                    elapsed_hours = (now - float(last_acc)) / 3600.0
                    c_eff = float(conf) * math.exp(-0.05 * elapsed_hours)
                    if c_eff < threshold:
                        # Strip known content prefixes before extracting the domain topic word
                        clean = content
                        for p in ("Query: ", "Fact: ", "Trace: "):
                            if clean.startswith(p):
                                clean = clean[len(p):]
                                break
                        first_words = [w for w in clean.replace(":", " ").replace("->", " ").split() if len(w) >= 3]
                        topic = first_words[0] if first_words else "algorithm"
                        # Sanity-check: skip generic English stop-topics that produce noise pages
                        _SKIP_TOPICS = {"query", "output", "result", "error", "delta", "system", "trace"}
                        if topic.lower() in _SKIP_TOPICS:
                            continue
                        decayed.append({"id": m_id, "content": content, "c_eff": c_eff, "topic": topic})
        except Exception:
            pass
        return decayed

    def record_telemetry(self, success: bool):
        if self.is_evaluating_optimization:
            self.post_opt_window.append(success)
        else:
            self.telemetry_window.append(success)

    def should_trigger_meta_optimization(self) -> bool:
        if len(self.telemetry_window) < (self.telemetry_window.maxlen or 1):
            return False
        failures = sum(1 for passed in self.telemetry_window if not passed)
        failure_rate = failures / len(self.telemetry_window)
        return failure_rate >= self.failure_threshold_ratio

    def _attempt_self_optimization(self) -> Optional[Any]:
        """Formulates an autonomous mutation proposal, evaluates risk, and executes canary validation."""
        synthesizer = getattr(self.autotelic, "synthesizer", None)
        if not synthesizer:
            synthesizer = getattr(self.engine, "synthesizer", None)
        if not synthesizer:
            return None

        # Ensure engine or governor can access synthesizer
        if not hasattr(self.engine, "synthesizer"):
            setattr(self.engine, "synthesizer", synthesizer)

        failures = sum(1 for passed in self.telemetry_window if not passed)
        self.pre_opt_failure_rate = (failures / len(self.telemetry_window)) if self.telemetry_window else 0.0

        current_depth = getattr(synthesizer, "max_depth", 3)
        self._pre_opt_depth = current_depth
        new_depth = min(current_depth + 1, 5)

        # 1. Execute live source-code mutation via SelfEvolvingKernel if configured
        if self.kernel:
            synth_file = None
            try:
                synth_file = inspect.getfile(synthesizer.__class__)
            except Exception:
                synth_file = None
            if synth_file and os.path.exists(synth_file):
                action = CandidateAction(
                    action_id=f"depth_expansion_{int(time.time())}",
                    target_code_file=synth_file,
                    proposed_patch=f"def get_search_depth(self) -> int:\n    return {new_depth}\n",
                    variables={"max_depth": new_depth},
                    impact_weight=0.70,
                    risk_variance=0.08,
                    irreversibility=0.05,
                    target_function="get_search_depth",
                )
                if self.kernel.execute_autonomous_adaptation([action]):
                    synthesizer.max_depth = new_depth
                    if self.optimizer:
                        self.optimizer.rollback_history.append((synthesizer, "max_depth", current_depth))
                    self.is_evaluating_optimization = True
                    self.post_opt_window.clear()
                    self.telemetry_window.clear()
                    self.last_optimization = AutonomousResult({
                        "status": "applied",
                        "applied": True,
                        "kernel_mutation": True,
                        "target_file": synth_file,
                        "new_depth": new_depth,
                    })
                    return self.last_optimization

        # 2. Formulate risk-bounded mutation proposal for in-memory Autonomous Governor
        proposal = MutationProposal(
            proposal_id=f"depth_expansion_{int(time.time())}",
            target_subsystem="synthesizer",
            description=f"Expand search depth bound from {current_depth} to {new_depth}",
            patch_code=f"def get_search_depth(self) -> int:\n    return {new_depth}\n",
            parameters_delta={"max_depth": new_depth},
            expected_accuracy_gain=0.70,
            expected_speedup=0.10,
            failure_probability=0.08,
            reversibility_score=0.95,
        )

        def canary_fn(fn: Any) -> bool:
            if not callable(fn):
                return False
            try:
                return fn(None) == new_depth
            except TypeError:
                try:
                    return fn() == new_depth
                except Exception:
                    return False
            except Exception:
                return False

        # 3. Execute through Autonomous Governor (in-memory fallback)
        result = self.governor.process_and_execute(proposal, canary_validator=canary_fn)

        if result.get("status") == "applied" or result.get("applied") is True:
            synthesizer.max_depth = new_depth
            self.is_evaluating_optimization = True
            self.post_opt_window.clear()
            self.telemetry_window.clear()

        self.last_optimization = result
        return result

    def check_post_optimization_degradation(self) -> bool:
        if not self.is_evaluating_optimization or len(self.post_opt_window) < (self.post_opt_window.maxlen or 1):
            return False

        post_fail_rate = sum(1 for ok in self.post_opt_window if not ok) / len(self.post_opt_window)
        if post_fail_rate > (self.pre_opt_failure_rate + self.degradation_tolerance):
            # Rollback memory and parameters
            if self.optimizer:
                self.optimizer.rollback_last()
            if self.autotelic and getattr(self.autotelic, "synthesizer", None):
                self.autotelic.synthesizer.max_depth = self._pre_opt_depth

            self.is_evaluating_optimization = False
            self.post_opt_window.clear()
            return True

        self.is_evaluating_optimization = False
        self.post_opt_window.clear()
        return False

    def find_epistemic_frontiers(self) -> List[str]:
        """Locate isolated or weakly-connected causal nodes requiring empirical ground tests."""
        graph = getattr(self.engine, "causal", None)
        if graph and hasattr(graph, "graph"):
            return [node for node, deg in graph.graph.degree() if deg <= 1]
        return []

    def formulate_inquiry(self, node: str) -> str:
        return f"Verify empirical causal properties of {node}"

    def _dispatch_intrinsic_inquiry(self, target: Any) -> Optional[Any]:
        """Dispatch target as an intrinsic goal to the engine process loop."""
        if not hasattr(self.engine, "process"):
            return None
        node_id = getattr(target, "node_id", str(target))
        ttype = getattr(target, "target_type", "frontier")
        ctx = getattr(target, "context", "")
        query = f"Verify empirical causal properties of {node_id} ({ttype}): {ctx}".strip()
        return self.engine.process(query)

    def step(self) -> Optional[Any]:
        # Non-blocking lock detection: immediately yield if foreground worker holds consolidation lock
        store = getattr(self.engine, "consolidation", None)
        if store and hasattr(store, "_lock"):
            acquired = store._lock.acquire(blocking=False)
            if not acquired:
                return None
            store._lock.release()

        # Non-blocking upgrade check: yield if architectural self-upgrade is actively mutating code
        upgrade_lock = getattr(self.engine, "upgrade_lock", None)
        if upgrade_lock:
            acquired_up = upgrade_lock.acquire(blocking=False)
            if not acquired_up:
                return None
            upgrade_lock.release()

        # Open-world curiosity exploration for decaying epistemic memories (C_eff < 0.60)
        decayed = self.scan_decayed_memories(threshold=0.60)
        if decayed:
            try:
                from ..core.compression_progress import CompressionProgressEvaluator
                cpe = CompressionProgressEvaluator()
                valid_targets = [d for d in decayed if not cpe.is_noise_trap(d.get("content", ""))]
                if not valid_targets:
                    return {"status": "noise_trap_aborted", "reason": "Target identified as uncompressible stochastic noise"}
                target = valid_targets[0]
            except Exception:
                target = decayed[0]

            topic = target["topic"]
            # 1. Epistemic conjecture & virtual staged synthesis loop
            if hasattr(self.engine, "execute_staged_autonomous_cycle"):
                goal = f"Synthesize and ground verified properties for {topic}"
                try:
                    res = self.engine.execute_staged_autonomous_cycle(goal)
                    if res and "Success" in str(res):
                        return {"status": "autonomous_staged_success", "topic": topic, "delta_s": 1.0, "result": res}
                except Exception:
                    pass

            # 2. Hybrid web scraper rejuvenation fallback
            if self.scraper:
                try:
                    res = self.scraper.fetch(f"https://en.wikipedia.org/wiki/{topic}")
                    if res.get("saliency_passed") and res.get("content"):
                        store = getattr(self.engine, "consolidation", None)
                        if store and hasattr(store, "add_memory"):
                            # Use "Rejuvenated:" prefix — not in _POLLUTION_PREFIXES, never self-referential
                            store.add_memory(
                                f"Rejuvenated: {topic} — {res['content'][:250]}",
                                confidence=0.90,
                                metadata={"source": "hybrid_scraper_curiosity", "url": res.get("url", "")}
                            )
                            return {"status": "rejuvenated", "topic": topic, "delta_s": 1.0}
                except Exception:
                    pass


        # Open-world curiosity exploration vs ARC macro synthesis alternation
        self._cycle_counter += 1
        if self.explorer and hasattr(self.explorer, "step_inquiry") and (self._cycle_counter % 2 == 1 or not self.autotelic):
            inquiry_res = self.explorer.step_inquiry()
            if inquiry_res is not None:
                return inquiry_res

        # Embodied physical sensory-motor exploration probe
        embodied = getattr(self.engine, "embodied", None)
        if embodied and self._cycle_counter % 5 == 0:
            try:
                nodes = embodied.eyes.inspect_ui_tree()
                clickable = [n for n in nodes if n.get("clickable") and n.get("name") and len(n["name"]) > 2]
                if clickable:
                    target_name = clickable[0]["name"]
                    target_x, target_y = clickable[0]["center"]
                    shift = embodied.eyes.compute_screen_delta(embodied.eyes.capture_frame()[0])
                    return {"status": "embodied_perception_active", "target": target_name, "coords": (target_x, target_y), "visual_shift": shift}
            except Exception:
                pass

        # 1. Epistemic autotelic experiments
        if self.autotelic and hasattr(self.engine, "consolidation"):
            gaps = getattr(self.autotelic, "active_hypotheses", [])
            if not gaps:
                causal_rules = getattr(getattr(self.engine, "causal", None), "rules", {})
                gaps = self.autotelic.scan_epistemic_gaps([], causal_rules)
            if gaps:
                gap = gaps.pop(0) if isinstance(gaps, list) else gaps[0]
                exp_res = self.autotelic.run_experiment(gap)
                self.record_telemetry(exp_res.success)

                # Online learning: train neural policy/value network on autotelic rollout
                synth = getattr(self.engine, "synthesizer", None)
                if synth and hasattr(synth, "train_policy_value") and getattr(exp_res, "input_grid", None) and exp_res.program:
                    from ..core.dsl import UNARY_PRIMITIVES
                    op_name = exp_res.program.steps[0][0] if exp_res.program.steps else "rot90"
                    op_keys = list(UNARY_PRIMITIVES.keys())
                    op_idx = op_keys.index(op_name) if op_name in op_keys else 0
                    synth.train_policy_value([[exp_res.input_grid]], [op_idx], [1.0 if exp_res.success else -0.5])

                # Check for post-optimization degradation
                if self.is_evaluating_optimization:
                    self.check_post_optimization_degradation()

                # Trigger sleep-phase library compression if batch threshold met
                if self.compressor:
                    successful = [
                        r.program for r in self.autotelic.experiment_history if r.success and r.program
                    ]
                    if len(successful) - self._last_compressed_count >= self.compression_batch_size:
                        mined = self.compressor.mine_abstractions(successful)
                        self.compressor.consolidate_into_dsl(mined)
                        macro_store = getattr(self.engine, "macro_store", None)
                        if macro_store:
                            macro_store.save(self.compressor.invented_library)
                            skills = getattr(self.engine, "skills", None)
                            if skills:
                                macro_store.export_to_skills(skills)
                        self._last_compressed_count = len(successful)

                # Bridge deep MCTS discoveries (len(steps) >= 3) into macro compression and skills
                if exp_res.success and exp_res.program and len(exp_res.program.steps) >= 3 and self.compressor:
                    mined_deep = self.compressor.mine_abstractions([exp_res.program])
                    if mined_deep:
                        self.compressor.consolidate_into_dsl(mined_deep)
                        macro_store = getattr(self.engine, "macro_store", None)
                        if macro_store:
                            macro_store.save(self.compressor.invented_library)
                            skills = getattr(self.engine, "skills", None)
                            if skills:
                                macro_store.export_to_skills(skills)

                # Trigger risk-governed autonomous self-modification if failure threshold reached
                if not self.is_evaluating_optimization and self.should_trigger_meta_optimization():
                    self._attempt_self_optimization()

                return exp_res

        # 2. Epistemic scanner frontier detection
        if self.scanner and not self.is_evaluating_optimization:
            epistemic_targets = self.scanner.scan_stale_or_uncertain_nodes(top_k=1)
            if epistemic_targets:
                return self._dispatch_intrinsic_inquiry(epistemic_targets[0])

        # 3. Cross-domain curriculum advancement via AutonomousDomainSpawner
        curriculum = getattr(self.engine, "curriculum", None)
        if curriculum and hasattr(curriculum, "execute_curriculum_step"):
            try:
                c_res = curriculum.execute_curriculum_step()
                if c_res and not str(c_res).startswith("Curriculum Status: Waiting"):
                    return {"status": "curriculum_advanced", "detail": c_res}
            except Exception:
                pass

        # 4. Frontier exploration fallback
        frontiers = self.find_epistemic_frontiers()
        if not frontiers or not hasattr(self.engine, "process"):
            return None
        target = frontiers[0]
        query = self.formulate_inquiry(target)
        return self.engine.process(query)

    def touch_activity(self) -> None:
        """Records user activity to defer curiosity steps until idle."""
        self._last_active_time = time.time()

    def pause(self) -> None:
        """Pauses curiosity steps during active computation or user prompt handling."""
        self._paused = True

    def resume(self) -> None:
        """Resumes curiosity steps and resets idle activity timer."""
        self._paused = False
        self.touch_activity()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_step_time = 0.0
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=1.0):
                break
            if getattr(self, "_paused", False):
                continue
            now = time.time()
            # Idle gating: require at least idle_delay_sec since last user interaction
            if now - getattr(self, "_last_active_time", 0.0) < self.idle_delay_sec:
                continue
            # Interval check: run at most once per interval_sec
            if now - last_step_time < self.interval_sec:
                continue
            try:
                self.step()
                last_step_time = time.time()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
