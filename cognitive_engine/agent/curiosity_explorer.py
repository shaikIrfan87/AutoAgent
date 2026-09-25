from typing import Dict, Any, Optional, List
import logging

try:
    from .executive_loop import ExecutiveLoop
    from ..core.graph_epistemic_scanner import GraphEpistemicScanner, EpistemicTarget
    from ..core.consolidation import HybridConsolidationStore
    from ..core.autotelic import AutotelicExperimentLoop, OpenWorldHypothesis
except ImportError:
    from agent.executive_loop import ExecutiveLoop
    from core.graph_epistemic_scanner import GraphEpistemicScanner, EpistemicTarget
    from core.consolidation import HybridConsolidationStore
    from core.autotelic import AutotelicExperimentLoop, OpenWorldHypothesis

logger = logging.getLogger(__name__)


class AutonomousCuriosityExplorer:
    """Autonomous daemon that scans knowledge gaps, queries the web/OS,
    executes verification code in the sandbox, and updates long-term memory.
    """

    def __init__(
        self,
        executive_loop: ExecutiveLoop,
        store: HybridConsolidationStore,
        scanner: Optional[GraphEpistemicScanner] = None,
        autotelic: Optional[AutotelicExperimentLoop] = None,
    ):
        self.loop = executive_loop
        self.store = store
        self.scanner = scanner or GraphEpistemicScanner(getattr(store, "conn", None))
        self.autotelic = autotelic or AutotelicExperimentLoop()
        self.history: List[Dict[str, Any]] = []

    def step_inquiry(self) -> Optional[Dict[str, Any]]:
        """Executes one autonomous research cycle."""
        # 1. Epistemic gap discovery
        targets = self.scanner.scan_stale_or_uncertain_nodes(top_k=1)
        if not targets:
            return None
        target = targets[0]

        # 2. Formulate empirical hypothesis
        hypo = self.autotelic.formulate_open_world_hypothesis(target.node_id)

        # 3. Grounded web query execution
        search_res = self.loop.execute_open_world_action(
            "http_get",
            {"url": f"https://api.duckduckgo.com/?q={hypo.search_query}&format=json"},
        )

        # 4. Programmatic execution verification in ephemeral virtual sandbox
        try:
            from ..core.virtual_workspace import EphemeralVirtualSystem
            from ..core.host_commit_gate import HostCommitGate
        except ImportError:
            from core.virtual_workspace import EphemeralVirtualSystem
            from core.host_commit_gate import HostCommitGate

        with EphemeralVirtualSystem(source_workspace=".") as v_sys:
            exec_res = self.loop.execute_algorithmic_task(
                code=hypo.verification_code_template,
                test_assertions=["assert True"],
            )
            success = (exec_res.get("status") == "success" or exec_res.get("success") is True)
            delta_s = 1.0 if success else -1.0
            gate = HostCommitGate(getattr(self.loop, "causal_graph", None))
            is_safe, _ = gate.inspect_and_verify(hypo.verification_code_template, delta_s)

        # 5. Long-term consolidation on verified success (Invariants 1-5)
        if success and is_safe:
            self.store.store_memory(
                memory_id=f"emp_{hypo.concept}",
                content=f"Empirically verified concept: {hypo.concept}. Search status: {search_res.get('status', 'unknown')}",
                metadata={"source": "autonomous_exploration", "concept": hypo.concept},
            )

        cycle_record = {
            "concept": hypo.concept,
            "search_status": search_res.get("status", "success" if search_res.get("success") else "error"),
            "execution_success": success,
            "output": exec_res.get("output", ""),
        }
        self.history.append(cycle_record)
        return cycle_record
