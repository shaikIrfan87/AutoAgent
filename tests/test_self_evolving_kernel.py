import os
import tempfile
import pytest
from cognitive_engine.core.dynamic_evaluator import CandidateAction, FastRiskAnalyzer
from cognitive_engine.core.dynamic_code_mutator import DynamicCodeMutator
from cognitive_engine.core.self_evolving_kernel import SelfEvolvingKernel


def test_fast_risk_analyzer_ranking():
    analyzer = FastRiskAnalyzer(risk_aversion=1.6, approval_threshold=0.20)

    # Option 1: Low impact, high risk -> Net = 0.3 - 1.6 * (0.8 * 1.5) = 0.3 - 1.92 = -1.62 (Reject)
    bad_opt = CandidateAction(
        action_id="bad",
        target_code_file="dummy.py",
        proposed_patch="def execute_task(): pass",
        variables={},
        impact_weight=0.3,
        risk_variance=0.8,
        irreversibility=0.5,
    )

    # Option 2: High impact, low risk -> Net = 0.9 - 1.6 * (0.1 * 1.1) = 0.9 - 0.176 = 0.724 (Approve)
    good_opt = CandidateAction(
        action_id="good",
        target_code_file="dummy.py",
        proposed_patch="def execute_task(): return 42",
        variables={},
        impact_weight=0.9,
        risk_variance=0.1,
        irreversibility=0.1,
    )

    ranked = analyzer.evaluate_and_rank([bad_opt, good_opt])
    assert len(ranked) == 1
    assert ranked[0].action_id == "good"


def test_self_evolving_kernel_successful_ast_mutation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("def execute_task():\n    return 'initial'\n")
        temp_path = f.name

    try:
        canary = lambda path: "return 'mutated_v2'" in open(path, "r", encoding="utf-8").read()
        kernel = SelfEvolvingKernel(canary_tester=canary)

        action = CandidateAction(
            action_id="upgrade_task",
            target_code_file=temp_path,
            proposed_patch="def execute_task():\n    return 'mutated_v2'\n",
            variables={},
            impact_weight=0.8,
            risk_variance=0.1,
            irreversibility=0.1,
        )

        applied = kernel.execute_autonomous_adaptation([action])
        assert applied is True

        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "mutated_v2" in content
        assert "initial" not in content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_self_evolving_kernel_canary_failure_rollback():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("def execute_task():\n    return 'safe_original'\n")
        temp_path = f.name

    try:
        # Canary rejects the change
        canary = lambda path: False
        kernel = SelfEvolvingKernel(canary_tester=canary)

        action = CandidateAction(
            action_id="risky_mutation",
            target_code_file=temp_path,
            proposed_patch="def execute_task():\n    return 'broken'\n",
            variables={},
            impact_weight=0.9,
            risk_variance=0.05,
            irreversibility=0.1,
        )

        applied = kernel.execute_autonomous_adaptation([action])
        assert applied is False

        # Verify rollback restored safe_original
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "safe_original" in content
        assert "broken" not in content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_curiosity_daemon_self_evolving_kernel_integration():
    from unittest.mock import MagicMock
    from cognitive_engine.agent.curiosity import CuriosityDaemon
    from cognitive_engine.core.autotelic import AutotelicExperimentLoop

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("class MockSynthesizer:\n    def get_search_depth(self):\n        return 3\n")
        temp_file = f.name

    try:
        class MockSynthesizer:
            max_depth = 3

        MockSynthesizer.__file__ = temp_file

        mock_engine = MagicMock()
        mock_kernel = MagicMock(spec=SelfEvolvingKernel)
        mock_kernel.execute_autonomous_adaptation.return_value = True

        mock_autotelic = MagicMock(spec=AutotelicExperimentLoop)
        synth_instance = MockSynthesizer()
        mock_autotelic.synthesizer = synth_instance

        daemon = CuriosityDaemon(
            engine=mock_engine,
            autotelic=mock_autotelic,
            kernel=mock_kernel,
        )

        daemon.telemetry_window.extend([False, False, False, False, False])
        result = daemon._attempt_self_optimization()

        assert result is not None
        assert result.get("applied") is True
        assert result.get("kernel_mutation") is True
        assert synth_instance.max_depth == 4
        assert mock_kernel.execute_autonomous_adaptation.call_count == 1
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
