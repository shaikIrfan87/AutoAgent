import pytest
from cognitive_engine.agent.curiosity import CuriosityDaemon
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer
from cognitive_engine.core.autotelic import AutotelicExperimentLoop
from cognitive_engine.core.autonomous_decision import RiskScoreGovernor
from cognitive_engine.core.autonomous_governor import AutonomousGovernor
from cognitive_engine.core.meta_optimizer import MetaSelfOptimizer


class MockEngine:

    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.consolidation = True
        self.causal = type("MockCausal", (), {"rules": {"k1": [("causes", "k2", True)]}})()


def test_autonomous_decision_cycle_integration():
    synth = ProgramSynthesizer(max_depth=2)
    engine = MockEngine(synthesizer=synth)
    autotelic = AutotelicExperimentLoop(synthesizer=synth)
    governor = AutonomousGovernor(target_engine=engine, governor=RiskScoreGovernor(approval_threshold=0.10))
    optimizer = MetaSelfOptimizer()

    daemon = CuriosityDaemon(
        engine=engine,
        autotelic=autotelic,
        governor=governor,
        optimizer=optimizer,
        telemetry_window_size=4,
        failure_threshold_ratio=0.75,
        eval_window_size=3,
    )

    # Induce simulated failures to trigger self-update
    for _ in range(3):
        daemon.record_telemetry(False)
    daemon.record_telemetry(True)

    assert daemon.should_trigger_meta_optimization() is True

    # Execute autonomous step
    result = daemon._attempt_self_optimization()

    assert result["status"] == "applied"
    assert synth.max_depth == 3
    assert len(governor.audit_log) == 1
    assert governor.audit_log[0]["decision"].approved is True
