import pytest
from cognitive_engine.agent.curiosity import CuriosityDaemon
from cognitive_engine.core.autotelic import AutotelicExperimentLoop, EpistemicGap
from cognitive_engine.core.meta_optimizer import MetaSelfOptimizer
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer


class MockEngine:
    def __init__(self):
        self.consolidation = True
        self.causal = type("MockCausal", (), {"rules": {}})()


def test_telemetry_threshold_triggers_meta_optimization():
    synthesizer = ProgramSynthesizer(max_depth=2)
    autotelic = AutotelicExperimentLoop(synthesizer=synthesizer)
    optimizer = MetaSelfOptimizer()
    engine = MockEngine()

    daemon = CuriosityDaemon(
        engine=engine,
        autotelic=autotelic,
        optimizer=optimizer,
        telemetry_window_size=4,
        failure_threshold_ratio=0.75,
    )

    # Simulate 3 failures and 1 success (75% failure rate)
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)
    daemon.record_telemetry(True)

    assert daemon.should_trigger_meta_optimization() is True

    # Trigger optimization pass
    res = daemon._attempt_self_optimization()
    assert res is not None
    assert res.applied is True
    assert synthesizer.max_depth == 3
    assert len(daemon.telemetry_window) == 0  # Telemetry reset after mutation
