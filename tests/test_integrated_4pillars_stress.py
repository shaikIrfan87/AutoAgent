import pytest
from cognitive_engine.core.dsl import UNARY_PRIMITIVES, to_grid
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer, SynthesizedProgram
from cognitive_engine.core.autotelic import AutotelicExperimentLoop, EpistemicGap
from cognitive_engine.core.compression import ProgramCompressor
from cognitive_engine.core.meta_optimizer import MetaSelfOptimizer
from cognitive_engine.agent.curiosity import CuriosityDaemon


class MockUnifiedEngine:
    def __init__(self):
        self.consolidation = True
        self.causal = type(
            "MockCausal",
            (),
            {"rules": {"gap_unbound": [("causes", "unverified_state", True)]}},
        )()


def test_four_pillar_full_lifecycle_stress():
    # Setup infrastructure
    synthesizer = ProgramSynthesizer(max_depth=2)
    autotelic = AutotelicExperimentLoop(synthesizer=synthesizer, uncertainty_threshold=0.30)
    compressor = ProgramCompressor(min_frequency=2, max_macro_len=2)
    optimizer = MetaSelfOptimizer()
    engine = MockUnifiedEngine()

    daemon = CuriosityDaemon(
        engine=engine,
        autotelic=autotelic,
        compressor=compressor,
        compression_batch_size=2,
        optimizer=optimizer,
        telemetry_window_size=4,
        failure_threshold_ratio=0.75,
        eval_window_size=3,
        degradation_tolerance=0.10,
    )

    # -------------------------------------------------------------
    # CYCLE 1: Pillar 1 (Synthesis) & Pillar 2 (Autotelic Inquiry)
    # -------------------------------------------------------------
    gap1 = EpistemicGap(gap_id="g1", concept="rot_flip_transform", uncertainty=0.8)
    task1 = [(to_grid(((1, 0), (0, 0))), to_grid(((0, 0), (0, 1))))]
    res1 = autotelic.run_experiment(gap1, custom_examples=task1)

    assert res1.success is True
    assert res1.program is not None
    daemon.record_telemetry(res1.success)

    # -------------------------------------------------------------
    # CYCLE 2: Pillar 3 (AST Subtree Mining & Macro Consolidation)
    # -------------------------------------------------------------
    shared_steps = [("rot90", ()), ("flip_h", ())]
    p1 = SynthesizedProgram(steps=shared_steps + [("rot180", ())], fn=lambda g: g, code="p1")
    p2 = SynthesizedProgram(steps=shared_steps, fn=lambda g: g, code="p2")

    autotelic.experiment_history.clear()
    autotelic.experiment_history.append(
        type("Exp", (), {"success": True, "program": p1})()
    )
    autotelic.experiment_history.append(
        type("Exp", (), {"success": True, "program": p2})()
    )

    mined = compressor.mine_abstractions([p1, p2])
    assert len(mined) >= 1
    added_macros = compressor.consolidate_into_dsl(mined)
    assert added_macros >= 1
    assert mined[0].name in UNARY_PRIMITIVES

    # -------------------------------------------------------------
    # CYCLE 3: Pillar 4 (Telemetry Trigger & Hot-Swap Mutation)
    # -------------------------------------------------------------
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)
    daemon.record_telemetry(True)

    assert daemon.should_trigger_meta_optimization() is True

    opt_res = daemon._attempt_self_optimization()
    assert opt_res is not None
    assert opt_res.applied is True
    assert synthesizer.max_depth == 3
    assert daemon.is_evaluating_optimization is True

    # -------------------------------------------------------------
    # CYCLE 4: Automated Degradation Rollback Verification
    # -------------------------------------------------------------
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)
    daemon.record_telemetry(False)

    rolled_back = daemon.check_post_optimization_degradation()
    assert rolled_back is True
    assert synthesizer.max_depth == 2
    assert daemon.is_evaluating_optimization is False
    assert len(daemon.post_opt_window) == 0
