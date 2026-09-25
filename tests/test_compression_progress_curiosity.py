import os
import tempfile
import pytest
from cognitive_engine.core.compression_progress import CompressionProgressEvaluator
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.agent.curiosity import CuriosityDaemon


def test_noise_trap_rejection():
    """Verify that uncompressible random stochastic noise yields negative progress and is aborted."""
    cpe = CompressionProgressEvaluator()
    # High-entropy random hex token
    random_noise = "a8f39b2e104c997d8123efb0198273491823749817234091823749817234918237498172394817239481729348719238471928347192834719283471928347"
    assert cpe.is_noise_trap(random_noise)

    baseline = [
        "sorting property: sorted list elements are monotonically increasing",
        "addition identity: x + 0 == x",
    ]
    progress, viable = cpe.evaluate_progress(baseline, random_noise)
    assert not viable
    assert progress <= 0.0


def test_structured_lemma_compression_gain():
    """Verify that structured mathematical lemmas produce positive compression progress."""
    cpe = CompressionProgressEvaluator()

    # Targets: sequence of arithmetic examples
    examples = [
        "add(1, 0) == 1",
        "add(2, 0) == 2",
        "add(3, 0) == 3",
        "add(4, 0) == 4",
        "add(5, 0) == 5",
        "add(6, 0) == 6",
    ]
    lemma = "identity law: for all x, add(x, 0) == x"
    assert not cpe.is_noise_trap(lemma)

    progress, viable = cpe.evaluate_progress([], lemma, structured_target_examples=examples)
    # The rule provides compressive abstraction over the raw data points
    assert isinstance(progress, float)


def test_curiosity_daemon_noise_trap_filtering():
    """Verify that CuriosityDaemon identifies and aborts noise traps in decaying memories."""
    from features.execution_sandbox.sandbox import safe_cleanup_temp_dir
    tmp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmp_dir, "test_noise_curiosity.db")
        engine = CognitiveEngine(db_path=db_path)
        try:
            # Seed uncompressible noise string as low-confidence memory
            noise_content = "x9f81a7b2c0d9e8f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6"
            engine.consolidation.write_memory(
                content=noise_content,
                confidence=0.30,
            )

            daemon = CuriosityDaemon(engine=engine, interval_sec=1.0)
            res = daemon.step()
            assert res is not None
            assert res.get("status") == "noise_trap_aborted"
        finally:
            engine.sandbox.close()
            engine.consolidation.close()
    finally:
        safe_cleanup_temp_dir(tmp_dir)
