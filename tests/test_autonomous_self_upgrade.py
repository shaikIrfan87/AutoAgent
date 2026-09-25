import pytest
import threading
from cognitive_engine.agent.orchestrator import CognitiveEngine, check_introspection
from cognitive_engine.core.dynamic_evaluator import FastRiskAnalyzer


def test_cognitive_engine_self_development_and_upgrade_wiring():
    engine = CognitiveEngine(db_path=":memory:")
    try:
        assert hasattr(engine, "evolving_kernel")
        assert engine.evolving_kernel is not None
        assert engine.curiosity.kernel is engine.evolving_kernel
        assert hasattr(engine, "core_evolver")
        assert engine.core_evolver is not None
        assert hasattr(engine, "upgrade_lock")

        result = engine.self_develop_and_upgrade()
        assert "self_development" in result
        assert "self_upgrade" in result
    finally:
        engine.sandbox.close()
        engine.consolidation.close()


def test_introspection_self_development_goal():
    ans = check_introspection("what is your main goal")
    assert ans is not None
    assert "self-development" in ans.lower()
    assert "self-upgrading" in ans.lower()

    ans2 = check_introspection("can you self develop and upgrade itself")
    assert ans2 is not None
    assert "self-development" in ans2.lower()


def test_dynamic_lambda_adaptation():
    analyzer = FastRiskAnalyzer(risk_aversion=1.6)
    # Consecutive failures should drive risk aversion (lambda) upwards
    for _ in range(5):
        analyzer.adapt_risk_aversion(success=False)
    assert analyzer.risk_aversion > 1.6

    # Consecutive successes flush failure window and relax risk aversion
    for _ in range(60):
        analyzer.adapt_risk_aversion(success=True)
    assert analyzer.risk_aversion < 1.6


def test_upgrade_lock_yield():
    engine = CognitiveEngine(db_path=":memory:")
    try:
        # Simulate active architectural mutation holding upgrade_lock
        with engine.upgrade_lock:
            # curiosity.step() must yield None without hanging
            res = engine.curiosity.step()
            assert res is None
    finally:
        engine.sandbox.close()
        engine.consolidation.close()


def test_cross_domain_curriculum_spawner_integration():
    engine = CognitiveEngine(db_path=":memory:")
    try:
        assert hasattr(engine, "curriculum")
        assert engine.curriculum is not None
        assert hasattr(engine.curriculum, "spawner")

        # Verify domain spawner can evaluate domains
        next_domain = engine.curriculum.spawner.evaluate_and_select_next_domain()
        assert "domain_id" in next_domain
        assert "beginner_topic" in next_domain
    finally:
        engine.sandbox.close()
        engine.consolidation.close()


def test_saliency_allowlist_and_codebase_audit_routing():
    engine = CognitiveEngine(db_path=":memory:")
    try:
        # Saliency pass for allowlist tokens
        for token in ["yes", "no", "ok", "help", "test", "status", "1"]:
            assert engine.saliency.evaluate(token).pass_filter is True

        # Diagnostic audit routing check
        resp = engine.process_interactive("analysis this entire project")
        assert "[System 2 Live Codebase Audit]" in resp
        assert "Working Memory Store" in resp
    finally:
        engine.sandbox.close()
        engine.consolidation.close()


def test_structured_tool_dispatch_in_cortex():
    from unittest.mock import MagicMock
    from cognitive_engine.core.generator import LocalLLMGenerator

    generator = LocalLLMGenerator()
    assert hasattr(generator, "generate_with_tools")

    engine = CognitiveEngine(db_path=":memory:")
    try:
        # Mock generator returning a tool call for audit_codebase on non-keyword prompt
        mock_gen = MagicMock()
        mock_gen.generate_with_tools.return_value = {
            "tool_calls": [{"function": {"name": "audit_codebase", "arguments": {}}}],
            "content": "",
        }
        engine.generator = mock_gen

        resp = engine.process_interactive("give me an overall assessment of technical failures")
        assert "[System 2 Live Codebase Audit]" in resp

        # Mock generator returning run_sandbox_code
        mock_gen.generate_with_tools.return_value = {
            "tool_calls": [{"function": {"name": "run_sandbox_code", "arguments": {"code": "print(40 + 2)"}}}],
            "content": "",
        }
        resp2 = engine.process_interactive("evaluate forty plus two in python")
        assert "[System 2 Sandbox Tool Output]" in resp2
        assert "42" in resp2
    finally:
        engine.sandbox.close()
        engine.consolidation.close()
