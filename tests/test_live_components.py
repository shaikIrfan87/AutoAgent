"""
Unit tests for LocalLLMGenerator, LiveWebSearch, and dynamic autonomous interaction loop.
"""

import pytest
from cognitive_engine.core.generator import LocalLLMGenerator
from cognitive_engine.core.search import LiveWebSearch
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_generator_code_cleaning():
    gen = LocalLLMGenerator()
    raw_markdown = "```python\ndef solution():\n    return 42\n```"
    cleaned = gen._clean_code(raw_markdown)
    assert cleaned == "def solution():\n    return 42"
    assert "```" not in cleaned


def test_generator_inductive_synthesis():
    gen = LocalLLMGenerator()
    code = gen.generate_code("IO: [(1, 2), (2, 3), (3, 4)]")
    assert "def solution" in code
    assert "x + 1" in code


def test_generator_zero_fallback_failure_registration():
    gen = LocalLLMGenerator()
    code = gen.generate_code("unsupported task without assertions or weights")
    assert "raise NotImplementedError" in code


def test_live_web_search_offline_resilience():
    # Should not raise an exception even under network failure
    res = LiveWebSearch.search("non_existent_random_token_query_99999", timeout=3.0)
    assert isinstance(res, str)
    assert len(res) > 0


def test_orchestrator_dynamic_interact_loop():
    engine = CognitiveEngine()
    try:
        # Dynamic query without hardcoded strings
        query = "Calculate kinetic energy for 1500kg car at 28 m/s"
        res = engine.interact(query)
        assert "Grounded Execution" in res
        assert "Joules" in res

        # Fast recall verification
        res_fast = engine.interact("kinetic energy of 1500kg car at 28 m/s")
        assert "System 1 Fast Recall" in res_fast
    finally:
        engine.sandbox.close()
        engine.consolidation.close()
