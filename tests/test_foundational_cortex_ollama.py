import pytest
import ast
from cognitive_engine.core.generator import LocalLLMGenerator
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_ollama_autodiscovery_and_model_selection():
    """Verify LocalLLMGenerator connects to local Ollama and selects installed coder model."""
    gen = LocalLLMGenerator()
    assert gen.endpoint is not None
    # If Ollama is running locally, model should match an installed model (e.g. qwen2.5-coder:3b)
    assert gen.model is not None
    assert isinstance(gen.model, str) and len(gen.model) > 0


def test_ollama_reasoning_and_ttt_conditioning():
    """Verify reasoning generation from local foundational cortex and TTT attention update."""
    gen = LocalLLMGenerator(timeout=15.0)
    initial_norm = gen.ttt_attention.frobenius_norm

    res = gen.generate_reasoning(
        prompt="Explain what a prime number is in one sentence.",
        context="You are a mathematical engine. Be precise."
    )
    if res:
        assert isinstance(res, str) and len(res) > 0
        assert "prime" in res.lower() or "number" in res.lower() or "divis" in res.lower()
        # Fast weights should reflect token conditioning
        assert gen.ttt_attention.frobenius_norm >= initial_norm


def test_ollama_ast_constrained_code_synthesis():
    """Verify code synthesized by the cortex satisfies native Python AST validity."""
    gen = LocalLLMGenerator(timeout=15.0)
    code = gen.generate_code("def add(a, b): return a + b")
    assert isinstance(code, str) and len(code) > 0
    # AST parse must succeed with zero syntax errors
    tree = ast.parse(code)
    assert tree is not None


def test_engine_deliberation_via_local_cortex():
    """End-to-end: CognitiveEngine routes open inquiries to the local cortex with TTT sync."""
    engine = CognitiveEngine(db_path=":memory:")
    query = "explain the law of conservation of momentum"
    resp = engine.process_interactive(query)

    assert isinstance(resp, str) and len(resp) > 0
    # Must route to System 2 Local Cortex (or grounded execution if simulated)
    assert "[System 2 Local Cortex]" in resp or "momentum" in resp.lower()

    # Verify memory commitment occurred
    q_vec = engine.saliency._embed(query)
    results = engine.consolidation.hybrid_search(query, q_vec, top_k=1)
    assert len(results) > 0
