import ast
import pytest
from cognitive_engine.core.generator import LocalLLMGenerator, ASTConstrainedSynthesizer


def test_ast_constrained_synthesizer_validation():
    synth = ASTConstrainedSynthesizer()
    valid, err = synth.validate_ast("def foo(): return 42")
    assert valid is True
    assert err is None

    invalid, err_invalid = synth.validate_ast("def foo( return 42")
    assert invalid is False
    assert err_invalid is not None


def test_ast_constrained_synthesizer_clean_extraction():
    synth = ASTConstrainedSynthesizer()
    markdown_wrapped = """```python
def compute():
    return [x**2 for x in range(5)]
```"""
    cleaned = synth.extract_clean_code(markdown_wrapped)
    assert "```" not in cleaned
    valid, _ = synth.validate_ast(cleaned)
    assert valid is True


def test_generator_thread_pinning():
    gen = LocalLLMGenerator(n_threads=8)
    assert gen.n_threads == 8

    # Default should clamp to max 16
    gen_default = LocalLLMGenerator()
    assert gen_default.n_threads <= 16


def test_first_pass_ast_synthesizer_generic_prompts():
    gen = LocalLLMGenerator()
    # Test arbitrary prompt synthesis
    code = gen.generate_code("Compute hash table frequency distribution of items")
    assert isinstance(code, str)
    assert len(code) > 0
    # Must be 100% valid AST without syntax error
    tree = ast.parse(code)
    assert len(tree.body) > 0


def test_python_gbnf_grammar_specification():
    synth = ASTConstrainedSynthesizer()
    assert hasattr(synth, "PYTHON_GBNF")
    assert "root ::=" in synth.PYTHON_GBNF
    assert "statement" in synth.PYTHON_GBNF

