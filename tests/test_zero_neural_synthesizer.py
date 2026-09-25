"""Unit test for ZeroNeuralSynthesizer learning from environmental feedback."""
import pytest
from cognitive_engine.agent.zero_neural_synthesizer import ZeroNeuralSynthesizer, ASTPolicyValueNet


def test_zero_neural_synthesizer_init():
    synth = ZeroNeuralSynthesizer()
    assert len(synth.tokens) > 20
    assert synth.net is not None


def test_zero_neural_synthesizer_sampling():
    synth = ZeroNeuralSynthesizer()
    code, seq, log_probs, _ = synth.generate_candidate_code(max_tokens=15)
    assert len(seq) >= 2
    assert log_probs.shape[0] == len(seq) - 1
    assert isinstance(code, str)


def test_zero_neural_synthesizer_sandbox():
    synth = ZeroNeuralSynthesizer()
    res_pass = synth.verify_in_sandbox("x = 5\ny = x * 2", ["y == 10"])
    assert res_pass == 1.0

    res_fail = synth.verify_in_sandbox("x = 5\ny = x * 2", ["y == 999"])
    assert res_fail == -1.0
