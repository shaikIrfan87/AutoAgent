import pytest
from cognitive_engine.agent.zero_neural_synthesizer import ZeroNeuralSynthesizer

@pytest.fixture
def synth():
    return ZeroNeuralSynthesizer(hidden_dim=128, beta_length_penalty=0.005)

def test_induction_parity_conditional(synth):
    """Test Level 1: Conditional branching and modulo parity (f(x) = 1 if x % 2 == 0 else 0)"""
    assertions = [
        "f(0) == 1",
        "f(1) == 0",
        "f(2) == 1",
        "f(3) == 0",
        "f(4) == 1",
        "f(10) == 1"
    ]
    res = synth.learn_task(assertions, max_episodes=500)
    assert res["status"] == "discovered", f"Failed conditional parity induction: {res}"

def test_induction_two_variable_nonlinear(synth):
    """Test Level 2: Multi-argument nonlinear synthesis (f(x, y) = x * y + 1)"""
    assertions = [
        "f(1, 2) == 3",
        "f(2, 3) == 7",
        "f(0, 5) == 1",
        "f(3, 3) == 10"
    ]
    res = synth.learn_task(assertions, max_episodes=600)
    assert res["status"] == "discovered", f"Failed multi-variable induction: {res}"

def test_induction_discrete_accumulator_loop(synth):
    """Test Level 3: Iterative loop / accumulator (f(x) = sum(range(x + 1)))"""
    assertions = [
        "f(1) == 1",
        "f(2) == 3",
        "f(3) == 6",
        "f(4) == 10"
    ]
    res = synth.learn_task(assertions, max_episodes=1000)
    assert res["status"] == "discovered", f"Failed accumulator induction: {res}"
