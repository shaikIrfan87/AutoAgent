import pytest
from cognitive_engine.core.world_model import (
    MentalSimulator,
    EnvironmentState,
    StateTransition,
    SimulationPrediction,
)


def test_environment_state_creation():
    state = EnvironmentState(memory_mb=12.5, open_file_descriptors=1, modified_variables=["x", "y"])
    d = state.to_dict()
    assert d["memory_mb"] == 12.5
    assert d["open_file_descriptors"] == 1
    assert "x" in d["modified_variables"]


def test_predict_transition_variables_and_duration():
    sim = MentalSimulator()
    code = """
a = 10
b = 20
c = a + b
for i in range(10):
    c += i
"""
    transition = sim.predict_transition(code)
    assert not transition.divergent
    assert transition.delta_s == 1.0
    assert "a" in transition.projected_state.modified_variables
    assert "b" in transition.projected_state.modified_variables
    assert "c" in transition.projected_state.modified_variables
    assert transition.projected_state.projected_duration_ms > 2.0


def test_predict_transition_memory_buffer_estimation():
    sim = MentalSimulator(energy_threshold=1.5)
    # Allocating 300MB buffer
    code = "buf = bytearray(300 * 1024 * 1024)"
    transition = sim.predict_transition(code)
    assert transition.projected_state.memory_mb >= 300.0
    assert "buf" in transition.projected_state.modified_variables
    assert transition.energy_score > 0.0


def test_energy_compatibility_rejection_on_divergence():
    # Low energy threshold to force rejection on large resource divergence
    sim = MentalSimulator(energy_threshold=0.5)
    code = "buf = bytearray(400 * 1024 * 1024)"
    pred = sim.simulate(code)
    assert pred.safe is False
    assert pred.predicted_delta_s == -1.0
    assert pred.energy_score > 0.5
    assert "energy" in pred.rejection_reason.lower()


def test_destructive_operation_energy_and_blast_radius():
    sim = MentalSimulator()
    code = "import shutil\nshutil.rmtree('/tmp/cache')"
    pred = sim.simulate(code)
    assert pred.safe is False
    assert pred.risk_score >= 0.70
    assert any("destructive_filesystem" in op for op in pred.blast_radius)
