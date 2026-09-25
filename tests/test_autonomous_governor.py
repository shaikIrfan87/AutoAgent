import pytest
from cognitive_engine.core.autonomous_decision import MutationProposal, RiskScoreGovernor
from cognitive_engine.core.autonomous_governor import AutonomousGovernor


class MockEngineSubsystem:

    def __init__(self):
        self.search_depth = 2
        self.entropy_threshold = 0.45


class MockEngine:

    def __init__(self):
        self.synthesizer = MockEngineSubsystem()


def test_autonomous_risk_rejection():
    engine = MockEngine()
    gov = AutonomousGovernor(engine)

    # Proposal with high failure risk and irreversibility
    risky_proposal = MutationProposal(
        proposal_id="prop_01",
        target_subsystem="synthesizer",
        description="Aggressive heuristic jump",
        patch_code=None,
        parameters_delta={"search_depth": 8},
        expected_accuracy_gain=0.1,
        expected_speedup=0.0,
        failure_probability=0.8,
        reversibility_score=0.2,
    )

    result = gov.process_and_execute(risky_proposal, canary_validator=lambda obj: True)
    assert result["status"] == "rejected"
    assert engine.synthesizer.search_depth == 2  # Left unchanged


def test_autonomous_approval_and_canary_rollback():
    engine = MockEngine()
    gov = AutonomousGovernor(engine)

    # High-utility proposal, but canary test detects a regression
    promising_but_broken_proposal = MutationProposal(
        proposal_id="prop_02",
        target_subsystem="synthesizer",
        description="Increase search depth",
        patch_code=None,
        parameters_delta={"search_depth": 4},
        expected_accuracy_gain=0.8,
        expected_speedup=0.2,
        failure_probability=0.05,
        reversibility_score=1.0,
    )

    # Canary function simulates a failure under search_depth == 4
    canary = lambda obj: obj.search_depth < 4

    result = gov.process_and_execute(promising_but_broken_proposal, canary_validator=canary)
    assert result["status"] == "canary_failed_rolled_back"
    assert engine.synthesizer.search_depth == 2  # Successfully rolled back


def test_autonomous_approval_and_canary_success():
    engine = MockEngine()
    gov = AutonomousGovernor(engine)

    good_proposal = MutationProposal(
        proposal_id="prop_03",
        target_subsystem="synthesizer",
        description="Safe parameter optimization",
        patch_code=None,
        parameters_delta={"search_depth": 3},
        expected_accuracy_gain=0.7,
        expected_speedup=0.3,
        failure_probability=0.05,
        reversibility_score=1.0,
    )

    canary = lambda obj: obj.search_depth == 3
    result = gov.process_and_execute(good_proposal, canary_validator=canary)
    assert result["status"] == "applied"
    assert engine.synthesizer.search_depth == 3
