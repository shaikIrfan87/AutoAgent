import torch
import pytest
from cognitive_engine.core.unified_rlcd import (
    UnifiedRLCDEngine,
    MacroActionCandidate,
)


def test_macro_gating_risk_and_approval():
    engine = UnifiedRLCDEngine(approval_threshold=0.60, risk_weight=1.0)

    # Low-risk ephemeral action with high reversibility
    cand_safe = MacroActionCandidate(
        action_id="act_safe_01",
        action_type="induct_code",
        target="virtual_scratchpad",
        payload="def solution(x): return x * 2",
        complexity=0.2,
        reversibility=1.0,  # 1.0 = fully reversible ephemeral sandbox
    )

    # High-risk irreversible action
    cand_danger = MacroActionCandidate(
        action_id="act_danger_01",
        action_type="system_telemetry",
        target="host_kernel",
        payload="os.system('wipe')",
        complexity=0.9,
        reversibility=0.0,  # 0.0 = completely irreversible
    )

    best_cand, best_p, best_net, approved = engine.select_macro_action([cand_safe, cand_danger])
    assert best_cand.action_id == "act_safe_01"
    assert 0.0 <= best_p <= 1.0
    assert isinstance(best_net, float)


def test_brier_calibration_alignment():
    engine = UnifiedRLCDEngine()

    cand1 = MacroActionCandidate(
        action_id="c1",
        action_type="induct_code",
        target="v1",
        payload="test",
        complexity=0.3,
        reversibility=0.9,
    )
    cand2 = MacroActionCandidate(
        action_id="c2",
        action_type="web_search",
        target="v2",
        payload="query",
        complexity=0.5,
        reversibility=0.7,
    )

    # Update calibration with positive ground truth delta_s = +1.0
    loss_pos = engine.update_macro_calibration(cand1, cand2, delta_s=1.0, tau=1.0, alpha=1.0)
    assert loss_pos >= 0.0

    # Update calibration with negative ground truth delta_s = -1.0
    loss_neg = engine.update_macro_calibration(cand2, cand1, delta_s=-1.0, tau=1.0, alpha=1.0)
    assert loss_neg >= 0.0


def test_micro_dpo_trajectory_distillation():
    engine = UnifiedRLCDEngine(gamma_occam=0.01)

    assertions = ["solution(2) == 4", "solution(5) == 10"]
    code, delta_s, micro_loss = engine.run_micro_trajectory_distillation(assertions, num_rollouts=4, beta=0.1)

    assert isinstance(code, str)
    assert delta_s in (-1.0, 0.0, 1.0)
    assert micro_loss >= 0.0
