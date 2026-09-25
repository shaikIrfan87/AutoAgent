import numpy as np
import pytest

from cognitive_engine.core.embodied_grounding import EmbodiedGUIGroundingEngine, VisualGroundingReceipt
from cognitive_engine.core.vision_sensor import ScreenPerceptionEngine
from cognitive_engine.core.os_actuator import OSActuator


class MockVisionSensor(ScreenPerceptionEngine):
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self._delta = delta
        self._f_pre = np.zeros((100, 100, 3), dtype=np.uint8)
        self._f_post = np.ones((100, 100, 3), dtype=np.uint8) * 200

    def capture_frame(self):
        return self._f_post, (100, 100)

    def wait_for_settle(self, max_wait_sec=0.35, interval_sec=0.04):
        return self._delta

    def compute_screen_delta(self, current_frame):
        return self._delta


def test_embodied_gui_grounding_success():
    """Verify closed-loop visual grounding: Delta S_visual == +1.0 when visual shift settles."""
    mock_eyes = MockVisionSensor(delta=0.08)
    hands = OSActuator()
    engine = EmbodiedGUIGroundingEngine(eyes=mock_eyes, hands=hands, change_threshold=0.01)

    receipt = engine.execute_and_ground_action(
        action_fn=lambda h: {"success": True},
        assertion_fn=lambda f_pre, f_post: True,
        target_coords=(100, 200),
        action_name="mock_click",
    )

    assert isinstance(receipt, VisualGroundingReceipt)
    assert receipt.delta_s == 1.0
    assert receipt.visual_shift >= 0.01
    assert receipt.settled


def test_embodied_gui_grounding_assertion_rejection():
    """Verify rejection: Delta S_visual == -1.0 when assertion_fn fails despite visual shift."""
    mock_eyes = MockVisionSensor(delta=0.08)
    hands = OSActuator()
    engine = EmbodiedGUIGroundingEngine(eyes=mock_eyes, hands=hands, change_threshold=0.01)

    receipt = engine.execute_and_ground_action(
        action_fn=lambda h: {"success": True},
        assertion_fn=lambda f_pre, f_post: False,  # Assertion fails
        target_coords=(100, 200),
    )

    assert receipt.delta_s == -1.0


def test_embodied_gui_grounding_action_out_of_bounds_containment():
    """Verify containment: out-of-bounds click returns Delta S_visual == -1.0 with error recorded."""
    mock_eyes = MockVisionSensor()
    hands = OSActuator()
    engine = EmbodiedGUIGroundingEngine(eyes=mock_eyes, hands=hands)

    # Click beyond screen boundaries
    receipt = engine.execute_and_ground_action(
        action_fn=lambda h: h.click_at(999999, 999999),
        target_coords=(999999, 999999),
        action_name="out_of_bounds_click",
    )

    assert receipt.delta_s == -1.0
    assert receipt.error is not None
    assert "bounds" in receipt.error.lower()
