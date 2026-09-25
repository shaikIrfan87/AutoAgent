"""
Closed-Loop Sensory-Motor GUI Grounding Engine.
Binds ScreenPerceptionEngine visual feedback with OSActuator physical input events
to compute empirical physical environmental grounding: Delta S_visual.
"""
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .vision_sensor import ScreenPerceptionEngine
from .os_actuator import OSActuator


@dataclass
class VisualGroundingReceipt:
    delta_s: float
    visual_shift: float
    settled: bool
    action_type: str
    target_coords: Tuple[int, int]
    error: Optional[str] = None


class EmbodiedGUIGroundingEngine:
    """
    Executes closed-loop sensory-motor actions on external operating system interfaces,
    settles UI animations, and verifies physical state transitions Delta S_visual.
    """

    def __init__(
        self,
        eyes: Optional[ScreenPerceptionEngine] = None,
        hands: Optional[OSActuator] = None,
        change_threshold: float = 0.005,
    ):
        self.eyes = eyes or ScreenPerceptionEngine()
        self.hands = hands or OSActuator()
        self.change_threshold = change_threshold

    def execute_and_ground_action(
        self,
        action_fn: Callable[[OSActuator], Any],
        assertion_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
        target_coords: Tuple[int, int] = (0, 0),
        action_name: str = "custom_action",
    ) -> VisualGroundingReceipt:
        """
        Executes physical action through OSActuator, monitors pixel shift settling,
        and computes Delta S_visual in {+1.0, -1.0}.
        """
        # 1. Capture pre-action visual baseline
        frame_pre, _ = self.eyes.capture_frame()

        # 2. Dispatch physical sensory-motor actuation
        try:
            action_res = action_fn(self.hands)
            if isinstance(action_res, dict) and not action_res.get("success", True):
                return VisualGroundingReceipt(
                    delta_s=-1.0,
                    visual_shift=0.0,
                    settled=False,
                    action_type=action_name,
                    target_coords=target_coords,
                    error=action_res.get("error", "Action dispatch failed"),
                )
        except Exception as e:
            return VisualGroundingReceipt(
                delta_s=-1.0,
                visual_shift=0.0,
                settled=False,
                action_type=action_name,
                target_coords=target_coords,
                error=str(e),
            )

        # 3. Adaptive visual settling loop
        settle_shift = self.eyes.wait_for_settle(max_wait_sec=0.35, interval_sec=0.04)

        # 4. Capture post-action frame and calculate empirical delta
        frame_post, _ = self.eyes.capture_frame()
        final_shift = self.eyes.compute_screen_delta(frame_post)
        visual_delta = max(settle_shift, final_shift)

        # 5. Evaluate empirical grounding Delta S_visual
        condition_passed = True
        if assertion_fn is not None:
            try:
                condition_passed = assertion_fn(frame_pre, frame_post)
            except Exception:
                condition_passed = False

        if visual_delta >= self.change_threshold and condition_passed:
            delta_s = 1.0
        else:
            delta_s = -1.0

        return VisualGroundingReceipt(
            delta_s=delta_s,
            visual_shift=visual_delta,
            settled=True,
            action_type=action_name,
            target_coords=target_coords,
        )
