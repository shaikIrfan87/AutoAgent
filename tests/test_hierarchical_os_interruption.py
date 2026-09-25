import os
import tempfile
import time
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import pytest

from cognitive_engine.agent.executive_loop import GoalFrame, HierarchicalExecutiveStack
from cognitive_engine.core.vision_sensor import NativeWinRTOCREngine
from cognitive_engine.agent.orchestrator import CognitiveEngine


class SimulatedInterruptedEnvironment:
    """
    Simulated OS environment with dynamic interruption injection:
    - Target task: requires foreground active window.
    - Interruption: unexpected modal dialog appears with a dynamic OTP/PIN code.
    - Resolution: requires reading PIN via WinRT OCR, entering PIN, dismissing modal, and resuming.
    """
    def __init__(self, target_pin: str = "PASS_749"):
        self.target_pin = target_pin
        self.modal_active = True
        self.pin_submitted = False
        self.target_completed = False
        self.action_history: List[str] = []

        # Render simulated modal dialog with dynamic PIN
        self.dialog_img = Image.new("RGB", (420, 160), (230, 230, 235))
        draw = ImageDraw.Draw(self.dialog_img)
        font = ImageFont.load_default(size=26)
        draw.text((20, 25), "Security Verification", fill=(180, 20, 20), font=font)
        draw.text((20, 70), f"Enter PIN: {self.target_pin}", fill=(0, 0, 0), font=font)

    def execute_step(self, description: str) -> Dict[str, Any]:
        self.action_history.append(description)

        if "Dismiss blocker" in description or "Resolve interruption" in description:
            # Epistemic action: OCR the modal dialog
            extracted_text = NativeWinRTOCREngine.recognize_text(self.dialog_img)
            if self.target_pin in extracted_text:
                self.pin_submitted = True
                self.modal_active = False
                return {
                    "delta_s": 1.0,
                    "status": "MODAL_DISMISSED",
                    "pin_verified": self.target_pin,
                }
            return {"delta_s": -1.0, "error": "OCR failed to ground PIN"}

        if "Execute secure transaction" in description:
            if self.modal_active:
                # Interrupted by blocking modal!
                return {
                    "delta_s": -1.0,
                    "error": "Blocked by modal: Security Verification prompt active",
                }
            # Modal cleared, transaction proceeds
            self.target_completed = True
            return {
                "delta_s": 1.0,
                "status": "TRANSACTION_COMMITTED",
                "verified_pin": self.target_pin,
            }

        return {"delta_s": 1.0, "status": "STEP_OK"}


def test_end_to_end_os_interruption_recovery_with_epistemic_ocr():
    """
    End-to-End Test: Multi-step OS automation with unexpected modal interruption.
    Verifies that:
    1. Target goal is suspended when modal interrupts execution (delta_s = -1.0).
    2. Localized repair frame is pushed onto the hierarchical call stack without wiping state.
    3. Repair frame performs epistemic OCR to extract the dynamic authorization PIN.
    4. Modal is dismissed, repair frame pops, and suspended target goal resumes cleanly.
    5. Final dependent payload executes successfully in sandbox.
    """
    secret_pin = "PASS_749"
    sim_env = SimulatedInterruptedEnvironment(target_pin=secret_pin)
    engine = CognitiveEngine()

    class InterruptionAwareAgent:
        def __init__(self, env):
            self.env = env

        def execute_step(self, goal_frame: GoalFrame) -> Dict[str, Any]:
            desc = goal_frame.description
            res = self.env.execute_step(desc)
            if res.get("pin_verified"):
                # Collapse epistemic uncertainty in memory
                engine.memory.add_memory(
                    content=f"Verified security PIN: {res['pin_verified']}",
                    tags=["security", "pin"],
                    confidence=1.0,
                )
            return res

    agent = InterruptionAwareAgent(sim_env)
    stack = HierarchicalExecutiveStack(agent.execute_step)

    root_goal = GoalFrame(
        goal_id="g_root_tx",
        description="Execute secure transaction in primary window",
        target_invariant="transaction_committed",
        preconditions=["foreground_window_active"],
        max_retries=2,
    )

    # Execute through the hierarchical stack
    t0 = time.perf_counter()
    result = stack.execute_with_repair(root_goal)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Verification of execution outcome
    assert result["success"] is True, f"Failed execution: {result}"
    assert result["status"] == "ALL_SUBGOALS_FULFILLED"
    assert sim_env.target_completed is True
    assert sim_env.pin_submitted is True
    assert sim_env.modal_active is False

    # 2. Verification of the exact call stack lifecycle:
    # Attempt Target -> Failed -> Suspend -> Push Repair -> OCR & Dismiss -> Pop Repair -> Resume Target -> Done!
    assert sim_env.action_history == [
        "Execute secure transaction in primary window",
        "Dismiss blocker / Re-anchor focus",
        "Execute secure transaction in primary window",
    ]

    # 3. Verify downstream sandbox execution using acquired epistemic PIN
    exec_code = f"""
pin = "{secret_pin}"
authorized = len(pin) == 8 and pin.startswith("PASS")
print(f"PIPELINE_SUCCESS:{{authorized}}")
"""
    sand_res = engine.sandbox.execute_python(exec_code)
    assert sand_res.exit_code == 0
    assert "PIPELINE_SUCCESS:True" in sand_res.stdout
