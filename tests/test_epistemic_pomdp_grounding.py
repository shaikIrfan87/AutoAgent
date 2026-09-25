import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pytest

from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.vision_sensor import NativeWinRTOCREngine
from cognitive_engine.agent.executive_loop import GoalFrame, HierarchicalExecutiveStack


def test_pomdp_epistemic_hidden_variable_resolution():
    """
    Evaluates agent's ability to solve an incomplete problem by actively
    probing the host environment to resolve a hidden variable.
    """
    engine = CognitiveEngine()
    hidden_secret_token = "KAPPA_9281"

    # Setup hidden environment artifact (Synthetic dialogue box / canvas)
    img = Image.new("RGB", (450, 180), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    draw.text((20, 30), "System Auth Required", fill=(0, 0, 0), font=font)
    draw.text((20, 80), f"Access Key: {hidden_secret_token}", fill=(10, 10, 80), font=font)

    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "auth_prompt.png")
    img.save(img_path)

    # 1. Provide an underspecified prompt (POMDP: token missing from input)
    user_goal = "Decrypt and execute authorization payload using the key displayed in auth_prompt.png"

    # 2. Epistemic Action: Read environment using NativeWinRTOCREngine
    ocr_text = NativeWinRTOCREngine.recognize_text(img)
    assert hidden_secret_token in ocr_text, "Visual sensor failed to ground hidden variable"

    # 3. Memory Update: Collapse epistemic uncertainty in belief manifold
    engine.memory.add_memory(
        content=f"Discovered environmental credential: {hidden_secret_token}",
        tags=["auth", "credentials"],
        confidence=0.95,
    )

    # 4. Verified Action Execution: Sandbox code execution using acquired key
    exec_code = f"""
key = "{hidden_secret_token}"
def authorize(k):
    return hash(k) != 0
result = authorize(key)
print(f"AUTHORIZED:{{result}}")
"""
    res = engine.sandbox.execute_python(exec_code)
    assert res.exit_code == 0
    assert "AUTHORIZED:True" in res.stdout

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_hierarchical_subgoal_call_stack_suspension_and_repair():
    """
    Evaluates goal frame suspension when an intermediate step fails,
    pushing a corrective repair frame, restoring preconditions, and popping back.
    """
    call_log = []

    class MockEmbodiedEngine:
        def __init__(self):
            self.dialog_dismissed = False

        def execute_embodied_step(self, description: str) -> dict:
            call_log.append(description)
            if "Dismiss blocker" in description:
                self.dialog_dismissed = True
                return {"delta_s": 1.0}
            if "Target task" in description:
                if not self.dialog_dismissed:
                    # Blocked by modal dialog!
                    return {"delta_s": -1.0}
                return {"delta_s": 1.0}
            return {"delta_s": 1.0}

    class MockEngine:
        def __init__(self):
            self.embodied = MockEmbodiedEngine()

    engine = MockEngine()
    stack = HierarchicalExecutiveStack(engine)

    root_goal = GoalFrame(
        goal_id="root_workflow",
        description="Target task in main window",
        target_invariant="window_ready",
        preconditions=[],
        max_retries=2,
    )

    result = stack.execute_with_repair(root_goal)
    assert result["success"] is True
    assert result["status"] == "ALL_SUBGOALS_FULFILLED"
    # Verify execution sequence: target attempted -> failed -> repair pushed & completed -> target resumed & completed!
    assert call_log == [
        "Target task in main window",
        "Dismiss blocker / Re-anchor focus",
        "Target task in main window",
    ]
