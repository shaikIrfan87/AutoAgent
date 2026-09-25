import pytest
import numpy as np
from cognitive_engine.core.vision_sensor import ScreenPerceptionEngine, VisualGeometryParser
from cognitive_engine.core.os_actuator import OSActuator
from cognitive_engine.agent.orchestrator import CognitiveEngine, EmbodiedCognitiveEngine


def test_screen_perception_engine():
    eyes = ScreenPerceptionEngine()
    frame, size = eyes.capture_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert size[0] > 0 and size[1] > 0

    # Initial call sets baseline frame (returns 1.0)
    init_delta = eyes.compute_screen_delta(frame)
    assert init_delta == 1.0

    # Frame divergence with identical frame should be 0.0
    delta_same = eyes.compute_screen_delta(frame)
    assert delta_same == 0.0

    # Frame divergence with completely inverted frame should be > 0.0
    diff_frame = 255 - frame
    delta_diff = eyes.compute_screen_delta(diff_frame)
    assert delta_diff > 0.0

    # Adaptive settle loop returns a valid float delta
    settle_delta = eyes.wait_for_settle(max_wait_sec=0.1, interval_sec=0.02)
    assert isinstance(settle_delta, float)

    # Inspect UI tree
    nodes = eyes.inspect_ui_tree()
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    first = nodes[0]
    assert "id" in first
    assert "bbox" in first
    assert "center" in first


def test_visual_geometry_parser():
    # Synthetic frame with a button-like high-contrast rectangle
    h, w = 400, 600
    synth_frame = np.zeros((h, w, 3), dtype=np.uint8)
    synth_frame[100:150, 150:350] = 255  # 200x50 white button

    regions = VisualGeometryParser.extract_bounding_boxes(synth_frame)
    assert isinstance(regions, list)
    assert len(regions) > 0
    box = regions[0]
    assert "bbox" in box
    assert "center" in box


def test_os_actuator_bounds_and_primitives():
    hands = OSActuator()
    assert hands.screen_w > 0
    assert hands.screen_h > 0

    # Negative coordinates must fail safely
    bad_res = hands.click_at(-100, -100)
    assert not bad_res["success"]

    # Out of bounds coordinates must fail safely
    bad_res2 = hands.click_at(hands.screen_w + 1000, hands.screen_h + 1000)
    assert not bad_res2["success"]

    # Drag out of bounds
    bad_drag = hands.drag_and_drop(-10, -10, 50, 50)
    assert not bad_drag["success"]

    # Scroll dispatch (0 clicks safe test)
    scroll_res = hands.scroll(clicks=0)
    assert scroll_res["success"]


def test_embodied_cognitive_engine_wire():
    engine = CognitiveEngine(db_path=":memory:")
    assert hasattr(engine, "embodied")
    assert engine.embodied is not None

    embodied = engine.embodied
    assert isinstance(embodied.eyes, ScreenPerceptionEngine)
    assert isinstance(embodied.hands, OSActuator)

    # Executing non-existent node returns graceful handled failure with delta_s = -1.0
    res = embodied.execute_embodied_step("__non_existent_impossible_ui_node_xyz__")
    assert not res["success"]
    assert res["delta_s"] == -1.0
    assert "not visible" in res["reason"]

    # Close resources
    engine.sandbox.close()
    engine.consolidation.close()
