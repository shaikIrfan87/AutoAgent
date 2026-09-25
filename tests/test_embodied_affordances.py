import pytest
import numpy as np
from cognitive_engine.core.vision_sensor import (
    SemanticAffordanceClassifier,
    AffordanceType,
    ScreenPerceptionEngine,
)


def test_semantic_affordance_classification():
    classifier = SemanticAffordanceClassifier()
    screen_size = (1920, 1080)

    # 1. Action Trigger (Button): compact rectangle
    btn_bbox = (500, 400, 650, 450)  # w=150, h=50, ar=3.0
    btn_type = classifier.classify_region(btn_bbox, screen_size=screen_size)
    assert btn_type == AffordanceType.ACTION_TRIGGER

    # 2. Text Input: elongated horizontal field
    input_bbox = (200, 200, 600, 240)  # w=400, h=40, ar=10.0
    input_type = classifier.classify_region(input_bbox, screen_size=screen_size)
    assert input_type == AffordanceType.TEXT_INPUT

    # 3. Navigation Control: top right icon
    nav_bbox = (1850, 20, 1890, 60)  # w=40, h=40, near corner
    nav_type = classifier.classify_region(nav_bbox, screen_size=screen_size)
    assert nav_type == AffordanceType.NAVIGATION_CONTROL

    # 4. Scroll Container: large area
    scroll_bbox = (100, 100, 1500, 900)  # huge area > 100000
    scroll_type = classifier.classify_region(scroll_bbox, screen_size=screen_size)
    assert scroll_type == AffordanceType.SCROLL_CONTAINER


def test_screen_perception_affordance_tagging():
    eyes = ScreenPerceptionEngine()
    elements = eyes.inspect_ui_tree()
    assert len(elements) > 0

    # Ensure every inspected node is annotated with semantic affordance
    for elem in elements:
        assert "affordance" in elem
        assert elem["affordance"] in [a.value for a in AffordanceType]

    # Test filtering by affordance
    action_triggers = eyes.find_by_affordance(AffordanceType.ACTION_TRIGGER)
    assert isinstance(action_triggers, list)


def test_native_winrt_ocr_perception():
    from PIL import Image, ImageDraw, ImageFont
    eyes = ScreenPerceptionEngine()

    # Draw a synthetic canvas button with rasterized text "CONFIRM"
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=32)
    draw.text((30, 30), "CONFIRM", fill=(0, 0, 0), font=font)
    frame_rgb = np.array(img)

    text = eyes.read_text_at((0, 0, 300, 100), frame_rgb=frame_rgb)
    assert "CONFIRM" in text.upper()
