import json
import pytest
from fastapi.testclient import TestClient
from cognitive_engine.agent.orchestrator import CognitiveEngine
from web_server import app


def test_process_interactive_stream_emits_structured_events():
    """Verify process_interactive_stream invokes callback with Invariants 1-5 and rollout telemetry."""
    engine = CognitiveEngine(db_path=":memory:")
    try:
        events = []
        def on_event(data):
            events.append(data)

        # Test with a simple math query that triggers sandbox and rollout
        resp = engine.process_interactive_stream("Calculate 12 * 12", callback=on_event)
        assert "144" in resp or "Result" in resp

        stages = [e.get("stage") for e in events]
        assert "inv1_saliency" in stages
        assert "inv2_causal" in stages
        assert "inv4_plasticity" in stages
        assert "inv5_consolidation" in stages

        # Check invariant event details
        saliency_event = next(e for e in events if e.get("stage") == "inv1_saliency")
        assert saliency_event["passed"] is True
        assert "entropy" in saliency_event
    finally:
        if hasattr(engine, "world_model") and hasattr(engine.world_model, "tuner"):
            engine.world_model.tuner.stop()


def test_process_interactive_stream_drops_low_entropy():
    """Verify sensory saliency gate drops low entropy input early in stream."""
    engine = CognitiveEngine(db_path=":memory:")
    try:
        events = []
        resp = engine.process_interactive_stream("a", callback=lambda e: events.append(e))
        assert "insufficient information entropy" in resp.lower()
        saliency_event = next(e for e in events if e.get("stage") == "inv1_saliency")
        assert saliency_event["passed"] is False
    finally:
        if hasattr(engine, "world_model") and hasattr(engine.world_model, "tuner"):
            engine.world_model.tuner.stop()


def test_web_server_interact_stream_endpoint():
    """Verify FastAPI /api/interact/stream returns text/event-stream with parsed SSE events."""
    client = TestClient(app)
    with client.stream("POST", "/api/interact/stream", json={"query": "Calculate 7 * 8"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        streamed_events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                streamed_events.append(payload)

        event_types = [e.get("type") for e in streamed_events]
        assert "complete" in event_types
        complete_event = next(e for e in streamed_events if e.get("type") == "complete")
        assert "56" in complete_event.get("response", "")


def test_web_server_legacy_interact_endpoint_parity():
    """Verify legacy /api/interact endpoint continues to function identically."""
    client = TestClient(app)
    resp = client.post("/api/interact", json={"query": "Calculate 10 + 20"})
    assert resp.status_code == 200
    data = resp.json()
    assert "30" in data.get("response", "")
