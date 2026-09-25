import time
from fastapi.testclient import TestClient
from cognitive_engine.service.api import app, engine
from cognitive_engine.service.supervisor import ProductionRuntimeSupervisor


def test_service_healthz():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "subsystems" in data
    assert data["subsystems"]["saliency"] is True
    assert data["subsystems"]["sandbox"] is True


def test_service_deliberate():
    client = TestClient(app)
    payload = {
        "goal": "Synthesize a palindrome checker: is_palindrome(\"radar\") == True; is_palindrome(\"hello\") == False",
        "tenant_id": "default_tenant",
        "timeout_sec": 30.0,
    }
    resp = client.post("/api/v1/deliberate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["verdict"] in ["attested_and_committed", "rejected"]
    assert "output" in data
    if data["verdict"] == "attested_and_committed":
        assert data["confidence"] > 0.0


def test_service_telemetry_ws():
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        data = ws.receive_json()
        assert "active_primitives" in data
        assert "memory_records" in data
        assert "timestamp" in data


def test_production_supervisor_lifecycle(tmp_path):
    test_db = tmp_path / "test_supervisor.db"
    supervisor = ProductionRuntimeSupervisor(
        max_rss_mb=10.0,  # Low threshold to trigger memory branch
        resource_interval=0.1,
        wal_interval=0.1,
        idle_interval=0.1,
        db_path=str(test_db),
    )

    # Test non-blocking background lifecycle with fast intervals
    supervisor.start(block=False)
    time.sleep(0.3)
    assert supervisor.is_running is True
    supervisor.stop(exit_process=False)
    assert supervisor.is_running is False


def test_service_quarantine_endpoints():
    client = TestClient(app)
    # 1. Check initial quarantine list
    resp = client.get("/api/v1/quarantine")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # 2. Add an item directly via consolidation store
    from cognitive_engine.service import api
    if api.engine is None:
        api.engine = CognitiveEngine()
    mid = api.engine.consolidation.quarantine_memory(
        content="Candidate API quarantine record",
        confidence=0.75,
        source="autonomous_test",
    )

    # 3. Verify visible in endpoint
    resp = client.get("/api/v1/quarantine")
    assert resp.status_code == 200
    items = resp.json()
    assert any(it["id"] == mid for it in items)

    # 4. Promote item via API
    resp_promo = client.post(f"/api/v1/quarantine/{mid}/promote?notes=api_verified")
    assert resp_promo.status_code == 200
    assert resp_promo.json()["status"] == "promoted"

    # 5. Reject a second item via API
    mid2 = api.engine.consolidation.quarantine_memory(content="Bad candidate", confidence=0.4)
    resp_rej = client.post(f"/api/v1/quarantine/{mid2}/reject?reason=low_confidence")
    assert resp_rej.status_code == 200
    assert resp_rej.json()["status"] == "rejected"

