from fastapi.testclient import TestClient
from cognitive_engine.api import app


def test_api_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_websocket_interact_endpoint():
    client = TestClient(app)
    with client.websocket_connect("/ws/interact") as ws:
        ws.send_text("Calculate the kinetic energy of a 1500kg car traveling at 28 m/s.")
        resp = ws.receive_text()
        assert "Kinetic energy" in resp or "Grounded Execution" in resp


def test_api_mcp_endpoints():
    client = TestClient(app)
    # Test tools/list via MCP RPC
    req = {"jsonrpc": "2.0", "id": 101, "method": "tools/list", "params": {}}
    resp = client.post("/mcp/rpc", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 101
    assert "tools" in data["result"]
    assert any(t["name"] == "deliberate" for t in data["result"]["tools"])

