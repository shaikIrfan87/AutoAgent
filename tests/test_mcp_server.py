import json
import threading
import time
import pytest
from cognitive_engine.core.mcp_server import MCPServer
from cognitive_engine.core.mcp_client import MCPClient


def test_mcp_server_initialize():
    server = MCPServer()
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = server.handle_rpc(req)
    assert resp["id"] == 1
    assert "protocolVersion" in resp["result"]
    assert resp["result"]["serverInfo"]["name"] == "AutoAgent Cognitive Engine"


def test_mcp_server_tools_list():
    server = MCPServer()
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = server.handle_rpc(req)
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "deliberate" in tool_names
    assert "recall_memory" in tool_names
    assert "verify_causal_hypothesis" in tool_names
    assert "predict_action_risk" in tool_names
    assert "interact" in tool_names


def test_mcp_server_predict_action_risk_call():
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "predict_action_risk",
            "arguments": {"code": "x = 42\nprint(x)"},
        },
    }
    resp = server.handle_rpc(req)
    assert resp["id"] == 3
    result = resp["result"]
    assert not result["isError"]
    payload = json.loads(result["content"][0]["text"])
    assert payload["safe"] is True
    assert payload["risk_score"] < 0.70


def test_mcp_server_verify_causal_hypothesis_call():
    server = MCPServer()
    # Test valid causal hypothesis
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "verify_causal_hypothesis",
            "arguments": {"subject": "human", "relation": "is_a", "target": "biologicalorganism"},
        },
    }
    resp = server.handle_rpc(req)
    assert resp["id"] == 4
    result = resp["result"]
    payload = json.loads(result["content"][0]["text"])
    assert payload["consistent"] is True


def test_mcp_server_network_client_integration():
    server = MCPServer()
    port = 8991
    http_server = server.create_http_server(port=port)
    server_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    client = MCPClient(url=f"http://127.0.0.1:{port}/sse", timeout=4.0)
    try:
        tools = client.list_tools()
        assert len(tools) >= 5
        tool_names = [t["name"] for t in tools]
        assert "deliberate" in tool_names
        assert "predict_action_risk" in tool_names

        # Call predict_action_risk over HTTP/SSE
        res = client.call_tool("predict_action_risk", {"code": "val = 100\nprint(val)"})
        assert not res.get("isError")
        payload = json.loads(res["content"][0]["text"])
        assert payload["safe"] is True
    finally:
        client.close()
        http_server.shutdown()
        http_server.server_close()
