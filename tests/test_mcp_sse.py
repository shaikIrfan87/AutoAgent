"""
Tests for Streamable HTTP/SSE MCP Transport.
Verifies MCPClient against mock SSE & HTTP endpoints.
"""

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from cognitive_engine.core.mcp_client import MCPClient


class MockMCPServerHandler(BaseHTTPRequestHandler):
    sse_clients: list = []
    response_mode: str = "sse"  # "sse" or "direct"

    def log_message(self, format, *args):
        # Suppress noisy HTTP logs during testing
        pass

    def do_GET(self):
        if self.path.startswith("/sse"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Handshake: notify endpoint for messages
            self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
            self.wfile.flush()

            client_queue = queue.Queue()
            MockMCPServerHandler.sse_clients.append(client_queue)

            try:
                while True:
                    msg = client_queue.get()
                    if msg is None:
                        break
                    sse_payload = f"event: message\ndata: {json.dumps(msg)}\n\n".encode("utf-8")
                    self.wfile.write(sse_payload)
                    self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
            finally:
                if client_queue in MockMCPServerHandler.sse_clients:
                    MockMCPServerHandler.sse_clients.remove(client_queue)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/messages"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8"))

            method = req.get("method")
            req_id = req.get("id")

            if method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {"name": "remote_weather", "description": "Fetch weather forecast"},
                            {"name": "remote_calculator", "description": "Execute calculation"},
                        ]
                    },
                }
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                if tool_name == "error_tool":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": "Simulated tool crash"},
                    }
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"output": f"Ran {tool_name} with {args}"},
                    }
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

            if MockMCPServerHandler.response_mode == "sse":
                # Acknowledge POST with 202 Accepted, then push response via SSE
                self.send_response(202)
                self.end_headers()
                for client_q in list(MockMCPServerHandler.sse_clients):
                    client_q.put(resp)
            else:
                # Direct JSON-RPC response in HTTP body
                resp_bytes = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture(scope="module")
def sse_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockMCPServerHandler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    for q in list(MockMCPServerHandler.sse_clients):
        q.put(None)
    server.shutdown()
    server.server_close()


def test_mcp_sse_transport_streaming(sse_server):
    MockMCPServerHandler.response_mode = "sse"
    client = MCPClient(url=f"{sse_server}/sse", timeout=5.0)
    try:
        tools = client.list_tools()
        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "remote_weather" in tool_names
        assert "remote_calculator" in tool_names

        res = client.call_tool("remote_calculator", {"expression": "2 + 2"})
        assert "Ran remote_calculator with {'expression': '2 + 2'}" in res["output"]
    finally:
        client.close()


def test_mcp_direct_http_response(sse_server):
    MockMCPServerHandler.response_mode = "direct"
    client = MCPClient(url=f"{sse_server}/sse", timeout=5.0)
    try:
        tools = client.list_tools()
        assert len(tools) == 2

        res = client.call_tool("remote_weather", {"city": "Zurich"})
        assert "Ran remote_weather with {'city': 'Zurich'}" in res["output"]
    finally:
        client.close()


def test_mcp_sse_tool_error_propagation(sse_server):
    MockMCPServerHandler.response_mode = "sse"
    client = MCPClient(url=f"{sse_server}/sse", timeout=5.0)
    try:
        with pytest.raises(RuntimeError, match="Simulated tool crash"):
            client.call_tool("error_tool", {})
    finally:
        client.close()


def test_mcp_client_argument_validation():
    with pytest.raises(ValueError, match="requires either a command"):
        MCPClient()


def test_mcp_client_stdio_backward_compat(tmp_path):
    server_script = tmp_path / "mock_mcp_stdio.py"
    server_script.write_text(
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line.strip())\n"
        "    if req['method'] == 'tools/list':\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'tools': [{'name': 'stdio_tool'}]}}\n"
        "    elif req['method'] == 'tools/call':\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'status': 'ok'}}\n"
        "    sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    client = MCPClient([sys.executable, str(server_script)])
    try:
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "stdio_tool"

        res = client.call_tool("stdio_tool", {})
        assert res["status"] == "ok"
    finally:
        client.close()
