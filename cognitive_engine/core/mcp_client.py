"""
Model Context Protocol (MCP) client for dynamic tool discovery and execution.
Supports standard JSON-RPC 2.0 protocol over:
1. stdio pipes (subprocess)
2. streamable HTTP/Server-Sent Events (SSE) network endpoints
"""

import json
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Union


class MCPClient:
    """Model Context Protocol (MCP) client supporting stdio and HTTP/SSE transports."""

    def __init__(
        self,
        target: Optional[Union[List[str], str]] = None,
        *,
        command: Optional[List[str]] = None,
        url: Optional[str] = None,
        timeout: float = 10.0,
    ):
        # Resolve transport mode
        if isinstance(target, list):
            self.command: Optional[List[str]] = target
            self.url: Optional[str] = None
        elif isinstance(target, str):
            self.command = None
            self.url = target
        else:
            self.command = command
            self.url = url

        if not self.command and not self.url:
            raise ValueError("MCPClient requires either a command (stdio) or url (HTTP/SSE)")

        self.timeout = timeout
        self._msg_id = 0

        # stdio state
        self.proc: Optional[subprocess.Popen] = None

        # SSE state
        self._closed = False
        self._post_url: Optional[str] = None
        self._endpoint_event = threading.Event()
        self._pending_requests: Dict[int, queue.Queue] = {}
        self._sse_thread: Optional[threading.Thread] = None
        self._sse_response: Optional[Any] = None

    def start(self) -> None:
        """Start the underlying transport if not already running."""
        if self.command:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
        elif self.url:
            if self._sse_thread is None or not self._sse_thread.is_alive():
                self._closed = False
                self._endpoint_event.clear()
                self._sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
                self._sse_thread.start()
                # Wait briefly for the endpoint event or handshake
                self._endpoint_event.wait(timeout=min(self.timeout, 2.0))
                if not self._post_url:
                    # Fallback: post directly to the base URL
                    self._post_url = self.url

    def _sse_listener(self) -> None:
        """Background thread that consumes the Server-Sent Events stream."""
        req = urllib.request.Request(
            self.url,
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        )
        try:
            self._sse_response = urllib.request.urlopen(req, timeout=self.timeout)
            current_event = "message"
            data_buffer: List[str] = []

            while not self._closed:
                line_bytes = self._sse_response.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")

                if not line:
                    # Blank line triggers event dispatch
                    if data_buffer:
                        combined_data = "\n".join(data_buffer)
                        self._handle_sse_event(current_event, combined_data)
                        data_buffer = []
                        current_event = "message"
                    continue

                if line.startswith(":"):
                    # Comment line / keepalive
                    continue
                elif line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].strip())
        except Exception:
            pass
        finally:
            self._endpoint_event.set()

    def _handle_sse_event(self, event: str, data: str) -> None:
        """Handle individual SSE events according to MCP protocol."""
        if event == "endpoint":
            raw_endpoint = data.strip()
            self._post_url = urllib.parse.urljoin(self.url, raw_endpoint)
            self._endpoint_event.set()
        elif event == "message" or not event:
            try:
                msg = json.loads(data)
                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending_requests:
                    self._pending_requests[req_id].put(msg)
            except Exception:
                pass

    def _send_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.start()
        self._msg_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method,
            "params": params or {},
        }

        if self.command:
            if not self.proc or not self.proc.stdin or not self.proc.stdout:
                raise RuntimeError("MCP process not initialized")

            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise ConnectionError(f"MCP server closed stream: {err}")
            return json.loads(line.strip())

        elif self.url:
            post_target = self._post_url or self.url
            resp_queue = queue.Queue()
            self._pending_requests[self._msg_id] = resp_queue

            try:
                post_req = urllib.request.Request(
                    post_target,
                    data=json.dumps(req).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                with urllib.request.urlopen(post_req, timeout=self.timeout) as resp:
                    body = resp.read()
                    if body:
                        try:
                            direct_resp = json.loads(body.decode("utf-8"))
                            if isinstance(direct_resp, dict) and (
                                direct_resp.get("id") == self._msg_id
                                or "result" in direct_resp
                                or "error" in direct_resp
                            ):
                                return direct_resp
                        except Exception:
                            pass

                # If not returned synchronously in HTTP POST body, wait on SSE stream
                try:
                    return resp_queue.get(timeout=self.timeout)
                except queue.Empty:
                    raise TimeoutError(f"Timed out waiting for SSE response to RPC {self._msg_id}")
            finally:
                self._pending_requests.pop(self._msg_id, None)

        raise RuntimeError("No valid transport available")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Discover available external tools and their input schemas."""
        res = self._send_rpc("tools/list")
        return res.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an MCP tool with structured arguments."""
        res = self._send_rpc("tools/call", {"name": name, "arguments": arguments})
        if "error" in res:
            raise RuntimeError(f"MCP Tool Error: {res['error']}")
        return res.get("result", {})

    def close(self) -> None:
        """Shut down transports and release resources."""
        self._closed = True
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                self.proc.kill()
            self.proc = None

        if self._sse_response:
            try:
                self._sse_response.close()
            except Exception:
                pass
            self._sse_response = None
