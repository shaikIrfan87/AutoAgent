"""
Model Context Protocol (MCP) Server for AutoAgent.
Exposes CognitiveEngine capabilities (Deliberation, Fast Recall, Causal Guardrails,
Predictive World Model) as standard JSON-RPC 2.0 tools over stdio and HTTP/SSE transports.
"""

import argparse
import json
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
import urllib.parse

from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.types import Triple


class MCPServer:
    """Model Context Protocol (MCP) server wrapping CognitiveEngine."""

    def __init__(self, engine: Optional[CognitiveEngine] = None):
        self.engine = engine or CognitiveEngine()
        self.tools = self._register_tools()

    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        return {
            "deliberate": {
                "name": "deliberate",
                "description": "Execute deliberate System 2 ReAct loop with causal validation, predictive state simulation, and sandboxed execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "The goal or task description to deliberate upon."},
                        "code": {"type": "string", "description": "Optional initial Python code snippet to execute and verify."},
                    },
                    "required": ["goal"],
                },
                "handler": self._handle_deliberate,
            },
            "recall_memory": {
                "name": "recall_memory",
                "description": "Perform System 1 instant hybrid recall across long-term episodic memory using FTS5 BM25 and semantic vector similarity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Semantic query or keywords to retrieve from memory."},
                        "top_k": {"type": "integer", "description": "Number of memories to recall.", "default": 3},
                    },
                    "required": ["query"],
                },
                "handler": self._handle_recall,
            },
            "verify_causal_hypothesis": {
                "name": "verify_causal_hypothesis",
                "description": "Verify directional causal hypotheses against Pearl's Causal Calculus axioms or evaluate Level 3 counterfactual inquiries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Source antecedent concept."},
                        "relation": {"type": "string", "description": "Relational verb, e.g. causes, is_a, cannot_be.", "default": "causes"},
                        "target": {"type": "string", "description": "Target consequence concept."},
                        "candidate_action": {"type": "string", "description": "Optional alternate action for Pearl Level 3 counterfactual testing."},
                    },
                    "required": ["subject", "target"],
                },
                "handler": self._handle_verify_causal,
            },
            "predict_action_risk": {
                "name": "predict_action_risk",
                "description": "Simulate candidate action side-effects in latent state space (estimating memory delta, file handles, loop bounds, and compatibility energy) before execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code snippet to simulate and estimate blast radius for."},
                    },
                    "required": ["code"],
                },
                "handler": self._handle_predict_risk,
            },
            "interact": {
                "name": "interact",
                "description": "End-to-end cognitive interaction through full 5-invariant pipeline (Sensory gating, System 1 recall, System 2 deliberation).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User prompt or command."},
                    },
                    "required": ["query"],
                },
                "handler": self._handle_interact,
            },
            "execute_open_world_action": {
                "name": "execute_open_world_action",
                "description": "Execute grounded real-world actions: OS shell commands, HTTP web queries, filesystem inspection, or system diagnostics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["shell", "web_request", "filesystem_read", "filesystem_write", "system_inspect"],
                            "description": "The category of open world action to perform.",
                        },
                        "params": {"type": "object", "description": "Parameters dictionary for the action."},
                    },
                    "required": ["action_type"],
                },
                "handler": self._handle_open_world_action,
            },
        }

    def _handle_open_world_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action_type = args["action_type"]
        params = args.get("params", {})
        res = self.engine.execute_open_world(action_type, params)
        return {
            "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
            "isError": not res.get("success", False),
        }

    def _handle_deliberate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        goal = args["goal"]
        code = args.get("code")
        res = self.engine.process(goal, code_action=code)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "status": res.get("status"),
                            "output": res.get("output"),
                            "delta_s": res.get("delta_s"),
                            "routed_system": res.get("routed_system"),
                            "error": res.get("error"),
                        },
                        indent=2,
                    ),
                }
            ],
            "isError": res.get("status") not in ("system_1", "system_2_success", "success"),
        }

    def _handle_recall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args["query"]
        top_k = int(args.get("top_k", 3))
        q_vec = self.engine.saliency._embed(query)
        memories = self.engine.consolidation.hybrid_search(query, q_vec, top_k=top_k)
        formatted = [
            {"id": rec.id, "content": rec.content, "confidence": rec.confidence, "score": score}
            for rec, score in memories
        ]
        return {
            "content": [{"type": "text", "text": json.dumps(formatted, indent=2)}],
            "isError": False,
        }

    def _handle_verify_causal(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sub = args["subject"]
        rel = args.get("relation", "causes")
        tgt = args["target"]
        cand = args.get("candidate_action")

        if cand:
            viable, conf, diag = self.engine.causal.evaluate_counterfactual(
                observed_state=sub, failed_action=rel, candidate_action=cand, target_invariant=tgt
            )
            data = {"counterfactual_viable": viable, "confidence": conf, "rationale": diag}
        else:
            valid, diag = self.engine.causal.verify_hypothesis(sub, rel, tgt)
            data = {"consistent": valid, "diagnostic": diag}

        return {
            "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
            "isError": False,
        }

    def _handle_predict_risk(self, args: Dict[str, Any]) -> Dict[str, Any]:
        code = args["code"]
        pred = self.engine.world_model.simulate(code)
        proj_dict = pred.transition.projected_state.to_dict() if pred.transition else {}
        data = {
            "safe": pred.safe,
            "risk_score": pred.risk_score,
            "predicted_delta_s": pred.predicted_delta_s,
            "energy_score": pred.energy_score,
            "blast_radius": pred.blast_radius,
            "projected_state": proj_dict,
            "rejection_reason": pred.rejection_reason,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
            "isError": not pred.safe,
        }

    def _handle_interact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args["query"]
        res = self.engine.interact(query)
        return {
            "content": [{"type": "text", "text": str(res)}],
            "isError": False,
        }

    def handle_rpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches JSON-RPC 2.0 requests according to Model Context Protocol specification."""
        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "AutoAgent Cognitive Engine",
                        "version": "0.3.0",
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                },
            }

        elif method == "tools/list":
            tools_list = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }
                for t in self.tools.values()
            ]
            if hasattr(self.engine, "skills"):
                for s_name in self.engine.skills.list_skills():
                    doc = self.engine.skills.get_skill_doc(s_name) if hasattr(self.engine.skills, "get_skill_doc") else ""
                    tools_list.append({
                        "name": f"skill_{s_name}",
                        "description": f"Compiled Python skill: {doc or s_name}",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "args": {"type": "object", "description": "Keyword arguments for the skill function"}
                            },
                        },
                    })
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": tools_list},
            }

        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})

            # Dynamic skill execution via MCP
            if hasattr(self.engine, "skills"):
                raw_skill_name = name[6:] if name.startswith("skill_") else name
                if raw_skill_name in self.engine.skills.list_skills():
                    try:
                        res = self.engine.skills.execute_skill(
                            raw_skill_name,
                            kwargs=arguments.get("args", arguments),
                            sandbox=self.engine.sandbox,
                        )
                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                                "isError": res.get("status") == "failed",
                            },
                        }
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [{"type": "text", "text": f"Skill execution error: {str(e)}"}],
                                "isError": True,
                            },
                        }

            if name not in self.tools:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool '{name}' not found"},
                }
            try:
                result = self.tools[name]["handler"](arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result,
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Execution error: {str(e)}"}],
                        "isError": True,
                    },
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method '{method}' not recognized"},
        }

    def serve_stdio(self) -> None:
        """Run line-delimited JSON-RPC loop over standard input/output."""
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                resp = self.handle_rpc(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()

    def create_http_server(self, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
        """Create a standard HTTP/SSE server handler for network MCP transport."""
        server_instance = self

        class MCPServerHandler(BaseHTTPRequestHandler):
            sse_clients: List[queue.Queue] = []
            _lock = threading.Lock()

            def log_message(self, format, *args):
                pass  # Suppress noisy logs

            def do_GET(self):
                if self.path.startswith("/sse"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    # Handshake
                    self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
                    self.wfile.flush()

                    client_q = queue.Queue()
                    with self._lock:
                        self.sse_clients.append(client_q)

                    try:
                        while True:
                            msg = client_q.get()
                            if msg is None:
                                break
                            payload = f"event: message\ndata: {json.dumps(msg)}\n\n".encode("utf-8")
                            self.wfile.write(payload)
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        with self._lock:
                            if client_q in self.sse_clients:
                                self.sse_clients.remove(client_q)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path.startswith("/messages"):
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        req = json.loads(body.decode("utf-8"))
                        resp = server_instance.handle_rpc(req)
                        resp_bytes = json.dumps(resp).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_bytes)))
                        self.end_headers()
                        self.wfile.write(resp_bytes)
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

        return ThreadingHTTPServer((host, port), MCPServerHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAgent Model Context Protocol (MCP) Server")
    parser.add_argument("--stdio", action="store_true", help="Serve via stdio JSON-RPC")
    parser.add_argument("--port", type=int, default=8765, help="HTTP/SSE server port")
    args = parser.parse_args()

    server = MCPServer()
    if args.stdio:
        server.serve_stdio()
    else:
        http_server = server.create_http_server(port=args.port)
        print(f"AutoAgent MCP Server running on http://127.0.0.1:{args.port}/sse")
        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            http_server.shutdown()
