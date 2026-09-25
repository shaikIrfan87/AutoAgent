import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.mcp_server import MCPServer

app = FastAPI(title="AutoAgent Daemon")
engine = CognitiveEngine()
mcp_server = MCPServer(engine=engine)


@app.get("/health")
def health_check():
    return {"status": "ok", "subsystems": ["saliency", "causal", "sandbox", "plasticity", "consolidation", "mcp_server"]}


@app.post("/mcp/messages")
@app.post("/mcp/rpc")
async def mcp_rpc_handler(request: Request):
    data = await request.json()
    resp = await asyncio.to_thread(mcp_server.handle_rpc, data)
    return JSONResponse(content=resp)


@app.get("/mcp/sse")
async def mcp_sse_endpoint():
    async def event_generator():
        yield "event: endpoint\ndata: /mcp/messages\n\n"
        while True:
            await asyncio.sleep(15.0)
            yield ": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws/interact")
async def ws_interact(websocket: WebSocket):
    # ponytail: built-in asyncio.to_thread prevents event loop stall during System 2 execution
    await websocket.accept()
    try:
        while True:
            task = await websocket.receive_text()
            if not task.strip():
                continue
            res = await asyncio.to_thread(engine.interact, task)
            await websocket.send_text(res)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(f"Error: {type(e).__name__} - {str(e)}")
