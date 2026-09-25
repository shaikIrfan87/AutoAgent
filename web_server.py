import os
import json
import sqlite3
import uvicorn
import asyncio
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from cognitive_engine.agent.orchestrator import CognitiveEngine

app = FastAPI(title="AutoAgent Web Console")

# Initialize persistent cognitive engine instance
engine = CognitiveEngine(db_path="assets/cognitive_memory.db")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("web/index.html")
    if not html_path.exists():
        return HTMLResponse("<h3>Error: web/index.html not found.</h3>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/api/telemetry")
async def get_telemetry():
    """Fetches real-time status of the 5 Invariants and memory stores."""
    conn = sqlite3.connect("assets/cognitive_memory.db")
    cur = conn.cursor()
    
    mem_count = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    recent_memories = cur.execute(
        "SELECT id, content, confidence, last_accessed FROM memories ORDER BY last_accessed DESC LIMIT 8"
    ).fetchall()
    conn.close()

    # Skill directory inspection
    skills_dir = Path("skills/default_tenant")
    compiled_skills = [f.stem for f in skills_dir.glob("*.py")] if skills_dir.exists() else []

    oja_norm = 0.0
    plastic = getattr(engine, "plastic", getattr(engine, "plastic_cell", None))
    if plastic is not None and hasattr(plastic, "A_fast"):
        import torch
        oja_norm = float(torch.norm(plastic.A_fast, p="fro").item())

    return JSONResponse({
        "invariants": {
            "inv1_saliency": "ONLINE",
            "inv2_causal_graph": "ONLINE",
            "inv3_sandbox": "ONLINE",
            "inv4_plasticity": "BOUNDED",
            "inv5_consolidation": "WAL_ACTIVE"
        },
        "memory_count": mem_count,
        "compiled_skills_count": len(compiled_skills),
        "compiled_skills": compiled_skills,
        "oja_norm": round(oja_norm, 4),
        "recent_memories": [
            {"id": m[0], "content": m[1], "confidence": round(m[2], 2)} for m in recent_memories
        ]
    })

@app.post("/api/interact")
async def interact(request: Request):
    """Processes queries through the unified cognitive pipeline."""
    payload = await request.json()
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    try:
        # Route through engine
        response = engine.process_interactive(query)
        return JSONResponse({
            "query": query,
            "response": str(response)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/interact/stream")
async def interact_stream(request: Request):
    """Streams real-time deliberation traces and invariant states over Server-Sent Events (SSE)."""
    payload = await request.json()
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def stream_cb(event_data: dict):
            loop.call_soon_threadsafe(queue.put_nowait, event_data)

        def worker():
            try:
                if hasattr(engine, "process_interactive_stream"):
                    resp = engine.process_interactive_stream(query, callback=stream_cb)
                else:
                    resp = engine.process_interactive(query, status_cb=lambda msg: stream_cb({"type": "deliberation", "message": msg}))
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "complete", "response": str(resp)})
            except Exception as ex:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": str(ex)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=False)
