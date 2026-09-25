import os
import re
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from pydantic import BaseModel
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.host_commit_gate import HostCommitGate

app = FastAPI(title="AutoAgent Production Runtime", version="2.0.0")

# Singleton Cognitive Engine instance
engine: Optional[CognitiveEngine] = None


class GoalRequest(BaseModel):
    goal: str
    tenant_id: str = "default_tenant"
    timeout_sec: float = 30.0


class IngressResult(BaseModel):
    status: str
    verdict: str
    confidence: float
    output: str


@app.on_event("startup")
def startup_event():
    global engine
    if engine is None:
        engine = CognitiveEngine()


@app.on_event("shutdown")
def shutdown_event():
    global engine
    if engine:
        if hasattr(engine, "sandbox") and hasattr(engine.sandbox, "close"):
            engine.sandbox.close()
        if hasattr(engine, "consolidation") and hasattr(engine.consolidation, "close"):
            engine.consolidation.close()


from pathlib import Path
from fastapi.responses import HTMLResponse


@app.get("/horizon", response_class=HTMLResponse)
async def get_3d_horizon():
    path = Path("web/cognitive_horizon_3d.html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<h3>3D Horizon Interface Not Found</h3>"


@app.get("/healthz")
def health_check():
    global engine
    if engine is None:
        engine = CognitiveEngine()
    return {
        "status": "healthy",
        "subsystems": {
            "saliency": True,
            "sandbox": True,
            "sqlite_wal": os.path.exists("assets/cognitive_memory.db"),
            "unified_rlcd": hasattr(engine, "unified_rlcd"),
        },
    }


@app.post("/api/v1/deliberate", response_model=IngressResult)
async def deliberate_endpoint(req: GoalRequest):
    """Executes full macro decision gating, virtual scratchpad induction, and atomic commit."""
    global engine
    if engine is None:
        engine = CognitiveEngine()

    try:
        loop = asyncio.get_event_loop()
        # Run CPU-bound synthesis inside executor to keep API non-blocking
        res = await loop.run_in_executor(None, engine.deliberate_and_act, req.goal)
        confidence = 0.0
        if "Success" in res:
            confidence = 0.65
            m = re.search(r"Confidence=([\d\.]+)", res)
            if m:
                try:
                    confidence = float(m.group(1))
                except Exception:
                    pass
        return IngressResult(
            status="completed",
            verdict="attested_and_committed" if "Success" in res else "rejected",
            confidence=confidence,
            output=res,
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/api/v1/quarantine")
def list_quarantine_endpoint(status: str = "PENDING", limit: int = 100):
    """Lists candidate memories held in quarantine staging awaiting human-in-the-loop review."""
    global engine
    if engine is None:
        engine = CognitiveEngine()
    if hasattr(engine, "consolidation") and hasattr(engine.consolidation, "list_quarantined"):
        return engine.consolidation.list_quarantined(status=status, limit=limit)
    return []


@app.post("/api/v1/quarantine/{mem_id}/promote")
def promote_quarantine_endpoint(mem_id: str, notes: Optional[str] = None):
    """Promotes a quarantined candidate fact into verified episodic storage."""
    global engine
    if engine is None:
        engine = CognitiveEngine()
    if hasattr(engine, "consolidation") and hasattr(engine.consolidation, "promote_quarantined"):
        ok = engine.consolidation.promote_quarantined(mem_id, reviewer_notes=notes or "api_promoted")
        if ok:
            return {"status": "promoted", "id": mem_id}
        raise HTTPException(status_code=404, detail="Quarantine record not found or not pending")
    raise HTTPException(status_code=500, detail="Consolidation store unavailable")


@app.post("/api/v1/quarantine/{mem_id}/reject")
def reject_quarantine_endpoint(mem_id: str, reason: str = "rejected_by_reviewer"):
    """Rejects a quarantined candidate fact without polluting episodic memory."""
    global engine
    if engine is None:
        engine = CognitiveEngine()
    if hasattr(engine, "consolidation") and hasattr(engine.consolidation, "reject_quarantined"):
        ok = engine.consolidation.reject_quarantined(mem_id, reason=reason)
        if ok:
            return {"status": "rejected", "id": mem_id}
        raise HTTPException(status_code=404, detail="Quarantine record not found or not pending")
    raise HTTPException(status_code=500, detail="Consolidation store unavailable")


@app.websocket("/ws/telemetry")
async def telemetry_stream(ws: WebSocket):
    """Streams live real-time cognitive metrics to external dashboards."""
    global engine
    if engine is None:
        engine = CognitiveEngine()

    await ws.accept()
    try:
        while True:
            await asyncio.sleep(1.0)
            active_prims = 0
            if engine and hasattr(engine, "skills"):
                if hasattr(engine.skills, "_mounted_skills"):
                    active_prims = len(engine.skills._mounted_skills)
                elif isinstance(engine.skills, dict) and "tools" in engine.skills:
                    active_prims = len(engine.skills["tools"])

            telemetry = {
                "active_primitives": active_prims,
                "memory_records": engine.consolidation.count_memories()
                if (engine and hasattr(engine, "consolidation") and hasattr(engine.consolidation, "count_memories"))
                else 0,
                "timestamp": asyncio.get_event_loop().time(),
            }
            await ws.send_json(telemetry)
    except WebSocketDisconnect:
        pass
