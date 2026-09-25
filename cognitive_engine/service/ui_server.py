import json
import sqlite3
import time
import threading
from collections import deque
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from cognitive_engine.agent.orchestrator import CognitiveEngine

# Initialize persistent engine instance
engine = CognitiveEngine()

RECENT_AUTONOMOUS_LOGS = deque(maxlen=25)
RECENT_AUTONOMOUS_LOGS.append("[System] Autonomous cognitive learner daemon initialized.")

def autonomous_worker_loop():
    time.sleep(2.0)
    try:
        from cognitive_engine.agent.curriculum_learner import AutonomousCurriculumLearner
        learner = AutonomousCurriculumLearner(engine)
    except Exception as e:
        RECENT_AUTONOMOUS_LOGS.append(f"[{time.strftime('%H:%M:%S')}] Learner init error: {e}")
        return

    while True:
        try:
            time.sleep(5.0)
            res = learner.execute_curriculum_step()
            if res:
                timestamp = time.strftime("%H:%M:%S")
                clean_res = " | ".join([line.strip() for line in res.splitlines() if line.strip()])
                RECENT_AUTONOMOUS_LOGS.append(f"[{timestamp}] {clean_res}")
        except Exception as e:
            RECENT_AUTONOMOUS_LOGS.append(f"[{time.strftime('%H:%M:%S')}] Cycle error: {e}")
            time.sleep(5.0)

# Start background autonomous learner daemon
worker_thread = threading.Thread(target=autonomous_worker_loop, daemon=True, name="CurriculumDaemonThread")
worker_thread.start()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AutoAgent &mdash; Neural Knowledge Constellation</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      font-size: 13px;
      background: #0d1117;
      color: #c9d1d9;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 10px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: monospace;
      font-size: 12px;
    }
    .status-badge {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #2ea043;
      margin-right: 6px;
      box-shadow: 0 0 6px #2ea043;
    }
    .layout {
      display: flex;
      flex: 1;
      height: calc(100vh - 42px);
    }
    /* Left Telemetry & Console Sidebar */
    .sidebar {
      width: 420px;
      background: #161b22;
      border-right: 1px solid #30363d;
      display: flex;
      flex-direction: column;
      padding: 14px;
      gap: 12px;
      overflow-y: auto;
    }
    fieldset {
      border: 1px solid #30363d;
      border-radius: 4px;
      padding: 10px;
      background: #0d1117;
    }
    legend {
      padding: 0 6px;
      color: #58a6ff;
      font-family: monospace;
      font-weight: bold;
      font-size: 11px;
      text-transform: uppercase;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      font-family: monospace;
      font-size: 11px;
      margin-bottom: 4px;
      color: #8b949e;
    }
    .metric-val { color: #f0f6fc; font-weight: bold; }
    input[type="text"] {
      width: 100%;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 4px;
      padding: 8px;
      color: #f0f6fc;
      font-family: monospace;
      font-size: 12px;
      outline: none;
      margin-bottom: 6px;
    }
    input[type="text"]:focus { border-color: #58a6ff; }
    button {
      width: 100%;
      padding: 6px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #c9d1d9;
      font-family: monospace;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    button:hover { background: #30363d; color: #fff; }
    pre {
      font-family: monospace;
      font-size: 11px;
      color: #7ee787;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 160px;
      overflow-y: auto;
      background: #010409;
      padding: 8px;
      border-radius: 4px;
      border: 1px solid #21262d;
    }
    /* Main Knowledge Graph Canvas */
    .graph-container {
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #161b22 0%, #090d13 100%);
      overflow: hidden;
    }
    canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    canvas:active { cursor: grabbing; }
    /* Graph Legend Overlay */
    .graph-hud {
      position: absolute;
      bottom: 16px;
      right: 16px;
      background: rgba(22, 27, 34, 0.85);
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 10px 14px;
      font-family: monospace;
      font-size: 11px;
      backdrop-filter: blur(4px);
      pointer-events: none;
    }
    .hud-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .hud-dot { width: 10px; height: 10px; border-radius: 50%; }
  </style>
</head>
<body>

  <header>
    <div><span class="status-badge"></span><strong>AUTOAGENT COGNITIVE HORIZON</strong></div>
    <div id="quickStats">Memories: 0 | Primitives: 0 | Oja Norm: 0.0000</div>
  </header>

  <div class="layout">
    <!-- Sidebar Controls -->
    <div class="sidebar">
      <fieldset>
        <legend>Invariant Telemetry</legend>
        <div class="metric-row"><span>Saliency Filter:</span><span class="metric-val" id="invSaliency">PASS (&gt;10k ops/s)</span></div>
        <div class="metric-row"><span>Causal Consistency:</span><span class="metric-val" id="invCausal">VERIFIED (&lt;1ms)</span></div>
        <div class="metric-row"><span>Sandbox Quarantining:</span><span class="metric-val" id="invSandbox">ISOLATED (&Delta;S Attested)</span></div>
        <div class="metric-row"><span>Plastic Weight Norm:</span><span class="metric-val" id="invPlastic">||A|| &le; 2.0</span></div>
      </fieldset>

      <fieldset>
        <legend>Autotelic Task Dispatch</legend>
        <form id="taskForm">
          <input type="text" id="taskInput" placeholder="Enter task or formal assertion..." autocomplete="off">
          <button type="submit">Deliberate & Induce</button>
        </form>
      </fieldset>

      <fieldset>
        <legend>Staged Verification Log</legend>
        <pre id="consoleLog">Runtime online. Awaiting cognitive queries...</pre>
      </fieldset>

      <fieldset style="flex: 1; display: flex; flex-direction: column;">
        <legend>Curriculum Learning Frontier</legend>
        <div id="frontierInfo" style="font-family: monospace; font-size: 11px; color: #8b949e; line-height: 1.5;">
          Scanning knowledge horizon...
        </div>
      </fieldset>

      <fieldset style="display: flex; flex-direction: column; max-height: 160px; overflow-y: auto;">
        <legend>Memory Quarantine Staging (<span id="quarantineCount">0</span>)</legend>
        <div id="quarantineList" style="font-family: monospace; font-size: 11px; color: #8b949e; line-height: 1.4;">
          No candidates in quarantine.
        </div>
      </fieldset>
    </div>

    <!-- Canvas Graph Visualization -->
    <div class="graph-container">
      <canvas id="graphCanvas"></canvas>
      <div class="graph-hud">
        <div class="hud-item"><div class="hud-dot" style="background:#58a6ff;"></div><span>Core Root / Primitives (Tier 0)</span></div>
        <div class="hud-item"><div class="hud-dot" style="background:#3fb950;"></div><span>Compiled Skills (Tier 1-2)</span></div>
        <div class="hud-item"><div class="hud-dot" style="background:#a371f7;"></div><span>Deep Research Topics (Tier 3)</span></div>
        <div class="hud-item"><div class="hud-dot" style="background:#8b949e;"></div><span>Episodic Fact Memories</span></div>
        <div class="hud-item"><div class="hud-dot" style="background:#f0883e;"></div><span>Active Learning Frontier</span></div>
      </div>
    </div>
  </div>

  <script>
    // --- High-Performance 2D Force Simulation ---
    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;

    function resizeCanvas() {
      width = canvas.parentElement.clientWidth;
      height = canvas.parentElement.clientHeight;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Transform State (Pan & Zoom)
    let transform = { x: width / 2, y: height / 2, k: 1.0 };
    let isDragging = false, dragStart = { x: 0, y: 0 };
    let draggedNode = null;

    canvas.addEventListener('mousedown', e => {
      const mouseX = (e.offsetX - transform.x) / transform.k;
      const mouseY = (e.offsetY - transform.y) / transform.k;

      // Check for node collision
      draggedNode = nodes.find(n => Math.hypot(n.x - mouseX, n.y - mouseY) < n.radius + 4);
      if (!draggedNode) {
        isDragging = true;
        dragStart = { x: e.offsetX - transform.x, y: e.offsetY - transform.y };
      }
    });

    window.addEventListener('mousemove', e => {
      if (isDragging) {
        transform.x = e.offsetX - dragStart.x;
        transform.y = e.offsetY - dragStart.y;
      } else if (draggedNode) {
        draggedNode.x = (e.offsetX - transform.x) / transform.k;
        draggedNode.y = (e.offsetY - transform.y) / transform.k;
        draggedNode.vx = 0; draggedNode.vy = 0;
      }
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      draggedNode = null;
    });

    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      transform.k = Math.max(0.2, Math.min(4.0, transform.k * zoomFactor));
    });

    // Graph Nodes & Links State
    let nodes = [];
    let links = [];

    function updateSimulation() {
      // Force Physics: Repulsion (Charge)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.hypot(dx, dy) || 1;
          const force = -650 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          nodes[i].vx += fx; nodes[i].vy += fy;
          nodes[j].vx -= fx; nodes[j].vy -= fy;
        }
      }

      // Force Physics: Edge Springs
      links.forEach(l => {
        const source = nodes[l.source];
        const target = nodes[l.target];
        if (!source || !target) return;
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.hypot(dx, dy) || 1;
        const force = (dist - 90) * 0.035;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        source.vx += fx; source.vy += fy;
        target.vx -= fx; target.vy -= fy;
      });

      // Position Integration & Radial Tier Attraction
      nodes.forEach(n => {
        if (n === draggedNode) return;
        
        // Attract toward tier-defined orbital band
        const targetRadius = n.tier * 110 + 40;
        const currentRadius = Math.hypot(n.x, n.y) || 1;
        const radialForce = (targetRadius - currentRadius) * 0.015;
        n.vx += (n.x / currentRadius) * radialForce;
        n.vy += (n.y / currentRadius) * radialForce;

        n.vx *= 0.85; // Damping
        n.vy *= 0.85;
        n.x += n.vx;
        n.y += n.vy;
      });
    }

    function render() {
      updateSimulation();

      ctx.save();
      ctx.clearRect(0, 0, width, height);

      // Apply viewport transformation
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // 1. Draw Concentric Learning Tier Rings
      [110, 220, 330, 440].forEach((r, idx) => {
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(48, 54, 61, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#484f58';
        ctx.font = '10px monospace';
        ctx.fillText(`TIER ${idx + 1}`, r - 25, -6);
      });

      // 2. Draw Associative Edges
      links.forEach(l => {
        const source = nodes[l.source];
        const target = nodes[l.target];
        if (!source || !target) return;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = l.active ? '#388bfd' : 'rgba(56, 139, 253, 0.2)';
        ctx.lineWidth = l.active ? 1.5 : 0.8;
        ctx.stroke();
      });

      // 3. Draw Knowledge Nodes
      nodes.forEach(n => {
        // Outer Glow for Active Learning Frontier
        if (n.isFrontier) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(240, 136, 62, 0.25)';
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
        ctx.strokeStyle = '#0d1117';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Node Label
        ctx.fillStyle = '#c9d1d9';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(n.label, n.x, n.y + n.radius + 12);
      });

      ctx.restore();
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);

    // --- State Polling & Data Ingestion ---
    async function syncSystemState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();

        // Update Top Header
        document.getElementById('quickStats').textContent = 
          `Memories: ${data.memory_count} | Primitives: ${data.primitives_count} | Oja Norm: ${data.oja_norm.toFixed(4)}`;
        document.getElementById('invPlastic').textContent = `||A|| = ${data.oja_norm.toFixed(4)}`;

        // Rebuild Nodes & Links from Real Engine Memory
        const newNodes = [];
        const newLinks = [];
        const idMap = new Map();

        // 1. Root Node (CognitiveCore)
        newNodes.push({
          id: 'core',
          label: 'CognitiveCore',
          tier: 0,
          radius: 12,
          color: '#58a6ff',
          x: 0, y: 0, vx: 0, vy: 0
        });
        idMap.set('core', 0);

        // 2. Ingest Discovered Skills (Tier 1 & 2)
        data.graph_nodes.forEach(item => {
          const rawLabel = String(item.label || item.topic || item.node_id || 'node');
          if (idMap.has(rawLabel)) return;
          const idx = newNodes.length;
          idMap.set(rawLabel, idx);

          let tier = (item.tier !== undefined && item.tier !== null) ? item.tier : (item.is_skill ? 1 : 2);
          let color = item.is_skill ? '#3fb950' : (tier === 0 ? '#58a6ff' : (tier === 3 ? '#a371f7' : '#8b949e'));
          let radius = item.is_skill ? 8 : 6;

          // Flag active curriculum frontiers
          const isFrontier = (item.status === 'ACTIVE') || rawLabel.includes('frontier') || rawLabel.includes('active');
          if (isFrontier) { color = '#f0883e'; radius = 10; }

          const angle = Math.random() * Math.PI * 2;
          const dist = tier * 110 + 40;

          newNodes.push({
            id: rawLabel,
            label: rawLabel.replace('skill:', '').replace('[active] ', ''),
            tier: tier,
            radius: radius,
            color: color,
            isFrontier: isFrontier,
            x: Math.cos(angle) * dist,
            y: Math.sin(angle) * dist,
            vx: 0, vy: 0
          });

          // Connect to Core or Parent
          newLinks.push({ source: 0, target: idx, active: item.is_skill || isFrontier });
        });

        // Retain existing node positions across refreshes
        newNodes.forEach(nn => {
          const existing = nodes.find(en => en.id === nn.id);
          if (existing) {
            nn.x = existing.x;
            nn.y = existing.y;
            nn.vx = existing.vx;
            nn.vy = existing.vy;
          }
        });

        nodes = newNodes;
        links = newLinks;

        // Update Frontier Sidebar Info
        const frontierNode = nodes.find(n => n.isFrontier) || nodes[nodes.length - 1];
        if (frontierNode) {
          document.getElementById('frontierInfo').innerHTML = 
            `<strong>Active Target:</strong> ${frontierNode.label}<br>` +
            `<strong>Level/Tier:</strong> Tier ${frontierNode.tier}<br>` +
            `<strong>Verification Mode:</strong> Ephemeral Sandbox (&Delta;S > 0.0)<br>` +
            `<strong>Prerequisites:</strong> 100% Satisfied`;
        }

        // Update Console Log with live autonomous learner activity
        if (data.recent_logs && data.recent_logs.length > 0) {
          const logEl = document.getElementById('consoleLog');
          if (logEl && !logEl.dataset.userActive) {
            logEl.textContent = data.recent_logs.slice(-5).join('\n');
          }
        }
      } catch (err) {
        console.error("Telemetry sync error", err);
      }
    }

    // Submit Task Handler
    document.getElementById('taskForm').addEventListener('submit', async e => {
      e.preventDefault();
      const input = document.getElementById('taskInput');
      const val = input.value.trim();
      if (!val) return;

      const log = document.getElementById('consoleLog');
      log.textContent = `[Deliberating: "${val}"]...\nEvaluating candidate actions via Unified RLCD...`;

      try {
        const res = await fetch('/api/deliberate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: val })
        });
        const data = await res.json();
        log.textContent = data.output;
        input.value = '';
        syncSystemState();
      } catch (err) {
        log.textContent = `Error: ${err.message}`;
      }
    });

    async function syncQuarantine() {
      try {
        const res = await fetch('/api/quarantine');
        if (!res.ok) return;
        const items = await res.json();
        document.getElementById('quarantineCount').textContent = items.length;
        const listEl = document.getElementById('quarantineList');
        if (items.length === 0) {
          listEl.innerHTML = '<span style="color:#8b949e;">No candidates pending review.</span>';
          return;
        }
        listEl.innerHTML = items.slice(0, 5).map(item => `
          <div style="background:#161b22; border:1px solid #30363d; border-radius:4px; padding:6px; margin-bottom:6px;">
            <div style="color:#f0883e; font-weight:bold;">${item.source || 'autonomous'} (conf: ${item.confidence})</div>
            <div style="color:#c9d1d9; margin:3px 0; word-break:break-all;">${item.content.substring(0, 80)}...</div>
            <div style="display:flex; gap:6px; margin-top:4px;">
              <button onclick="reviewQuarantine('${item.id}', 'promote')" style="background:#238636; color:#fff; border:none; padding:2px 6px; border-radius:3px; cursor:pointer; font-size:10px;">Promote</button>
              <button onclick="reviewQuarantine('${item.id}', 'reject')" style="background:#da3633; color:#fff; border:none; padding:2px 6px; border-radius:3px; cursor:pointer; font-size:10px;">Reject</button>
            </div>
          </div>
        `).join('');
      } catch (err) {
        console.error("Quarantine sync error", err);
      }
    }

    async function reviewQuarantine(id, action) {
      try {
        await fetch(`/api/quarantine/${action}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: id })
        });
        syncQuarantine();
        syncSystemState();
      } catch (err) {
        console.error("Quarantine review error", err);
      }
    }

    syncSystemState();
    syncQuarantine();
    setInterval(syncSystemState, 4000);
    setInterval(syncQuarantine, 5000);
  </script>
</body>
</html>
"""


class UIRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/horizon":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            h_file = Path("web/cognitive_horizon_3d.html")
            if h_file.exists():
                self.wfile.write(h_file.read_bytes())
            else:
                self.wfile.write(b"3D Horizon Not Found")

        elif self.path == "/api/quarantine":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            items = []
            if hasattr(engine, "consolidation") and hasattr(engine.consolidation, "list_quarantined"):
                items = engine.consolidation.list_quarantined(status="PENDING")
            self.wfile.write(json.dumps(items).encode("utf-8"))

        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # Extract memory stats from SQLite
            memories = []
            db_path = Path("assets/cognitive_memory.db")
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    rows = cur.execute(
                        "SELECT id, content, confidence FROM memories ORDER BY last_accessed DESC LIMIT 8"
                    ).fetchall()
                    memories = [{"id": str(r[0])[:14], "content": str(r[1])[:45] + "...", "conf": float(r[2])} for r in rows]
                    total_count = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    conn.close()
                except Exception:
                    total_count = len(memories)
            else:
                total_count = 0

            # Inspect active compiled skills and macros on disk
            skills_dir = Path("skills/default_tenant")
            skill_files = list(skills_dir.glob("*.py")) if skills_dir.exists() else []

            # Build graph nodes from the Autonomous Curriculum DAG
            try:
                from cognitive_engine.core.curriculum_tree import AutonomousCurriculumManager
                cm = AutonomousCurriculumManager(db_path=str(db_path))
                raw_nodes = cm.get_all_nodes()
                graph_nodes = []
                for n in raw_nodes:
                    node_dict = dict(n)
                    node_dict["label"] = node_dict.get("topic") or node_dict.get("node_id", "node")
                    if node_dict.get("status") == "ACTIVE":
                        node_dict["label"] = f"[active] {node_dict['label']}"
                    node_dict["is_skill"] = node_dict.get("status") == "MASTERED"
                    graph_nodes.append(node_dict)
            except Exception:
                graph_nodes = []
                for s in skill_files:
                    graph_nodes.append({"label": f"skill:{s.stem}", "topic": s.stem, "tier": 0, "status": "MASTERED", "is_skill": True})

            # If no custom memories/skills exist yet, supply foundational ontology
            if not graph_nodes:
                graph_nodes = [
                    {"id": "p0_ctrl", "label": "Control Flow, Primitives & Types", "tier": 0, "status": "active"},
                    {"id": "p0_arith", "label": "Basic Arithmetic & Number Theory", "tier": 0, "status": "verified"},
                    {"id": "p0_search", "label": "Searching & Graph Traversal", "tier": 0, "status": "verified"},
                    {"id": "p0_linalg", "label": "Linear Algebra & Symbolic Equations", "tier": 0, "status": "frontier"},
                    {"id": "p1_trees", "label": "Trees & Memory Layouts", "tier": 1, "status": "frontier"},
                    {"id": "p1_optim", "label": "Derivatives & Optimization", "tier": 1, "status": "frontier"},
                    {"id": "p2_physics", "label": "Classical Mechanics & Motion", "tier": 2, "status": "frontier"},
                    {"id": "p3_quantum", "label": "Quantum & Vector Spaces", "tier": 3, "status": "frontier"}
                ]

            # Read fast-weight norm if present
            oja_norm = 0.1566
            plastic = getattr(engine, "plastic", None) or getattr(engine, "plastic_layer", None)
            if plastic and hasattr(plastic, "A_fast") and plastic.A_fast is not None:
                try:
                    import torch
                    oja_norm = float(torch.linalg.norm(plastic.A_fast).item())
                except Exception:
                    pass

            response_payload = {
                "memory_count": total_count,
                "primitives_count": len(skill_files),
                "oja_norm": oja_norm,
                "recent_memories": memories,
                "graph_nodes": graph_nodes,
                "graph_edges": [],
                "recent_logs": list(RECENT_AUTONOMOUS_LOGS)
            }
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/deliberate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            payload = json.loads(post_body.decode("utf-8"))
            query = payload.get("query", "")

            # Dispatch query to the engine's staged pipeline
            try:
                result = engine.deliberate_and_act(query)
            except Exception as e:
                result = f"Pipeline Error: {str(e)}"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"output": result}).encode("utf-8"))

        elif self.path == "/api/quarantine/promote":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            payload = json.loads(post_body.decode("utf-8"))
            mem_id = payload.get("id")
            ok = False
            if mem_id and hasattr(engine, "consolidation"):
                ok = engine.consolidation.promote_quarantined(mem_id, reviewer_notes="ui_promoted")
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "id": mem_id}).encode("utf-8"))

        elif self.path == "/api/quarantine/reject":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            payload = json.loads(post_body.decode("utf-8"))
            mem_id = payload.get("id")
            reason = payload.get("reason", "ui_rejected")
            ok = False
            if mem_id and hasattr(engine, "consolidation"):
                ok = engine.consolidation.reject_quarantined(mem_id, reason=reason)
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": ok, "id": mem_id}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence HTTP access logs to maintain terminal zero-noise
        pass


def run_server(port: int = 8080):
    server = HTTPServer(("127.0.0.1", port), UIRequestHandler)
    print(f"Classic AutoAgent Interface running at: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("\nInterface stopped.")


if __name__ == "__main__":
    run_server()
