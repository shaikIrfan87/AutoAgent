"""
Tests for SkillLibrary and CuriosityDaemon (Pillars 1 & 2 of Industry-Grade Autonomous Intelligence).
"""

import time
import numpy as np
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.skills import SkillLibrary
from cognitive_engine.agent.curiosity import CuriosityDaemon
from cognitive_engine.core.types import Triple


def test_skill_library_synthesis(tmp_path):
    skill_dir = str(tmp_path / "skills")
    lib = SkillLibrary(storage_dir=skill_dir)

    code = "def add(a, b):\n    return a + b"
    saved_path = lib.register_skill("math_add", code, doc="Add two numbers")

    assert "math_add" in lib.list_skills()
    retrieved = lib.get_skill_code("math_add")
    assert "def add(a, b):" in retrieved
    assert "Add two numbers" in retrieved


def test_cognitive_engine_auto_skill_compilation(tmp_path):
    skill_dir = str(tmp_path / "engine_skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    engine = CognitiveEngine(skill_library=lib)

    query = "Matrix diagonal sum computation"
    code = "def diag_sum(m):\n    return sum(m[i][i] for i in range(len(m)))\nprint(diag_sum([[1,2],[3,4]]))"

    res = engine.process(query, code_action=code)
    assert res["status"] == "system_2_success"

    # Verify skill was automatically compiled and registered in library
    skills = engine.skills.list_skills()
    assert len(skills) >= 1
    assert any("matrix" in s.lower() for s in skills)


def test_curiosity_daemon_frontier_discovery():
    engine = CognitiveEngine()
    # Add an isolated node to the causal graph
    engine.causal.graph.add_node("superconductive_plasma")

    daemon = CuriosityDaemon(engine=engine, interval_sec=10.0)
    frontiers = daemon.find_epistemic_frontiers()

    assert "superconductive_plasma" in frontiers
    inquiry = daemon.formulate_inquiry("superconductive_plasma")
    assert "superconductive_plasma" in inquiry

    # Execute one active curiosity step
    res = daemon.step()
    assert res is not None
    assert "status" in res


def test_skill_library_tenant_isolation(tmp_path):
    skill_dir = str(tmp_path / "partitioned_skills")
    lib = SkillLibrary(storage_dir=skill_dir)

    # Tenant A registers solution
    lib.register_skill("solution", "def run(): return 'A'", tenant_id="tenant_A")
    # Tenant B registers conflicting solution
    lib.register_skill("solution", "def run(): return 'B'", tenant_id="tenant_B")

    # Verify no collision across tenant namespaces
    assert lib.get_skill_code("solution", tenant_id="tenant_A").strip().endswith("'A'")
    assert lib.get_skill_code("solution", tenant_id="tenant_B").strip().endswith("'B'")
    assert "solution" in lib.list_skills(tenant_id="tenant_A")
    assert "solution" in lib.list_skills(tenant_id="tenant_B")


def test_dynamic_causal_edge_induction():
    engine = CognitiveEngine()
    # Dynamic induction from observed transition
    induced = engine.causal.induce_edge("matrix_multiplication", "positive_state_delta", polarity=True)
    assert induced is True
    assert engine.causal.get_edge_weight("matrix_multiplication", "positive_state_delta") > 0.0

    # Contradiction rejection: attempting to induce violation of physical axiom
    contradictory = engine.causal.induce_edge(
        "entropy decrease", "without energy dissipation", polarity=False, relation="cannot_be"
    )
    assert contradictory is False


def test_saliency_synonym_cluster_alignment():
    from cognitive_engine.core.saliency import _semantic_topological_embed
    v_exploit = _semantic_topological_embed("malicious exploit payload")
    v_attack = _semantic_topological_embed("malicious attack payload")
    cos_sim = float(np.dot(v_exploit, v_attack) / (np.linalg.norm(v_exploit) * np.linalg.norm(v_attack)))
    assert cos_sim > 0.60, f"Synonym alignment failed: cos_sim={cos_sim}"


def test_curiosity_daemon_non_blocking_yield():
    import threading
    engine = CognitiveEngine()
    daemon = CuriosityDaemon(engine=engine)
    lock_held = threading.Event()
    release_lock = threading.Event()

    def lock_holder():
        with engine.consolidation._lock:
            lock_held.set()
            release_lock.wait()

    t = threading.Thread(target=lock_holder, daemon=True)
    t.start()
    lock_held.wait()
    try:
        # Step should detect lock held by other thread and yield immediately returning None
        result = daemon.step()
        assert result is None
    finally:
        release_lock.set()
        t.join()


def test_mcp_client_stdio_jsonrpc(tmp_path):
    import sys
    from cognitive_engine.core.mcp_client import MCPClient
    server_script = tmp_path / "mock_mcp_server.py"
    server_script.write_text(
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line.strip())\n"
        "    if req['method'] == 'tools/list':\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'tools': [{'name': 'fetch_data', 'description': 'Fetches data'}]}}\n"
        "    elif req['method'] == 'tools/call':\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'content': f\"Executed {req['params']['name']}\"}}\n"
        "    else:\n"
        "        resp = {'jsonrpc': '2.0', 'id': req['id'], 'error': 'Unknown'}\n"
        "    sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    client = MCPClient([sys.executable, str(server_script)])
    try:
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "fetch_data"

        res = client.call_tool("fetch_data", {"query": "test"})
        assert "Executed fetch_data" in res["content"]
    finally:
        client.close()


def test_goal_tree_htn_decomposition():
    from cognitive_engine.agent.goal_tree import GoalTree
    tree = GoalTree(root_id="audit_project", root_description="Audit Entire Codebase")
    subgoals = [("lint", "Run static linter"), ("security", "Check CVE vulnerabilities")]
    created = tree.decompose("audit_project", subgoals)
    assert created == ["lint", "security"]

    act = tree.get_next_actionable_goal()
    assert act is not None
    assert act.id == "lint"

    tree.mark_completed("lint", "0 errors")
    act2 = tree.get_next_actionable_goal()
    assert act2 is not None
    assert act2.id == "security"

    tree.mark_completed("security", "0 CVEs")
    assert tree.nodes["audit_project"].status == "completed"

    serialized = tree.serialize()
    deserialized = GoalTree.deserialize(serialized)
    assert deserialized.nodes["audit_project"].status == "completed"
    assert deserialized.nodes["lint"].result == "0 errors"


def test_counterfactual_mental_simulator():
    from cognitive_engine.core.world_model import MentalSimulator
    sim = MentalSimulator(risk_threshold=0.70)

    safe_code = "def add(a, b):\n    return a + b\nprint(add(2, 3))"
    pred_safe = sim.simulate(safe_code)
    assert pred_safe.safe is True
    assert pred_safe.risk_score < 0.70
    assert pred_safe.predicted_delta_s > 0

    harmful_code = "import os\nos.remove('important_database.db')"
    pred_harmful = sim.simulate(harmful_code)
    assert pred_harmful.safe is False
    assert pred_harmful.risk_score >= 0.70
    assert any("destructive_filesystem" in b for b in pred_harmful.blast_radius)
    assert "rejected" in pred_harmful.rejection_reason.lower()


def test_nested_plasticity_continuum():
    import torch
    from cognitive_engine.core.plastic_layer import FastPlasticLinear
    layer = FastPlasticLinear(in_features=384, out_features=384)
    k = torch.randn(384)
    v = torch.randn(384)

    layer.absorb(k, v, tenant_id="t1", session_id="s1")

    fast_state = layer._get_session_state("t1", "s1")
    mid_state = layer._get_mid_state("t1", "s1")
    assert torch.linalg.norm(fast_state) > 0
    assert torch.linalg.norm(mid_state) > 0

    layer.reset_session("t1", "s1", reset_mid=False)
    assert torch.linalg.norm(layer._get_session_state("t1", "s1")) == 0
    assert torch.linalg.norm(layer._get_mid_state("t1", "s1")) > 0

    layer.reset_session("t1", "s1", reset_mid=True)
    assert torch.linalg.norm(layer._get_mid_state("t1", "s1")) == 0


def test_cognitive_engine_htn_orchestration():
    engine = CognitiveEngine()
    engine.goal_tree.decompose("root", [
        ("task_1", "Calculate square of 5: print(5 * 5)"),
        ("task_2", "Calculate cube of 3: print(3 ** 3)")
    ])

    res1 = engine.execute_next_htn_goal()
    assert res1 is not None
    assert engine.goal_tree.nodes["task_1"].status == "completed"

    res2 = engine.execute_next_htn_goal()
    assert res2 is not None
    assert engine.goal_tree.nodes["task_2"].status == "completed"
    assert engine.goal_tree.nodes["root"].status == "completed"



