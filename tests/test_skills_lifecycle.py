"""
Lifecycle tests for dynamic skill compilation, AST safety, sandbox verification,
live ReAct mounting, utility decay, and MCP exposure.
"""

import time
from pathlib import Path
import pytest

from cognitive_engine.core.skills import SkillLibrary, SkillManager, validate_skill_ast
from cognitive_engine.core.sandbox import EnvironmentalSandbox
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.mcp_server import MCPServer


def test_skills_ast_validation_rejects_unsafe_code():
    # 1. Prohibited subprocess import
    unsafe_subprocess = "import subprocess\ndef run():\n    subprocess.run(['ls'])"
    is_safe, msg = validate_skill_ast(unsafe_subprocess)
    assert not is_safe
    assert "subprocess" in msg.lower()

    # 2. Prohibited socket import
    unsafe_socket = "from socket import socket\ndef run():\n    s = socket()"
    is_safe, msg = validate_skill_ast(unsafe_socket)
    assert not is_safe
    assert "socket" in msg.lower()

    # 3. Prohibited eval call
    unsafe_eval = "def run(expr):\n    return eval(expr)"
    is_safe, msg = validate_skill_ast(unsafe_eval)
    assert not is_safe
    assert "eval" in msg.lower()

    # 4. Clean verified function passes
    clean_code = "def calculate_hypotenuse(a, b):\n    return (a**2 + b**2)**0.5"
    is_safe, msg = validate_skill_ast(clean_code)
    assert is_safe
    assert "passed" in msg.lower()


def test_skill_compilation_and_persistence(tmp_path):
    skill_dir = str(tmp_path / "skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    sandbox = EnvironmentalSandbox()

    code = "def solution():\n    return 42 * 2\n"
    success, path = lib.compile_and_persist(
        name="double_answer",
        code=code,
        doc="Compute double the universal answer",
        sandbox=sandbox,
    )
    sandbox.close()

    assert success is True
    assert Path(path).exists()
    assert "double_answer" in lib.list_skills()
    assert "42 * 2" in lib.get_skill_code("double_answer")


def test_skill_execution_in_sandbox(tmp_path):
    skill_dir = str(tmp_path / "skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    sandbox = EnvironmentalSandbox()

    code = "def solution():\n    return [x**2 for x in range(5)]\n"
    lib.compile_and_persist("square_series", code, doc="Generate square series", sandbox=sandbox)

    res = lib.execute_skill("square_series", sandbox=sandbox)
    sandbox.close()

    assert res["status"] == "success"
    assert "[0, 1, 4, 9, 16]" in res["output"]
    assert res["delta_s"] == 1.0


def test_skill_live_react_integration(tmp_path):
    skill_dir = str(tmp_path / "react_skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    engine = CognitiveEngine(skill_library=lib)

    # Trigger System 2 execution that produces positive state delta
    query = "Calculate factorial of 6"
    code = "def solution():\n    f = 1\n    for i in range(1, 7): f *= i\n    return f\nprint(solution())"

    res = engine.process(query, code_action=code)
    assert res["status"] == "system_2_success"

    # Verify skill was automatically compiled and mounted
    skills = engine.skills.list_skills()
    assert any("factorial" in s.lower() for s in skills)

    # Verify skill can be directly executed by the engine's library
    skill_name = [s for s in skills if "factorial" in s.lower()][0]
    exec_res = engine.skills.execute_skill(skill_name, sandbox=engine.sandbox)
    assert exec_res["status"] == "success"
    assert "720" in exec_res["output"]


def test_skill_utility_decay_and_pruning(tmp_path):
    skill_dir = str(tmp_path / "decay_skills")
    lib = SkillLibrary(storage_dir=skill_dir)

    lib.register_skill("stale_tool", "def solution(): return 1", doc="Old tool")
    assert "stale_tool" in lib.list_skills()

    # Simulate 100 hours passing without access (decay_lambda=0.05)
    # U_eff = 0.85 * exp(-0.05 * 100) = 0.85 * 0.0067 = 0.0057 < 0.20
    simulated_future = time.time() + (100 * 3600)
    pruned = lib.prune_stale_skills(decay_lambda=0.05, threshold=0.20, current_time=simulated_future)

    assert "stale_tool" in pruned
    assert "stale_tool" not in lib.list_skills()


def test_mcp_dynamic_skill_exposure_and_call(tmp_path):
    skill_dir = str(tmp_path / "mcp_skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    engine = CognitiveEngine(skill_library=lib)

    # Register a new skill
    lib.compile_and_persist(
        name="custom_adder",
        code="def solution():\n    return 10 + 25\n",
        doc="Custom adder utility",
    )

    server = MCPServer(engine=engine)

    # 1. tools/list must dynamically expose the new skill
    req_list = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp_list = server.handle_rpc(req_list)
    tool_names = [t["name"] for t in resp_list["result"]["tools"]]
    assert "skill_custom_adder" in tool_names

    # 2. tools/call must execute the skill and return the result
    req_call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "skill_custom_adder",
            "arguments": {},
        },
    }
    resp_call = server.handle_rpc(req_call)
    assert resp_call["result"]["isError"] is False
    content_text = resp_call["result"]["content"][0]["text"]
    assert "35" in content_text


def test_semantic_skill_retrieval_and_reuse(tmp_path):
    skill_dir = str(tmp_path / "semantic_skills")
    lib = SkillLibrary(storage_dir=skill_dir)

    # Register math matrix skill
    lib.compile_and_persist(
        name="matrix_inversion_solver",
        code="def solution():\n    return 'matrix_inverted_safely'\nprint(solution())",
        doc="Invert a 2D square matrix and solve linear equations",
    )

    # Search using a paraphrase query
    query = "Solve linear system by inverting matrix"
    matches = lib.retrieve_relevant_skills(query, top_k=1, threshold=0.50)

    assert len(matches) == 1
    assert matches[0]["name"] == "matrix_inversion_solver"
    assert matches[0]["score"] >= 0.50

    # Test ExecutiveLoop auto-reuse
    from cognitive_engine.agent.executive_loop import ExecutiveLoop
    sandbox = EnvironmentalSandbox()
    executive = ExecutiveLoop(sandbox=sandbox, skill_manager=lib)

    res = executive.run(goal=query, code="")
    sandbox.close()

    assert res["success"] is True
    assert "matrix_inverted_safely" in res["output"]
    assert res["scratchpad"][0].get("reused_skill") == "matrix_inversion_solver"


def test_macro_to_skill_export_and_execution(tmp_path):
    from cognitive_engine.core.macro_store import PersistentMacroStore
    from cognitive_engine.core.compression import MacroPrimitive

    store_file = str(tmp_path / "test_macros.json")
    macro_store = PersistentMacroStore(store_path=store_file)

    # Create dummy macro
    macro = MacroPrimitive(
        name="rot_and_transpose",
        operations=(("rot90", ()), ("transpose", ())),
        fn=lambda g: g,
        utility_score=0.9,
    )
    macro_store.save({"rot_and_transpose": macro})

    # Export to skills library
    skill_dir = str(tmp_path / "exported_skills")
    lib = SkillLibrary(storage_dir=skill_dir)
    exported = macro_store.export_to_skills(lib)

    assert "arc_rot_and_transpose" in exported
    assert "arc_rot_and_transpose" in lib.list_skills()

    # Execute exported skill in sandbox
    sandbox = EnvironmentalSandbox()
    exec_res = lib.execute_skill("arc_rot_and_transpose", sandbox=sandbox)
    sandbox.close()

    assert exec_res["status"] == "success"
    assert "[[3, 4], [1, 2]]" in exec_res["output"]

