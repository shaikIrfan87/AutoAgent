import os
import tempfile
import pytest
from cognitive_engine.core.virtual_workspace import EphemeralVirtualSystem
from cognitive_engine.core.host_commit_gate import HostCommitGate
from features.protocol_guard.guard import TypeSafeProtocolGuard
from cognitive_engine.agent.orchestrator import CognitiveEngine


def test_ephemeral_virtual_workspace_lifecycle():
    with tempfile.TemporaryDirectory() as src:
        test_file = os.path.join(src, "module.py")
        with open(test_file, "w") as f:
            f.write("x = 42\n")

        with EphemeralVirtualSystem(source_workspace=src) as v_sys:
            assert v_sys.virtual_dir is not None
            assert v_sys.virtual_dir.exists()
            assert (v_sys.virtual_dir / "module.py").exists()

            v_sys.log_mutation("file_write", "test.py", "x = 10")
            assert len(v_sys.mutation_journal) == 1
            virtual_path = v_sys.virtual_dir

        # Verifies virtual directory was destroyed upon exit
        assert not virtual_path.exists()


def test_host_commit_gate_rejections():
    guard = TypeSafeProtocolGuard()
    gate = HostCommitGate(causal_verifier=None, protocol_guard=guard)

    # 1. Delta S <= 0 rejection
    passed, reason = gate.inspect_and_verify("print(1)", delta_s=-1.0)
    assert not passed
    assert "execution failed" in reason

    # 2. Forbidden host destructive operations
    passed, reason = gate.inspect_and_verify("import os; os.system('calc.exe')", delta_s=1.0)
    assert not passed
    assert "Security Violation" in reason

    # 3. Safe verified code
    passed, reason = gate.inspect_and_verify("x = 10 + 5\nprint(x)", delta_s=1.0)
    assert passed
    assert "Attestation Passed" in reason


def test_host_commit_gate_atomic_apply():
    gate = HostCommitGate()
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = os.path.join(tmp_dir, "verified_result.py")
        gate.apply_to_real_host(target, "VERIFIED = True\n")
        assert os.path.exists(target)
        with open(target, "r") as f:
            assert f.read() == "VERIFIED = True\n"


def test_orchestrator_staged_autonomous_cycle():
    engine = CognitiveEngine()
    try:
        res = engine.execute_staged_autonomous_cycle("Calculate kinetic energy of 1000kg car at 10m/s")
        assert "Success: Action verified and committed safely to host" in res
    finally:
        engine.sandbox.close()
        engine.consolidation.close()
