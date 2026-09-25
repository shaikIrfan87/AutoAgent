import os
import shutil
import unittest
from cognitive_engine.core.host_commit_gate import HostCommitGate
from cognitive_engine.core.unified_rlcd import UnifiedRLCDEngine, MacroActionCandidate


class TestMultiFilePackageInduction(unittest.TestCase):
    def setUp(self):
        self.target_dir = ".test_induced_pkg"
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir, ignore_errors=True)

    def test_package_attestation_and_atomic_commit(self):
        gate = HostCommitGate()
        pkg = {
            "__init__.py": "from .core import calculate_metrics\n__all__ = ['calculate_metrics']\n",
            "core.py": "def calculate_metrics(values):\n    return sum(values) / len(values) if values else 0.0\n",
            "test_core.py": "from .core import calculate_metrics\ndef test_basic():\n    assert calculate_metrics([1, 2, 3]) == 2.0\n",
        }

        # Safe package passes attestation
        is_safe, reason = gate.inspect_and_verify_package(pkg, delta_s=1.0)
        self.assertTrue(is_safe)
        self.assertIn("Safe to commit", reason)

        # Apply atomically to host
        gate.apply_package_to_real_host(self.target_dir, pkg)

        # Verify all files exist
        self.assertTrue(os.path.exists(os.path.join(self.target_dir, "__init__.py")))
        self.assertTrue(os.path.exists(os.path.join(self.target_dir, "core.py")))
        self.assertTrue(os.path.exists(os.path.join(self.target_dir, "test_core.py")))

    def test_package_security_rejection(self):
        gate = HostCommitGate()
        malicious_pkg = {
            "__init__.py": "# valid init",
            "core.py": "import os\nos.system('del /f /q C:\\*')",
            "test_core.py": "pass",
        }

        is_safe, reason = gate.inspect_and_verify_package(malicious_pkg, delta_s=1.0)
        self.assertFalse(is_safe)
        self.assertIn("Security Violation", reason)

    def test_unified_pipeline_package_induction(self):
        engine = UnifiedRLCDEngine(num_shards=2, max_shard_capacity=10)
        gate = HostCommitGate()

        pkg = {
            "__init__.py": "# auto package",
            "solver.py": "def solve(x): return x * 2",
            "test_solver.py": "assert True",
        }

        candidate = MacroActionCandidate(
            action_id="pkg_test_01",
            action_type="induct_package",
            target=self.target_dir,
            payload=pkg,
            complexity=0.3,
            reversibility=0.0,
        )

        result = engine.execute_unified_pipeline(
            candidates=[candidate],
            assertions_map={"pkg_test_01": []},
            host_gate=gate,
            entropy=0.1,
        )

        self.assertTrue(result.committed_to_host)
        self.assertTrue(os.path.exists(os.path.join(self.target_dir, "solver.py")))


if __name__ == "__main__":
    unittest.main()
