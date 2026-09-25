import unittest
from cognitive_engine.core.unified_rlcd import (
    UnifiedRLCDEngine,
    MacroActionCandidate,
    ExecutiveDecisionNet,
)
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.host_commit_gate import HostCommitGate


class TestUnifiedRLCD(unittest.TestCase):
    def setUp(self):
        self.engine = CognitiveEngine()
        self.rlcd = UnifiedRLCDEngine(synthesizer=self.engine.zero_synthesizer)

    def test_macro_selection_and_calibration(self):
        candidates = [
            MacroActionCandidate("induct", "induct_code", "skills/default_tenant/test_skill.py", "goal", 0.4, 1.0),
            MacroActionCandidate("web", "web_search", "memory_store", "goal", 0.2, 1.0),
        ]
        chosen, conf, net_u, approved = self.rlcd.select_macro_action(candidates)
        self.assertIsNotNone(chosen)
        self.assertGreater(conf, 0.0)

        # Verify Brier calibration update
        alt = candidates[1] if chosen == candidates[0] else candidates[0]
        loss = self.rlcd.update_macro_calibration(chosen, alt, delta_s=1.0)
        self.assertGreaterEqual(loss, 0.0)

    def test_micro_trajectory_distillation(self):
        assertions = ["double(3) == 6", "double(5) == 10", "double(0) == 0"]
        code, delta_s, loss = self.rlcd.run_micro_trajectory_distillation(assertions)
        self.assertTrue(len(code) > 0)
        self.assertIn(delta_s, [-1.0, 1.0])

    def test_end_to_end_deliberate_and_act(self):
        goal = "Synthesize an optimized gcd algorithm: gcd(48, 18) == 6; gcd(101, 103) == 1"
        res = self.engine.deliberate_and_act(goal)
        self.assertIn("Success", res)

        from skills.default_tenant.gcd import gcd
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(101, 103), 1)


if __name__ == "__main__":
    unittest.main()
