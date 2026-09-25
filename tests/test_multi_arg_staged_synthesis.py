import unittest
from cognitive_engine.agent.orchestrator import CognitiveEngine


class TestMultiArgStagedSynthesis(unittest.TestCase):
    def setUp(self):
        self.engine = CognitiveEngine()

    def test_staged_gcd_synthesis(self):
        result = self.engine.execute_staged_autonomous_cycle(
            "Synthesize an optimized gcd algorithm: gcd(48, 18) == 6; gcd(101, 103) == 1; gcd(54, 24) == 6"
        )
        self.assertIn("Success", result)

        from skills.default_tenant.gcd import gcd
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(101, 103), 1)
        self.assertEqual(gcd(54, 24), 6)

    def test_staged_binary_search_synthesis(self):
        result = self.engine.execute_staged_autonomous_cycle(
            "Synthesize binary search: binary_search([1, 3, 5, 7, 9], 7) == 3; binary_search([1, 2, 4, 8], 5) == -1"
        )
        self.assertIn("Success", result)

        from skills.default_tenant.binary_search import binary_search
        self.assertEqual(binary_search([1, 3, 5, 7, 9], 7), 3)
        self.assertEqual(binary_search([1, 2, 4, 8], 5), -1)


if __name__ == "__main__":
    unittest.main()
