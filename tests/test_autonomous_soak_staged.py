import unittest
from cognitive_engine.agent.orchestrator import CognitiveEngine


class TestAutonomousSoakStaged(unittest.TestCase):
    def test_autonomous_soak_cycles(self):
        engine = CognitiveEngine()
        daemon = engine.curiosity

        # Execute 5 consecutive autonomous curiosity cycles with staged verification
        cycles_completed = 0
        for _ in range(5):
            res = daemon.step()
            if res is not None:
                cycles_completed += 1

        self.assertGreater(cycles_completed, 0)

        # Test staged execution cycle directly on an autonomous exploration goal
        result = engine.execute_staged_autonomous_cycle("Synthesize helper for prime factorization: factorize(12) == [2, 2, 3]")
        self.assertIn("Success", result)

        engine.sandbox.close()
        engine.consolidation.close()


if __name__ == "__main__":
    unittest.main()
