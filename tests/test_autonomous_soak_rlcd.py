import unittest
from cognitive_engine.agent.orchestrator import CognitiveEngine


class TestAutonomousSoakRLCD(unittest.TestCase):
    def test_continuous_soak_with_rlcd_and_sharded_replay(self):
        engine = CognitiveEngine()
        daemon = engine.curiosity

        # 1. Execute multiple autonomous curiosity cycles
        cycles = 0
        for _ in range(5):
            res = daemon.step()
            if res is not None:
                cycles += 1

        self.assertGreater(cycles, 0)

        # 2. Autonomous RLCD deliberate and act execution
        goal = "Synthesize squares dictionary: squares_dict([2, 4]) == {2: 4, 4: 16}"
        status = engine.deliberate_and_act(goal)
        self.assertIn("Success", status)

        # 3. Verify Sharded Replay Buffer can replay and distill
        loss = engine.unified_rlcd.replay_and_distill(batch_size=2)
        self.assertIsInstance(loss, float)

        # 4. Clean resource shutdown
        engine.sandbox.close()
        engine.consolidation.close()


if __name__ == "__main__":
    unittest.main()
