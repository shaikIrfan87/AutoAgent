import unittest
from cognitive_engine.agent.orchestrator import CognitiveEngine


class TestMultiClauseControlFlow(unittest.TestCase):
    def setUp(self):
        self.engine = CognitiveEngine()

    def test_nested_loops_matrix_flattening(self):
        goal = "Synthesize matrix flattening: flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]; flatten([[5], [6, 7]]) == [5, 6, 7]"
        res = self.engine.deliberate_and_act(goal)
        self.assertIn("Success", res)

        from skills.default_tenant.flatten import flatten
        self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
        self.assertEqual(flatten([[5], [6, 7]]), [5, 6, 7])

    def test_dictionary_comprehension(self):
        goal = "Synthesize squares dictionary: squares_dict([1, 2, 3]) == {1: 1, 2: 4, 3: 9}; squares_dict([4]) == {4: 16}"
        res = self.engine.deliberate_and_act(goal)
        self.assertIn("Success", res)

        from skills.default_tenant.squares_dict import squares_dict
        self.assertEqual(squares_dict([1, 2, 3]), {1: 1, 2: 4, 3: 9})
        self.assertEqual(squares_dict([4]), {4: 16})

    def test_compound_exception_handling(self):
        goal = "Synthesize safe division with exception handling: safe_divide(10, 2) == 5; safe_divide(10, 0) == 0; safe_divide(9, 3) == 3"
        res = self.engine.deliberate_and_act(goal)
        self.assertIn("Success", res)

        from skills.default_tenant.safe_divide import safe_divide
        self.assertEqual(safe_divide(10, 2), 5)
        self.assertEqual(safe_divide(10, 0), 0)
        self.assertEqual(safe_divide(9, 3), 3)


if __name__ == "__main__":
    unittest.main()
