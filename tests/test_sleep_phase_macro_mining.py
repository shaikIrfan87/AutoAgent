import json
import os
import shutil
import unittest
import torch

from cognitive_engine.core.dsl import UNARY_PRIMITIVES, to_grid, crop_nonzero, keep_largest_object
from cognitive_engine.core.macro_store import PersistentMacroStore
from cognitive_engine.core.sharded_replay_buffer import ShardedReplayBuffer
from cognitive_engine.core.sleep_phase_miner import SleepPhaseMacroMiner
from cognitive_engine.agent.program_synthesizer import ProgramSynthesizer


class TestSleepPhaseMacroMining(unittest.TestCase):
    def setUp(self):
        self.test_store_path = ".test_assets/dsl_macros.json"
        if os.path.exists(".test_assets"):
            shutil.rmtree(".test_assets", ignore_errors=True)
        os.makedirs(".test_assets", exist_ok=True)
        self.store = PersistentMacroStore(store_path=self.test_store_path)
        self.miner = SleepPhaseMacroMiner(macro_store=self.store, store_path=self.test_store_path)

    def tearDown(self):
        if os.path.exists(".test_assets"):
            shutil.rmtree(".test_assets", ignore_errors=True)

    def test_sleep_cycle_ast_mining_and_persistence(self):
        buffer = ShardedReplayBuffer(num_shards=2, max_shard_capacity=10)

        # 1. Deposit solutions from ARC tasks 32e9702f and 358ba94e into replay buffer
        code_32e = "replace_color(0, 5)(shift_left(replace_color(0, 5)(g)))"
        code_358 = "if_then_else(has_color(2), crop_nonzero(keep_largest_object(g)), crop_nonzero(keep_smallest_object(g)))(g)"

        buffer.push(
            pos_code=code_32e,
            pos_log_probs=torch.tensor([0.9]),
            neg_code="identity(g)",
            neg_log_probs=torch.tensor([0.1]),
            margin=2.0,
            domain="arc_visual_induction",
        )
        buffer.push(
            pos_code=code_358,
            pos_log_probs=torch.tensor([0.95]),
            neg_code="identity(g)",
            neg_log_probs=torch.tensor([0.05]),
            margin=2.0,
            domain="arc_visual_induction",
        )

        # 2. Run sleep cycle
        diagnostics = self.miner.run_sleep_cycle(replay_buffer=buffer)

        self.assertGreaterEqual(diagnostics["new_macros_discovered"], 1)
        self.assertTrue(os.path.exists(self.test_store_path))

        # 3. Verify dsl_macros.json content
        with open(self.test_store_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertGreaterEqual(len(saved_data), 1)

        # 4. Verify macros were injected into UNARY_PRIMITIVES
        for macro_name in diagnostics["macro_names"]:
            self.assertIn(macro_name, UNARY_PRIMITIVES)

    def test_search_depth_collapse_with_mined_macro(self):
        # Target transformation: keep_largest_object then crop_nonzero
        input_grid = to_grid([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        expected_output = crop_nonzero(keep_largest_object(input_grid))

        # Run sleep cycle directly on 358ba94e code to inject macro
        code_358 = "crop_nonzero(keep_largest_object(g))"
        res = self.miner.run_sleep_cycle(candidate_codes=[code_358])
        self.assertGreaterEqual(res["new_macros_discovered"], 1)

        # Verify that ProgramSynthesizer now solves this composite task in depth D=1!
        synth = ProgramSynthesizer(max_depth=2, max_expansions=20)
        prog = synth.synthesize([(input_grid, expected_output)])

        self.assertIsNotNone(prog)
        self.assertEqual(prog(input_grid), expected_output)
        # Depth is 1 because the composite operation is now an atomic unary primitive
        self.assertEqual(len(prog.steps), 1)


if __name__ == "__main__":
    unittest.main()
