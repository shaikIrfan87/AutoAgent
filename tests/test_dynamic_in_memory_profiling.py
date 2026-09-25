import os
import shutil
import unittest
import torch
from cognitive_engine.core.sharded_replay_buffer import ShardedReplayBuffer


class TestDynamicInMemoryProfiling(unittest.TestCase):
    def setUp(self):
        self.test_cold_dir = ".test_cold_storage"
        if os.path.exists(self.test_cold_dir):
            shutil.rmtree(self.test_cold_dir, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.test_cold_dir):
            shutil.rmtree(self.test_cold_dir, ignore_errors=True)

    def test_eviction_to_cold_disk_storage(self):
        buffer = ShardedReplayBuffer(
            num_shards=2,
            max_shard_capacity=3,
            cold_storage_dir=self.test_cold_dir,
        )

        # Push 10 items into shard 0 (capacity is 3)
        for i in range(10):
            buffer.push(
                pos_code=f"def pos_{i}(): return {i}",
                pos_log_probs=torch.tensor([0.9, 0.8]),
                neg_code="def neg(): return 0",
                neg_log_probs=torch.tensor([0.1, 0.2]),
                margin=float(i + 1),
                domain="domain_0",
            )

        # Hot size should not exceed max capacity for that shard (3)
        self.assertLessEqual(buffer.hot_size(), 3)
        # Evicted items should be saved to cold storage
        self.assertGreater(buffer.cold_size(), 0)

        # Hydrate from disk
        hydrated = buffer.hydrate_from_disk(max_items=3)
        self.assertGreater(len(hydrated), 0)
        self.assertTrue(hasattr(hydrated[0], "pos_code"))

    def test_adaptive_memory_pressure_offload(self):
        # Set artificially low RAM threshold to trigger memory pressure offload
        buffer = ShardedReplayBuffer(
            num_shards=2,
            max_shard_capacity=10,
            ram_threshold_mb=0.001,  # Force trigger on any memory usage
            cold_storage_dir=self.test_cold_dir,
        )

        for i in range(6):
            buffer.push(
                pos_code=f"def p_{i}(): pass",
                pos_log_probs=torch.ones(3),
                neg_code="syntax_err",
                neg_log_probs=torch.zeros(3),
                margin=float(i),
                domain=f"d_{i % 2}",
            )

        # Memory pressure offload should have moved lower margin items to cold storage
        self.assertGreater(buffer.cold_size(), 0)

        # Test clear removes cold files
        buffer.clear()
        self.assertEqual(buffer.cold_size(), 0)
        self.assertEqual(buffer.hot_size(), 0)


if __name__ == "__main__":
    unittest.main()
