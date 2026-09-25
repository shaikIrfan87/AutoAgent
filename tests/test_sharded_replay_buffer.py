import threading
import unittest
import torch
from cognitive_engine.core.sharded_replay_buffer import ShardedReplayBuffer
from cognitive_engine.core.unified_rlcd import UnifiedRLCDEngine


class TestShardedReplayBuffer(unittest.TestCase):
    def test_sharded_concurrency_and_eviction(self):
        buffer = ShardedReplayBuffer(num_shards=4, max_shard_capacity=10)

        # Concurrently push items from 8 threads
        def worker_push(thread_id: int):
            for i in range(20):
                buffer.push(
                    pos_code=f"def f_{thread_id}_{i}(): pass",
                    pos_log_probs=torch.randn(5),
                    neg_code="syntax_err",
                    neg_log_probs=torch.randn(5),
                    margin=float(i + thread_id),
                    domain=f"domain_{thread_id % 4}",
                )

        threads = [threading.Thread(target=worker_push, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total size must be strictly bounded by capacity (4 shards * 10 = 40 max)
        self.assertLessEqual(buffer.total_size(), 40)
        self.assertGreater(buffer.total_size(), 0)

        # Sample batch
        batch = buffer.sample_batch(batch_size=8)
        self.assertLessEqual(len(batch), 8)
        self.assertGreater(len(batch), 0)

    def test_replay_and_distill_optimization(self):
        engine = UnifiedRLCDEngine(num_shards=2, max_shard_capacity=10)
        # Push synthetic high-margin contrastive pairs
        for i in range(5):
            engine.replay_buffer.push(
                pos_code="square = lambda x: (x * x)",
                pos_log_probs=torch.randn(4, requires_grad=True),
                neg_code="square = lambda x: 0",
                neg_log_probs=torch.randn(4, requires_grad=True),
                margin=2.0 + i,
            )

        loss = engine.replay_and_distill(batch_size=4)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
