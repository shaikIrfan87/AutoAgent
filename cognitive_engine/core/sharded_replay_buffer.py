import glob
import heapq
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class ContrastiveTrajectory:
    """Encapsulates a contrastive (verified vs failing) trajectory pair for DPO distillation."""
    pos_code: str
    pos_log_probs: torch.Tensor
    neg_code: str
    neg_log_probs: torch.Tensor
    margin: float
    domain: str = "general"
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ReplayShard:
    """Individual thread-safe memory shard with capacity limit and priority eviction."""

    def __init__(self, shard_id: int, max_capacity: int = 250):
        self.shard_id = shard_id
        self.max_capacity = max_capacity
        self.lock = threading.Lock()
        self.entries: List[ContrastiveTrajectory] = []

    def push(self, item: ContrastiveTrajectory) -> Optional[ContrastiveTrajectory]:
        """Pushes an item; returns evicted item if capacity is exceeded, or None."""
        with self.lock:
            if len(self.entries) >= self.max_capacity:
                # Evict lowest margin item
                min_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].margin)
                if item.margin > self.entries[min_idx].margin:
                    evicted = self.entries[min_idx]
                    self.entries[min_idx] = item
                    return evicted
                return item
            else:
                self.entries.append(item)
                return None

    def sample(self, k: int) -> List[ContrastiveTrajectory]:
        with self.lock:
            if not self.entries:
                return []
            if len(self.entries) <= k:
                return list(self.entries)
            # Prioritized weighting by margin
            weights = [max(e.margin, 0.01) for e in self.entries]
            return random.choices(self.entries, weights=weights, k=k)

    def pop_lowest_margin(self, count: int) -> List[ContrastiveTrajectory]:
        """Pops up to `count` lowest-margin items for cold disk offload."""
        with self.lock:
            if not self.entries:
                return []
            count = min(count, len(self.entries))
            # Sort ascending by margin so lowest margins are first
            self.entries.sort(key=lambda e: e.margin)
            evicted = self.entries[:count]
            self.entries = self.entries[count:]
            return evicted

    def size(self) -> int:
        with self.lock:
            return len(self.entries)

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()


class ShardedReplayBuffer:
    """Multi-shard asynchronous replay buffer distributing contrastive pairs across workers

    with dynamic in-memory profiling and adaptive cold disk offloading.
    """

    def __init__(
        self,
        num_shards: int = 4,
        max_shard_capacity: int = 250,
        ram_threshold_mb: Optional[float] = None,
        cold_storage_dir: str = ".cold_replay_storage",
    ):
        self.num_shards = num_shards
        self.max_shard_capacity = max_shard_capacity
        self.ram_threshold_mb = ram_threshold_mb
        self.cold_storage_dir = cold_storage_dir
        self.shards = [ReplayShard(i, max_shard_capacity) for i in range(num_shards)]
        self._cold_files: List[str] = []
        self._cold_lock = threading.Lock()
        os.makedirs(self.cold_storage_dir, exist_ok=True)

    def _get_shard_index(self, domain: str) -> int:
        return abs(hash(domain)) % self.num_shards

    def get_current_process_memory_mb(self) -> float:
        """Returns current process RSS memory in megabytes."""
        if psutil is not None:
            try:
                return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                pass
        return 0.0

    def is_memory_pressure_high(self) -> bool:
        """Checks whether process RAM consumption crosses the configured threshold."""
        if self.ram_threshold_mb is not None:
            current_mb = self.get_current_process_memory_mb()
            if current_mb > 0.0 and current_mb >= self.ram_threshold_mb:
                return True
        return False

    def offload_to_disk(self, count_per_shard: int = 2) -> int:
        """Offloads lowest-priority items across shards into cold disk storage."""
        offloaded_count = 0
        timestamp = int(time.time() * 1000)

        for s in self.shards:
            items = s.pop_lowest_margin(count_per_shard)
            if items:
                filepath = os.path.join(
                    self.cold_storage_dir,
                    f"shard_{s.shard_id}_t{timestamp}_{random.randint(1000, 9999)}.pt",
                )
                torch.save(items, filepath)
                with self._cold_lock:
                    self._cold_files.append(filepath)
                offloaded_count += len(items)

        return offloaded_count

    def hydrate_from_disk(self, max_items: int = 5) -> List[ContrastiveTrajectory]:
        """Loads and restores cold trajectories from disk storage."""
        with self._cold_lock:
            if not self._cold_files:
                return []
            filepath = self._cold_files.pop(0)

        try:
            items: List[ContrastiveTrajectory] = torch.load(filepath, weights_only=False)
            if os.path.exists(filepath):
                os.remove(filepath)
            return items[:max_items]
        except Exception:
            return []

    def push(
        self,
        pos_code: str,
        pos_log_probs: torch.Tensor,
        neg_code: str,
        neg_log_probs: torch.Tensor,
        margin: float,
        domain: str = "general",
    ) -> None:
        item = ContrastiveTrajectory(
            pos_code=pos_code,
            pos_log_probs=pos_log_probs.detach(),
            neg_code=neg_code,
            neg_log_probs=neg_log_probs.detach(),
            margin=margin,
            domain=domain,
        )
        idx = self._get_shard_index(domain)
        evicted = self.shards[idx].push(item)

        # Offload evicted item directly to cold storage if present
        if evicted is not None:
            filepath = os.path.join(
                self.cold_storage_dir,
                f"evicted_s{idx}_{int(time.time()*1000)}_{random.randint(100, 999)}.pt",
            )
            torch.save([evicted], filepath)
            with self._cold_lock:
                self._cold_files.append(filepath)

        # Adaptive memory offload if RAM pressure threshold is triggered
        if self.is_memory_pressure_high():
            self.offload_to_disk(count_per_shard=2)

    def sample_batch(self, batch_size: int = 8, allow_cold_hydration: bool = False) -> List[ContrastiveTrajectory]:
        """Samples uniformly across non-empty shards, weighting items by margin within shards."""
        active_shards = [s for s in self.shards if s.size() > 0]
        if not active_shards:
            if allow_cold_hydration and self.cold_size() > 0:
                hydrated = self.hydrate_from_disk(max_items=batch_size)
                return hydrated[:batch_size]
            return []

        per_shard_k = max(1, batch_size // len(active_shards))
        sampled = []
        for s in active_shards:
            sampled.extend(s.sample(per_shard_k))
        random.shuffle(sampled)
        return sampled[:batch_size]

    def hot_size(self) -> int:
        return sum(s.size() for s in self.shards)

    def cold_size(self) -> int:
        with self._cold_lock:
            return len(self._cold_files)

    def total_size(self) -> int:
        return self.hot_size()

    def clear(self) -> None:
        for s in self.shards:
            s.clear()
        with self._cold_lock:
            for f in self._cold_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            self._cold_files.clear()
