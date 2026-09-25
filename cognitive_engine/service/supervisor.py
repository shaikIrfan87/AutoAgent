import os
import sys
import time
import psutil
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional
from cognitive_engine.agent.orchestrator import CognitiveEngine
from cognitive_engine.core.sleep_phase_miner import SleepPhaseMacroMiner

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Supervisor] %(message)s"
)
logger = logging.getLogger("Supervisor")


class ProductionRuntimeSupervisor:
    """Supervises AutoAgent background threads, RSS thresholds, SQLite WAL checkpoints, and idle sleep mining."""

    def __init__(
        self,
        max_rss_mb: float = 1536.0,
        resource_interval: float = 10.0,
        wal_interval: float = 300.0,
        idle_interval: float = 60.0,
        db_path: str = "assets/cognitive_memory.db",
    ):
        self.max_rss_mb = max_rss_mb
        self.resource_interval = resource_interval
        self.wal_interval = wal_interval
        self.idle_interval = idle_interval
        self.db_path = Path(db_path)
        self.engine = CognitiveEngine()
        self.macro_miner = SleepPhaseMacroMiner()
        self.is_running = True
        self.process = psutil.Process(os.getpid())

    def _monitor_resources(self):
        """Monitors working set memory and forces GC/cold storage offload."""
        while self.is_running:
            try:
                rss_mb = self.process.memory_info().rss / (1024 * 1024)
                if rss_mb > self.max_rss_mb:
                    logger.warning(
                        f"Memory threshold exceeded: {rss_mb:.2f} MB > {self.max_rss_mb} MB. Evacuating replay buffers."
                    )
                    # Force dynamic replay buffer evacuation if available
                    if hasattr(self.engine, "replay_buffer") and self.engine.replay_buffer is not None:
                        self.engine.replay_buffer.offload_to_disk()
            except Exception as ex:
                logger.error(f"Resource monitoring error: {ex}")

            # Sleep in short increments to allow rapid responsive shutdown
            slept = 0.0
            while self.is_running and slept < self.resource_interval:
                time.sleep(min(0.5, self.resource_interval - slept))
                slept += 0.5

    def _wal_maintenance_loop(self):
        """Performs non-blocking WAL truncate checkpoints periodically."""
        while self.is_running:
            slept = 0.0
            while self.is_running and slept < self.wal_interval:
                time.sleep(min(1.0, self.wal_interval - slept))
                slept += 1.0

            if not self.is_running:
                break

            if self.db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.execute("PRAGMA optimize;")
                    conn.close()
                    logger.info("SQLite WAL maintenance checkpoint completed.")
                except Exception as ex:
                    logger.error(f"WAL maintenance error: {ex}")

    def _autonomous_idle_worker(self):
        """Runs background autotelic inquiry and sleep-phase macro mining when idle."""
        while self.is_running:
            slept = 0.0
            while self.is_running and slept < self.idle_interval:
                time.sleep(min(1.0, self.idle_interval - slept))
                slept += 1.0

            if not self.is_running:
                break

            logger.info("Executing periodic sleep-phase macro induction...")
            try:
                # Sleep cycle mines newly verified trajectories into reusable primitives
                res = self.macro_miner.run_sleep_cycle(replay_buffer=getattr(self.engine, "replay_buffer", None))
                if res.get("new_macros_discovered", 0) > 0:
                    logger.info(f"Mined {res['new_macros_discovered']} new AST macros into dsl_macros.json")
            except Exception as ex:
                logger.error(f"Autonomous idle cycle error: {ex}")

    def start(self, block: bool = True):
        logger.info("Starting AutoAgent Production Supervisor...")
        t_res = threading.Thread(target=self._monitor_resources, daemon=True)
        t_wal = threading.Thread(target=self._wal_maintenance_loop, daemon=True)
        t_auto = threading.Thread(target=self._autonomous_idle_worker, daemon=True)

        t_res.start()
        t_wal.start()
        t_auto.start()

        logger.info("All supervisor daemons active. Engine ready for live traffic.")
        if block:
            try:
                while self.is_running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.stop()

    def stop(self, exit_process: bool = False):
        logger.info("Shutting down supervisor...")
        self.is_running = False
        if hasattr(self.engine, "sandbox") and hasattr(self.engine.sandbox, "close"):
            self.engine.sandbox.close()
        if hasattr(self.engine, "consolidation") and hasattr(self.engine.consolidation, "close"):
            self.engine.consolidation.close()
        if exit_process:
            sys.exit(0)


if __name__ == "__main__":
    supervisor = ProductionRuntimeSupervisor()
    supervisor.start()
