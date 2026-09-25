import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List


class EphemeralVirtualSystem:
    """Creates an isolated virtual scratchpad for inspecting untrusted actions."""

    def __init__(self, source_workspace: str = "."):
        self.source_dir = Path(source_workspace).resolve()
        self.virtual_dir: Path | None = None
        self.mutation_journal: List[Dict[str, Any]] = []

    def __enter__(self):
        self.virtual_dir = Path(tempfile.mkdtemp(prefix="autoagent_virtual_system_"))
        # Clone repo/workspace into scratchpad (ignoring Git/virtualenv/cache bloat)
        shutil.copytree(
            self.source_dir,
            self.virtual_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "assets/models", "*.db*"),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.virtual_dir and self.virtual_dir.exists():
            shutil.rmtree(self.virtual_dir, ignore_errors=True)

    def log_mutation(self, action_type: str, target: str, payload: str):
        self.mutation_journal.append({
            "action": action_type,
            "target": target,
            "payload": payload,
            "timestamp": os.path.getmtime(self.virtual_dir) if self.virtual_dir else 0.0,
        })


class VirtualWorkspace(EphemeralVirtualSystem):
    """
    Two-Tier Staged Virtual Workspace.
    Encapsulates isolated filesystem clone and execution sandbox for untrusted execution.
    """
    def __init__(self, source_workspace: str = "."):
        super().__init__(source_workspace=source_workspace)
        from features.execution_sandbox.sandbox import IsolatedSandboxExecutor
        self.sandbox = IsolatedSandboxExecutor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.virtual_dir and self.virtual_dir.exists():
            from features.execution_sandbox.sandbox import safe_cleanup_temp_dir
            safe_cleanup_temp_dir(str(self.virtual_dir))

