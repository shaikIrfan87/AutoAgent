import ast
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class AutonomousCodebaseController:
    """Empowers AutoAgent to inspect, modify, test, and commit changes

    to its own source tree based on empirical accuracy metrics.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.root = Path(workspace_root or Path(__file__).resolve().parents[2])
        self.backup_dir = self.root / ".snapshots"
        self.backup_dir.mkdir(exist_ok=True)
        self.audit_trail: List[Dict[str, Any]] = []

    def inspect_file(self, rel_path: str) -> Dict[str, Any]:
        """Reads file source and extracts top-level functions and classes via AST."""
        file_path = self.root / rel_path
        if not file_path.exists():
            return {"error": f"File {rel_path} not found"}

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            return {
                "rel_path": rel_path,
                "lines": len(source.splitlines()),
                "functions": functions,
                "classes": classes,
                "source": source,
            }
        except SyntaxError as e:
            return {"error": f"Syntax error in existing file: {e}"}

    def evaluate_benchmark(self, command: Optional[List[str]] = None) -> float:
        """Runs the test suite or accuracy benchmark, returning pass ratio (0.0 to 1.0)."""
        cmd = command or [sys.executable, "-m", "pytest", "tests/test_saliency.py", "tests/test_sandbox.py", "-q"]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def apply_patch_and_verify(
        self,
        rel_path: str,
        new_source: str,
        target_fn: Optional[str] = None,
        benchmark_cmd: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Applies a patch, compiles it, runs benchmarks, and either commits or rolls back."""
        target_file = self.root / rel_path
        if not target_file.exists():
            return {"status": "error", "reason": "Target file does not exist"}

        # 1. Measure baseline accuracy
        baseline_score = self.evaluate_benchmark(benchmark_cmd)

        # 2. Create atomic snapshot
        snapshot_path = self.backup_dir / f"{target_file.name}.snapshot"
        shutil.copyfile(target_file, snapshot_path)

        # 3. Compile validation prior to writing
        try:
            compile(new_source, str(target_file), "exec")
        except SyntaxError as e:
            return {"status": "rejected", "reason": f"AST syntax validation failed: {e}"}

        try:
            # 4. Write new source to disk
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(new_source)

            # 5. Measure post-mutation accuracy
            post_score = self.evaluate_benchmark(benchmark_cmd)

            # 6. Guardrail: Require strict non-degradation (post_score >= baseline_score)
            if post_score < baseline_score:
                shutil.copyfile(snapshot_path, target_file)
                record = {
                    "status": "rolled_back",
                    "reason": f"Benchmark regressed: {baseline_score} -> {post_score}",
                    "file": rel_path,
                }
                self.audit_trail.append(record)
                return record

            # 7. Success: Hot-reload modified module into running Python interpreter
            for mod_name, module in list(sys.modules.items()):
                mod_file = getattr(module, "__file__", None)
                if mod_file and os.path.exists(mod_file) and os.path.samefile(mod_file, target_file):
                    try:
                        importlib.reload(module)
                    except Exception:
                        pass
                    break

            record = {
                "status": "committed",
                "file": rel_path,
                "score_delta": post_score - baseline_score,
            }
            self.audit_trail.append(record)
            return record

        except Exception as ex:
            # Hard crash recovery
            shutil.copyfile(snapshot_path, target_file)
            return {"status": "failed_exception_rolled_back", "error": str(ex)}
