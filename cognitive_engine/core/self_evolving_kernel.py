import os
import shutil
from typing import Any, Callable, List, Optional
from .dynamic_code_mutator import DynamicCodeMutator
from .dynamic_evaluator import CandidateAction, FastRiskAnalyzer


class SelfEvolvingKernel:
    """Closed-loop dynamic execution kernel evaluating risk and applying safe AST source mutations."""

    def __init__(
        self,
        canary_tester: Callable[[str], bool],
        analyzer: Optional[FastRiskAnalyzer] = None,
    ):
        self.analyzer = analyzer or FastRiskAnalyzer()
        self.canary_tester = canary_tester

    def execute_autonomous_adaptation(self, options: List[CandidateAction]) -> bool:
        """Evaluates N options by risk, writes AST mutation to disk, executes canary, and commits or rolls back."""
        # Step 1: Fast risk and variable analysis over N options
        ranked_options = self.analyzer.evaluate_and_rank(options)
        if not ranked_options:
            return False  # All options rejected due to excessive risk

        best_action = ranked_options[0]
        filepath = best_action.target_code_file
        backup_path = f"{filepath}.bak"

        # Backup code file before mutation
        shutil.copyfile(filepath, backup_path)

        try:
            # Step 2: Dynamically rewrite its own target code file
            target_fn = getattr(best_action, "target_function", "execute_task") or "execute_task"
            DynamicCodeMutator.mutate_function(
                filepath=filepath,
                fn_name=target_fn,
                new_code_str=best_action.proposed_patch,
            )

            # Step 3: Run isolated canary validation
            if self.canary_tester(filepath):
                # Success: mutation adopted permanently, clean backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                return True
            else:
                # Canary failed: restore original code
                shutil.copyfile(backup_path, filepath)
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                return False

        except Exception:
            # Revert immediately on error
            if os.path.exists(backup_path):
                shutil.copyfile(backup_path, filepath)
                os.remove(backup_path)
            return False


def make_default_canary_harness(sandbox: Optional[Any] = None) -> Callable[[str], bool]:
    """Isolated smoke and syntax test for mutated source files."""
    def canary_tester(filepath: str) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, filepath, "exec")
            if sandbox and hasattr(sandbox, "execute_python"):
                res = sandbox.execute_python(f"import py_compile; py_compile.compile(r'{filepath}', doraise=True)")
                return getattr(res, "exit_code", 1) == 0
            return True
        except Exception:
            return False
    return canary_tester
