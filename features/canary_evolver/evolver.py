import ast
import shutil
from typing import Callable, List
from .models import KernelPatchAction

class SelfEvolvingKernelTCB:
    """
    Continuous AST Self-Evolver Canary Harness.
    Evaluates AST modifications using utility-risk optimization: E[U] = ΔP - (λ · Risk · (1 + Irrev)).
    Enforces atomic .bak rollback on failed canary execution.
    """
    def __init__(self, risk_aversion: float = 1.6, approval_cutoff: float = 0.20):
        self.risk_aversion = risk_aversion
        self.approval_cutoff = approval_cutoff

    def rank_patches(self, options: List[KernelPatchAction]) -> List[KernelPatchAction]:
        approved = []
        for opt in options:
            expected_utility = opt.impact_weight - (
                self.risk_aversion * (opt.risk_variance * (1.0 + opt.irreversibility))
            )
            if expected_utility >= self.approval_cutoff:
                approved.append((expected_utility, opt))
        approved.sort(key=lambda x: x[0], reverse=True)
        return [opt for _, opt in approved]

    def apply_patch_safely(self, patch: KernelPatchAction, canary_test_fn: Callable[[str], bool]) -> bool:
        backup_file = f"{patch.target_filepath}.bak"
        try:
            ast.parse(patch.proposed_ast_str)
        except SyntaxError:
            return False

        shutil.copyfile(patch.target_filepath, backup_file)
        try:
            with open(patch.target_filepath, "w", encoding="utf-8") as f:
                f.write(patch.proposed_ast_str)

            passed = canary_test_fn(patch.target_filepath)
            if not passed:
                shutil.copyfile(backup_file, patch.target_filepath)
                return False
            return True
        except Exception:
            shutil.copyfile(backup_file, patch.target_filepath)
            return False
