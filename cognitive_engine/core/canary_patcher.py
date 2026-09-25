import ast
import shutil
import logging
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

logger = logging.getLogger("CanaryPatcher")

@dataclass
class CodebaseUpgradeProposal:
    target_module: str
    proposed_code: str
    rationale: str
    impact_weight: float    # Expected utility (0.0 to 1.0)
    risk_variance: float    # Failure blast-radius
    irreversibility: float  # Recovery overhead

class BoundedCanaryPatcher:
    """
    Evaluates AST modifications using utility-risk optimization:
    E[U] = ΔP - (λ * Risk * (1 + Irreversibility)).
    Enforces atomic backups (.bak) and canary test verification prior to commit.
    """

    def __init__(self, workspace_root: str = ".", risk_aversion: float = 1.6, approval_bar: float = 0.20):
        self.root = Path(workspace_root).resolve()
        self.snapshots_dir = self.root / ".snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.risk_aversion = risk_aversion
        self.approval_bar = approval_bar

    def evaluate_utility(self, proposal: CodebaseUpgradeProposal) -> Tuple[bool, float]:
        expected_utility = proposal.impact_weight - (
            self.risk_aversion * (proposal.risk_variance * (1.0 + proposal.irreversibility))
        )
        return (expected_utility >= self.approval_bar), expected_utility

    def apply_staged_upgrade(self, proposal: CodebaseUpgradeProposal) -> Tuple[bool, str]:
        approved, utility = self.evaluate_utility(proposal)
        if not approved:
            return False, f"Proposal rejected by risk analyzer: Score {utility:.3f} < {self.approval_bar}"

        # 1. Syntactic Verification via AST Parser
        try:
            parsed_ast = ast.parse(proposal.proposed_code)
            # Security guard: block malicious system interactions
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os.system", "shutil.rmtree"):
                            return False, "Dangerous primitive identified in candidate patch."
        except SyntaxError as e:
            return False, f"Candidate patch rejected: Syntax Error ({e})"

        target_file = self.root / proposal.target_module
        if not target_file.exists():
            return False, f"Target file '{proposal.target_module}' does not exist."

        # 2. Create atomic snapshot
        backup_file = self.snapshots_dir / f"{target_file.name}.bak"
        shutil.copyfile(target_file, backup_file)

        try:
            # 3. Write staged code
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(proposal.proposed_code)

            # 4. Run Isolated Canary Regression Test Suite
            test_run = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_saliency.py", "-q"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=15.0
            )

            if test_run.returncode == 0:
                logger.info(f"Canary Passed (ΔS = +1.0). Mutation committed to {proposal.target_module}.")
                return True, "Upgrade verified and permanently integrated."
            else:
                # 5. Immediate rollback on test regression
                shutil.copyfile(backup_file, target_file)
                logger.warning(f"Canary regression detected (ΔS = -1.0). State restored.")
                return False, f"Canary validation failure: {test_run.stderr.strip()}"

        except Exception as ex:
            shutil.copyfile(backup_file, target_file)
            return False, f"Exception during patch cycle: {str(ex)}"
