import os
import re
from typing import Dict, Any, Tuple, Optional


class HostCommitGate:
    """Attests whether virtual modifications are safe to apply to the real system."""

    FORBIDDEN_OPERATIONS = [
        r"\bformat\s+[a-zA-Z]:",
        r"\brmdir\s+/[sS]",
        r"\bdel\s+/[fFqsS]\s+[cC]:\\",
        r"socket\.",
        r"os\.system",
        r"subprocess\.Popen",
        r"chrome\.exe",
    ]

    def __init__(self, causal_verifier=None, protocol_guard=None):
        self.causal = causal_verifier
        self.guard = protocol_guard

    def inspect_and_verify(self, candidate_code: str, delta_s: float) -> Tuple[bool, str]:
        # 1. Reject if sandboxed execution failed
        if delta_s <= 0.0:
            return False, f"Rejected: Virtual sandbox execution failed (Delta_S = {delta_s})"

        # 2. Invariant Check: Scan for forbidden OS mutations or host escapes
        for pattern in self.FORBIDDEN_OPERATIONS:
            if re.search(pattern, candidate_code, re.IGNORECASE):
                return False, f"Security Violation: Prohibited host pattern matched: {pattern}"

        # 3. Protocol Egress Schema Compliance
        if self.guard:
            try:
                egress_decision = self.guard.enforce_egress_contract({
                    "action": "commit_to_host",
                    "code_payload": candidate_code,
                })
                if not egress_decision.get("valid", False):
                    return False, "Security Violation: Egress schema validation failed"
            except Exception as e:
                return False, f"Security Violation: Egress schema validation failed ({e})"

        return True, "Attestation Passed: Safe to commit to physical host system"

    def apply_to_real_host(self, target_filepath: str, verified_content: str):
        """Atomically applies the verified artifact to the real host system."""
        parent = os.path.dirname(target_filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_dest = f"{target_filepath}.atomic"
        with open(temp_dest, "w", encoding="utf-8") as f:
            f.write(verified_content)
        # Atomic rename prevents file corruption during unexpected termination
        os.replace(temp_dest, target_filepath)

    def attest_and_commit(self, target_rel_path: str, source_code: str, delta_s: float = 1.0) -> Tuple[bool, str]:
        """Attests candidate code safety and commits atomically to the host filesystem."""
        passed, reason = self.inspect_and_verify(source_code, delta_s=delta_s)
        if not passed:
            return False, reason
        self.apply_to_real_host(target_rel_path, source_code)
        return True, "Attestation Passed: Committed atomically to host"

    def inspect_and_verify_package(
        self, package_files: Dict[str, str], delta_s: float
    ) -> Tuple[bool, str]:
        """Attests whether a multi-file package is safe to commit to the real system."""
        if delta_s <= 0.0:
            return False, f"Rejected: Virtual sandbox execution failed (Delta_S = {delta_s})"
        if not package_files:
            return False, "Rejected: Package files dictionary is empty"

        for filepath, content in package_files.items():
            for pattern in self.FORBIDDEN_OPERATIONS:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Security Violation in '{filepath}': Prohibited host pattern matched: {pattern}"

            if self.guard:
                try:
                    egress_decision = self.guard.enforce_egress_contract({
                        "action": "commit_to_host",
                        "target_file": filepath,
                        "code_payload": content,
                    })
                    if not egress_decision.get("valid", False):
                        return False, f"Security Violation in '{filepath}': Egress schema validation failed"
                except Exception as e:
                    return False, f"Security Violation in '{filepath}': Egress schema validation failed ({e})"

        return True, f"Attestation Passed: Safe to commit multi-file package ({len(package_files)} files)"

    def apply_package_to_real_host(self, target_dir: str, package_files: Dict[str, str]) -> None:
        """Atomically applies a verified multi-file package to the real host system."""
        import shutil
        import uuid
        os.makedirs(target_dir, exist_ok=True)
        staging_dir = os.path.join(target_dir, f".staging_{uuid.uuid4().hex[:8]}")
        os.makedirs(staging_dir, exist_ok=True)
        try:
            for rel_path, content in package_files.items():
                dest_path = os.path.join(staging_dir, rel_path)
                parent = os.path.dirname(dest_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(content)

            for rel_path in package_files.keys():
                src = os.path.join(staging_dir, rel_path)
                dst = os.path.join(target_dir, rel_path)
                parent = os.path.dirname(dst)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                os.replace(src, dst)
        finally:
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

