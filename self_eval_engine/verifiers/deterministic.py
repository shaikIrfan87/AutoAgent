import ast
import os
import subprocess
import sys
import tempfile
from ..schemas import VerificationResult


class PythonExecutionSandbox:
    def __init__(self, timeout_sec: int = 5):
        self.timeout_sec = timeout_sec

    def verify_code(self, code_snippet: str, unit_test: str = "") -> VerificationResult:
        full_script = f"{code_snippet}\n\n{unit_test}".strip()
        # Fast AST syntax pre-check (stdlib, zero process overhead)
        try:
            ast.parse(full_script)
        except SyntaxError as e:
            return VerificationResult(
                is_valid=False, score=0.0, failed_assertions=[f"SyntaxError: {e}"]
            )

        if not unit_test:
            return VerificationResult(
                is_valid=True, score=1.0, execution_logs="AST syntax valid"
            )

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as temp_f:
            temp_f.write(full_script)
            temp_path = temp_f.name

        proc = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_sec)
            if proc.returncode == 0:
                return VerificationResult(
                    is_valid=True, score=1.0, execution_logs=stdout
                )
            return VerificationResult(
                is_valid=False,
                score=0.0,
                failed_assertions=[stderr.strip() or "Assertion failed"],
                execution_logs=stdout,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # drain pipes and prevent zombie handles
            return VerificationResult(
                is_valid=False,
                score=0.0,
                failed_assertions=["Process timed out (infinite loop protection)"],
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
