import subprocess
import sys
from typing import Tuple


class EnvironmentalVerifier:
    """Grounds symbolic hypotheses against empirical execution."""
    def __init__(self, timeout_sec: float = 2.5):
        self.timeout_sec = timeout_sec

    def verify_hypothesis(self, test_code: str) -> Tuple[float, str]:
        """Runs candidate logic in an isolated subprocess with stripped environment."""
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", test_code],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            if proc.returncode == 0:
                return 1.0, proc.stdout.strip()
            return -1.0, proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1.0, "TimeoutExpired"
        except Exception as ex:
            return -1.0, str(ex)


if __name__ == "__main__":
    verifier = EnvironmentalVerifier()
    delta_s, out = verifier.verify_hypothesis("print(1 + 1)")
    assert delta_s == 1.0 and out == "2"
    delta_fail, _ = verifier.verify_hypothesis("assert 1 == 2")
    assert delta_fail == -1.0
    print("EnvironmentalVerifier check passed.")
