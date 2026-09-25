import ast
from typing import Tuple, Any


def verify_executable_grounding(test_code: str, sandbox: Any) -> Tuple[float, str]:
    """
    Verifies that the candidate assertion contains falsifiable unit tests
    rather than trivial variable assignments, and executes it in isolation.
    Returns: (delta_s: float [1.0 for verified, -1.0 for failure], message: str)
    """
    if not test_code or not test_code.strip():
        return -1.0, "Rejected: Grounding payload is empty."

    try:
        tree = ast.parse(test_code)
    except SyntaxError as e:
        return -1.0, f"Rejected: Grounding payload syntax error ({e})"

    # Disallow trivial code that only contains assignments and no assertions/checks
    has_assert_or_test = any(
        isinstance(node, (ast.Assert, ast.Raise))
        or (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("test", "verify", "check"))
        for node in ast.walk(tree)
    )
    if not has_assert_or_test:
        return -1.0, "Rejected: Grounding payload contains no falsifiable assertions."

    # Run in isolated execution sandbox
    try:
        if hasattr(sandbox, "execute_python"):
            result = sandbox.execute_python(test_code)
            if getattr(result, "exit_code", 1) == 0:
                return 1.0, "Verified: Assertion passed runtime execution."
            stderr = getattr(result, "stderr", "")
            return -1.0, f"Failed: Runtime verification error ({str(stderr).strip()})"
        elif hasattr(sandbox, "execute"):
            result = sandbox.execute(test_code)
            if isinstance(result, tuple):
                delta, msg = result
                if delta > 0.0:
                    return 1.0, "Verified: Assertion passed runtime execution."
                return -1.0, f"Failed: Runtime verification error ({msg})"
            elif getattr(result, "exit_code", 1) == 0:
                return 1.0, "Verified: Assertion passed runtime execution."
            else:
                stderr = getattr(result, "stderr", "")
                return -1.0, f"Failed: Runtime verification error ({str(stderr).strip()})"
        else:
            return -1.0, "Rejected: Invalid sandbox interface."
    except Exception as ex:
        return -1.0, f"Failed: Sandbox execution exception ({str(ex)})"
