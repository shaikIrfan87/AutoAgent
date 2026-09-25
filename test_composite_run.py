"""
Composite skill test: verify quicksort -> list_reverse produces descending sort.
Uses the promoted benchmark skills via HostCommitGate-attested files.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from cognitive_engine.core.skills import SkillLibrary
from features.execution_sandbox.sandbox import IsolatedSandboxExecutor
from cognitive_engine.core.grounding_verifier import verify_executable_grounding

lib = SkillLibrary()
lib.scan_and_mount()

# Load the two promoted skills by name
qs_code  = lib.get_skill_code("quicksort")
rev_code = lib.get_skill_code("list_reverse")

print(f"quicksort skill:\n{qs_code}\n")
print(f"list_reverse skill:\n{rev_code}\n")

# Compose: descending_sort(x) = list_reverse(quicksort(x))
composed = f"""{qs_code.strip()}

{rev_code.strip()}

def solution(x):
    return reverse_fn(quicksort(x))
"""

io_pairs = [
    ([3, 1, 4, 1, 5, 9], [9, 5, 4, 3, 1, 1]),
    ([10, -2, 5],         [10, 5, -2]),
    ([42],                [42]),
    ([],                  []),
]

test_suite = composed + "\n" + "\n".join(
    f"assert solution({repr(inp)}) == {repr(out)}" for inp, out in io_pairs
)

sandbox = IsolatedSandboxExecutor()
delta_s, status = verify_executable_grounding(test_suite, sandbox)

if delta_s == 1.0:
    print("[SUCCESS] Composed skill pipeline verified (Delta S: +1.0)")
    print(composed)
else:
    print(f"[FAILED] Delta S: {delta_s}, status: {status}")
