# test_composite_run.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

from cognitive_engine.core.skills import SkillLibrary
from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer
from features.execution_sandbox.sandbox import IsolatedSandboxExecutor
from cognitive_engine.core.grounding_verifier import verify_executable_grounding

# 1. Mount persistent skill primitives
lib = SkillLibrary()
lib.scan_and_mount()

# 2. Instantiate synthesizer wired to skill library
synth = DynamicASTSynthesizer(skill_library=lib)
primitives = synth.load_skill_primitives("default_tenant")
print(f"Loaded {len(primitives)} primitive definitions into AST synthesizer.")

# 3. Target: Sort descending (quicksort -> list_reverse)
candidates = synth.generate_skill_composed_candidates(primitives)
print(f"Generated {len(candidates)} candidate composition ASTs.")

# 4. Search for the candidate that satisfies descending order
io_pairs = [
    ([3, 1, 4, 1, 5, 9], [9, 5, 4, 3, 1, 1]),
    ([10, -2, 5], [10, 5, -2]),
    ([42], [42]),
    ([], [])
]

sandbox = IsolatedSandboxExecutor()
found_solution = None

for cand_code in candidates:
    test_suite = cand_code + "\n" + "\n".join(
        f"assert solution({repr(inp)}) == {repr(out)}" for inp, out in io_pairs
    )
    delta_s, status = verify_executable_grounding(test_suite, sandbox)
    if delta_s == 1.0:
        found_solution = cand_code
        break

if found_solution:
    print("\n[SUCCESS] Composed skill pipeline discovered and verified (Delta S: +1.0):")
    print(found_solution)
else:
    print("\n[FAILED] No candidate satisfied the descending sort assertions.")