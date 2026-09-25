import ast
from typing import Any, Optional


class RecursiveSelfCompiler:
    """Discovers working logic and compiles it into reusable skills."""
    def __init__(self, skill_library: Any):
        self.skills = skill_library

    def compile_and_persist(
        self, task_name: str, code_snippet: str, vector_embedding: Optional[Any] = None
    ) -> bool:
        try:
            tree = ast.parse(code_snippet)
            if not any(isinstance(node, (ast.FunctionDef, ast.Assign)) for node in tree.body):
                return False

            if hasattr(self.skills, "register_skill"):
                self.skills.register_skill(name=task_name, code=code_snippet, doc="Autonomous induction")
            elif hasattr(self.skills, "add_skill"):
                self.skills.add_skill(
                    name=task_name,
                    code=code_snippet,
                    embedding=vector_embedding,
                    metadata={"origin": "autonomous_induction"},
                )
            return True
        except SyntaxError:
            return False


if __name__ == "__main__":
    class MockSkills:
        def __init__(self):
            self.saved = {}
        def register_skill(self, name, code, doc=""):
            self.saved[name] = code
            return f"skills/{name}.py"

    compiler = RecursiveSelfCompiler(MockSkills())
    ok = compiler.compile_and_persist("add_fn", "def add(a, b):\n    return a + b")
    assert ok
    fail = compiler.compile_and_persist("bad_syntax", "def bad(")
    assert not fail
    print("RecursiveSelfCompiler check passed.")
