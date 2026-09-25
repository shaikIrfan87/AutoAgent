import ast
import importlib
import os
import sys
from typing import Any, Callable, Optional


class DynamicCodeMutator:
    """Modifies its own source code files while running via AST metaprogramming."""

    @staticmethod
    def inspect_file(filepath: str) -> ast.AST:
        with open(filepath, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=filepath)

    @staticmethod
    def mutate_function(filepath: str, fn_name: str, new_code_str: str, allow_append: bool = True) -> bool:
        """
        Parses source file, swaps function definition with new logic (or appends if not found and allow_append=True),
        writes back to disk, and reloads into the running Python runtime.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        parsed_new = ast.parse(new_code_str.strip())
        if not parsed_new.body or not isinstance(parsed_new.body[0], ast.FunctionDef):
            raise ValueError(f"Proposed patch does not define a valid function AST: {new_code_str}")
        new_fn_ast = parsed_new.body[0]

        class FunctionReplacer(ast.NodeTransformer):
            def __init__(self):
                self.replaced = False

            def visit_FunctionDef(self, node):
                if node.name == fn_name:
                    self.replaced = True
                    return ast.copy_location(new_fn_ast, node)
                return self.generic_visit(node)

        replacer = FunctionReplacer()
        modified_tree = replacer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        if not replacer.replaced:
            if allow_append:
                modified_tree.body.append(new_fn_ast)
                ast.fix_missing_locations(modified_tree)
            else:
                raise ValueError(f"Target function '{fn_name}' not found in {filepath}")

        # 1. Compile test to confirm no syntax/AST corruption
        compile(modified_tree, filename=filepath, mode="exec")

        # 2. Write mutated code to disk so changes persist
        new_source = ast.unparse(modified_tree)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)

        # 3. Reload module into active memory
        norm_target = os.path.abspath(filepath).lower()
        for mod_name, module in list(sys.modules.items()):
            mod_file = getattr(module, "__file__", None)
            if mod_file and os.path.abspath(mod_file).lower() == norm_target:
                importlib.reload(module)
                break
        return True
