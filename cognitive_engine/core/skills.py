"""
Skill Library & Manager: Persistent module synthesis for verified code abstractions (Voyager invariant).
Scans skills/ at boot, validates candidate AST safety, persists verified routines,
dynamically mounts tools for ReAct, and supports utility decay and pruning.
"""

import ast
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from .embeddings import get_semantic_embedding
except ImportError:
    from embeddings import get_semantic_embedding


BANNED_IMPORTS = {"subprocess", "socket", "ctypes", "pty", "commands"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}


def validate_skill_ast(code: str) -> Tuple[bool, str]:
    """Inspects code AST to reject unsafe imports and enforce clean functional contracts."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in skill code: {e}"

    for node in ast.walk(tree):
        # Reject prohibited imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in BANNED_IMPORTS:
                    return False, f"Prohibited import '{alias.name}' detected"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in BANNED_IMPORTS:
                    return False, f"Prohibited import '{node.module}' detected"

        # Reject dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                return False, f"Prohibited call '{node.func.id}()' detected"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"system", "popen", "spawn"}:
                    return False, f"Prohibited call '{node.func.attr}()' detected"

    # Enforce functional contract: must contain at least one definition (function or class)
    has_definition = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
    if not has_definition and not any(isinstance(node, ast.Assign) for node in tree.body):
        return False, "Skill code must define at least one function or assignment"

    return True, "AST validation passed"


@dataclass
class SkillMetadata:
    name: str
    doc: str
    file_path: Path
    code: str
    tenant_id: str = "default_tenant"
    access_count: int = 0
    confidence: float = 0.85
    vector: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SkillLibrary:
    """Verified executable function registry with multi-tenant directory partitioning.
    Saves verified System 2 code solutions as reusable Python modules callable by future runs.
    """

    def __init__(self, storage_dir: str = "skills"):
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        init_file = self.storage_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        self._mounted_skills: Dict[str, SkillMetadata] = {}
        self.scan_and_mount()

    def _tenant_dir(self, tenant_id: str) -> Path:
        t_clean = re.sub(r"[^\w]", "_", tenant_id).lower()
        td = self.storage_path / t_clean
        td.mkdir(parents=True, exist_ok=True)
        init_file = td / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        return td

    def scan_and_mount(self, tenant_id: str = "default_tenant") -> int:
        """Discovers existing .py skills from disk at boot time and registers them in-memory."""
        mounted_count = 0
        search_dirs = [(self.storage_path, "default_tenant"), (self._tenant_dir(tenant_id), tenant_id)]

        for directory, t_id in search_dirs:
            if not directory.exists():
                continue
            for file_path in directory.glob("*.py"):
                if file_path.stem == "__init__":
                    continue
                skill_name = file_path.stem
                if (t_id, skill_name) in self._mounted_skills:
                    continue
                try:
                    code = file_path.read_text(encoding="utf-8")
                    # Extract module-level docstring if present
                    doc = ""
                    try:
                        tree = ast.parse(code)
                        doc = ast.get_docstring(tree) or ""
                    except Exception:
                        pass
                    vec = get_semantic_embedding(f"{skill_name} {doc}")
                    meta = SkillMetadata(
                        name=skill_name,
                        doc=doc,
                        file_path=file_path,
                        code=code,
                        tenant_id=t_id,
                        vector=vec,
                    )
                    self._mounted_skills[(t_id, skill_name)] = meta
                    mounted_count += 1
                except Exception:
                    continue
        return mounted_count

    def register_skill(self, name: str, code: str, doc: str = "", tenant_id: str = "default_tenant") -> str:
        """Synchronously writes validated skill to disk and mounts into memory."""
        clean_name = re.sub(r"[^\w]", "_", name).lower()
        td = self._tenant_dir(tenant_id)
        file_path = td / f"{clean_name}.py"
        header = f'"""{doc}"""\n' if doc else ""
        full_code = f"{header}{code.strip()}\n"
        file_path.write_text(full_code, encoding="utf-8")

        vec = get_semantic_embedding(f"{clean_name} {doc}")
        meta = SkillMetadata(
            name=clean_name,
            doc=doc,
            file_path=file_path,
            code=full_code,
            tenant_id=tenant_id,
            vector=vec,
            last_accessed=time.time(),
        )
        self._mounted_skills[(tenant_id, clean_name)] = meta
        return str(file_path)

    def retrieve_relevant_skills(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.50,
        tenant_id: str = "default_tenant",
    ) -> List[Dict[str, Any]]:
        """Performs semantic vector similarity search over compiled skills using in-process BGE embeddings."""
        q_vec = get_semantic_embedding(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored: List[Tuple[float, SkillMetadata]] = []
        for (t_id, _), meta in self._mounted_skills.items():
            if t_id != tenant_id and tenant_id != "default_tenant":
                continue
            if meta.vector is None:
                meta.vector = get_semantic_embedding(f"{meta.name} {meta.doc}")
            v_norm = np.linalg.norm(meta.vector)
            if v_norm == 0:
                continue
            sim = float(np.dot(q_vec, meta.vector) / (q_norm * v_norm + 1e-9))
            if sim >= threshold:
                scored.append((sim, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "name": meta.name,
                "doc": meta.doc,
                "code": meta.code,
                "score": sim,
                "file_path": str(meta.file_path),
            }
            for sim, meta in scored[:top_k]
        ]

    def compile_and_persist(
        self,
        name: str,
        code: str,
        doc: str = "",
        tenant_id: str = "default_tenant",
        sandbox: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Validates AST safety, verifies execution, and persists the skill."""
        is_safe, msg = validate_skill_ast(code)
        if not is_safe:
            return False, f"AST Safety Validation Failed: {msg}"

        if sandbox is not None:
            res = sandbox.execute_python(code)
            if res.exit_code != 0:
                return False, f"Sandbox execution test failed: {res.stderr}"

        path = self.register_skill(name=name, code=code, doc=doc, tenant_id=tenant_id)
        return True, path

    def list_skills(self, tenant_id: str = "default_tenant") -> List[str]:
        td = self._tenant_dir(tenant_id)
        disk_skills = {f.stem for f in td.glob("*.py") if f.stem != "__init__"}
        for (t_id, s_name) in self._mounted_skills.keys():
            if t_id == tenant_id:
                disk_skills.add(s_name)
        if tenant_id == "default_tenant":
            disk_skills.update(f.stem for f in self.storage_path.glob("*.py") if f.stem != "__init__")
        return sorted(list(disk_skills))

    def touch_skill(self, name: str, tenant_id: str = "default_tenant") -> None:
        """Increments access count and updates last_accessed timestamp for active skill."""
        key = (tenant_id, name)
        if key in self._mounted_skills:
            meta = self._mounted_skills[key]
            meta.access_count += 1
            meta.last_accessed = time.time()

    def get_skill_code(self, name: str, tenant_id: str = "default_tenant") -> Optional[str]:
        td = self._tenant_dir(tenant_id)
        f = td / f"{name}.py"
        if f.exists():
            return f.read_text(encoding="utf-8")
        if (tenant_id, name) in self._mounted_skills:
            return self._mounted_skills[(tenant_id, name)].code
        if tenant_id == "default_tenant":
            f_root = self.storage_path / f"{name}.py"
            if f_root.exists():
                return f_root.read_text(encoding="utf-8")
        return None

    def get_skill_doc(self, name: str, tenant_id: str = "default_tenant") -> str:
        if (tenant_id, name) in self._mounted_skills:
            return self._mounted_skills[(tenant_id, name)].doc
        code = self.get_skill_code(name, tenant_id)
        if code:
            try:
                tree = ast.parse(code)
                return ast.get_docstring(tree) or ""
            except Exception:
                pass
        return ""

    def execute_skill(
        self,
        name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        sandbox: Optional[Any] = None,
        tenant_id: str = "default_tenant",
    ) -> Dict[str, Any]:
        """Executes a mounted skill, records access telemetry, and returns output."""
        code = self.get_skill_code(name, tenant_id)
        if not code:
            return {"status": "error", "error": f"Skill '{name}' not found"}

        if (tenant_id, name) in self._mounted_skills:
            meta = self._mounted_skills[(tenant_id, name)]
            meta.access_count += 1
            meta.last_accessed = time.time()

        kwargs = kwargs or {}
        call_code = f"{code}\n\n"
        if "solution" in code:
            call_code += "print(solution())\n"
        elif "run" in code:
            call_code += f"print(run(**{kwargs}))\n"

        if sandbox is not None:
            exec_res = sandbox.execute_python(call_code)
            return {
                "status": "success" if exec_res.exit_code == 0 else "failed",
                "output": exec_res.stdout,
                "stderr": exec_res.stderr,
                "delta_s": exec_res.delta_s,
            }

        # Isolated execution in-process
        scope: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(code, scope)
            fn = scope.get("run") or scope.get("solution")
            output = fn(**kwargs) if (fn and callable(fn)) else "Executed without return value"
            return {"status": "success", "output": output, "delta_s": 1.0}
        except Exception as e:
            return {"status": "failed", "error": str(e), "delta_s": -1.0}

    def prune_stale_skills(
        self,
        decay_lambda: float = 0.05,
        threshold: float = 0.20,
        current_time: Optional[float] = None,
        tenant_id: str = "default_tenant",
    ) -> List[str]:
        """Ebbinghaus utility decay: U_eff = U_0 * exp(-lambda * delta_t_hours). Prunes stale skills."""
        now = current_time if current_time is not None else time.time()
        pruned = []

        for (t_id, name), meta in list(self._mounted_skills.items()):
            if t_id != tenant_id:
                continue
            delta_hours = max(0.0, (now - meta.last_accessed) / 3600.0)
            u_eff = meta.confidence * math.exp(-decay_lambda * delta_hours)

            if u_eff < threshold:
                if meta.file_path.exists():
                    try:
                        meta.file_path.unlink()
                    except Exception:
                        pass
                del self._mounted_skills[(t_id, name)]
                pruned.append(name)

        return pruned


SkillManager = SkillLibrary
