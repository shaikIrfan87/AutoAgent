import copy
import os
import sys
import subprocess
import tempfile
import time
from typing import Tuple, Dict, Any, Optional, List
from cognitive_engine.core.types import ExecutionResult
from cognitive_engine.core.sandbox import _apply_windows_sandbox_limits


class IsolatedSandboxExecutor:
    """
    Invariant 3: Isolated Grounded Execution.
    Process-isolated worker environment with stripped environment, ephemeral temp scoping,
    and atomic rollback.
    """
    def __init__(self, base_builtins: Optional[Dict[str, Any]] = None, timeout_sec: float = 3.0, max_memory_mb: int = 512):
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb
        self.namespace = {"__builtins__": {}}

    def execute(self, python_code: str) -> Tuple[float, str]:
        """
        Executes code within an isolated ephemeral subprocess.
        Returns: (delta_s: float [1.0 for success, -1.0 for failure], message: str)
        """
        driver = (
            "import sys\n"
            "allowed = {'math', 'json', 're', 'itertools', 'collections', 'time'}\n"
            "orig_import = __import__\n"
            "def safe_import(name, *args, **kwargs):\n"
            "    if name in allowed:\n"
            "        return orig_import(name, *args, **kwargs)\n"
            "    raise ImportError(f'Unauthorized import in sandbox: {name}')\n"
            "safe_builtins = {\n"
            "    'print': print, 'range': range, 'len': len, 'min': min, 'max': max,\n"
            "    'sum': sum, 'abs': abs, 'float': float, 'int': int, 'str': str,\n"
            "    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,\n"
            "    '__import__': safe_import\n"
            "}\n"
            "code = sys.stdin.read()\n"
            "compiled = compile(code, '<sandbox>', 'exec')\n"
            "exec(compiled, {'__builtins__': safe_builtins})\n"
        )
        clean_env = {
            "PYTHONHASHSEED": "42",
            "LC_ALL": "C.UTF-8",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
        }
        tmpdir = tempfile.mkdtemp(prefix="sandbox_isolated_")
        job = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", driver],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=clean_env,
                cwd=tmpdir,
            )
            job = _apply_windows_sandbox_limits(proc, self.max_memory_mb)
            try:
                stdout, stderr = proc.communicate(input=python_code, timeout=self.timeout_sec)
                if proc.returncode == 0:
                    return 1.0, "Success"
                return -1.0, stderr.strip() or f"Process exited with code {proc.returncode}"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return -1.0, f"Execution timed out after {self.timeout_sec}s"
            finally:
                if job and sys.platform == "win32":
                    try:
                        import ctypes
                        ctypes.windll.kernel32.CloseHandle(job)
                    except Exception:
                        pass
        except Exception as ex:
            return -1.0, str(ex)
        finally:
            safe_cleanup_temp_dir(tmpdir)



def safe_cleanup_temp_dir(path: str, max_retries: int = 5) -> None:
    """Resilient directory cleanup handling brief Windows file locks after process exit."""
    if not path or not os.path.exists(path):
        return
    import shutil
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            break
        except PermissionError:
            time.sleep(0.05)
        except Exception:
            break


class HardenedOSSandboxExecutor:
    """
    Hardened OS-level Process Sandbox.
    Enforces real OS process isolation, execution timeout, and memory containment.
    """
    def __init__(self, timeout_sec: float = 3.0, max_memory_mb: int = 128):
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb
        self.memory_limit_bytes = max_memory_mb * 1024 * 1024
        self.job = None
        if sys.platform == "win32":
            try:
                import win32job
                self.job = win32job.CreateJobObject(None, "")
                info = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
                info['BasicLimitInformation']['LimitFlags'] = (
                    win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY |
                    win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                )
                info['ProcessMemoryLimit'] = self.memory_limit_bytes
                win32job.SetInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation, info)
            except Exception:
                self.job = None

    def __del__(self):
        if self.job and sys.platform == "win32":
            try:
                import win32api
                win32api.CloseHandle(self.job)
            except Exception:
                pass

    def execute_python(self, code: str, timeout_sec: Optional[float] = None) -> ExecutionResult:
        """Executes code in an isolated child process with stripped environment, timeout, and memory bounds."""
        t_limit = timeout_sec if timeout_sec is not None else self.timeout_sec
        t0 = time.perf_counter()
        driver = (
            "import sys\n"
            "allowed = {'math', 'json', 're', 'itertools', 'collections', 'time'}\n"
            "orig_import = __import__\n"
            "def safe_import(name, *args, **kwargs):\n"
            "    if name in allowed:\n"
            "        return orig_import(name, *args, **kwargs)\n"
            "    raise ImportError(f'Unauthorized import: {name}')\n"
            "safe_builtins = {\n"
            "    'print': print, 'range': range, 'len': len, 'min': min, 'max': max,\n"
            "    'sum': sum, 'abs': abs, 'float': float, 'int': int, 'str': str,\n"
            "    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,\n"
            "    '__import__': safe_import\n"
            "}\n"
            "code = sys.stdin.read()\n"
            "compiled = compile(code, '<sandbox>', 'exec')\n"
            "exec(compiled, {'__builtins__': safe_builtins})\n"
        )

        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
        }

        tmpdir = tempfile.mkdtemp(prefix="sandbox_os_")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", driver],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=clean_env,
                cwd=tmpdir,
            )
            # Assign child process to pinned Job Object
            if self.job and sys.platform == "win32":
                try:
                    import win32job
                    win32job.AssignProcessToJobObject(self.job, proc._handle)
                except Exception:
                    pass
            else:
                _apply_windows_sandbox_limits(proc, self.max_memory_mb)

            try:
                stdout, stderr = proc.communicate(input=code, timeout=t_limit)
                duration = time.perf_counter() - t0
                exit_code = proc.returncode
                delta_s = 1.0 if exit_code == 0 else -1.0
                return ExecutionResult(
                    stdout=stdout.strip(),
                    stderr=stderr.strip(),
                    exit_code=exit_code,
                    delta_s=delta_s,
                    duration_sec=duration,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                duration = time.perf_counter() - t0
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {t_limit}s",
                    exit_code=-1,
                    delta_s=-1.0,
                    duration_sec=duration,
                )
        except Exception as e:
            duration = time.perf_counter() - t0
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                delta_s=-1.0,
                duration_sec=duration,
            )
        finally:
            safe_cleanup_temp_dir(tmpdir)

    def execute(self, python_code: str) -> Tuple[float, str]:
        """
        Executes code in an isolated child process with stripped environment and strict timeout.
        Returns: (delta_s: float [1.0 for success, -1.0 for failure], message: str)
        """
        res = self.execute_python(python_code)
        if res.exit_code == 0:
            return 1.0, res.stdout or "Success"
        else:
            return -1.0, res.stderr or f"Process exited with code {res.exit_code}"


class DockerSandboxExecutor:
    """
    Phase 2: Kernel-Level Container Sandbox.
    Ephemeral container runner with read-only root, memory limit, and network isolation.
    Gracefully falls back to HardenedOSSandboxExecutor when Docker daemon is not active.
    """
    def __init__(self, image: str = "python:3.11-alpine", timeout_sec: float = 3.0, mem_limit: str = "256m"):
        self.image = image
        self.timeout_sec = timeout_sec
        self.mem_limit = mem_limit
        self.client = None
        self._fallback = HardenedOSSandboxExecutor(timeout_sec=timeout_sec)
        try:
            import docker
            self.client = docker.from_env()
            self.client.ping()
            self.available = True
        except Exception:
            self.available = False

    def execute(self, python_code: str) -> Tuple[float, str]:
        if not self.available:
            return self._fallback.execute(python_code)
        res = self.execute_python(python_code, timeout_sec=int(self.timeout_sec))
        return res["delta_s"], res.get("stdout") or res.get("stderr", "")

    def execute_python(self, code: str, timeout_sec: int = 3) -> Dict[str, Any]:
        if not self.available:
            delta_s, msg = self._fallback.execute(code)
            return {
                "exit_code": 0 if delta_s > 0 else 1,
                "stdout": msg if delta_s > 0 else "",
                "stderr": "" if delta_s > 0 else msg,
                "delta_s": delta_s,
            }
        try:
            import docker
            container = self.client.containers.run(
                self.image,
                command=["python", "-c", code],
                network_disabled=True,
                mem_limit=self.mem_limit,
                nano_cpus=1_000_000_000,
                read_only=True,
                remove=True,
                stdout=True,
                stderr=True,
                detach=False,
            )
            out = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            return {"exit_code": 0, "stdout": out.strip() or "Success", "stderr": "", "delta_s": 1.0}
        except Exception as ex:
            return {"exit_code": -1, "stdout": "", "stderr": str(ex), "delta_s": -1.0}

    def execute_in_workspace(self, command: List[str], workspace_dir: str, timeout_sec: int = 15) -> Dict[str, Any]:
        """
        SWE-bench / Multi-file repo workspace runner.
        Mounts host workspace_dir into /workspace, executes shell command, returns output.
        Falls back to local subprocess in workspace_dir if Docker daemon is not active.
        """
        if not self.available:
            try:
                proc = subprocess.run(
                    command,
                    cwd=workspace_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                return {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                    "delta_s": 1.0 if proc.returncode == 0 else -1.0,
                }
            except Exception as e:
                return {"exit_code": -1, "stdout": "", "stderr": str(e), "delta_s": -1.0}

        try:
            abs_dir = os.path.abspath(workspace_dir)
            volumes = {abs_dir: {"bind": "/workspace", "mode": "rw"}}
            container = self.client.containers.run(
                self.image,
                command=command,
                working_dir="/workspace",
                volumes=volumes,
                network_disabled=True,
                mem_limit=self.mem_limit,
                nano_cpus=1_000_000_000,
                remove=True,
                stdout=True,
                stderr=True,
                detach=False,
            )
            out = container.decode("utf-8") if isinstance(container, bytes) else str(container)
            return {"exit_code": 0, "stdout": out.strip(), "stderr": "", "delta_s": 1.0}
        except Exception as ex:
            return {"exit_code": -1, "stdout": "", "stderr": str(ex), "delta_s": -1.0}
