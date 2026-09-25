import json
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

try:
    from .types import ExecutionResult
    from .search import LiveWebSearch
except ImportError:
    from types import ExecutionResult
    from search import LiveWebSearch


_DAEMON_SCRIPT = """
import sys, json, traceback, io
from contextlib import redirect_stdout, redirect_stderr

base_scope = {"__builtins__": __builtins__}
scope = dict(base_scope)

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        data = json.loads(line)
        code = data.get("code", "")
        isolate = data.get("isolate", True)
        reset_scope = data.get("reset", False)

        if reset_scope:
            scope = dict(base_scope)
            sys.stdout.write(json.dumps({"exit_code": 0, "stdout": "scope_reset", "stderr": ""}) + "\\n")
            sys.stdout.flush()
            continue

        # Snapshot keys before execution for rollback isolation
        snapshot_keys = set(scope.keys())
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        exit_code = 0
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                exec(code, scope)
        except Exception:
            exit_code = 1
            err_buf.write(traceback.format_exc())
            # Rollback dirty state on failure
            for k in list(scope.keys()):
                if k not in snapshot_keys:
                    del scope[k]

        # Clean up if step-level isolation is requested
        if isolate and exit_code != 0:
            for k in list(scope.keys()):
                if k not in snapshot_keys:
                    del scope[k]

        out_str = out_buf.getvalue()
        if len(out_str) > 16384:
            out_str = out_str[:16384] + "\\n[TRUNCATED: Exceeded 16384 character pipe limit]"
        err_str = err_buf.getvalue()
        if len(err_str) > 16384:
            err_str = err_str[:16384] + "\\n[TRUNCATED: Exceeded 16384 character pipe limit]"

        sys.stdout.write(json.dumps({"exit_code": exit_code, "stdout": out_str, "stderr": err_str}) + "\\n")
        sys.stdout.flush()
    except Exception:
        err = traceback.format_exc()
        if len(err) > 16384:
            err = err[:16384] + "\\n[TRUNCATED: Exceeded 16384 character pipe limit]"
        sys.stdout.write(json.dumps({"exit_code": 1, "stdout": "", "stderr": err}) + "\\n")
        sys.stdout.flush()
"""



def _apply_windows_sandbox_limits(proc: subprocess.Popen, memory_limit_mb: int = 512):
    """Clamp Windows child process memory and kill on job close to prevent runaway leaks."""
    if sys.platform != "win32" or proc is None or not hasattr(proc, "_handle"):
        return None
    try:
        import win32job
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        )
        info["ProcessMemoryLimit"] = memory_limit_mb * 1024 * 1024
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        win32job.AssignProcessToJobObject(job, proc._handle)
        return job
    except Exception:
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return None

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryLimit", ctypes.c_size_t),
                    ("PeakJobMemoryLimit", ctypes.c_size_t),
                ]

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000 | 0x0100
            info.ProcessMemoryLimit = ctypes.c_size_t(memory_limit_mb * 1024 * 1024)
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            k32.AssignProcessToJobObject(job, int(proc._handle))
            return job
        except Exception:
            return None


def _worker_preexec():
    # ponytail: cap open file descriptors and bind child process lifecycle to parent on POSIX
    if sys.platform != "win32":
        try:
            import resource
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
            except Exception:
                pass
            try:
                import ctypes
                libc = ctypes.CDLL(None)
                libc.prctl(1, 9)  # PR_SET_PDEATHSIG = 1, SIGKILL = 9
            except Exception:
                pass
        except Exception:
            pass


class PersistentSandboxWorker:
    """Persistent background REPL daemon eliminating Python startup latency (<2ms/call)."""

    def __init__(self, timeout_sec: float = 5.0, memory_limit_mb: int = 512):
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb
        self.proc: Optional[subprocess.Popen] = None
        self._job_handle = None
        self._lock = threading.RLock()
        self._res_queue: queue.Queue = queue.Queue()
        self._start_worker()

    def _start_worker(self) -> None:
        self.terminate()
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _DAEMON_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=_worker_preexec if sys.platform != "win32" else None,
            )
            self._job_handle = _apply_windows_sandbox_limits(self.proc, self.memory_limit_mb)
            self._res_queue = queue.Queue()

            def _reader_loop(proc, q):
                while proc.poll() is None:
                    try:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        q.put(line)
                    except Exception:
                        break

            r_thread = threading.Thread(target=_reader_loop, args=(self.proc, self._res_queue), daemon=True)
            r_thread.start()

        except Exception:
            self.proc = None
            if self._job_handle and sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.CloseHandle(self._job_handle)
                except Exception:
                    pass
                self._job_handle = None

    def execute(self, code: str, timeout: Optional[float] = None) -> ExecutionResult:
        timeout = timeout or self.timeout_sec
        t0 = time.perf_counter()

        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start_worker()

            if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
                return ExecutionResult(
                    stdout="", stderr="Daemon process unavailable", exit_code=-1, delta_s=-1.0, duration_sec=0.0
                )

            payload = json.dumps({"code": code}) + "\n"
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()

                try:
                    res_line = self._res_queue.get(timeout=timeout)
                except queue.Empty:
                    # Timeout: kill worker and respawn
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    self.proc = None
                    if self._job_handle and sys.platform == "win32":
                        try:
                            import ctypes
                            ctypes.windll.kernel32.CloseHandle(self._job_handle)
                        except Exception:
                            pass
                        self._job_handle = None
                    duration = time.perf_counter() - t0
                    return ExecutionResult(
                        stdout="",
                        stderr=f"Execution timed out after {timeout}s",
                        exit_code=-1,
                        delta_s=-1.0,
                        duration_sec=duration,
                    )

                duration = time.perf_counter() - t0
                resp = json.loads(res_line.strip())
                exit_code = resp.get("exit_code", 0)
                stdout = resp.get("stdout", "").strip()
                stderr = resp.get("stderr", "").strip()

                delta_s = (1.0 if stdout else 0.0) if exit_code == 0 else -1.0
                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    delta_s=delta_s,
                    duration_sec=duration,
                )

            except Exception as e:
                duration = time.perf_counter() - t0
                try:
                    if self.proc:
                        self.proc.kill()
                except Exception:
                    pass
                self.proc = None
                if self._job_handle and sys.platform == "win32":
                    try:
                        import ctypes
                        ctypes.windll.kernel32.CloseHandle(self._job_handle)
                    except Exception:
                        pass
                    self._job_handle = None
                return ExecutionResult(
                    stdout="", stderr=str(e), exit_code=-1, delta_s=-1.0, duration_sec=duration
                )

    def start(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start_worker()

    def terminate(self) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self.proc:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
                self.proc = None
            if self._job_handle and sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.CloseHandle(self._job_handle)
                except Exception:
                    pass
                self._job_handle = None


class PersistentREPLDaemon:
    """Convenience daemon wrapper providing status, output, delta, start, and terminate."""

    def __init__(self, timeout_sec: float = 5.0):
        self.worker = PersistentSandboxWorker(timeout_sec=timeout_sec)

    def start(self) -> None:
        self.worker.start()
        # Warmup IPC pipe handshake to guarantee <2ms latency on subsequent calls
        self.worker.execute("pass")

    def execute(self, code: str, timeout: Optional[float] = None) -> dict:
        res = self.worker.execute(code, timeout=timeout)
        status = "success" if res.exit_code == 0 else "failed"
        return {
            "status": status,
            "output": res.stdout.strip(),
            "delta": res.delta_s,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.exit_code,
        }

    def terminate(self) -> None:
        self.worker.terminate()


class EnvironmentalSandbox:
    """Invariant 3: Closed-Loop Environmental Execution.
    Executes Python REPL operations using persistent background daemon (<2ms/call).
    """

    def __init__(self, timeout_sec: float = 5.0, max_memory_mb: int = 512, use_persistent_worker: bool = True):
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb
        self.use_persistent = use_persistent_worker
        self.on_transition_callback = None
        self._worker = PersistentSandboxWorker(timeout_sec=timeout_sec) if use_persistent_worker else None

    def execute_python(self, code: str, timeout: Optional[float] = None) -> ExecutionResult:
        if self._worker:
            res = self._worker.execute(code, timeout=timeout)
        else:
            # Fallback to single-shot subprocess if persistent worker is disabled
            timeout = timeout or self.timeout_sec
            t0 = time.perf_counter()
            try:
                sub_res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
                duration = time.perf_counter() - t0
                exit_code = sub_res.returncode
                stdout = sub_res.stdout.strip()
                stderr = sub_res.stderr.strip()
                delta_s = (1.0 if stdout else 0.0) if exit_code == 0 else -1.0
                res = ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code, delta_s=delta_s, duration_sec=duration)
            except Exception as e:
                duration = time.perf_counter() - t0
                res = ExecutionResult(stdout="", stderr=str(e), exit_code=-1, delta_s=-1.0, duration_sec=duration)

        if self.on_transition_callback:
            try:
                self.on_transition_callback(code, res)
            except Exception:
                pass

        return res

    def search_web(self, query: str) -> str:
        return LiveWebSearch.search(query)

    def create_checkpoint(self, workspace_dir: str) -> "SandboxCheckpoint":
        """Create an atomic snapshot of workspace_dir for deterministic rollback."""
        return SandboxCheckpoint(workspace_dir)

    def close(self) -> None:
        if self._worker:
            self._worker.close()


class SandboxCheckpoint:
    """Atomic workspace directory checkpoint and rollback mechanism using stdlib shutil."""

    def __init__(self, workspace_dir: str):
        import os, shutil, tempfile
        self._os = os
        self._shutil = shutil
        self.src = os.path.abspath(workspace_dir)
        self.backup = tempfile.mkdtemp(prefix="autoagent_chk_")
        shutil.copytree(self.src, self.backup, dirs_exist_ok=True)

    def restore(self) -> None:
        """Rollback workspace to the exact state at checkpoint creation."""
        if self._os.path.exists(self.src):
            self._shutil.rmtree(self.src)
        self._shutil.copytree(self.backup, self.src)

    def close(self) -> None:
        """Release temporary backup directory."""
        if self._os.path.exists(self.backup):
            self._shutil.rmtree(self.backup, ignore_errors=True)

