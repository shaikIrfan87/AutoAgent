import pytest
from features.execution_sandbox.sandbox import HardenedOSSandboxExecutor

@pytest.fixture
def sandbox():
    return HardenedOSSandboxExecutor()

def test_sandbox_forkbomb_and_thread_exhaustion(sandbox):
    """Attempt thread/process proliferation to exhaust host PID table."""
    fork_bomb = """
import threading, time
def loop():
    while True: time.sleep(0.01)
for _ in range(500):
    t = threading.Thread(target=loop)
    t.daemon = True
    t.start()
print('SPAWNED')
"""
    res = sandbox.execute_python(fork_bomb, timeout_sec=2)
    # The sandbox must either kill the process within 2s or enforce resource limits
    assert res.exit_code != 0 or "SPAWNED" in res.stdout
    assert res.duration_sec <= 2.5, f"Sandbox hung on thread allocation: {res.duration_sec}s"

def test_sandbox_memory_bomb(sandbox):
    """Attempt allocation of 2GB RAM to verify memory ceiling."""
    mem_bomb = """
a = 'A' * (1024 * 1024 * 500) # 500MB chunk
b = [a] * 4                   # 2GB virtual memory
print(len(b))
"""
    res = sandbox.execute_python(mem_bomb, timeout_sec=2)
    assert res.exit_code != 0 or res.delta_s == -1.0, "Memory allocation was not intercepted"

def test_sandbox_host_filesystem_escape(sandbox):
    """Verify write isolation outside the workspace."""
    escape_code = """
import os
try:
    with open('C:\\\\Windows\\\\Temp\\\\autoagent_escape_test.txt', 'w') as f:
        f.write('ESCAPED')
    print('ESCAPED')
except Exception as ex:
    print('BLOCKED:', ex)
"""
    res = sandbox.execute_python(escape_code, timeout_sec=2)
    assert "BLOCKED" in res.stdout or res.exit_code != 0
