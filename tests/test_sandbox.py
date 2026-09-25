from cognitive_engine.core.sandbox import EnvironmentalSandbox
from cognitive_engine.agent.executive_loop import ExecutiveLoop


def test_sandbox_broken_code_delta():
    sandbox = EnvironmentalSandbox()
    broken_code = "x = 10 / 0\nprint(x)"
    res = sandbox.execute_python(broken_code)
    assert res.exit_code != 0
    assert res.delta_s == -1.0
    assert "ZeroDivisionError" in res.stderr


def test_sandbox_successful_code_delta():
    sandbox = EnvironmentalSandbox()
    valid_code = "print('State change verified')"
    res = sandbox.execute_python(valid_code)
    assert res.exit_code == 0
    assert res.delta_s == 1.0
    assert res.stdout == "State change verified"


def test_executive_loop_auto_backtrack_and_correction():
    executive = ExecutiveLoop()
    # Initial code has ZeroDivisionError
    broken_code = "val = 42 / 0\nprint(f'Result: {val}')"
    res = executive.run(goal="Compute division", code=broken_code, max_retries=3)

    assert res["success"] is True
    assert res["delta_s"] == 1.0
    assert res["iterations"] >= 2
    # Verify scratchpad captured initial error
    first_attempt = res["scratchpad"][0]
    assert first_attempt["delta_s"] == -1.0
    assert "ZeroDivisionError" in first_attempt["stderr"]


def test_executive_loop_unresolved_grounding_failure():
    # Loop configured with no-op corrector that fails to fix a fatal error
    executive = ExecutiveLoop(code_corrector=lambda code, err: "raise RuntimeError('Unfixable fault')")
    res = executive.run(goal="Impossible task", code="raise RuntimeError('Fatal crash')", max_retries=3)

    assert res["success"] is False
    assert res["delta_s"] == -1.0
    assert "failure" in res
    failure = res["failure"]
    assert failure["goal"] == "Impossible task"
    assert failure["iterations"] == 3
    assert "Unfixable fault" in failure["error_trace"] or "Fatal crash" in failure["error_trace"]


def test_persistent_worker_latency_benchmark():
    import time
    sandbox = EnvironmentalSandbox(use_persistent_worker=True)

    # Warm-up call
    sandbox.execute_python("x = 1")

    t0 = time.perf_counter()
    n_iters = 20
    for i in range(n_iters):
        res = sandbox.execute_python(f"print('iteration_{i}')")
        assert res.exit_code == 0
        assert res.stdout == f"iteration_{i}"
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_iters) * 1000.0
    print(f"\nPersistent Sandbox Worker Latency: {latency_ms:.2f} ms/call (target: <10 ms)")
    sandbox.close()
    assert latency_ms < 20.0, f"Expected <20ms per call, got {latency_ms:.2f} ms"


def test_workspace_isolation_rollback():
    sandbox = EnvironmentalSandbox(use_persistent_worker=True)

    # Step 1: Failed execution defining dirty variable
    res1 = sandbox.execute_python("leaked_val = 999\nraise ValueError('Intentional crash')")
    assert res1.exit_code != 0
    assert res1.delta_s == -1.0

    # Step 2: Unrelated execution checking if leaked_val persisted
    res2 = sandbox.execute_python("try:\n    print(leaked_val)\nexcept NameError:\n    print('Isolated clean state')")
    assert res2.exit_code == 0
    assert "Isolated clean state" in res2.stdout
    sandbox.close()


if __name__ == "__main__":
    test_sandbox_broken_code_delta()
    test_sandbox_successful_code_delta()
    test_executive_loop_auto_backtrack_and_correction()
    test_executive_loop_unresolved_grounding_failure()
    test_persistent_worker_latency_benchmark()
    test_workspace_isolation_rollback()
    print("All sandbox & executive loop tests passed.")



