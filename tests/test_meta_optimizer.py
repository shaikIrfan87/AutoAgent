import pytest
from cognitive_engine.core.meta_optimizer import MetaSelfOptimizer


class TargetSubsystem:
    def __init__(self, factor: int = 2):
        self.factor = factor

    def calculate(self, x: int) -> int:
        return x * self.factor


def test_successful_hotswap():
    optimizer = MetaSelfOptimizer()
    subsystem = TargetSubsystem(factor=2)
    assert subsystem.calculate(5) == 10

    patch_code = """
def calculate(self, x: int) -> int:
    return (x ** 2) + self.factor
"""

    def custom_canary(fn):
        return fn(3) == (3**2) + 2

    res = optimizer.test_and_hotswap(
        target_obj=subsystem,
        attr_name="calculate",
        new_source=patch_code,
        fn_name="calculate",
        custom_canary=custom_canary,
    )

    assert res.applied is True
    assert subsystem.calculate(5) == (25 + 2)


def test_failed_canary_prevents_hotswap():
    optimizer = MetaSelfOptimizer()
    subsystem = TargetSubsystem(factor=2)

    bad_code = """
def calculate(self, x: int) -> int:
    return -1
"""

    def custom_canary(fn):
        return fn(5) == 25

    res = optimizer.test_and_hotswap(
        target_obj=subsystem,
        attr_name="calculate",
        new_source=bad_code,
        fn_name="calculate",
        custom_canary=custom_canary,
    )

    assert res.applied is False
    assert "Canary validation suite failed" in res.error
    assert subsystem.calculate(5) == 10


def test_atomic_rollback():
    optimizer = MetaSelfOptimizer()
    subsystem = TargetSubsystem(factor=2)

    patch_code = """
def calculate(self, x: int) -> int:
    return x + 100
"""
    optimizer.test_and_hotswap(
        target_obj=subsystem,
        attr_name="calculate",
        new_source=patch_code,
        fn_name="calculate",
    )
    assert subsystem.calculate(1) == 101

    reverted = optimizer.rollback_last()
    assert reverted is True
    assert subsystem.calculate(1) == 2
