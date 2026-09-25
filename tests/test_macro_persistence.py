import os
import pytest
from cognitive_engine.core.dsl import UNARY_PRIMITIVES, rot90, flip_h, to_grid
from cognitive_engine.core.compression import MacroPrimitive
from cognitive_engine.core.macro_store import PersistentMacroStore


def test_cold_reboot_macro_persistence(tmp_path):
    store_file = str(tmp_path / "macros.json")
    store = PersistentMacroStore(store_path=store_file)

    # 1. Define synthetic macro: rot90 -> flip_h
    ops = (("rot90", ()), ("flip_h", ()))
    macro_name = "test_cold_macro_v1"
    macro = MacroPrimitive(
        name=macro_name,
        operations=ops,
        fn=lambda g: flip_h(rot90(g)),
        utility_score=4.0,
    )

    # 2. Save macro to file
    store.save({macro_name: macro})
    assert os.path.exists(store_file)

    # 3. Simulate cold boot: clear from UNARY_PRIMITIVES if present
    if macro_name in UNARY_PRIMITIVES:
        del UNARY_PRIMITIVES[macro_name]

    # 4. Restore on boot
    loaded = store.load_into_dsl()
    assert loaded == 1
    assert macro_name in UNARY_PRIMITIVES

    # 5. Verify functional equivalence
    g = to_grid(((1, 2), (3, 4)))
    assert UNARY_PRIMITIVES[macro_name](g) == flip_h(rot90(g))
