import os
import shutil
import tempfile
from pathlib import Path
import pytest
from cognitive_engine.agent.ast_policy_mcts import DynamicASTSynthesizer
from cognitive_engine.core.skills import SkillLibrary


@pytest.fixture
def temp_skills_dir():
    """Provides a temporary skills directory with test skills."""
    d = tempfile.mkdtemp(prefix="autoagent_skills_test_")
    tenant_dir = Path(d) / "default_tenant"
    tenant_dir.mkdir(parents=True, exist_ok=True)

    # Skill 1: cube(x)
    (tenant_dir / "cube_skill.py").write_text(
        "def cube(x):\n    \"\"\"Computes cube of x.\"\"\"\n    return x ** 3\n",
        encoding="utf-8",
    )

    # Skill 2: is_mult5(x)
    (tenant_dir / "mult5_skill.py").write_text(
        "def is_mult5(x):\n    \"\"\"Checks if multiple of 5.\"\"\"\n    return x % 5 == 0\n",
        encoding="utf-8",
    )

    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_skill_primitives_discovers_compiled_skills(temp_skills_dir):
    """Verify load_skill_primitives discovers compiled skills from disk."""
    synth = DynamicASTSynthesizer(skills_dir=temp_skills_dir)
    primitives = synth.load_skill_primitives()
    names = [p[0] for p in primitives]
    fn_names = [p[1] for p in primitives]

    assert "cube_skill" in names
    assert "mult5_skill" in names
    assert "cube" in fn_names
    assert "is_mult5" in fn_names


def test_synthesize_dynamic_composes_skill_map(temp_skills_dir):
    """Verify MCTS synthesizes a list mapping using a pre-compiled skill."""
    synth = DynamicASTSynthesizer(skills_dir=temp_skills_dir)
    # Task: [1, 2, 3] -> [1, 8, 27] (requires cube)
    io_pairs = [
        ([1, 2, 3], [1, 8, 27]),
        ([2, 4], [8, 64]),
    ]
    sol_code = synth.synthesize_dynamic(io_pairs)
    assert sol_code is not None
    assert "cube" in sol_code

    # Verify execution
    env = {}
    exec(sol_code, env)
    assert env["solution"]([1, 2, 3]) == [1, 8, 27]
    assert env["solution"]([3]) == [27]


def test_synthesize_dynamic_composes_skill_filter(temp_skills_dir):
    """Verify MCTS synthesizes a list filter using pre-compiled predicate skill."""
    synth = DynamicASTSynthesizer(skills_dir=temp_skills_dir)
    # Task: [2, 5, 9, 15, 20] -> [5, 15, 20] (requires is_mult5)
    io_pairs = [
        ([2, 5, 9, 15, 20], [5, 15, 20]),
        ([3, 10, 7], [10]),
    ]
    sol_code = synth.synthesize_dynamic(io_pairs)
    assert sol_code is not None
    assert "is_mult5" in sol_code

    env = {}
    exec(sol_code, env)
    assert env["solution"]([1, 5, 25, 3]) == [5, 25]


def test_synthesize_from_assertions_composes_skill(temp_skills_dir):
    """Verify synthesize_from_assertions composes compiled skills to satisfy unit tests."""
    synth = DynamicASTSynthesizer(skills_dir=temp_skills_dir)
    # Task: solution(2) == 10 (cube(2) + 2 == 10)
    assertions = [
        "assert solution(1) == 3",   # cube(1) + 2 = 3
        "assert solution(2) == 10",  # cube(2) + 2 = 10
        "assert solution(3) == 29",  # cube(3) + 2 = 29
    ]
    sol_code = synth.synthesize_from_assertions(assertions)
    assert sol_code is not None
    assert "cube" in sol_code

    env = {}
    exec(sol_code, env)
    assert env["solution"](2) == 10


def test_skill_library_integration_with_dynamic_synthesizer(temp_skills_dir):
    """Verify SkillLibrary instance automatically feeds compiled functions into DynamicASTSynthesizer."""
    lib = SkillLibrary(storage_dir=temp_skills_dir)
    synth = DynamicASTSynthesizer(skill_library=lib, skills_dir=temp_skills_dir)

    primitives = synth.load_skill_primitives()
    assert len(primitives) >= 2
    fn_names = [p[1] for p in primitives]
    assert "cube" in fn_names
