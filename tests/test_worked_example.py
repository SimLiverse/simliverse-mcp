# MIT License
#
# Copyright (c) 2026 SimLiverse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""`controller.example()` is copied, so what it returns is a contract too.

The prompt tells the programmer to read it before writing anything, which makes
it the highest-leverage text in the repository: whatever shape it has becomes
the shape of every controller written afterwards.

It used to return `demo/stack_cubes.py`, which never re-homes the arm. The
Franka's wrist winds up across a sequence of solves, and once joint 6 sits near
its 3.752 rad limit a demanded DOWN orientation is only satisfiable by driving
into the stop -- so RMPflow trades position away for it and every target after
the first lands 9 to 22 cm short. Measured: `[0.45, 0.2, 0.135]` unreachable at
0.095 m error with the wrist wound, 0.009 m after homing. A run read that as a
kinematic limit and gave up on the last cube.

Read as text and parsed, never executed: these import `carb`, which exists only
inside Isaac Sim.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = (_ROOT / "simliverse_sim" / "controller.py").read_text(encoding="utf-8")


def _preferred() -> Path:
    """The file `example()` would return, resolved the way `example()` resolves it."""
    block = re.search(r"_WORKED_EXAMPLES = \((.*?)\n\)", _SOURCE, re.S)
    assert block, "_WORKED_EXAMPLES is gone — example() no longer has a preference order"
    entries = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', block.group(1))
    assert entries, "no worked examples are listed"
    for subdirectory, filename in entries:
        candidate = _ROOT / subdirectory / filename
        if candidate.is_file():
            return candidate
    pytest.fail(f"none of the listed worked examples exist in the repo: {entries}")


@pytest.fixture(scope="module")
def example_source() -> str:
    return _preferred().read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(example_source: str) -> ast.Module:
    return ast.parse(example_source)


def test_the_preferred_example_exists_and_parses(tree: ast.Module) -> None:
    """A worked example that does not parse teaches nothing and writes nothing."""
    assert isinstance(tree, ast.Module)


def test_it_re_homes_the_arm(example_source: str) -> None:
    """The omission that cost a run, and the reason the preference order exists."""
    assert "HOME" in example_source, (
        "the worked example does not home the arm — an agent copying it will wind "
        "the wrist and read the resulting 9-22 cm shortfall as a kinematic limit"
    )
    assert "set_joint_positions" in example_source, (
        "HOME is named but never commanded"
    )


def test_homing_is_not_a_one_off_at_startup(example_source: str) -> None:
    """The wind-up accumulates across picks; pick two is where it first bites.

    A single home in INIT looks like homing and does not survive a sequence.
    """
    assert example_source.count("HOME_ARM") >= 3, (
        "HOME_ARM should be a state that is re-entered before every pick, not a "
        "step passed through once on the way out of INIT"
    )


def test_it_measures_before_it_says_done(example_source: str) -> None:
    """Reaching the last state is not the same as having done the task.

    A controller once reported DONE having stacked two of three cubes and left
    the third on the floor.
    """
    assert "CHECK" in example_source, "no state between the last placement and DONE"
    assert "FAILED" in example_source, "DONE is the only exit"


def test_it_does_not_step_physics_from_inside_compute(example_source: str) -> None:
    """`move_ee_to` steps the world; a ScriptNode doing that deadlocks the graph."""
    assert "servo_to" in example_source
    assert "move_ee_to" not in example_source.split('"""', 2)[-1], (
        "the example calls move_ee_to outside its docstring — that double-advances "
        "the world from inside compute()"
    )


def test_no_state_can_wait_forever(example_source: str) -> None:
    """A servo that cannot converge would otherwise neither finish nor fail:
    the graph runs, the timeline plays, and nothing happens."""
    assert "LIMIT" in example_source, "no per-state frame limit"
