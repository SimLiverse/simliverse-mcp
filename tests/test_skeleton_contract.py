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

"""`controller.skeleton()` is copied verbatim, so its shape is a contract.

Agents start from it rather than rediscovering the ScriptNode rules, which means
anything missing here is missing from every controller written afterwards. The
omission that cost a run: no FAILED state. With DONE as the only exit, a
controller that stacked two of three cubes and left the third untouched on the
floor still ended in DONE and reported success -- reaching the last state is not
the same as having done the task.

Read as text and parsed, never executed: it imports `carb`, which only exists
inside Isaac Sim.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_sl_controller_src", _ROOT / "simliverse_sim" / "controller.py"
)


def _skeleton() -> str:
    """The template, taken from the source rather than by importing it."""
    source = (_ROOT / "simliverse_sim" / "controller.py").read_text(encoding="utf-8")
    return source.split("SKELETON = '''", 1)[1].split("'''", 1)[0]


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_skeleton())


@pytest.fixture(scope="module")
def text() -> str:
    return _skeleton()


# ── The rules that make a ScriptNode run at all ───────────────────────────────


def test_it_is_valid_python(tree) -> None:
    """It is pasted into a ScriptNode as-is; a syntax error here is total."""
    assert tree.body


def test_both_entry_points_exist(tree) -> None:
    """Without `compute` the node falls back to legacy mode and never runs."""
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"setup", "compute"} <= functions


def test_state_is_reset_on_timeline_stop(text) -> None:
    """Or the second Play resumes mid-task instead of starting over."""
    assert "TimelineEventType.STOP" in text


def test_compute_never_steps_physics(text) -> None:
    """Stepping from inside the graph's own callback deadlocks it."""
    for forbidden in ("scene.step(", "move_ee_to(", "simulation_app.update("):
        assert forbidden not in text, f"the skeleton demonstrates {forbidden}"


def test_it_servos_rather_than_blocking(text) -> None:
    assert "servo_to(" in text


# ── The part that was missing ─────────────────────────────────────────────────


def test_there_is_a_way_to_fail(text) -> None:
    """DONE as the only exit is how "two of three cubes" reported success."""
    assert "FAILED" in text, "the skeleton offers no failure state"
    assert "_fail(" in text, "nothing shows how to reach it"


def test_the_outcome_is_checked_before_declaring_done(tree, text) -> None:
    """There is a state between the work and DONE that measures the world."""
    assert "CHECK" in text

    # And it must read something back, not just arrive.
    check_body = text.split("if _state == CHECK:", 1)[1]
    assert "ee_position" in check_body or "position" in check_body, (
        "the CHECK state declares success without reading the world"
    )


def test_failure_carries_a_reason(text) -> None:
    """"It failed" is not actionable; which part failed is."""
    assert "_why" in text
    assert "log_warn" in text, "a failure that is not logged is invisible on a replay"


def test_no_state_can_wait_forever(text) -> None:
    """A servo that cannot converge otherwise neither finishes nor fails.

    That is the hardest failure to read from outside: the graph is running, the
    timeline is playing, and nothing is happening.
    """
    assert "_frame >" in text, "no per-state frame bound"


def test_terminal_states_stop_doing_work(text) -> None:
    assert "if _state in (DONE, FAILED):" in text
