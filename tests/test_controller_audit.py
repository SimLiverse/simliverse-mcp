# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
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

"""A recorded verification is only reusable while it still describes the stage.

`audit` exists so the agent checking a motion claim can read the measurement
instead of replaying it, which is the slowest step in a session. That trade is
only sound if a record goes stale the instant anything it depended on changes —
so these tests are mostly about the ways it must refuse, not the way it accepts.

No Isaac Sim here: the two seams that touch the stage (`_wired_script`, which
reads a graph's ScriptNode, and `graphs`, which enumerates them) are replaced,
and everything else is real.
"""

import os

import pytest

from simliverse_sim import controller

REPORT = {
    "moved": ["/World/Cube1"],
    "reproduced": True,
    "diverged": [],
    "at_rest": True,
}


@pytest.fixture
def delivered(tmp_path, monkeypatch):
    """A controller on disk, a report beside it, and a graph wired to it."""
    script = tmp_path / "stack.py"
    script.write_text("def setup(db=None):\n    pass\n\n\ndef compute(db=None):\n    return True\n")
    monkeypatch.setattr(controller, "_CANDIDATE_DIRECTORIES", (str(tmp_path),))
    controller._record(REPORT, script_path=str(script), graph_path="/World/TaskGraph")

    stage = {"/World/TaskGraph": str(script)}
    monkeypatch.setattr(controller, "_wired_script", lambda path: stage.get(path))
    monkeypatch.setattr(
        controller,
        "graphs",
        lambda: [{"graph": g, "script": s} for g, s in stage.items()],
    )
    return {"script": script, "stage": stage, "dir": tmp_path}


def test_record_lands_beside_its_controller(delivered):
    assert os.path.isfile(str(delivered["script"]).replace(".py", ".report.json"))


def test_unchanged_delivery_is_current(delivered):
    result = controller.audit()
    assert result["found"] and result["current"]
    assert result["stale_because"] == []
    # The numbers have to survive the round trip — they are the whole point.
    assert result["moved"] == ["/World/Cube1"]
    assert result["reproduced"] is True


def test_named_lookup_accepts_the_bare_name(delivered):
    for name in ("stack", "stack.py"):
        assert controller.audit(name)["current"], name


def test_missing_record_is_not_current(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "_CANDIDATE_DIRECTORIES", (str(tmp_path),))
    result = controller.audit()
    assert result["found"] is False and result["current"] is False
    assert "hint" in result


def test_editing_the_controller_voids_the_record(delivered):
    """The most likely way to get a false pass: fix the script, keep the report."""
    delivered["script"].write_text("def setup(db=None):\n    pass\n\n\ndef compute(db=None):\n    return False\n")
    result = controller.audit()
    assert not result["current"]
    assert any("edited" in reason for reason in result["stale_because"])


def test_deleting_the_controller_voids_the_record(delivered):
    os.remove(delivered["script"])
    result = controller.audit()
    assert not result["current"]
    assert any("gone" in reason for reason in result["stale_because"])


def test_unwiring_the_graph_voids_the_record(delivered):
    """`usePath` False reads as unwired: the node runs its empty inline source."""
    delivered["stage"].clear()
    result = controller.audit()
    assert not result["current"]
    assert any("no graph" in reason for reason in result["stale_because"])


def test_repointing_the_graph_voids_the_record(delivered):
    delivered["stage"]["/World/TaskGraph"] = str(delivered["dir"] / "something_else.py")
    result = controller.audit()
    assert not result["current"]
    assert any("not the script that was verified" in r for r in result["stale_because"])


def test_a_second_graph_on_the_same_script_voids_the_record(delivered):
    """Two graphs means two state machines commanding one robot every frame.

    This happened: a failed re-attach left the old graph in place, the task ran
    at roughly double speed, and it read as the controller working.
    """
    delivered["stage"]["/World/StackTaskGraph"] = str(delivered["script"])
    result = controller.audit()
    assert not result["current"]
    assert any("more than one graph" in reason for reason in result["stale_because"])


def test_articulation_roots_are_measured_as_robots(monkeypatch):
    """Passing a robot in `objects` is a routing problem, not a caller error."""
    monkeypatch.setattr(controller, "_is_articulation", lambda p: p == "/World/Franka")
    bodies, robots, rerouted = controller._split_by_kind(
        ["/World/Cube1", "/World/Franka"], None
    )
    assert bodies == ["/World/Cube1"]
    assert robots == ["/World/Franka"]
    assert rerouted == ["/World/Franka"]


def test_routing_does_not_duplicate_an_already_named_robot(monkeypatch):
    monkeypatch.setattr(controller, "_is_articulation", lambda p: p == "/World/Franka")
    _, robots, _ = controller._split_by_kind(["/World/Franka"], ["/World/Franka"])
    assert robots == ["/World/Franka"]
