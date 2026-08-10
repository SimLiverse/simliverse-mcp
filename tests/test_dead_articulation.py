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

"""A robot whose physics view has gone must say so, not answer `nan`.

Reconstructed from a live run. The agent stacked three cubes past an obstacle
and reported real measurements; then the physics simulation view was torn down
under it, and every reading it took afterwards was false in a way nothing
named:

    robot.ee_position          -> nan
    robot.describe()           -> IndexError: too many indices for array:
                                  array is 0-dimensional
    get_simulation_state()     -> {"timeline_state": "playing"}

Isaac said what had happened, in its own log and nowhere else:

    [Warning] [articulation] Physics Simulation View is not created yet in
              order to use get_joint_positions
    [Error]   [articulation_kinematics_solver] Attempted to compute forward
              kinematics for an uninitialized robot Articulation

The cause is that `get_joint_positions()` does not raise when the view is gone.
It returns `None`, and `np.asarray(None, dtype=float)` is `array(nan)` — a
zero-dimensional array that survives every check made of it until something
indexes it, three frames deep in this library.

No simulator here: `Robot` reads its articulation through a handful of methods,
so a stub that behaves the way Isaac does is enough to pin the behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from simliverse_sim.robots.base import Robot, StaleArticulation


class _Articulation:
    """The parts of an Isaac articulation that `Robot` actually touches."""

    def __init__(self, positions, *, num_dof=7, names=None):
        self._positions = positions
        self.num_dof = num_dof
        self.dof_names = names or [f"joint{i}" for i in range(num_dof)]
        self.rebuilt = False

    def get_joint_positions(self):
        return self._positions

    def get_joint_velocities(self):
        return self._positions

    def initialize(self):
        self.rebuilt = True


class _Robot(Robot):
    """A `Robot` bound to a stub, with rebinding under our control."""

    def __init__(self, articulation, replacement=None):
        self.prim_path = "/World/Arm"
        self._articulation = articulation
        self._replacement = replacement

    def _rebind(self):
        if self._replacement is None:
            raise RuntimeError("no articulation at /World/Arm")
        return self._replacement

    # The rest of what describe() gathers. Stubbed so the healthy case
    # exercises the dictionary this method builds rather than the simulator
    # underneath it — the failing case is the one with something to prove.
    class _Groups:
        @staticmethod
        def to_dict(names):
            return {"arm": list(names)}

    groups = _Groups()

    @property
    def base_position(self):
        return np.zeros(3)

    def drive_health(self):
        return []

    def asset_problems(self):
        return []

    def capabilities(self):
        return ["cartesian"]


@pytest.fixture(autouse=True)
def _no_isaac(monkeypatch):
    """`_joint_state` re-binds through `single_articulation`; stub that."""
    import simliverse_sim.robots.base as base

    monkeypatch.setattr(
        base, "single_articulation",
        lambda path: _current["robot"]._rebind(),
    )
    yield


_current: dict = {}


def _make(positions, replacement=None, **kwargs):
    robot = _Robot(_Articulation(positions, **kwargs), replacement)
    _current["robot"] = robot
    return robot


# ── The reading that means nothing ────────────────────────────────────────────


def test_a_healthy_read_is_returned_unchanged() -> None:
    robot = _make([0.1] * 7)
    assert robot.joint_positions.tolist() == pytest.approx([0.1] * 7)


def test_none_is_not_a_joint_reading() -> None:
    """The exact production failure: Isaac returns None, numpy makes it nan."""
    # Proof that the raw conversion is the trap, not a hypothetical.
    assert np.asarray(None, dtype=float).ndim == 0

    robot = _make(None)
    with pytest.raises(StaleArticulation) as caught:
        robot.joint_positions

    message = str(caught.value)
    assert "/World/Arm" in message
    assert "physics view" in message
    assert "Robot.attach()" in message


@pytest.mark.parametrize(
    "positions, why",
    [
        pytest.param(None, "None becomes a 0-d nan array", id="none"),
        pytest.param([], "an empty read is not seven joints", id="empty"),
        pytest.param([0.1, 0.2], "two values for a seven-DOF arm", id="wrong-length"),
        pytest.param([float("nan")] * 7, "right shape, no information", id="all-nan"),
        pytest.param([0.1] * 6 + [float("inf")], "one joint unreadable", id="one-infinite"),
    ],
)
def test_every_unusable_shape_is_refused(positions, why) -> None:
    robot = _make(positions)
    with pytest.raises(StaleArticulation):
        robot.joint_positions


def test_a_stale_view_is_retried_once_before_giving_up() -> None:
    """The original behaviour, kept: a view stale across a timeline cycle."""
    healthy = _Articulation([0.3] * 7)
    robot = _make(None, replacement=healthy)

    assert robot.joint_positions.tolist() == pytest.approx([0.3] * 7)
    assert healthy.rebuilt, "the replacement view was never initialised"


def test_a_rebind_that_fails_names_the_robot() -> None:
    robot = _make(None, replacement=None)
    with pytest.raises(StaleArticulation) as caught:
        robot.joint_positions
    assert "no articulation at /World/Arm" in str(caught.value)


# ── describe() is the call you make because something is wrong ────────────────


def test_describe_diagnoses_instead_of_crashing() -> None:
    """It used to raise IndexError from three frames down. That call is the
    one an agent reaches for *after* something has gone wrong."""
    robot = _make(None)
    described = robot.describe()

    assert described["usable"] is False
    assert "physics view" in described["problem"]
    assert "Robot.attach" in described["recover"]
    # Facts that are still true are still reported.
    assert described["prim_path"] == "/World/Arm"


def test_describe_on_a_healthy_robot_is_untouched() -> None:
    robot = _make([0.0] * 7)
    described = robot.describe()

    assert "usable" not in described, "the diagnosis leaked into the healthy path"
    assert described["joint_positions"] == [0.0] * 7
