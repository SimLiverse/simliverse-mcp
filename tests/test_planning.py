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

"""The parts of planning that hold without a GPU or an Isaac Sim session.

The obstacle tables are the interesting ones. They were measured — by reading
which `add_*` methods the world interface actually implements rather than which
ones it declares — and getting them wrong is not a cosmetic error: an obstacle
the reactive policy silently cannot represent gets driven through, and one the
planner cannot represent makes every later plan fail while it stays registered.
"""

import numpy as np
import pytest

from simliverse_sim.robots import planning


class _FakeJoints:
    def __init__(self, positions, velocities):
        self.positions = positions
        self.velocities = velocities


class _FakeState:
    def __init__(self, t):
        self.joints = _FakeJoints(np.full(7, t), np.full(7, -t))


class _FakeTrajectory:
    """Stands in for a cuMotion trajectory: samples report the time asked for."""

    duration = 2.0

    def get_target_state(self, t):
        return _FakeState(t)


def test_servo_sees_no_more_than_the_planner():
    """Lula's world is a subset of the planner's; a wider set would be a lie."""
    assert planning.SERVO_OBSTACLE_TYPES <= planning.PLANNER_OBSTACLE_TYPES


def test_unrepresentable_types_are_in_neither_backend():
    """A type listed as supported *and* unrepresentable would be screened wrongly."""
    assert not (planning.UNREPRESENTABLE_TYPES & planning.PLANNER_OBSTACLE_TYPES)
    assert not (planning.UNREPRESENTABLE_TYPES & planning.SERVO_OBSTACLE_TYPES)


def test_cylinders_and_cones_are_screened():
    """Both are declared by the world interface and both raise when called."""
    assert "Cylinder" in planning.UNREPRESENTABLE_TYPES
    assert "Cone" in planning.UNREPRESENTABLE_TYPES


def test_plan_reports_its_shape():
    plan = planning.MotionPlan(_FakeTrajectory(), [f"j{i}" for i in range(7)], np.zeros((2, 7)))
    assert plan.duration == 2.0
    assert len(plan.joint_names) == 7
    assert "2 waypoints" in repr(plan)


def test_sampling_is_clamped_to_the_trajectory():
    """`follow` advances a clock past the end on the tick it finishes.

    Sampling past `duration` has to hold the final state rather than extrapolate,
    or the last tick of every planned motion commands a pose off the end of the
    plan.
    """
    plan = planning.MotionPlan(_FakeTrajectory(), [f"j{i}" for i in range(7)], np.zeros((2, 7)))
    positions, _ = plan.sample(99.0)
    assert positions == pytest.approx(np.full(7, 2.0))

    positions, _ = plan.sample(-5.0)
    assert positions == pytest.approx(np.zeros(7))


def test_sampling_returns_positions_and_velocities():
    plan = planning.MotionPlan(_FakeTrajectory(), [f"j{i}" for i in range(7)], np.zeros((2, 7)))
    positions, velocities = plan.sample(1.0)
    assert positions == pytest.approx(np.ones(7))
    assert velocities == pytest.approx(np.full(7, -1.0))


def test_available_is_false_without_isaac_sim():
    """It must answer rather than raise, so callers can degrade deliberately."""
    assert planning.available() is False


def test_supported_robots_is_empty_rather_than_raising():
    assert planning.supported_robots() == []
