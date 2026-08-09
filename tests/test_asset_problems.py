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

"""What `describe()` reports as wrong with the robot itself.

Driven over a stub rather than a live articulation, so it runs without Isaac Sim.

The case that motivated this file was measured, not imagined. A UR10 spawned
from the catalogue — where it is listed as "no gripper by default" — reports:

    has .gripper   : True
    gripper object : <Gripper 0 joints: []>
    asset_problems : None

A clean bill of health on an arm with nothing on the flange. The check that
existed looked for gripper joints whose travel limits were missing, and it
iterates over the finger joints, so zero finger joints passed it silently: the
worse defect was invisible while the milder one was reported. An agent following
its instructions — "check `describe()` for `asset_problems` before you write
control code" — was told everything was fine, and the next honest step from
there is to build a gripper nobody asked for.
"""

import pytest

from simliverse_sim.robots.base import Robot

# The real joint sets.
UR10 = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FRANKA = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]


class _Gripper:
    def __init__(self, joint_indices):
        self.joint_indices = joint_indices


class _Arm:
    """The surface of `Robot` that `asset_problems` actually reads."""

    def __init__(self, names, *, gripper_joints=None, limits=None, has_gripper=True):
        self.prim_path = "/World/Arm"
        self.joint_names = names
        self.joint_limits = limits or [(-3.14, 3.14)] * len(names)
        self.base_position = [0.0, 0.0, 0.0]
        self._pose_source = "physics (articulation view)"
        self.gripper = _Gripper(gripper_joints or []) if has_gripper else None

    def _pose_feedback_problems(self):
        return []


def problems(arm) -> list[dict[str, str]]:
    return Robot.asset_problems(arm)


def issues(arm) -> list[str]:
    return [p["issue"] for p in problems(arm)]


def test_an_arm_with_no_end_effector_says_so() -> None:
    """The UR10 case. It used to report nothing at all."""
    found = issues(_Arm(UR10))
    assert "this arm has no end effector" in found


def test_the_report_says_fitting_one_is_the_users_call() -> None:
    """The whole point: the next step is a question, not a suction cup."""
    fault = next(p for p in problems(_Arm(UR10)) if "no end effector" in p["issue"])
    assert "ask rather than authoring one" in fault["consequence"]
    assert "different arm" in fault["consequence"]


def test_the_consequence_matches_what_the_gripper_actually_does() -> None:
    """Measured on a live UR10: close() raises, it does not quietly do nothing.

    The first draft of this text said open() and close() "return without doing
    anything". They raise MotionError — `Gripper.set_position` guards on
    `exists`. A consequence that misdescribes the symptom sends whoever reads it
    looking for a silent no-op that never happens.
    """
    fault = next(p for p in problems(_Arm(UR10)) if "no end effector" in p["issue"])
    assert "raise MotionError" in fault["consequence"]


def test_an_arm_with_a_working_gripper_is_clean() -> None:
    """A Franka must not be flagged, or the check is noise and gets ignored."""
    assert issues(_Arm(FRANKA, gripper_joints=[7, 8])) == []


def test_a_robot_with_no_gripper_attribute_is_not_flagged() -> None:
    """A quadruped or a rover is not a broken arm."""
    assert issues(_Arm(UR10, has_gripper=False)) == []


def test_missing_travel_limits_are_still_reported() -> None:
    """The check this one sits in front of still works."""
    limits = [(-3.14, 3.14)] * 7 + [(None, None), (None, None)]
    found = issues(_Arm(FRANKA, gripper_joints=[7, 8], limits=limits))
    assert "gripper joints have no travel limits" in found


def test_no_end_effector_outranks_the_limits_check() -> None:
    """Reported first, because it is the one that decides whether to continue."""
    found = issues(_Arm(UR10))
    assert found[0] == "this arm has no end effector"


@pytest.mark.parametrize("empty", [[], None])
def test_an_empty_joint_list_counts_as_no_end_effector(empty) -> None:
    assert "this arm has no end effector" in issues(_Arm(UR10, gripper_joints=empty))
