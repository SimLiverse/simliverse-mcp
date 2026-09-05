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

"""A six-bar linkage jaw is not six fingers.

`Gripper` treated every gripper as a set of independent fingers: it commanded
`[value] * len(joints)` and took "open" to be the upper limit. That is right for
a Panda, whose two prismatic fingers really are independent and really do open
at 0.04 m.

A Cobotta Pro 900 carries an OnRobot RG6: six *revolute* joints, +/-0.628 and
+/-0.873 rad, one driven (`finger_joint`) and five sided followers coupled to it
by the mechanism. Commanding all six independently sets them fighting it, and
taking the upper limit as "open" is backwards -- the driven joint sits near zero
open and rotates positive to close.

Neither error raises. `close()` opens the jaw, nothing is ever gripped, and it
reads as a control problem. Measured on a live worker: the agent under test gave
up on `grasp()`, `open()` and `close()` entirely, drove `finger_joint` by index
by hand, swept it 0.45 -> 0.31 to find the grip, and settled on 0.34. It passed
T4 -- in 65 tool calls and $3.90, by not using the library.
"""

import numpy as np
import pytest

from simliverse_sim.robots.manipulator import Gripper, _is_sided

COBOTTA = [
    "finger_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
]
COBOTTA_LIMITS = [(-0.628, 0.628)] * 4 + [(-0.873, 0.873)] * 2

PANDA = ["panda_finger_joint1", "panda_finger_joint2"]
PANDA_LIMITS = [(0.0, 0.04)] * 2

SHADOW = [f"{finger}_j{n}" for finger in ("ff", "mf", "rf", "th") for n in (1, 2, 3)]


class FakeRobot:
    prim_path = "/World/Arm"

    def __init__(self, names, limits, *, links=()):
        self.joint_names = list(names)
        self.joint_limits = list(limits)
        self.joint_positions = np.zeros(len(names))
        self._links = list(links)
        self.commands = []

    def links(self):
        return list(self._links)

    def drive_health(self):
        return []

    def set_joint_positions(self, targets, *, indices=None, settle_steps=0):
        self.commands.append((list(indices) if indices else None, list(targets)))
        for slot, value in zip(indices or range(len(self.joint_names)), targets):
            self.joint_positions[slot] = value


def cobotta_gripper(**kwargs):
    robot = FakeRobot(COBOTTA, COBOTTA_LIMITS, **kwargs)
    return robot, Gripper(robot, list(range(len(COBOTTA))))


def panda_gripper():
    robot = FakeRobot(PANDA, PANDA_LIMITS)
    return robot, Gripper(robot, [0, 1])


# ── telling the two apart ────────────────────────────────────────────────────


def test_a_sided_joint_is_recognised_by_its_name():
    assert _is_sided("left_inner_finger_joint")
    assert _is_sided("right_outer_knuckle_joint")
    assert not _is_sided("finger_joint")
    assert not _is_sided("panda_finger_joint1")


def test_a_robotiq_style_jaw_is_a_linkage_with_one_driven_joint():
    _, gripper = cobotta_gripper()

    assert gripper.is_linkage
    assert gripper.joint_names[gripper.primary_index] == "finger_joint"


def test_two_independent_fingers_are_not_a_linkage():
    _, gripper = panda_gripper()

    assert not gripper.is_linkage
    assert gripper.primary_index is None


def test_a_many_fingered_hand_is_not_a_linkage():
    robot = FakeRobot(SHADOW, [(0.0, 1.5)] * len(SHADOW))
    gripper = Gripper(robot, list(range(len(SHADOW))))

    # No unsided joint at all, so there is nothing to single out as driven.
    assert not gripper.is_linkage


# ── commanding it ────────────────────────────────────────────────────────────


def test_a_linkage_is_driven_by_one_joint_not_six():
    robot, gripper = cobotta_gripper()

    gripper.set_position(0.34, settle_steps=0)

    assert robot.commands == [([0], [0.34])], (
        "the followers are moved by the mechanism; commanding them independently sets them fighting it"
    )


def test_independent_fingers_are_still_all_commanded():
    robot, gripper = panda_gripper()

    gripper.set_position(0.02, settle_steps=0)

    assert robot.commands == [([0, 1], [0.02, 0.02])]


def test_position_reports_the_driven_joint_of_a_linkage():
    robot, gripper = cobotta_gripper()
    robot.joint_positions = np.array([0.34, 0.30, 0.30, -0.30, 0.10, 0.10])

    # The mean of those six is not a value `set_position` would accept.
    assert gripper.position == pytest.approx(0.34)


def test_position_averages_independent_fingers():
    robot, gripper = panda_gripper()
    robot.joint_positions = np.array([0.02, 0.04])

    assert gripper.position == pytest.approx(0.03)


# ── which end closes ─────────────────────────────────────────────────────────


def test_the_closing_end_is_whichever_brings_the_pads_together(monkeypatch):
    """The jaw is driven to both ends and the pads are measured at each."""
    robot, gripper = cobotta_gripper(links=["/World/Arm/left_inner_finger", "/World/Arm/right_inner_finger"])
    # An RG6: near zero it is open, positive rotation closes it.
    monkeypatch.setattr(Gripper, "_pad_gap", lambda self, pads: 0.09 - 0.1 * float(robot.joint_positions[0]))

    opened, closed = gripper._ends_by_measurement(-0.628, 0.628)

    assert closed == pytest.approx(0.628)
    assert opened == pytest.approx(-0.628)


def test_the_jaw_is_left_where_it_was_found(monkeypatch):
    robot, gripper = cobotta_gripper(links=["/World/Arm/left_inner_finger", "/World/Arm/right_inner_finger"])
    robot.joint_positions[0] = 0.21
    monkeypatch.setattr(Gripper, "_pad_gap", lambda self, pads: 0.09 - 0.1 * float(robot.joint_positions[0]))

    gripper._ends_by_measurement(-0.628, 0.628)

    assert robot.commands[-1] == ([0], [pytest.approx(0.21)])


def test_a_jaw_with_no_measurable_pads_says_it_is_guessing(caplog):
    _, gripper = cobotta_gripper(links=["/World/Arm/base_link"])

    opened, closed = gripper._ends_by_measurement(-0.628, 0.628)

    # The industrial convention, and named as an assumption rather than applied
    # silently -- the Panda convention would be exactly backwards here.
    assert (opened, closed) == (-0.628, 0.628)
    assert "guess" in caplog.text.lower()


def test_open_and_close_use_the_measured_ends(monkeypatch):
    robot, gripper = cobotta_gripper(links=["/World/Arm/left_inner_finger", "/World/Arm/right_inner_finger"])
    monkeypatch.setattr(Gripper, "_pad_gap", lambda self, pads: 0.09 - 0.1 * float(robot.joint_positions[0]))

    gripper.close(settle_steps=0)
    assert robot.commands[-1] == ([0], [pytest.approx(0.628)])

    gripper.open(settle_steps=0)
    assert robot.commands[-1] == ([0], [pytest.approx(-0.628)])


def test_a_panda_still_opens_at_its_upper_limit():
    """The case that worked by luck, and must keep working."""
    robot, gripper = panda_gripper()

    gripper.open(settle_steps=0)

    assert robot.commands[-1] == ([0, 1], [pytest.approx(0.04), pytest.approx(0.04)])


# ── finding the motion config ────────────────────────────────────────────────

from simliverse_sim.robots.manipulator import _match_motion_config  # noqa: E402

SUPPORTED = [
    "Cobotta_Pro_1300",
    "Cobotta_Pro_900",
    "FR3",
    "Fanuc_CRX10IAL",
    "FestoCobot",
    "Franka",
    "Kuka_KR210",
    "RS007L",
    "RS007N",
    "Rizon4",
    "Techman_TM12",
    "UR10",
    "UR10e",
    "UR16e",
    "UR3",
    "UR3e",
    "UR5",
    "UR5e",
]

# What `_asset_identity` and the joint/leaf readers produce: lowercased, no
# underscores.
COBOTTA_ASSET = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/assets/"
    "isaac/6.0/isaac/robots/denso/cobottapro900/cobottapro900.usd"
)
COBOTTA_JOINTS = "joint1 joint2 joint3 joint4 joint5 joint6 fingerjoint"


def test_a_cobotta_is_found_by_its_asset_when_its_joints_say_nothing():
    """The failure this fixes: generic joint names and a user-chosen prim path.

    It raised `No RMPflow configuration matches /World/Arm` while listing
    `Cobotta_Pro_900` among the robots it supported, in the same message.
    """
    assert _match_motion_config(SUPPORTED, COBOTTA_JOINTS, COBOTTA_ASSET, "arm") == ("Cobotta_Pro_900")


def test_a_panda_is_not_this_functions_job():
    """`panda_*` joints reach `Franka` through the alias table above this call.

    Worth pinning: the config is named for the vendor and the joints for the
    product, so nothing here can bridge them by substring.
    """
    joints = "pandajoint1 pandajoint2 pandafingerjoint1"
    assert _match_motion_config(SUPPORTED, joints, "", "arm") is None


def test_a_franka_is_found_when_its_asset_says_franka():
    asset = ".../isaac/robots/frankarobotics/frankapanda/franka.usd"
    assert _match_motion_config(SUPPORTED, "", asset, "arm") == "Franka"


def test_the_prim_path_still_works_when_it_is_the_only_clue():
    assert _match_motion_config(SUPPORTED, "j1 j2 j3", "", "ur10e") == "UR10e"


def test_a_longer_name_is_not_claimed_by_a_shorter_one():
    """`UR5` is a substring of `ur5e`; first-match-wins would take the wrong one."""
    assert _match_motion_config(SUPPORTED, "", ".../robots/universalrobots/ur5e/ur5e.usd", "arm") == "UR5e"
    assert _match_motion_config(SUPPORTED, "", ".../denso/cobottapro1300/x.usd", "arm") == ("Cobotta_Pro_1300")


def test_an_unknown_robot_matches_nothing():
    assert _match_motion_config(SUPPORTED, "a1 a2", ".../vendor/mystery/m.usd", "arm") is None
