"""A finger gripper bolted to a bare arm: the arithmetic and the bookkeeping.

The asset facts here were read off the shipped USDs (Isaac Sim 6.0.1): a
Robotiq 2F-85 is one driven `finger_joint` (0-47 deg) whose `base_link` is
body0 of both knuckle joints and body1 of nothing; a Schunk EGK-25 is one
driven prismatic `Jaw_Drive` with a mimic follower; a Hand-E is two undriven
sliders named `Slider_1` and `Slider_2`, which no joint-name token finds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simliverse_sim.robots.manipulator import (
    Z_ONTO,
    Gripper,
    fitted_gripper_indices,
    gripper_mount,
    gripper_root_body,
)


def _rotate(q, v):
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ np.asarray(v, dtype=float)


def _axis(name):
    out = np.zeros(3)
    out["XYZ".index(name[-1])] = -1.0 if name.startswith("-") else 1.0
    return out


@pytest.mark.parametrize("axis", sorted(Z_ONTO))
def test_the_fingers_point_along_the_flange_axis(axis):
    """Gripper +Z lands on the flange's tool axis, whichever one it is."""
    position, orientation = gripper_mount(axis, 0.05)
    assert np.linalg.norm(orientation) == pytest.approx(1.0)
    np.testing.assert_allclose(_rotate(orientation, [0, 0, 1]), _axis(axis), atol=1e-6)
    np.testing.assert_allclose(position, _axis(axis) * 0.05, atol=1e-12)


def test_yaw_spins_about_the_fingers_not_the_flange():
    """A UR flange (Z) yawed 90 degrees: +Z stays +Z, the jaw's X turns to Y."""
    _, q = gripper_mount("Z", 0.0, 90.0)
    np.testing.assert_allclose(_rotate(q, [0, 0, 1]), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-6)
    # And on a Fanuc CRX flange (-Y) the same yaw turns the jaw about -Y.
    _, q = gripper_mount("-Y", 0.0, 90.0)
    np.testing.assert_allclose(_rotate(q, [0, 0, 1]), [0, -1, 0], atol=1e-6)
    turned = _rotate(q, [1, 0, 0])
    assert turned[1] == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(turned[0], turned[2]) == pytest.approx(1.0)


def test_a_bad_axis_is_refused():
    with pytest.raises(ValueError):
        gripper_mount("W", 0.0)


ROBOTIQ_2F85 = [
    ("/G/left_outer_knuckle", "/G/left_outer_finger"),
    ("/G/right_outer_knuckle", "/G/right_outer_finger"),
    ("/G/base_link", "/G/left_outer_knuckle"),
    ("/G/base_link", "/G/right_outer_knuckle"),
    ("/G/right_outer_finger", "/G/right_inner_finger"),
    ("/G/right_inner_finger", "/G/right_inner_knuckle"),
    ("/G/left_inner_finger", "/G/left_inner_knuckle"),
    ("/G/left_outer_finger", "/G/left_inner_finger"),
]

SCHUNK_EGK = [
    ("/G/SCHUNK_1491752_EGK_25_PN_M_B_000ohne0", "/G/SCHUNK_1500102Grundbacke_EGK_25_3"),
    ("/G/SCHUNK_1491752_EGK_25_PN_M_B_000ohne0", "/G/SCHUNK_1500102Grundbacke_EGK_25_4"),
]


def test_the_flange_joint_takes_the_link_nothing_hangs_off():
    assert gripper_root_body(ROBOTIQ_2F85) == "/G/base_link"
    assert gripper_root_body(SCHUNK_EGK) == "/G/SCHUNK_1491752_EGK_25_PN_M_B_000ohne0"


def test_a_root_joint_to_the_world_does_not_confuse_the_root():
    """Schunk assets carry a `root_joint` with an empty body0; it is not a link."""
    assert gripper_root_body([("", "/G/housing")] + SCHUNK_EGK) == "/G/SCHUNK_1491752_EGK_25_PN_M_B_000ohne0"
    assert gripper_root_body([]) is None


UR10_WITH_HAND_E = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "Slider_1",
    "Slider_2",
]


def test_fitted_joints_are_found_even_when_their_names_say_nothing():
    """The tokens find nothing in `Slider_1`; the stamp does."""
    assert fitted_gripper_indices(UR10_WITH_HAND_E, [], ["Slider_1", "Slider_2"]) == [6, 7]


def test_fitted_joints_add_to_what_the_tokens_found_and_never_duplicate():
    names = UR10_WITH_HAND_E[:6] + ["finger_joint", "right_outer_knuckle_joint"]
    assert fitted_gripper_indices(names, [6, 7], ["finger_joint", "right_outer_knuckle_joint"]) == [6, 7]
    assert fitted_gripper_indices(names, [6], None) == [6]


class _Robot:
    prim_path = "/World/UR"

    def __init__(self, names, links, extra):
        self.joint_names = names
        self._links = links
        self._extra = extra

    def links(self):
        return list(self._links)

    def _fitted_gripper_links(self):
        return list(self._extra)


def test_pad_measurement_sees_the_fitted_gripper_beside_the_arm():
    """The gripper lives at /World/URGripper, not under /World/UR, so
    `robot.links()` cannot see its pads; the arm hands them over."""
    robot = _Robot(
        ["shoulder_pan_joint", "finger_joint", "right_outer_knuckle_joint"],
        ["/World/UR/base_link", "/World/UR/wrist_3_link"],
        ["/World/URGripper/base_link", "/World/URGripper/left_inner_finger", "/World/URGripper/right_inner_finger"],
    )
    gripper = Gripper(robot, [1, 2])
    assert gripper._pad_links() == ["/World/URGripper/left_inner_finger", "/World/URGripper/right_inner_finger"]


def test_a_robot_without_the_hook_still_measures_its_own_pads():
    class Bare(_Robot):
        _fitted_gripper_links = None

    robot = Bare(["a"], ["/World/F/panda_leftfinger", "/World/F/panda_rightfinger"], [])
    gripper = Gripper(robot, [0])
    assert len(gripper._pad_links()) == 2
