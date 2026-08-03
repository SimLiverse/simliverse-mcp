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

"""Morphology classification.

Pure logic over joint-name lists, so it runs without Isaac Sim. The joint sets
below are the real ones from each robot's USD.

This matters because the previous implementation classified quadrupeds by looking
for "spot" or "go2" in the *prim path*, which misclassifies any robot a user
spawned under a different name.
"""

import pytest

from simliverse_sim.robots.base import JointGroups, Morphology, classify_morphology

FRANKA = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]

UR10 = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

CARTER = ["left_wheel", "right_wheel", "rear_pivot", "rear_axle"]

ANYMAL_C = [
    "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE",
]

UNITREE_H1 = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

ALLEGRO_HAND = [
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
    "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
]

QUADCOPTER = ["rotor0_joint", "rotor1_joint", "rotor2_joint", "rotor3_joint"]

MOBILE_MANIPULATOR = [
    "left_wheel_joint", "right_wheel_joint",
    "arm_shoulder_pan", "arm_shoulder_lift", "arm_elbow",
    "arm_wrist_1", "arm_wrist_2",
    "gripper_finger_left", "gripper_finger_right",
]


@pytest.mark.parametrize(
    ("joint_names", "expected"),
    [
        (FRANKA, Morphology.MANIPULATOR),
        (UR10, Morphology.MANIPULATOR),
        (CARTER, Morphology.WHEELED),
        (ANYMAL_C, Morphology.QUADRUPED),
        (UNITREE_H1, Morphology.HUMANOID),
        (ALLEGRO_HAND, Morphology.DEXTEROUS_HAND),
        (QUADCOPTER, Morphology.AERIAL),
        (MOBILE_MANIPULATOR, Morphology.MOBILE_MANIPULATOR),
    ],
)
def test_classifies_real_robots(joint_names, expected):
    groups = JointGroups.classify(joint_names)
    assert classify_morphology(joint_names, groups) is expected


def test_classification_ignores_prim_path_naming():
    """A quadruped stays a quadruped whatever the user called it."""
    groups = JointGroups.classify(ANYMAL_C)
    assert classify_morphology(ANYMAL_C, groups) is Morphology.QUADRUPED
    # No prim path is consulted at all — the signature takes joint names only.


def test_franka_gripper_joints_are_found():
    groups = JointGroups.classify(FRANKA)
    assert [FRANKA[i] for i in groups.gripper] == [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    # The seven arm joints must not be swept into the gripper group.
    assert len(groups.gripper) == 2


def test_allegro_fingers_group_separately():
    groups = JointGroups.classify(ALLEGRO_HAND)
    assert len(groups.gripper) == 16
    assert not groups.arms, "a standalone hand has no arm chain"


def test_wheel_and_steering_split():
    groups = JointGroups.classify(CARTER)
    assert [CARTER[i] for i in groups.wheels] == ["left_wheel", "right_wheel"]


def test_steering_joints_are_not_mistaken_for_fingers():
    """"steering" contains "ring", which is also a finger name."""
    joints = ["front_steering_joint", "rear_steering_joint", "left_wheel", "right_wheel"]
    groups = JointGroups.classify(joints)

    assert not groups.gripper, "a steering joint is not a finger"
    assert [joints[i] for i in groups.steering] == [
        "front_steering_joint",
        "rear_steering_joint",
    ]
    assert classify_morphology(joints, groups) is Morphology.WHEELED


def test_shadow_hand_abbreviated_joints():
    """Shadow Hand names joints FFJ1/MFJ2/RFJ3/LFJ4/THJ5 rather than spelling them out."""
    joints = [f"{prefix}J{n}" for prefix in ("FF", "MF", "RF", "LF", "TH") for n in range(4)]
    groups = JointGroups.classify(joints)

    assert len(groups.gripper) == 20
    assert classify_morphology(joints, groups) is Morphology.DEXTEROUS_HAND


def test_unknown_robot_degrades_gracefully():
    joints = ["mystery_a", "mystery_b"]
    groups = JointGroups.classify(joints)
    assert classify_morphology(joints, groups) is Morphology.UNKNOWN
    assert groups.other == [0, 1]


def test_groups_serialize_only_non_empty():
    groups = JointGroups.classify(UR10)
    rendered = groups.to_dict(UR10)
    assert "arms" in rendered
    assert "wheels" not in rendered, "empty groups should be omitted from describe()"
