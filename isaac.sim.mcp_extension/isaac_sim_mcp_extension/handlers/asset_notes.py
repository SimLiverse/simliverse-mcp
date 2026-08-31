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

"""Plain-language notes about what a spawned asset turned out to be.

Kept apart from `robots.py` and free of imports on purpose: the extension
package cannot be imported without Isaac Sim (its `__init__` reaches for
`carb`), so anything that lives there can only be tested by parsing it. This
is ordinary logic over a list of joint names and deserves ordinary tests.
"""

from __future__ import annotations


def _what_this_actually_is(joint_names: list[str], key: str) -> str:
    """Say so when an asset is not the robot its name implies.

    `lite6_gripper` reads as "a Lite6 arm with a gripper on it". It is the
    UFACTORY gripper by itself: `uf_lite_gripper.usd`, three links,
    `finger_joint1` and `finger_joint2`, and no arm. Spawning it returned
    `status: success` with `num_dof: 2` and nothing else, so a task that asked
    to "grasp the cylinder with the arm at /World/Arm" ran against a floating
    hand -- and the failure was recorded against the controller rather than
    against the scene.

    Two DOF is technically all the information needed to work that out. It is
    not information anyone reads in time.
    """
    fingers = [j for j in joint_names if "finger" in j.lower()]
    arm = [j for j in joint_names if j not in fingers]
    if joint_names and not arm:
        return (
            f"'{key}' is an end effector on its own, not an arm: its only joints "
            f"are {joint_names}. It has nothing to carry it, so it cannot reach "
            f"for anything. Spawn an arm as well, or pick an asset that includes "
            f"one."
        )
    if arm and not fingers:
        return (
            f"'{key}' has no gripper joints -- {len(arm)} arm joints and nothing "
            f"to close. It can be positioned but not made to hold anything."
        )
    return ""
