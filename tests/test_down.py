"""A tool-down orientation for any flange axis, not just the two written out.

A Fanuc CRX's tool axis is -Y. With only X and Z handled, -Y fell through to
the Z formula: the cup was mounted correctly, the arm arrived to 0.2 mm, and
the cup pointed at the wall.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simliverse_sim.robots.manipulator import down_quaternion


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
    sign = -1.0 if name.startswith("-") else 1.0
    out = np.zeros(3)
    out["XYZ".index(name.lstrip("-"))] = sign
    return out


def test_the_two_hand_written_cases_are_unchanged():
    np.testing.assert_allclose(down_quaternion("Z", 0.0), [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    root = math.sqrt(0.5)
    np.testing.assert_allclose(down_quaternion("X", 0.0), [root, 0.0, root, 0.0], atol=1e-12)


@pytest.mark.parametrize("axis", ["X", "-X", "Y", "-Y", "Z", "-Z"])
def test_every_signed_axis_ends_up_pointing_at_the_floor(axis):
    q = down_quaternion(axis, 0.0)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(_rotate(q, _axis(axis)), [0.0, 0.0, -1.0], atol=1e-9)


@pytest.mark.parametrize("axis", ["Z", "-Y", "X"])
def test_yaw_turns_about_world_z_and_keeps_the_tool_down(axis):
    q = down_quaternion(axis, 90.0)
    np.testing.assert_allclose(_rotate(q, _axis(axis)), [0.0, 0.0, -1.0], atol=1e-9)
    # A tool-frame vector perpendicular to the approach axis rotates by 90
    # degrees in the floor plane between yaw 0 and yaw 90.
    perp = np.cross(_axis(axis), [0.3, 0.5, 0.7])
    perp /= np.linalg.norm(perp)
    a = _rotate(down_quaternion(axis, 0.0), perp)
    b = _rotate(q, perp)
    assert a[2] == pytest.approx(b[2], abs=1e-9)
    angle = math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])) % 360
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_a_nonsense_axis_is_refused():
    with pytest.raises(ValueError):
        down_quaternion("W", 0.0)
