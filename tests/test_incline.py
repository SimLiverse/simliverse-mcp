"""The geometry of a pitched belt, before physics.

Measured live first: a 1 kg carton rode a kinematic slab up 10-25 degrees with
mu=0.9 and settled against the stop; below mu=tan(pitch) it slid back. These
tests pin the arithmetic that positions the slab, the stop and the cartons so
that result is reproducible without a GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simliverse_sim.conveyor import Conveyor, _tilt_quat


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


def _belt(pitch_deg, heading=(1.0, 0.0, 0.0)):
    b = Conveyor.__new__(Conveyor)
    b.direction = np.asarray(heading, dtype=float) / np.linalg.norm(heading)
    b.pitch = math.radians(pitch_deg)
    b.top_z = 0.5
    b._origin = np.array([0.0, 0.0, 0.5])
    b.speed = 0.4
    return b


def test_flat_is_the_flat_path_untouched():
    b = _belt(0.0)
    assert b.pitch == 0.0
    assert b.deck_z([3.0, 0.0]) == 0.5
    np.testing.assert_allclose(b.heading3(), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("pitch", [10, 15, 20, 25])
def test_the_surface_rises_toward_the_gate(pitch):
    b = _belt(pitch)
    # A metre along the heading gains tan(pitch) in height.
    assert b.deck_z([1.0, 0.0]) - b.top_z == pytest.approx(math.tan(math.radians(pitch)))
    assert b.deck_z([-1.0, 0.0]) - b.top_z == pytest.approx(-math.tan(math.radians(pitch)))


@pytest.mark.parametrize("pitch", [10, 20, 30])
def test_heading3_lifts_by_the_pitch(pitch):
    b = _belt(pitch)
    h = b.heading3()
    assert np.linalg.norm(h) == pytest.approx(1.0)
    assert h[2] == pytest.approx(math.sin(math.radians(pitch)))
    assert math.degrees(math.atan2(h[2], math.hypot(h[0], h[1]))) == pytest.approx(pitch)


@pytest.mark.parametrize("yaw,pitch", [(0, 15), (90, 20), (37, 25), (180, 10)])
def test_the_slab_local_x_points_up_the_slope(yaw, pitch):
    q = _tilt_quat(yaw, math.radians(pitch))
    assert np.linalg.norm(q) == pytest.approx(1.0)
    local_x = _rotate(q, [1.0, 0.0, 0.0])
    # Rises by sin(pitch), and its horizontal part points along the heading.
    assert local_x[2] == pytest.approx(math.sin(math.radians(pitch)), abs=1e-9)
    horiz = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    assert np.dot(local_x[:2] / np.linalg.norm(local_x[:2]), horiz) == pytest.approx(1.0, abs=1e-6)


def test_the_stop_sits_above_the_pitched_far_end():
    """tan(pitch) * half-length is how far the far end has risen."""
    pitch, length = math.radians(20), 2.0
    rise = math.tan(pitch) * (length / 2.0 + 0.02)
    assert rise == pytest.approx(0.371, abs=0.01)


def test_half_run_is_horizontal_not_slope():
    """The bug the A42 ramp exposed: a slope half-length used as a horizontal
    reach put a carton past the belt's end at 30 degrees, and it fell."""
    b = _belt(0.0)
    b.length = 2.7
    assert b.half_run() == pytest.approx(1.35)
    b30 = _belt(30.0)
    b30.length = 2.7
    assert b30.half_run() == pytest.approx(1.35 * math.cos(math.radians(30)))
    assert b30.half_run() < 1.35
