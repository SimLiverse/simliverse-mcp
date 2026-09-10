"""The arc geometry of a curved belt, before physics.

Measured live first: a 1 kg carton rode a fan of tangent chord slabs around a
90-degree bend (radius 1.0-1.5 m) and pressed against a stop at the exit.
These pin the angle bookkeeping so that is reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simliverse_sim.conveyor import CURVES, CurvedConveyor, curve_spec


def _curve(centre=(0.0, 1.55), radius=1.55, start=-90.0, turn=90.0, segments=8):
    c = CurvedConveyor.__new__(CurvedConveyor)
    c.centre = np.asarray(centre, dtype=float)
    c.radius = radius
    c.a0 = math.radians(start)
    c.a1 = math.radians(start + turn)
    c.width = 0.5
    c.speed = 0.4
    c.deck_z = 0.5
    c.box_size = np.array([0.15, 0.15, 0.15])
    # Reproduce the segment fan the builder makes.
    step = (c.a1 - c.a0) / segments
    c.segments = []
    for i in range(segments):
        amid = c.a0 + (i + 0.5) * step
        tang = np.array([-math.sin(amid), math.cos(amid)]) * np.sign(step)
        c.segments.append({"path": "/s%d" % i, "angle": float(amid), "tangent": tang, "yaw": 0.0})
    c._boxes = []
    return c


class _Body:
    def __init__(self, pos, speed=0.0):
        self.position = np.asarray(pos, dtype=float)
        self.speed = speed
        self.prim_path = "/World/Box"


def _at_angle(c, deg, dz=0.0, dr=0.0, speed=0.0):
    a = math.radians(deg)
    p = c.centre + (c.radius + dr) * np.array([math.cos(a), math.sin(a)])
    return _Body([p[0], p[1], c.deck_z + c.box_size[2] / 2.0 + dz], speed)


def test_segments_fan_across_the_turn():
    c = _curve(segments=6)
    assert len(c.segments) == 6
    angles = [math.degrees(s["angle"]) for s in c.segments]
    assert angles[0] == pytest.approx(-90 + 90 / 6 / 2)
    assert angles[-1] == pytest.approx(0 - 90 / 6 / 2)


def test_each_tangent_is_unit_and_perpendicular_to_the_radius():
    c = _curve()
    for s in c.segments:
        assert np.linalg.norm(s["tangent"]) == pytest.approx(1.0)
        radial = np.array([math.cos(s["angle"]), math.sin(s["angle"])])
        assert abs(np.dot(s["tangent"], radial)) < 1e-9


def test_nearest_segment_picks_by_angle():
    c = _curve()
    body = _at_angle(c, -45)  # middle of a -90..0 sweep
    seg = c._nearest_segment(body.position)
    assert abs(math.degrees(seg["angle"]) + 45) < 90 / 8


def test_box_at_gate_finds_the_settled_carton_at_the_exit():
    c = _curve()
    c._boxes = [_at_angle(c, -80), _at_angle(c, -2)]  # one near entry, one at exit
    got = c.box_at_gate()
    assert got is c._boxes[1]


def test_box_at_gate_refuses_a_moving_carton():
    c = _curve()
    c._boxes = [_at_angle(c, -1, speed=0.5)]
    assert c.box_at_gate() is None


def test_box_at_gate_refuses_one_flung_off_the_arc():
    c = _curve()
    c._boxes = [_at_angle(c, -1, dr=0.8)]  # right angle but far off-radius
    assert c.box_at_gate() is None


def test_a_carton_on_the_floor_is_not_at_the_gate():
    c = _curve()
    c._boxes = [_at_angle(c, -1, dz=-0.4)]
    assert c.box_at_gate() is None


def test_curve_spec_reads_the_a01_prop():
    assert curve_spec("conveyorbelt_a01")["radius"] == pytest.approx(1.55)
    assert curve_spec("conveyorbelt_a05") is None
    assert "conveyorbelt_a01" in CURVES
