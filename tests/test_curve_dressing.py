"""The curve prop's origin is its entry port, not the arc centre (measured off
ConveyorBelt_A02: rollers local x 0..1.95, y -1.57..0.45, centreline radius
1.55 round (0, -1.55)). `dress()` used to put the prop at the arc centre, 1.55 m
off its own slabs. Now a clockwise arc gets the prop on its entry point yawed to
the entry heading, and an anticlockwise arc runs the prop backwards from its
exit point."""

import math

import numpy as np
import pytest

import simliverse_sim.conveyor as conveyor
from simliverse_sim.conveyor import CURVES, CurvedConveyor


def _curve(a0_deg: float, turn_deg: float) -> CurvedConveyor:
    return CurvedConveyor(
        "/World/Curve",
        centre=(4.0, -2.0),
        radius=1.55,
        a0=math.radians(a0_deg),
        a1=math.radians(a0_deg + turn_deg),
        width=0.9,
        speed=0.2,
        deck_z=0.769,
        segments=[],
        gate_path=None,
        scene=object(),
    )


@pytest.fixture
def spawned(monkeypatch):
    calls: list[dict] = []

    def fake_spawn(prop, prim_path, position, orientation, scene):
        calls.append({"prop": prop, "path": prim_path, "position": list(position), "orientation": list(orientation)})

    import simliverse_sim.props as props

    monkeypatch.setattr(props, "spawn_prop", fake_spawn)
    monkeypatch.setattr(conveyor, "_strip_physics", lambda scene, path: 0)
    return calls


def _local_to_world(call, local_xy):
    yaw = math.radians(call["orientation"][2])
    x, y = local_xy
    return (
        call["position"][0] + math.cos(yaw) * x - math.sin(yaw) * y,
        call["position"][1] + math.sin(yaw) * x + math.cos(yaw) * y,
    )


@pytest.mark.parametrize("prop", ["conveyorbelt_a01", "conveyorbelt_a02", "conveyorbelt_a03"])
def test_the_three_curves_share_the_measured_geometry(prop):
    assert CURVES[prop] == {"deck": 0.769, "radius": 1.55, "footprint": 2.07}


def test_a_clockwise_arc_puts_the_prop_on_the_entry_point_facing_the_entry_heading(spawned):
    # Entry at 90 deg round (4, -2): the point (4, -0.45), heading +X; the arc turns right to (5.55, -2) heading -Y.
    curve = _curve(90.0, -90.0)
    curve.dress("conveyorbelt_a02")
    (call,) = spawned
    assert call["prop"] == "conveyorbelt_a02"
    assert np.allclose(call["position"], [4.0, -0.45, 0.0], atol=1e-6)
    assert abs(call["orientation"][2] - 0.0) < 1e-6
    # The authored arc centre (0, -1.55) lands on ours, and the authored exit on our exit.
    assert np.allclose(_local_to_world(call, (0.0, -1.55)), (4.0, -2.0), atol=1e-6)
    assert np.allclose(_local_to_world(call, (1.55, -1.55)), (5.55, -2.0), atol=1e-6)


def test_an_anticlockwise_arc_runs_the_prop_backwards_from_the_exit_point(spawned):
    # Entry at -90 deg: the point (4, -3.55) heading +X; the arc turns left to (5.55, -2) heading +Y.
    curve = _curve(-90.0, 90.0)
    curve.dress("conveyorbelt_a01")
    (call,) = spawned
    assert np.allclose(call["position"], [5.55, -2.0, 0.0], atol=1e-6)
    assert abs(call["orientation"][2] - (-90.0)) < 1e-6
    # The authored entry sits on our exit, the authored exit on our entry, the centres coincide.
    assert np.allclose(_local_to_world(call, (0.0, -1.55)), (4.0, -2.0), atol=1e-6)
    assert np.allclose(_local_to_world(call, (1.55, -1.55)), (4.0, -3.55), atol=1e-6)


def test_the_prop_is_dropped_by_its_deck_so_the_rollers_meet_the_slabs(spawned):
    curve = _curve(90.0, -90.0)
    curve.deck_z = 1.0
    curve.dress("conveyorbelt_a03")
    (call,) = spawned
    assert abs(call["position"][2] - (1.0 - 0.769)) < 1e-6
