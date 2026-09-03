"""Does the guarding actually fit the cell it was built around?

The fence is sized from the cell's own description rather than typed in, and
these are the checks that the arithmetic connecting the two is right. They
matter because every failure in this area is silent: a crossing that misses
the belt puts a panel across it, and a panel across a conveyor is edge-on from
the camera angle anyone would take and stops every carton.
"""
from __future__ import annotations

import numpy as np
import pytest

from simliverse_sim.guarding import SafetyFence


class _Spawned:
    def __init__(self, prim_path, **kwargs):
        self.prim_path = prim_path
        self.kwargs = kwargs

    @property
    def position(self):
        return np.asarray(self.kwargs["position"], dtype=float)

    @property
    def scale(self):
        return np.asarray(self.kwargs["scale"], dtype=float)


class _FakeScene:
    def __init__(self):
        self.spawned: list[_Spawned] = []

    def spawn_rigid(self, prim_path, **kwargs):
        made = _Spawned(prim_path, **kwargs)
        self.spawned.append(made)
        return made


#: The measured cell, as `Conveyor.describe()` reports it.
BELT = {
    "centre": [-0.05, -0.40, 0.45],
    "length": 1.6,
    "width": 0.4,
    "direction": [1.0, 0.0, 0.0],
}
PALLET_Y = 0.75
REACH = 1.3
MARGIN = 0.55


def _fence(scene, *, margin=MARGIN, belt=None):
    """Rebuild the footprint arithmetic `demo.guarded_cell.guard` uses."""
    belt = belt or BELT
    belt_y = belt["centre"][1]
    belt_far_x = belt["centre"][0] + belt["length"] / 2.0
    envelope = REACH + margin
    west = -envelope
    east = envelope
    south = min(belt_y - belt["width"] / 2.0 - margin, -envelope)
    north = max(PALLET_Y + 0.605 + margin, envelope)
    reaches = belt_far_x >= east - 1e-6
    return SafetyFence.build(
        "/World/Fence",
        centre=((west + east) / 2.0, (south + north) / 2.0),
        size=(east - west, north - south),
        gate="south", gate_width=1.0,
        crossings=([{"side": "east", "centre": belt_y,
                     "width": belt["width"] + 0.30}] if reaches else []),
        scene=scene)


def test_the_conveyor_leaves_through_a_gap_not_a_panel() -> None:
    """The expensive failure: a panel across the belt, invisible from front."""
    scene = _FakeScene()
    belt = dict(BELT, centre=[0.9, -0.4, 0.45], length=3.0)
    fence = _fence(scene, belt=belt)

    belt_y = belt["centre"][1]
    half = belt["width"] / 2.0
    for panel in scene.spawned:
        if "East" not in panel.prim_path or "Post" in panel.prim_path:
            continue
        low = panel.position[1] - panel.scale[1]
        high = panel.position[1] + panel.scale[1]
        assert high <= belt_y - half or low >= belt_y + half, (
            "%s stands in the belt's path at y=%.2f" % (panel.prim_path, belt_y))


def test_a_belt_that_reaches_the_line_gets_a_slot_wider_than_itself() -> None:
    """Cut to the exact belt width, a crossing clips the guide rails."""
    scene = _FakeScene()
    long_enough = dict(BELT, centre=[0.9, -0.4, 0.45], length=3.0)  # ends 2.4
    fence = _fence(scene, belt=long_enough)
    start, end = fence.openings["east"][0]

    assert end - start > long_enough["width"], "no room either side"
    assert start < long_enough["centre"][1] < end, "the gap is not on the belt"


def test_a_belt_that_stops_short_gets_no_slot_at_all() -> None:
    """A doorway onto nothing is a hole in the guarding that nothing uses.

    The measured cell's belt ends at 0.75 m and the east line stands at 1.85,
    because that side is set by the arm's reach rather than by the conveyor.
    Cutting the slot anyway - which is what the first version did - leaves an
    opening no carton ever passes through.
    """
    scene = _FakeScene()
    fence = _fence(scene)

    assert fence.openings["east"] == [], (
        "the belt ends inside the guarding, so nothing should be cut for it")


def test_the_arm_and_the_pallet_are_both_inside_the_guarding() -> None:
    scene = _FakeScene()
    fence = _fence(scene)

    assert fence.contains((0.0, 0.0)), "the arm is outside its own fence"
    assert fence.contains((0.0, PALLET_Y)), "the pallet is outside the fence"
    assert fence.contains((0.0, PALLET_Y + 0.605)), "the pallet's far edge"


def test_the_belt_run_is_inside_the_guarding_up_to_its_exit() -> None:
    scene = _FakeScene()
    fence = _fence(scene)
    belt_y = BELT["centre"][1]
    near = BELT["centre"][0] - BELT["length"] / 2.0

    assert fence.contains((near, belt_y)), "the belt's near end is outside"


def test_the_arm_does_not_reach_the_fence() -> None:
    """Guarding on the envelope is a fence the arm polishes."""
    scene = _FakeScene()
    fence = _fence(scene)
    verdict = fence.fits((0.0, 0.0), reach=REACH)

    assert verdict["inside"]
    assert not verdict["touches_fence"], (
        "the arm reaches the guarding with %.2f m to spare" % verdict["clearance"])


@pytest.mark.parametrize("margin", [0.05, 0.2, 0.55, 1.0, 2.0])
def test_the_guarding_clears_the_envelope_at_any_margin(margin) -> None:
    """The invariant the sizing exists to hold.

    It did not hold before: sizing the east line off the belt's far end put
    the guarding 0.95 m from a 1.3 m arm, because the belt was the only thing
    on that side and it stopped short of the reach. Taking the larger of the
    equipment extent and the envelope is what makes the margin mean what it
    says on every side rather than on three of them.
    """
    fence = _fence(_FakeScene(), margin=margin)
    verdict = fence.fits((0.0, 0.0), reach=REACH)

    assert verdict["inside"]
    assert not verdict["touches_fence"], (
        "at margin %.2f the arm reaches guarding %.2f m away"
        % (margin, verdict["clearance"]))
    assert verdict["clearance"] >= REACH + margin - 1e-6


def test_the_operator_stands_outside_the_gate() -> None:
    scene = _FakeScene()
    fence = _fence(scene)
    gate_line = fence.centre[1] - fence.size[1] / 2.0
    station = (float(fence.centre[0]), float(gate_line - 1.0))

    assert not fence.contains(station)


def test_a_longer_belt_moves_the_crossing_with_it() -> None:
    """Sized from the cell, not typed in - so a different belt still exits."""
    scene = _FakeScene()
    longer = dict(BELT, centre=[0.4, -0.9, 0.45], length=3.0, width=0.6)
    fence = _fence(scene, belt=longer)
    start, end = fence.openings["east"][0]

    assert start < -0.9 < end, "the crossing did not follow the belt"
    assert end - start > 0.6
    assert fence.contains((0.4, -0.9)), "the belt is not inside the fence"


def test_a_short_belt_ends_inside_the_guarding() -> None:
    """The measured cell's belt does not leave the fence, and that is fine.

    It is a layout question - how do cartons get in - not a defect, so it is
    reported rather than corrected. What would be a defect is cutting a slot
    for a conveyor that never arrives at it.
    """
    scene = _FakeScene()
    fence = _fence(scene)
    far_x = BELT["centre"][0] + BELT["length"] / 2.0
    east_line = fence.centre[0] + fence.size[0] / 2.0

    assert east_line > far_x
    assert fence.openings["east"] == []
