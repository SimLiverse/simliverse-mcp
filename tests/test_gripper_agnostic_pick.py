# MIT License
#
# Copyright (c) 2026 SimLiverse

"""One palletising loop, two end effectors.

A suction cup grips a carton's top FACE; a finger jaw squeezes its SIDES. The
geometry is different (a box hangs below a cup, but sits between pads) and so is
what "is it held" means (a cup reports a seal; a jaw is read off contact), yet
`pick_waiting_box`/`place_on_slot` must run either. These are the pieces that
can be checked without a simulator: the datum arithmetic and, most importantly,
the jaw's refusal to promise a grip on a box wider than it opens - the one fact
about finger grippers the agent cannot guess and a demo must not paper over.
"""

from __future__ import annotations

import inspect

import demo.ur10_palletizing as demo


class _FakeCup:
    tip_offset = 0.05
    status = "Open"

    def __init__(self):
        self.holding = False
        self.gripped_objects = []


class _FakeJaw:
    """A linkage jaw stand-in: an open width and a measured pad drop."""

    def __init__(self, open_width=0.085, tip_offset=0.155):
        self.open_width = open_width
        self.tip_offset = tip_offset
        self.position = 0.0


def test_build_lets_the_caller_choose_a_gripper() -> None:
    """The whole point: `build(gripper=...)` exists and defaults to suction."""
    params = inspect.signature(demo.build).parameters
    assert "gripper" in params, "build() cannot be told which end effector to fit"
    assert params["gripper"].default == "suction"


def test_a_cup_reaches_the_top_face_and_holds_the_box_below_it() -> None:
    ee = demo._SuctionEE(_FakeCup())
    here = [0.0, 0.0, 0.50]  # a 0.10 m box, centre at z=0.50
    size = 0.10
    # The flange sits a cup above the top face: 0.50 + 0.05 (half box) + 0.05.
    assert abs(ee.approach_tool_z(here, size) - 0.60) < 1e-9
    # While held, the box centre hangs half a box plus a cup below the flange.
    assert abs(ee.hold_center_offset(size) - 0.10) < 1e-9


def test_a_jaw_straddles_the_box_centre_not_its_face() -> None:
    ee = demo._JawEE(arm=None, jaw=_FakeJaw(tip_offset=0.155), key="2f_85")
    here = [0.0, 0.0, 0.50]
    size = 0.06
    # The pad plane goes to the box CENTRE, so the flange sits one tip_offset
    # above the centre - the half-box term that suction carries is absent.
    assert abs(ee.approach_tool_z(here, size) - (0.50 + 0.155)) < 1e-9
    assert abs(ee.hold_center_offset(size) - 0.155) < 1e-9


def test_a_jaw_refuses_a_box_wider_than_it_opens() -> None:
    """The fact the agent must be told: a 2F-85 palletises parcels, not cartons."""
    ee = demo._JawEE(arm=None, jaw=_FakeJaw(open_width=0.085), key="2f_85")

    fits_small, _ = ee.fits(0.06)
    assert fits_small, "a 60 mm parcel is inside an 85 mm jaw"

    fits_carton, why = ee.fits(0.15)
    assert not fits_carton, "a 150 mm carton does not fit an 85 mm jaw"
    assert "mm" in why and "jaw opens" in why, "the refusal must say the numbers"


def test_a_cup_never_gates_on_carton_width() -> None:
    """Suction grips a face, so a wide carton is fine where a jaw would refuse."""
    ee = demo._SuctionEE(_FakeCup())
    fits, _ = ee.fits(0.30)
    assert fits, "a suction cup should take a 300 mm carton"


def test_a_native_gripper_targets_the_grasp_point_not_a_flange() -> None:
    """A Franka's IK frame is the point between its fingers, so `pose_to(centre)`
    already puts the fingers around the box and the drop is 0 - the working
    finger-jaw palletise. A bolted-on jaw's frame is the flange, drop is real."""
    ee = demo._JawEE(arm=None, jaw=_FakeJaw(), key="native")
    here = [0.0, 0.0, 0.50]
    # The tool frame goes straight to the box centre: no flange offset.
    assert abs(ee.approach_tool_z(here, 0.05) - 0.50) < 1e-9
    assert abs(ee.hold_center_offset(0.05)) < 1e-9
    fits, _ = ee.fits(0.05)
    assert fits, "a 50 mm box fits a 75 mm native jaw"
    assert not ee.fits(0.12)[0], "a 120 mm box does not"


def test_placement_squares_a_box_gripped_askew() -> None:
    """A cup carries whatever yaw a box drifted to on the belt; the place rotates
    the wrist by that offset so the box lands on the slot's angle. A carton is
    90-deg symmetric, so the correction wraps into +/-45 deg."""
    # 11 deg off square -> rotate the wrist -11 deg to bring it back.
    assert abs(demo._square_delta(11.0, {"yaw": 0.0}) - (-11.0)) < 1e-6
    # 88 deg reads as 2 deg off, not 88 (the symmetry wrap).
    assert abs(demo._square_delta(88.0, {"yaw": 0.0}) - 2.0) < 1e-6
    # A slot that wants a non-zero angle is matched, not just zero.
    assert abs(demo._square_delta(11.0, {"yaw": 45.0}) - 34.0) < 1e-6


def test_old_cells_that_carry_only_a_cup_still_resolve_an_end_effector() -> None:
    """`_ee_of` must wrap a bare cup, so a cell built before this change runs."""
    cell = {"cup": _FakeCup()}
    ee = demo._ee_of(cell)
    assert isinstance(ee, demo._SuctionEE)
    assert ee.kind == "suction"
