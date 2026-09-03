"""The cell, built at sizes it was never measured at.

Every constant in `demo.ur10_palletizing` came from one geometry. These
tests do not check that a 22 cm carton palletises - that needs a GPU and a
running stage - they check the arithmetic that a different carton feeds,
which is where a demo-fitted number does its damage silently: a gate too
short to stop anything, a cup wider than the face it seals against, a queue
spaced tighter than the cartons are wide.
"""
from __future__ import annotations

import inspect

import pytest

import demo.ur10_palletizing as cell_mod
from simliverse_sim.palletizing import pallet_slots

SIZES = [0.08, 0.10, 0.15, 0.22, 0.30]


def _defaults():
    sig = inspect.signature(cell_mod.build)
    return {name: p.default for name, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty}


def test_the_cell_is_buildable_at_more_than_one_size() -> None:
    """A build that takes only `boxes` cannot be asked this question at all."""
    defaults = _defaults()
    for knob in ("box", "box_mass", "deck", "stop_x", "offset_y", "speed",
                 "pallet_y", "rows", "cols", "layers", "robot"):
        assert knob in defaults, (
            "%s is still a module constant, so no scenario can vary it" % knob)


def test_the_measured_cell_is_still_the_default() -> None:
    """Parameterising must not quietly move the cell the numbers came from."""
    defaults = _defaults()
    assert defaults["box"] == cell_mod.BOX
    assert defaults["box_mass"] == cell_mod.BOX_MASS
    assert defaults["deck"] == cell_mod.DECK
    assert defaults["stop_x"] == cell_mod.STOP_X
    assert defaults["offset_y"] == cell_mod.OFFSET_Y
    assert defaults["speed"] == cell_mod.SPEED
    assert defaults["pallet_y"] == cell_mod.PALLET_Y
    assert defaults["robot"] == "ur10"


@pytest.mark.parametrize("size", SIZES)
def test_the_pick_reads_the_carton_the_cell_was_built_with(size) -> None:
    """Reading the module constant computes a 15 cm box's top every time.

    At 22 cm that puts the target 35 mm inside the carton the cup is meant to
    seal against, and the failure looks like a bad grip rather than bad
    arithmetic.
    """
    assert cell_mod._box_of({"box_size": size}) == pytest.approx(size)


def test_a_cell_without_a_recorded_size_falls_back_to_the_measured_one() -> None:
    assert cell_mod._box_of({}) == pytest.approx(cell_mod.BOX)


@pytest.mark.parametrize("size", SIZES)
def test_a_gate_is_taller_than_what_it_stops(size) -> None:
    """A stop shorter than the carton is a ramp."""
    assert cell_mod.cell_geometry(size)["gate_height"] > size


@pytest.mark.parametrize("size", SIZES)
def test_a_cup_never_overhangs_the_face_it_seals(size) -> None:
    """A cup wider than the top face grips the corner it hangs over."""
    cup_radius = cell_mod.cell_geometry(size)["cup_radius"]
    assert cup_radius * 2.0 <= size, (
        "a %.0f mm cup on a %.0f mm face seals on air" % (
            cup_radius * 2000.0, size * 1000.0))


@pytest.mark.parametrize("size", SIZES)
def test_a_queue_is_spaced_wider_than_the_cartons(size) -> None:
    """Cartons spaced tighter than they are wide arrive as one block."""
    assert cell_mod.cell_geometry(size)["spacing"] > size


@pytest.mark.parametrize("size", SIZES)
def test_the_belt_is_wider_than_the_carton_with_room_to_sit_askew(size) -> None:
    assert cell_mod.cell_geometry(size)["width"] >= size + 0.10


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 2), (2, 3), (3, 3)])
def test_slots_never_overlap_at_any_carton_size(size, rows, cols) -> None:
    """The layout has to follow the carton, not the demo's 2x2 of 15 cm boxes."""
    slots = pallet_slots(origin=[0.0, 0.75, 0.1425], box=(size, size, size),
                         rows=rows, cols=cols, layers=1, gap=0.01)
    assert len(slots) == rows * cols
    places = [s["place"] for s in slots]
    for i, a in enumerate(places):
        for b in places[i + 1:]:
            apart = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            assert apart >= size - 1e-9, (
                "two %.0f mm cartons are %.0f mm apart, so they intersect" % (
                    size * 1000.0, apart * 1000.0))


@pytest.mark.parametrize("size", SIZES)
def test_every_slot_rests_a_carton_on_the_deck_not_in_it(size) -> None:
    deck_z = 0.1425
    slots = pallet_slots(origin=[0.0, 0.75, deck_z], box=(size, size, size),
                         rows=2, cols=2, layers=1, gap=0.01)
    for slot in slots:
        rest = deck_z + size / 2.0
        assert slot["place"][2] >= rest - 1e-6, (
            "a carton placed at %.4f sits inside a deck at %.4f" % (
                slot["place"][2], deck_z))


def test_home_is_not_handed_to_an_arm_with_different_joints() -> None:
    """Six angles measured on a UR10 are not a pose on another chain."""

    class _Arm:
        dof = 7

    home = cell_mod._home_of({"spec": {"robot": "kuka"}, "arm": _Arm()})
    assert len(home) == 7
    assert cell_mod._home_of({"spec": {"robot": "ur10"}}) == list(cell_mod.HOME)


def test_a_carton_with_no_size_is_refused_rather_than_derived_from() -> None:
    with pytest.raises(ValueError):
        cell_mod.cell_geometry(0.0)
