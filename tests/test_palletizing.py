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

"""The pallet pattern.

Pure arithmetic over the pattern spec, so this runs without Isaac Sim — which is
the point of computing the pattern outside the controller. The failures these
cover are the ones that do not raise: a stack that leans, a layer that overhangs,
two boxes assigned the same slot. Every one of those used to be discovered by
watching a tower fall over.
"""

import math

import pytest

from simliverse_sim.palletizing import PalletError, pallet_slots, verify_pallet

BOX = (0.18, 0.13, 0.11)
DECK = [0.0, 1.20, 0.145]


class _FakeBody:
    """Just enough of a RigidObject for `verify_pallet` to read."""

    def __init__(self, position, speed=0.0):
        self.position = position
        self.speed = speed


def test_a_two_by_two_layer_is_four_slots_centred_on_the_pallet() -> None:
    slots = pallet_slots(DECK, BOX, rows=2, cols=2)
    assert len(slots) == 4

    # The pattern's own centre of mass must sit on the pallet centre, or the
    # load is off-axis and the stack leans as it grows.
    mean_x = sum(s["rest"][0] for s in slots) / 4
    mean_y = sum(s["rest"][1] for s in slots) / 4
    assert mean_x == pytest.approx(DECK[0], abs=1e-9)
    assert mean_y == pytest.approx(DECK[1], abs=1e-9)


def test_the_first_layer_rests_half_a_box_above_the_deck() -> None:
    """A box rests on its bottom face; its centre is half its height up."""
    slot = pallet_slots(DECK, BOX)[0]
    assert slot["rest"][2] == pytest.approx(DECK[2] + BOX[2] / 2, abs=1e-9)


def test_layers_stack_by_exactly_one_box_height() -> None:
    slots = pallet_slots(DECK, BOX, rows=2, cols=2, layers=3)
    assert len(slots) == 12
    by_layer = {}
    for slot in slots:
        by_layer.setdefault(slot["layer"], []).append(slot["rest"][2])
    for layer, heights in by_layer.items():
        assert len(set(round(h, 9) for h in heights)) == 1, "a layer is not level"
        expected = DECK[2] + layer * BOX[2] + BOX[2] / 2
        assert heights[0] == pytest.approx(expected, abs=1e-9)


def test_release_is_above_rest_but_only_just() -> None:
    """3 mm, not 12. Letting go higher knocked a two-high stack apart."""
    slot = pallet_slots(DECK, BOX, clearance=0.003)[0]
    assert slot["place"][2] - slot["rest"][2] == pytest.approx(0.003, abs=1e-9)
    assert slot["approach"][2] > slot["place"][2]


def test_no_two_boxes_are_sent_to_the_same_place() -> None:
    """The bug that stacks two boxes into one another and calls it a layer."""
    slots = pallet_slots(DECK, BOX, rows=3, cols=2, layers=2)
    seen = {tuple(round(v, 6) for v in s["rest"]) for s in slots}
    assert len(seen) == len(slots)


def test_boxes_in_a_layer_do_not_overlap() -> None:
    slots = [s for s in pallet_slots(DECK, BOX, rows=2, cols=2, gap=0.01) if s["layer"] == 0]
    for i, a in enumerate(slots):
        for b in slots[i + 1 :]:
            dx = abs(a["rest"][0] - b["rest"][0])
            dy = abs(a["rest"][1] - b["rest"][1])
            # Axis-aligned boxes are clear if they are separated on either axis.
            assert dx >= BOX[0] - 1e-9 or dy >= BOX[1] - 1e-9, f"slots {a['index']} and {b['index']} overlap"


def test_interlock_turns_alternate_layers_and_grid_does_not() -> None:
    plain = pallet_slots(DECK, BOX, rows=2, cols=2, layers=2)
    assert {s["yaw"] for s in plain} == {0.0}

    tied = pallet_slots(DECK, BOX, rows=2, cols=2, layers=2, interlock=True)
    assert {s["yaw"] for s in tied if s["layer"] == 0} == {0.0}
    assert {s["yaw"] for s in tied if s["layer"] == 1} == {90.0}


def test_a_pattern_that_fits_the_deck_is_allowed() -> None:
    """4x4 of these boxes is 0.75 x 0.55 m, which a 0.8 x 0.6 m pallet carries."""
    assert len(pallet_slots(DECK, BOX, rows=4, cols=4, deck=(0.8, 0.6))) == 16


def test_a_pattern_that_overhangs_the_deck_is_refused() -> None:
    """The outer column would rest on nothing. Better to raise than to stack it."""
    with pytest.raises(PalletError, match="overhang"):
        pallet_slots(DECK, BOX, rows=6, cols=6, deck=(0.8, 0.6))


def test_the_deck_check_looks_at_the_rotated_layer_too() -> None:
    """A footprint that fits flat can overhang once a layer is turned 90 degrees."""
    long_box = (0.30, 0.10, 0.10)
    # 2 cols x 1 row flat = 0.61 x 0.10; turned it becomes 0.21 x 0.30.
    pallet_slots(DECK, long_box, rows=1, cols=2, layers=1, deck=(0.65, 0.15))
    with pytest.raises(PalletError, match="overhang"):
        pallet_slots(
            DECK,
            long_box,
            rows=1,
            cols=2,
            layers=2,
            interlock=True,
            deck=(0.65, 0.15),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"cols": -1},
        {"layers": 0},
        {"gap": -0.01},
    ],
)
def test_nonsense_patterns_raise(kwargs) -> None:
    with pytest.raises(PalletError):
        pallet_slots(DECK, BOX, **kwargs)


def test_a_zero_sized_box_raises_rather_than_producing_flat_slots() -> None:
    with pytest.raises(PalletError):
        pallet_slots(DECK, (0.18, 0.0, 0.11))


# ── verify_pallet ────────────────────────────────────────────────────────────


def test_verify_passes_when_every_box_sits_on_its_slot() -> None:
    slots = pallet_slots(DECK, BOX, rows=2, cols=2)
    bodies = [_FakeBody(s["rest"]) for s in slots]
    report = verify_pallet(bodies, slots)
    assert report["complete"] is True
    assert report["placed"] == 4


def test_verify_names_the_box_that_missed_and_by_how_far() -> None:
    slots = pallet_slots(DECK, BOX, rows=2, cols=2)
    bodies = [_FakeBody(s["rest"]) for s in slots]
    bodies[2] = _FakeBody([slots[2]["rest"][0] + 0.09, *slots[2]["rest"][1:]])

    report = verify_pallet(bodies, slots)
    assert report["complete"] is False
    assert report["placed"] == 3
    bad = [s for s in report["slots"] if not s["ok"]]
    assert len(bad) == 1
    assert bad[0]["index"] == 2
    assert bad[0]["error"] == pytest.approx(0.09, abs=1e-6)
    assert "from its slot" in bad[0]["reason"]


def test_a_box_still_falling_is_not_placed_however_close_it_is() -> None:
    """Measured one frame too early, a collapsing stack reads as a success."""
    slots = pallet_slots(DECK, BOX, rows=1, cols=1)
    report = verify_pallet([_FakeBody(slots[0]["rest"], speed=0.5)], slots)
    assert report["complete"] is False
    assert report["slots"][0]["reason"] == "still moving"


def test_an_unreadable_body_is_reported_not_raised() -> None:
    class Dead:
        @property
        def position(self):
            raise RuntimeError("stale handle")

        speed = 0.0

    slots = pallet_slots(DECK, BOX, rows=1, cols=1)
    report = verify_pallet([Dead()], slots)
    assert report["complete"] is False
    assert "unreadable" in report["slots"][0]["reason"]


def test_an_empty_pallet_is_not_a_complete_one() -> None:
    """`all([])` is True, and that once made a run with nothing placed pass."""
    assert verify_pallet([], [])["complete"] is False


def test_placement_order_is_bottom_layer_first() -> None:
    slots = pallet_slots(DECK, BOX, rows=2, cols=2, layers=2)
    layers = [s["layer"] for s in slots]
    assert layers == sorted(layers), "a box would be placed under one already there"


def test_within_a_layer_the_arm_does_not_cross_back_over_itself() -> None:
    """Serpentine order: consecutive slots are always adjacent, never diagonal."""
    slots = [s for s in pallet_slots(DECK, BOX, rows=3, cols=3) if s["layer"] == 0]
    for a, b in zip(slots, slots[1:]):
        step = math.hypot(a["rest"][0] - b["rest"][0], a["rest"][1] - b["rest"][1])
        assert step <= max(BOX[0], BOX[1]) + 0.02 + 1e-9, "order jumps across the pallet"
