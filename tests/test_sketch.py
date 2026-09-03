"""Reading a drawing, and building what it says.

The payloads here are the real thing, header and all, copied from the format
`dashboard/src/lib/sketch.ts` emits. That matters: the parser's job is to
survive a block of prose followed by shape lines, and a test that feeds it
only the shape lines would not be testing the case that actually arrives.
"""
from __future__ import annotations

import numpy as np
import pytest

from simliverse_sim import sketch as S
from simliverse_sim.sketch import SketchError

HEADER = """[LAYOUT SKETCH - plan view of the floor, all values in metres.
 The simulator is Z-up, so this is the XY plane: +X is right, +Y is away
 from the viewer, and (0, 0) is the marked origin on the canvas.
 These coordinates are the user's own measurements, taken off a metre grid.
 Treat them as the requested layout, not as an approximation to re-derive.]
"""

CELL = HEADER + """
rect   "cell" centre (0.00, 0.00) 6.50 x 6.50 m (x -3.25..3.25, y -3.25..3.25)
arrow  "infeed conveyor" (5.00, -0.40) -> (-1.00, -0.40) length 6.00 m heading 180.00 deg (-X)
circle "pallet" centre (0.00, 0.75) radius 0.60 m
"""


class _Spawned:
    def __init__(self, prim_path, **kw):
        self.prim_path = prim_path
        self.kwargs = kw

    @property
    def position(self):
        return np.asarray(self.kwargs["position"], dtype=float)


class _FakeScene:
    def __init__(self):
        self.spawned = []

    def spawn_rigid(self, prim_path, **kw):
        made = _Spawned(prim_path, **kw)
        self.spawned.append(made)
        return made


@pytest.fixture
def scene():
    return _FakeScene()


# ── Reading the payload ──────────────────────────────────────────────────────


def test_the_prose_header_does_not_stop_the_parser() -> None:
    """The header is there for the agent. A parser that chokes on it is a
    parser that forces the header out, and the header is what stops anyone
    re-deriving the frame."""
    shapes = S.parse_sketch(CELL)

    assert len(shapes["rects"]) == 1
    assert len(shapes["arrows"]) == 1
    assert len(shapes["circles"]) == 1


def test_a_rectangle_keeps_the_numbers_the_user_saw() -> None:
    """Nothing rescales. The grid was in metres and Isaac is Z-up."""
    rect = S.parse_sketch(CELL)["rects"][0]

    assert rect["label"] == "cell"
    assert rect["centre"] == pytest.approx((0.0, 0.0))
    assert rect["size"] == pytest.approx((6.5, 6.5))


def test_negative_coordinates_survive_the_round_trip() -> None:
    text = HEADER + '\nrect "cell" centre (-2.40, -1.05) 3.20 x 1.80 m\n'
    rect = S.parse_sketch(text)["rects"][0]

    assert rect["centre"] == pytest.approx((-2.4, -1.05))
    assert rect["size"] == pytest.approx((3.2, 1.8))


def test_an_arrow_keeps_both_ends() -> None:
    arrow = S.parse_sketch(CELL)["arrows"][0]

    assert arrow["from"] == pytest.approx((5.0, -0.4))
    assert arrow["to"] == pytest.approx((-1.0, -0.4))
    assert arrow["length"] == pytest.approx(6.0)


def test_a_circle_is_a_place_with_a_size() -> None:
    circle = S.parse_sketch(CELL)["circles"][0]

    assert circle["label"] == "pallet"
    assert circle["centre"] == pytest.approx((0.0, 0.75))
    assert circle["radius"] == pytest.approx(0.6)


def test_an_empty_sketch_is_refused_rather_than_built() -> None:
    for text in ("", "   \n\n "):
        with pytest.raises(SketchError, match="nothing to build"):
            S.parse_sketch(text)


def test_an_unlabelled_shape_still_parses() -> None:
    """The canvas emits `unlabelled` rather than dropping the shape."""
    text = HEADER + '\nrect "unlabelled" centre (0.00, 0.00) 4.00 x 4.00 m\n'
    assert S.parse_sketch(text)["rects"][0]["label"] == "unlabelled"


# ── Which rectangle is the cell ──────────────────────────────────────────────


def test_a_label_beats_size() -> None:
    """The failure this prevents: someone draws the building around the cell.

    'Biggest rectangle' is a fine rule until that happens, and then it fences
    the site instead of the machine and nothing says so.
    """
    text = HEADER + """
rect "warehouse floor" centre (0.00, 0.00) 40.00 x 25.00 m
rect "cell" centre (2.00, 1.00) 6.00 x 6.00 m
"""
    picked = S.pick_footprint(S.parse_sketch(text)["rects"])

    assert picked["label"] == "cell"
    assert picked["chosen_by"] == "label"


@pytest.mark.parametrize("word", ["fence", "cell", "guard", "enclosure",
                                  "perimeter", "cage", "safety"])
def test_the_words_people_actually_use_are_recognised(word) -> None:
    text = HEADER + ('\nrect "big" centre (0,0) 30.00 x 30.00 m'
                     '\nrect "%s line" centre (1,1) 5.00 x 5.00 m\n' % word)
    assert S.pick_footprint(S.parse_sketch(text)["rects"])["label"].startswith(
        word)


def test_one_unlabelled_rectangle_is_taken_as_the_cell() -> None:
    text = HEADER + '\nrect "unlabelled" centre (0,0) 5.00 x 4.00 m\n'
    picked = S.pick_footprint(S.parse_sketch(text)["rects"])

    assert picked["chosen_by"] == "the only rectangle"


def test_guessing_between_unlabelled_rectangles_says_it_guessed() -> None:
    """A guess that does not announce itself is the one that gets trusted."""
    text = HEADER + """
rect "unlabelled" centre (0,0) 5.00 x 4.00 m
rect "unlabelled" centre (0,0) 9.00 x 9.00 m
"""
    picked = S.pick_footprint(S.parse_sketch(text)["rects"])

    assert picked["size"] == pytest.approx((9.0, 9.0))
    assert picked["chosen_by"] == "largest, unlabelled"


def test_a_sketch_with_no_rectangle_says_what_to_draw() -> None:
    text = HEADER + '\ncircle "pallet" centre (0,0) radius 0.60 m\n'
    with pytest.raises(SketchError, match="draw the guarded area"):
        S.pick_footprint(S.parse_sketch(text)["rects"])


# ── Building the fence ───────────────────────────────────────────────────────


def test_a_drawn_rectangle_becomes_the_fence_line(scene) -> None:
    out = S.fence_from_sketch(CELL, scene=scene)
    fence = out["fence"]

    assert fence.centre.tolist() == pytest.approx([0.0, 0.0])
    assert fence.size.tolist() == pytest.approx([6.5, 6.5])
    assert fence.panels, "a fence with no panels is a floor plan"


def test_the_fence_encloses_what_was_drawn_inside_it(scene) -> None:
    out = S.fence_from_sketch(CELL, scene=scene)
    fence = out["fence"]

    assert fence.contains((0.0, 0.75)), "the pallet is outside its own cell"
    assert not fence.contains((5.0, -0.4)), "the infeed starts outside"


def test_an_arrow_crossing_the_line_opens_the_guarding(scene) -> None:
    """This is what someone means by drawing a conveyor running in."""
    out = S.fence_from_sketch(CELL, scene=scene)

    assert len(out["crossings"]) == 1
    crossing = out["crossings"][0]
    assert crossing["side"] == "east", "the arrow enters from +X"
    assert crossing["centre"] == pytest.approx(-0.4)
    assert crossing["for"] == "infeed conveyor"
    assert out["fence"].openings["east"], "the line was never opened"


def test_an_arrow_drawn_inside_the_cell_is_not_a_doorway(scene) -> None:
    """Inside, an arrow means travel direction, not an entry point."""
    text = HEADER + """
rect  "cell" centre (0.00, 0.00) 6.50 x 6.50 m
arrow "travel" (-1.00, 0.00) -> (1.00, 0.00) length 2.00 m heading 0.00 deg (+X)
"""
    out = S.fence_from_sketch(text, scene=scene)

    assert out["crossings"] == []
    assert out["fence"].openings["east"] == []


def test_an_arrow_entering_near_a_corner_still_picks_one_side(scene) -> None:
    """Picking the wrong side puts the gap round the corner from the belt."""
    text = HEADER + """
rect  "cell" centre (0.00, 0.00) 6.00 x 6.00 m
arrow "belt" (4.00, 2.90) -> (0.00, 2.90) length 4.00 m heading 180.00 deg (-X)
"""
    out = S.fence_from_sketch(text, scene=scene)

    assert len(out["crossings"]) == 1
    assert out["crossings"][0]["side"] == "east"


@pytest.mark.parametrize("start,side", [
    ((6.0, 0.0), "east"),
    ((-6.0, 0.0), "west"),
    ((0.0, 6.0), "north"),
    ((0.0, -6.0), "south"),
])
def test_a_feed_from_each_direction_opens_the_right_side(scene, start, side) -> None:
    text = HEADER + (
        '\nrect  "cell" centre (0.00, 0.00) 6.00 x 6.00 m'
        '\narrow "belt" (%.2f, %.2f) -> (0.00, 0.00) length 6.00 m heading 0 deg\n'
        % start)
    out = S.fence_from_sketch(text, scene=scene, gate=None)

    assert out["crossings"][0]["side"] == side
    assert out["fence"].openings[side]


def test_the_gate_is_still_cut_alongside_a_crossing(scene) -> None:
    out = S.fence_from_sketch(CELL, scene=scene, gate="south")
    assert out["fence"].openings["south"], "a person still has to get in"


def test_the_result_says_how_the_footprint_was_chosen(scene) -> None:
    """A sketch is ambiguous; picking a reading silently is how a drawing
    becomes a cell nobody recognises."""
    out = S.fence_from_sketch(CELL, scene=scene)
    assert out["footprint"]["chosen_by"] == "label"
    assert out["footprint"]["label"] == "cell"


def test_what_was_ignored_is_reported(scene) -> None:
    out = S.fence_from_sketch(CELL, scene=scene)
    assert "pallet" in out["ignored"]["circles"]


# ── The rest of the drawing ──────────────────────────────────────────────────


def test_zones_hands_back_placeable_numbers() -> None:
    zones = S.zones_from_sketch(CELL)

    assert zones["cell"]["size"] == pytest.approx([6.5, 6.5])
    assert zones["spots"][0]["label"] == "pallet"
    assert zones["spots"][0]["centre"] == pytest.approx([0.0, 0.75])
    assert zones["flows"][0]["length"] == pytest.approx(6.0)


def test_zones_does_not_decide_what_a_circle_means() -> None:
    """What "pallet here" implies is the agent's call, not this module's."""
    zones = S.zones_from_sketch(CELL)
    spot = zones["spots"][0]

    assert set(spot) == {"label", "centre", "radius"}


def test_a_sketch_of_only_spots_still_reports_them() -> None:
    text = HEADER + '\ncircle "drop zone" centre (1.00, 2.00) radius 0.50 m\n'
    zones = S.zones_from_sketch(text)

    assert zones["cell"] is None
    assert zones["spots"][0]["label"] == "drop zone"
