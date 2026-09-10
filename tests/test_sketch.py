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

CELL = (
    HEADER
    + """
rect   "cell" centre (0.00, 0.00) 6.50 x 6.50 m (x -3.25..3.25, y -3.25..3.25)
arrow  "infeed conveyor" (5.00, -0.40) -> (-1.00, -0.40) length 6.00 m heading 180.00 deg (-X)
circle "pallet" centre (0.00, 0.75) radius 0.60 m
"""
)


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
    text = (
        HEADER
        + """
rect "warehouse floor" centre (0.00, 0.00) 40.00 x 25.00 m
rect "cell" centre (2.00, 1.00) 6.00 x 6.00 m
"""
    )
    picked = S.pick_footprint(S.parse_sketch(text)["rects"])

    assert picked["label"] == "cell"
    assert picked["chosen_by"] == "label"


@pytest.mark.parametrize("word", ["fence", "cell", "guard", "enclosure", "perimeter", "cage", "safety"])
def test_the_words_people_actually_use_are_recognised(word) -> None:
    text = HEADER + ('\nrect "big" centre (0,0) 30.00 x 30.00 m\nrect "%s line" centre (1,1) 5.00 x 5.00 m\n' % word)
    assert S.pick_footprint(S.parse_sketch(text)["rects"])["label"].startswith(word)


def test_one_unlabelled_rectangle_is_taken_as_the_cell() -> None:
    text = HEADER + '\nrect "unlabelled" centre (0,0) 5.00 x 4.00 m\n'
    picked = S.pick_footprint(S.parse_sketch(text)["rects"])

    assert picked["chosen_by"] == "the only rectangle"


def test_guessing_between_unlabelled_rectangles_says_it_guessed() -> None:
    """A guess that does not announce itself is the one that gets trusted."""
    text = (
        HEADER
        + """
rect "unlabelled" centre (0,0) 5.00 x 4.00 m
rect "unlabelled" centre (0,0) 9.00 x 9.00 m
"""
    )
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
    text = (
        HEADER
        + """
rect  "cell" centre (0.00, 0.00) 6.50 x 6.50 m
arrow "travel" (-1.00, 0.00) -> (1.00, 0.00) length 2.00 m heading 0.00 deg (+X)
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert out["crossings"] == []
    assert out["fence"].openings["east"] == []


def test_an_arrow_entering_near_a_corner_still_picks_one_side(scene) -> None:
    """Picking the wrong side puts the gap round the corner from the belt."""
    text = (
        HEADER
        + """
rect  "cell" centre (0.00, 0.00) 6.00 x 6.00 m
arrow "belt" (4.00, 2.90) -> (0.00, 2.90) length 4.00 m heading 180.00 deg (-X)
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert len(out["crossings"]) == 1
    assert out["crossings"][0]["side"] == "east"


@pytest.mark.parametrize(
    "start,side",
    [
        ((6.0, 0.0), "east"),
        ((-6.0, 0.0), "west"),
        ((0.0, 6.0), "north"),
        ((0.0, -6.0), "south"),
    ],
)
def test_a_feed_from_each_direction_opens_the_right_side(scene, start, side) -> None:
    text = HEADER + (
        '\nrect  "cell" centre (0.00, 0.00) 6.00 x 6.00 m'
        '\narrow "belt" (%.2f, %.2f) -> (0.00, 0.00) length 6.00 m heading 0 deg\n' % start
    )
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


# ── Where the gate goes ──────────────────────────────────────────────────────


def test_the_gate_opens_nearest_the_operator(scene) -> None:
    """The bug this exists for: the gate always opened south, whatever was
    drawn. It only ever matched an operator drawn south of the cell by
    coincidence - the same default would have fired with nobody there."""
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "operator" centre (5.00, 0.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert out["gate"]["side"] == "east"
    assert out["fence"].openings["east"]
    assert out["fence"].openings["south"] == []


@pytest.mark.parametrize(
    "point,side",
    [
        ((0.0, 5.0), "north"),
        ((0.0, -5.0), "south"),
        ((5.0, 0.0), "east"),
        ((-5.0, 0.0), "west"),
    ],
)
def test_the_gate_follows_the_operator_to_every_side(scene, point, side) -> None:
    text = HEADER + (
        '\nrect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m\ncircle "worker" centre (%.2f, %.2f) radius 0.60 m\n' % point
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert out["gate"]["side"] == side


def test_no_operator_falls_back_to_south_and_says_so(scene) -> None:
    out = S.fence_from_sketch(CELL, scene=scene)

    assert out["gate"]["side"] == "south"
    assert "no operator" in out["gate"]["chosen_by"]


def test_an_explicit_gate_wins_over_a_drawn_operator(scene) -> None:
    """A caller that says where the gate goes should never be second-guessed."""
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "operator" centre (5.00, 0.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene, gate="north")

    assert out["gate"]["side"] == "north"
    assert out["gate"]["chosen_by"] == "explicit"


def test_explicit_none_still_means_no_gate_at_all(scene) -> None:
    """None is a real answer, not "unset" - it must not be reinterpreted."""
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "operator" centre (5.00, 0.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene, gate=None)

    assert out["fence"].openings == {"north": [], "south": [], "east": [], "west": []}


def test_the_operator_used_for_the_gate_is_not_reported_as_ignored(scene) -> None:
    """It was used. Calling it ignored would be a second, quieter lie."""
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "operator" centre (5.00, 0.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert "operator" not in out["ignored"]["circles"]


def test_a_circle_that_is_not_an_operator_does_not_move_the_gate(scene) -> None:
    """A pallet drawn east of the cell is not a person and must not steer it."""
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "pallet" centre (5.00, 0.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert out["gate"]["side"] == "south"
    assert "pallet" in out["ignored"]["circles"]


def test_the_first_operator_wins_when_more_than_one_is_drawn(scene) -> None:
    text = (
        HEADER
        + """
rect   "cell" centre (0.00, 0.00) 6.00 x 6.00 m
circle "operator 1" centre (5.00, 0.00) radius 0.60 m
circle "operator 2" centre (0.00, 5.00) radius 0.60 m
"""
    )
    out = S.fence_from_sketch(text, scene=scene)

    assert out["gate"]["side"] == "east"


# ── Nearest-side, on its own ─────────────────────────────────────────────────


def test_nearest_side_does_not_tie_off_a_corner(scene) -> None:
    """The exact failure `_crosses` had before it solved real intersections:
    a point square off one side of a square footprint is equidistant from
    the infinite lines of both neighbours, and the wrong one used to win by
    list order.

    (5, 1) sits east of a 6x6 square, and its y (1) is within the east edge's
    own span (-3..3) - so the true nearest point on the east edge is a
    perpendicular drop, distance 2. The north edge's nearest point clamps to
    the shared corner (3, 3), distance ~2.83. East is not a tie here; a
    version that measured distance to the infinite lines instead would call
    it one, because both lines pass equally near the corner.
    """
    side = S._nearest_side((0.0, 0.0), (6.0, 6.0), (5.0, 1.0))
    assert side == "east"

    side = S._nearest_side((0.0, 0.0), (6.0, 6.0), (1.0, 5.0))
    assert side == "north"


def test_nearest_side_ties_are_a_real_ambiguity_not_an_artefact(scene) -> None:
    """Exactly off a corner, on the diagonal, the two neighbouring sides are
    genuinely equidistant. This is not the bug the fix above addresses - it
    is what an honest tie looks like once the artefact one is gone."""
    side = S._nearest_side((0.0, 0.0), (6.0, 6.0), (5.0, 5.0))
    assert side in ("north", "east")


def test_a_drawn_path_becomes_drive_waypoints():
    """A `path` polyline is a route a mobile robot follows: first point the
    start, last the goal, the bends the waypoints that keep it off the racks."""
    text = (
        "[LAYOUT SKETCH]\n"
        'path "route" (-2.0, -2.0) -> (2.0, -2.0) -> (2.0, 1.5) -> (5.5, 1.5)\n'
    )
    route = S.route_from_sketch(text)
    assert route["start"] == [-2.0, -2.0]
    assert route["waypoints"] == [[2.0, -2.0], [2.0, 1.5]]
    assert route["goal"] == [5.5, 1.5]
    assert route["chosen_by"] == "label"


def test_a_labelled_arrow_is_a_two_point_route():
    """A single straight run needs no polyline; a labelled arrow is a route."""
    text = '[LAYOUT SKETCH]\narrow "drive" (0.0, 0.0) -> (4.0, 0.0)\n'
    route = S.route_from_sketch(text)
    assert route["start"] == [0.0, 0.0]
    assert route["goal"] == [4.0, 0.0]
    assert route["waypoints"] == []


def test_a_start_circle_overrides_where_the_route_begins():
    text = (
        "[LAYOUT SKETCH]\n"
        'path "route" (0.0, 0.0) -> (3.0, 0.0)\n'
        'circle "start" centre (-1.0, -1.0) radius 0.4 m\n'
    )
    route = S.route_from_sketch(text)
    assert route["start"] == [-1.0, -1.0]


def test_a_sketch_with_no_route_says_so():
    with pytest.raises(SketchError):
        S.route_from_sketch('[LAYOUT SKETCH]\ncircle "pallet" centre (0,0) radius 0.5 m\n')


# ── Writing the sketch back ──────────────────────────────────────────────────

FULL = (
    '[LAYOUT SKETCH]\n'
    'rect   "cell" centre (0.00, 0.00) 6.50 x 6.50 m (x -3.25..3.25, y -3.25..3.25)\n'
    'arrow  "infeed" (4.00, -0.40) -> (-1.00, -0.40) length 5.00 m heading 180.00 deg (-X)\n'
    'circle "pallet" centre (0.00, 0.75) radius 0.60 m\n'
    'path   "route" (-2.00, -2.00) -> (2.00, -2.00) -> (2.00, 1.50)\n'
)


def test_render_round_trips_through_parse():
    """What render emits, parse reads back identically - and a second render
    of that is byte-identical, so an edited sketch never drifts on re-emit."""
    shapes = S.parse_sketch(FULL)
    text = S.render_sketch(shapes)
    again = S.parse_sketch(text)
    assert again == shapes
    assert S.render_sketch(again) == text


def test_render_emits_the_dashboard_format_exactly():
    """Byte-compatible with dashboard/src/lib/sketch.ts describeSketch, so the
    dashboard can redraw an agent-edited sketch as if it drew it."""
    text = S.render_sketch(S.parse_sketch(FULL))
    assert 'rect   "cell" centre (0.00, 0.00) 6.50 x 6.50 m (x -3.25..3.25, y -3.25..3.25)' in text
    assert 'arrow  "infeed" (4.00, -0.40) -> (-1.00, -0.40) length 5.00 m heading 180.00 deg (-X)' in text
    assert 'circle "pallet" centre (0.00, 0.75) radius 0.60 m' in text
    assert text.startswith("[LAYOUT SKETCH - plan view of the floor")


def test_edit_adds_moves_resizes_relabels_and_removes():
    out = S.edit_sketch(
        FULL,
        [
            {"op": "add", "kind": "circle", "label": "operator", "centre": [0, -3.1], "radius": 0.5},
            {"op": "move", "label": "pallet", "centre": [1.0, 2.0]},
            {"op": "resize", "label": "cell", "size": [8.0, 6.0]},
            {"op": "relabel", "label": "infeed", "to": "belt"},
            {"op": "remove", "label": "route"},
        ],
    )
    shapes = S.parse_sketch(out)
    labels = {s["label"] for kind in shapes.values() for s in kind}
    assert labels == {"cell", "belt", "pallet", "operator"}
    assert shapes["circles"][0]["centre"] == (1.0, 2.0)  # pallet moved
    assert shapes["rects"][0]["size"] == (8.0, 6.0)  # cell resized
    assert shapes["paths"] == []  # route removed


def test_moving_a_path_translates_every_point():
    out = S.edit_sketch(FULL, [{"op": "move", "label": "route", "centre": [10.0, 10.0]}])
    pts = np.asarray(S.parse_sketch(out)["paths"][0]["points"])
    # The renderer rounds to 2 decimals (the dashboard's format), so a mean
    # of thirds lands within a centimetre of the target, not on it exactly.
    assert np.allclose(pts.mean(axis=0), [10.0, 10.0], atol=0.02)
    # Shape preserved: the same L, just translated.
    assert np.allclose(pts[1] - pts[0], [4.0, 0.0], atol=0.02)


def test_editing_a_missing_label_refuses_rather_than_duplicating():
    """'Move the pallet' on a sketch with no pallet must not quietly add one."""
    with pytest.raises(SketchError):
        S.edit_sketch(FULL, [{"op": "move", "label": "forklift", "centre": [0, 0]}])


def test_an_empty_sketch_can_be_started_from_nothing():
    out = S.edit_sketch("", [{"op": "add", "kind": "rect", "label": "cell", "centre": [0, 0], "size": [4, 4]}])
    assert len(S.parse_sketch(out)["rects"]) == 1


# ── A line somebody drew is never dropped in silence ────────────────────────
#
# Found by acting as the agent: a hand-written `arrow "infeed" from (3.5, 0)
# to (0.5, 0)` parsed as nothing, so the fence had no opening for the belt,
# zones reported no flow, and an edit re-emitted the block without the arrow.
# Nothing said a word.

HAND_WRITTEN = (
    '[LAYOUT SKETCH]\n'
    'rect   "cell" centre (0.0, 0.0) 4.0 x 3.0 m\n'
    'arrow  "infeed" from (3.5, 0.0) to (0.5, 0.0)\n'
    'circle "operator" centre (0.0, 2.6) radius 0.3 m\n'
)


def test_an_arrow_written_from_to_is_read_like_the_dashboards_form():
    shapes = S.parse_sketch(HAND_WRITTEN)
    assert shapes["unparsed"] == []
    (arrow,) = shapes["arrows"]
    assert arrow["from"] == (3.5, 0.0) and arrow["to"] == (0.5, 0.0)
    # And it does what an infeed arrow is for: an opening in the fence line.
    zones = S.zones_from_sketch(HAND_WRITTEN)
    assert zones["flows"][0]["label"] == "infeed"


def test_a_shape_line_nothing_could_read_is_reported_not_skipped():
    text = HAND_WRITTEN + 'arrow "outfeed" starts (0, 0) ends (4, 0)\ncircle pallet at (1, 1)\n'
    shapes = S.parse_sketch(text)
    assert shapes["unparsed"] == [
        'arrow "outfeed" starts (0, 0) ends (4, 0)',
        "circle pallet at (1, 1)",
    ]
    assert S.zones_from_sketch(text)["unparsed"] == shapes["unparsed"]
    # The prose header is still free to say "rect" mid-sentence.
    assert S.parse_sketch("[LAYOUT SKETCH - a rect is a footprint]\n" + HAND_WRITTEN)["unparsed"] == []


def test_editing_refuses_rather_than_re_emit_a_sketch_missing_a_drawn_line():
    text = HAND_WRITTEN + 'circle pallet at (1, 1)\n'
    with pytest.raises(SketchError, match="circle pallet at"):
        S.edit_sketch(text, [{"op": "move", "label": "operator", "centre": [0, -2.6]}])
    # The same edit on the readable sketch goes through, and nothing is lost.
    out = S.edit_sketch(HAND_WRITTEN, [{"op": "move", "label": "operator", "centre": [0, -2.6]}])
    assert {s["label"] for k in ("rects", "arrows", "circles") for s in S.parse_sketch(out)[k]} == {
        "cell",
        "infeed",
        "operator",
    }
