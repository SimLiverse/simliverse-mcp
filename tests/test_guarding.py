"""Perimeter guarding: the fence line, its openings, and what it encloses.

The failures worth catching here are layout failures, and layout failures are
quiet. A panel authored across the conveyor stops every carton and is
invisible from the camera angle anyone would take. A gate that runs off the
end of a side leaves a corner open rather than a doorway. A fence sized to the
robot's reach rather than past it puts guarding inside the working envelope,
where the arm hits it.

None of those raise. All of them are arithmetic, and all of them are cheaper
to find here than in a render.
"""
from __future__ import annotations

import numpy as np
import pytest

from simliverse_sim import guarding as G
from simliverse_sim.guarding import GuardingError, SafetyFence


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

    def bounds(self):
        """World min/max, whatever primitive this is.

        Cubes carry a scale; cylinders and spheres carry radius and size, so a
        harness that only understands `scale` cannot measure a person.
        """
        kw = self.kwargs
        if "scale" in kw:
            half = np.asarray(kw["scale"], dtype=float)
        elif kw.get("radius") is not None and kw.get("size") is not None:
            r = float(kw["radius"])
            half = np.array([r, r, float(kw["size"]) / 2.0])
        elif kw.get("radius") is not None:
            r = float(kw["radius"])
            half = np.array([r, r, r])
        else:
            side = float(kw.get("size", 0.0)) / 2.0
            half = np.array([side, side, side])
        return self.position - half, self.position + half


class _FakeScene:
    """Records spawns instead of making prims."""

    def __init__(self):
        self.spawned: list[_Spawned] = []

    def spawn_rigid(self, prim_path, **kwargs):
        made = _Spawned(prim_path, **kwargs)
        self.spawned.append(made)
        return made

    def by_prefix(self, text):
        return [s for s in self.spawned if text in s.prim_path]


@pytest.fixture
def scene():
    return _FakeScene()


# ── The fence line ───────────────────────────────────────────────────────────


def test_a_fence_is_specified_by_the_area_it_guards(scene) -> None:
    """Not by a centre and a guess. The recurring mistake in this codebase."""
    fence = SafetyFence.build(size=(6.0, 4.0), centre=(1.0, 2.0), scene=scene)

    assert fence.contains((1.0, 2.0))
    assert fence.contains((3.9, 3.9))
    assert not fence.contains((4.1, 2.0)), "just outside the east line"
    assert not fence.contains((1.0, 4.1)), "just outside the north line"


def test_panels_and_posts_are_actually_authored(scene) -> None:
    fence = SafetyFence.build(size=(4.0, 4.0), scene=scene)
    assert fence.panels, "a fence with no panels is a floor plan"
    assert fence.posts, "guarding is bolted down at posts"
    assert len(scene.spawned) == len(fence.panels) + len(fence.posts)


def test_every_panel_stands_on_the_perimeter(scene) -> None:
    """A panel inboard of the line is an obstacle, not guarding."""
    fence = SafetyFence.build(size=(5.0, 3.0), centre=(0.5, -1.0), scene=scene)

    for panel in scene.by_prefix("Fence_"):
        if "Post" in panel.prim_path:
            continue
        gap = fence.clearance(panel.position[:2])
        assert abs(gap) < 1e-6, (
            "%s sits %.3f m from the fence line" % (panel.prim_path, gap))


def test_panels_clear_the_floor_and_stand_full_height(scene) -> None:
    fence = SafetyFence.build(size=(4.0, 4.0), height=2.4, scene=scene)
    panel = next(p for p in scene.spawned if "Post" not in p.prim_path)
    low, high = panel.bounds()
    assert low[2] == pytest.approx(G.PANEL_GROUND_GAP)
    assert high[2] == pytest.approx(G.PANEL_GROUND_GAP + 2.4)


# ── Openings ─────────────────────────────────────────────────────────────────


def test_a_gate_leaves_a_real_gap_in_the_line(scene) -> None:
    fence = SafetyFence.build(size=(4.0, 4.0), gate="south",
                              gate_width=1.2, scene=scene)

    south = [p for p in scene.spawned
             if "South" in p.prim_path and "Post" not in p.prim_path]
    assert south, "the south side still needs panels either side of the gate"
    for panel in south:
        low, high = panel.bounds()
        assert high[0] <= -0.6 + 1e-6 or low[0] >= 0.6 - 1e-6, (
            "%s crosses the gateway" % panel.prim_path)


def test_no_gate_means_a_closed_perimeter(scene) -> None:
    closed = SafetyFence.build(size=(4.0, 4.0), gate=None, scene=scene)
    assert closed.openings["south"] == []


def test_a_conveyor_crossing_is_left_open(scene) -> None:
    """A panel through the belt stops every carton and cannot be seen.

    This is the expensive one: from the usual camera angle the guarding is
    edge-on and the belt looks clear, so the symptom is a conveyor that has
    stopped delivering for no visible reason.
    """
    fence = SafetyFence.build(
        size=(4.0, 4.0), gate="south",
        crossings=[{"side": "east", "centre": -0.4, "width": 0.6}],
        scene=scene)

    assert len(fence.openings["east"]) == 1
    assert list(fence.openings["east"][0]) == pytest.approx([-0.7, -0.1])
    for panel in scene.by_prefix("East"):
        low, high = panel.bounds()
        assert high[1] <= -0.7 + 1e-6 or low[1] >= -0.1 - 1e-6, (
            "%s is authored across the conveyor slot" % panel.prim_path)


def test_a_crossing_and_a_gate_on_the_same_side_both_survive(scene) -> None:
    fence = SafetyFence.build(
        size=(6.0, 4.0), gate="south", gate_width=1.0, gate_offset=-2.0,
        crossings=[{"side": "south", "centre": 2.0, "width": 0.8}],
        scene=scene)

    got = sorted(fence.openings["south"])
    flat = [v for pair in got for v in pair]
    assert flat == pytest.approx([-2.5, -1.5, 1.6, 2.4])
    for panel in scene.by_prefix("South"):
        low, high = panel.bounds()
        for start, end in fence.openings["south"]:
            assert high[0] <= start + 1e-6 or low[0] >= end - 1e-6


def test_posts_stand_at_both_edges_of_every_opening(scene) -> None:
    """Guarding ends at a post. An opening with no post is a torn panel."""
    fence = SafetyFence.build(size=(4.0, 4.0), gate="south", gate_width=1.0,
                              scene=scene)
    posts = {tuple(np.round(p.position[:2], 4))
             for p in scene.by_prefix("Post")}
    assert (-0.5, -2.0) in posts
    assert (0.5, -2.0) in posts
    for corner in ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)):
        assert corner in posts


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_a_gate_running_off_the_end_is_refused(scene) -> None:
    """It would leave a corner unguarded rather than a doorway."""
    with pytest.raises(GuardingError, match="off the south side"):
        SafetyFence.build(size=(4.0, 4.0), gate="south", gate_width=1.0,
                          gate_offset=1.8, scene=scene)


def test_a_gate_too_narrow_to_walk_through_is_refused(scene) -> None:
    with pytest.raises(GuardingError, match="hatch"):
        SafetyFence.build(size=(4.0, 4.0), gate_width=0.4, scene=scene)


def test_an_unknown_side_is_refused(scene) -> None:
    with pytest.raises(GuardingError, match="expected one of"):
        SafetyFence.build(size=(4.0, 4.0), gate="starboard", scene=scene)
    with pytest.raises(GuardingError, match="expected one of"):
        SafetyFence.build(
            size=(4.0, 4.0),
            crossings=[{"side": "up", "centre": 0.0, "width": 0.5}],
            scene=scene)


@pytest.mark.parametrize("size", [(0.0, 4.0), (-1.0, 4.0), (4.0, 0.0)])
def test_a_fence_with_no_area_is_refused(scene, size) -> None:
    with pytest.raises(GuardingError, match="positive extents"):
        SafetyFence.build(size=size, scene=scene)


def test_a_crossing_with_no_width_is_refused(scene) -> None:
    with pytest.raises(GuardingError, match="positive width"):
        SafetyFence.build(
            size=(4.0, 4.0),
            crossings=[{"side": "east", "centre": 0.0, "width": 0.0}],
            scene=scene)


# ── Does the cell actually fit inside it ─────────────────────────────────────


def test_an_arm_that_would_reach_through_the_mesh_is_reported(scene) -> None:
    """Guarding inside the working envelope is a collision waiting to happen."""
    fence = SafetyFence.build(size=(2.0, 2.0), scene=scene)
    verdict = fence.fits((0.0, 0.0), reach=1.3)

    assert verdict["inside"]
    assert verdict["touches_fence"], "a 1.3 m reach crosses a 1.0 m half-width"
    assert verdict["overhang"] == pytest.approx(0.3)


def test_an_arm_comfortably_inside_is_reported_as_such(scene) -> None:
    fence = SafetyFence.build(size=(6.0, 6.0), scene=scene)
    verdict = fence.fits((0.0, 0.0), reach=1.3)

    assert verdict["inside"] and not verdict["touches_fence"]
    assert verdict["clearance"] == pytest.approx(3.0)


def test_an_arm_outside_the_guarding_is_not_quietly_accepted(scene) -> None:
    fence = SafetyFence.build(size=(2.0, 2.0), centre=(5.0, 0.0), scene=scene)
    verdict = fence.fits((0.0, 0.0), reach=0.5)

    assert not verdict["inside"]
    assert verdict["clearance"] < 0.0


def test_margin_asks_the_question_that_matters(scene) -> None:
    """Not 'inside the fence' but 'far enough in to not reach through it'."""
    fence = SafetyFence.build(size=(4.0, 4.0), scene=scene)

    assert fence.contains((1.9, 0.0))
    assert not fence.contains((1.9, 0.0), margin=0.3)


def test_a_margin_larger_than_the_cell_contains_nothing(scene) -> None:
    fence = SafetyFence.build(size=(1.0, 1.0), scene=scene)
    assert not fence.contains((0.0, 0.0), margin=0.6)


# ── Panelling ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("side_length", [1.0, 2.0, 3.5, 6.0, 12.0])
def test_long_runs_are_split_into_panels(scene, side_length) -> None:
    """Guarding is bought in widths; a 12 m slab does not read as a fence."""
    SafetyFence.build(size=(side_length, side_length), gate=None,
                      panel_max=1.5, scene=scene)
    for panel in scene.spawned:
        if "Post" in panel.prim_path:
            continue
        along = max(panel.scale[0], panel.scale[1]) * 2.0
        assert along <= 1.5 + 1e-6, (
            "%s is %.2f m wide" % (panel.prim_path, along))


def test_panels_tile_a_run_without_leaving_a_gap(scene) -> None:
    """A gap between panels is a hole in the guarding that nothing reports."""
    SafetyFence.build(size=(5.0, 4.0), gate=None, panel_max=1.5, scene=scene)

    north = sorted((p for p in scene.by_prefix("North")),
                   key=lambda p: p.position[0])
    edges = [(p.position[0] - p.scale[0], p.position[0] + p.scale[0])
             for p in north]
    assert edges[0][0] == pytest.approx(-2.5)
    assert edges[-1][1] == pytest.approx(2.5)
    for (_, end), (start, _) in zip(edges, edges[1:]):
        assert start == pytest.approx(end), "gap between panels"


# ── Describing it ────────────────────────────────────────────────────────────


def test_describe_round_trips_the_layout(scene) -> None:
    fence = SafetyFence.build(
        size=(6.0, 4.0), centre=(1.0, 0.0), height=2.2, gate="west",
        crossings=[{"side": "east", "centre": 0.0, "width": 0.5}],
        scene=scene)
    described = fence.describe()

    assert described["size"] == [6.0, 4.0]
    assert described["centre"] == [1.0, 0.0]
    assert described["height"] == 2.2
    assert "west" in described["openings"] and "east" in described["openings"]
    assert described["panels"] == len(fence.panels)
    assert described["posts"] == len(fence.posts)


# ── Cell furniture ───────────────────────────────────────────────────────────


def test_a_cabinet_stands_on_the_floor_rather_than_half_in_it(scene) -> None:
    """The commonest way this furniture gets placed wrong."""
    G.spawn_cabinet(position=(2.0, 1.0, 0.0), size=(0.6, 0.5, 1.2),
                    scene=scene)
    low, high = scene.spawned[-1].bounds()

    assert low[2] == pytest.approx(0.0)
    assert high[2] == pytest.approx(1.2)


def test_a_beacon_stands_on_the_floor(scene) -> None:
    G.spawn_beacon(position=(1.0, 1.0, 0.0), height=0.9, scene=scene)
    made = scene.spawned[-1]
    assert made.position[2] == pytest.approx(0.45)


def test_an_operator_platform_sits_on_the_floor_not_in_it(scene) -> None:
    """A platform sunk into the floor renders as a stain."""
    G.spawn_operator_platform(position=(0.0, -3.0, 0.0), thickness=0.06,
                              scene=scene)
    low, high = scene.spawned[-1].bounds()

    assert low[2] == pytest.approx(0.0)
    assert high[2] == pytest.approx(0.06)


def test_furniture_can_be_placed_outside_the_gate(scene) -> None:
    """The layout question the operator platform exists to answer."""
    fence = SafetyFence.build(size=(4.0, 4.0), gate="south", scene=scene)
    station = (0.0, -3.0)

    assert not fence.contains(station), "the operator stands outside"
    assert fence.clearance(station) < 0.0


# ── The pedestal ─────────────────────────────────────────────────────────────


def test_a_pedestal_stands_on_the_floor(scene) -> None:
    G.spawn_pedestal(position=(0.0, 0.0, 0.0), height=0.4, scene=scene)
    low, high = scene.spawned[-1].bounds()

    assert low[2] == pytest.approx(0.0), "a plinth sunk into the floor"
    assert high[2] == pytest.approx(0.4)


@pytest.mark.parametrize("height", [0.15, 0.4, 0.75, 1.2])
def test_the_pedestal_reports_where_the_base_actually_goes(scene, height) -> None:
    """The top is the number the rest of the layout needs.

    An arm spawned anywhere but here intersects the plinth it is supposedly
    bolted to, and the render shows a robot growing out of a crate.
    """
    made = G.spawn_pedestal(position=(0.3, -0.2, 0.0), height=height,
                            scene=scene)
    _, high = scene.spawned[-1].bounds()

    assert made["top"] == pytest.approx(height)
    assert made["top"] == pytest.approx(high[2]), (
        "the reported base height is not the top of the plinth")


def test_a_pedestal_on_a_raised_floor_still_reports_its_own_top(scene) -> None:
    made = G.spawn_pedestal(position=(0.0, 0.0, 0.25), height=0.4, scene=scene)
    assert made["top"] == pytest.approx(0.65)


def test_a_pedestal_with_no_height_is_refused(scene) -> None:
    """Zero height is an arm on the floor with a decal under it."""
    with pytest.raises(GuardingError, match="not a mounting"):
        G.spawn_pedestal(height=0.0, scene=scene)
    with pytest.raises(GuardingError, match="not a mounting"):
        G.spawn_pedestal(height=-0.3, scene=scene)


# ── The operator ─────────────────────────────────────────────────────────────


def test_the_operator_is_a_real_character_not_a_stack_of_primitives() -> None:
    """A cell full of hand-built furniture reads as a mock-up.

    The first version of this was two capsules and a sphere. It was the right
    shape and the wrong answer: the asset library ships 23 people and the
    index simply never scanned `/Isaac/People`.
    """
    from simliverse_sim.props import find_prop

    entry = find_prop(G.DEFAULT_OPERATOR)
    assert entry["path"].startswith("/Isaac/People/Characters/")
    assert entry["category"] == "people"


def test_the_indexed_people_are_person_sized() -> None:
    """Measured on the asset server, not guessed."""
    from simliverse_sim.props import list_props

    people = [e for e in list_props("person") if e["category"] == "people"]
    assert people, "no people in the index"
    for entry in people:
        height = entry["extent"][2]
        assert 1.5 <= height <= 2.1, (
            "%s stands %.2f m" % (entry["key"], height))


def test_a_worker_can_be_found_by_the_words_anyone_would_use() -> None:
    from simliverse_sim.props import find_prop

    for query in ("worker", "operator", "person"):
        entry = find_prop(query)
        assert entry["category"] == "people", (
            "%r found %s instead of a person" % (query, entry["key"]))
