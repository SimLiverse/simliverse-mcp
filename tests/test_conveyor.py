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

"""The belt: its geometry, its queue, and what it refuses to do.

Isaac is not importable here, so everything that touches USD is faked and what
is tested is the arithmetic and the decisions — where a box is laid down, which
box counts as having arrived, and which requests are refused rather than
silently producing a belt that does not convey.

The constants asserted at the bottom were measured from the shipped USD files
(`ConveyorBelt_A01`, `ConveyorBelt_A09`, `pallet`), downloaded and opened with
usd-core. They are in the test because they are the numbers a scene is built
from, and a silent change to one of them moves every box in every scene.
"""

import numpy as np
import pytest

from simliverse_sim import conveyor as C
from simliverse_sim.conveyor import Conveyor, ConveyorError


class _FakeBody:
    def __init__(self, prim_path, position, speed=0.0):
        self.prim_path = prim_path
        self.position = np.asarray(position, dtype=float)
        self.speed = speed
        self.velocity_writes = []

    def set_velocity(self, linear=None, angular=None):
        self.velocity_writes.append(
            (None if linear is None else np.asarray(linear, dtype=float),
             None if angular is None else np.asarray(angular, dtype=float)))


class _FakeScene:
    """Records spawns instead of making prims."""

    def __init__(self):
        self.spawned = []

    def spawn_rigid(self, prim_path, **kwargs):
        self.spawned.append({"prim_path": prim_path, **kwargs})
        return _FakeBody(prim_path, kwargs.get("position", (0, 0, 0)))

    def spawn_box(self, prim_path, **kwargs):
        self.spawned.append({"prim_path": prim_path, "_box": True, **kwargs})
        return _FakeBody(prim_path, kwargs.get("position", (0, 0, 0)))

    def apply_physics_material(self, *args, **kwargs):
        return "/World/PhysicsMaterials/fake"


@pytest.fixture
def belt(monkeypatch):
    """A 4 m belt along +X, decked at 1.0 m, without touching USD."""
    monkeypatch.setattr(C, "_body_of", lambda path: path)
    monkeypatch.setattr(C, "drive_surface", lambda *a, **k: {"enabled": True})
    made = Conveyor(
        "/World/Conveyor",
        direction=(1, 0, 0),
        speed=0.3,
        top_z=1.0,
        length=4.0,
        width=0.9,
        gate_path="/World/ConveyorGate",
        scene=_FakeScene(),
    )
    made._origin = np.array([0.0, 0.0, 1.0])
    return made


# ── Direction ────────────────────────────────────────────────────────────────


def test_direction_is_normalised_and_flattened() -> None:
    assert C._unit((5, 0, 3)).tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_a_belt_with_no_horizontal_direction_is_refused() -> None:
    """Straight up is not a direction of travel."""
    with pytest.raises(ConveyorError, match="horizontal"):
        C._unit((0, 0, 1))


# ── Laying boxes on the belt ─────────────────────────────────────────────────


def test_load_places_boxes_on_the_deck_not_through_it(belt) -> None:
    """A box rests on its bottom face: centre is half a height above the deck."""
    box = (0.18, 0.13, 0.11)
    belt.load(3, box=box)
    for spawn in belt.scene.spawned:
        assert spawn["position"][2] == pytest.approx(1.0 + 0.11 / 2, abs=0.005)


def test_load_passes_the_full_box_size_and_never_a_scale(belt) -> None:
    """Cartons are authored at size, not scaled.

    This is a regression test for a silent pick failure, not a style
    preference. Isaac's surface gripper detects a grip by raycasting, and its
    own sample warns that the raycast does not reliably hit a *scaled* box
    collider. The cup arrives, closes, attaches to nothing, and lifts away
    empty while every pose reads correct — so the symptom points at the
    gripper, the pose, or the tolerances, and never at the box.

    The old call scaled a size-2 `UsdGeom.Cube` by half the requested size,
    which is also why the half-extent factor of two used to be a live hazard.
    Authoring at real size removes both problems at once.
    """
    belt.load(1, box=(0.18, 0.13, 0.11))
    spawn = belt.scene.spawned[0]
    assert spawn["_box"], "cartons must come from spawn_box, not spawn_rigid"
    assert spawn["size"] == pytest.approx([0.18, 0.13, 0.11])
    assert "scale" not in spawn


def test_boxes_are_queued_back_from_the_gate_in_arrival_order(belt) -> None:
    belt.load(3, box=(0.2, 0.13, 0.11), spacing=0.5, start_offset=0.25)
    xs = [s["position"][0] for s in belt.scene.spawned]
    # Box 0 is nearest the far end and arrives first; the rest trail behind it.
    assert xs == sorted(xs, reverse=True)
    assert xs[0] == pytest.approx(4.0 / 2 - 0.25)
    assert xs[0] - xs[1] == pytest.approx(0.5)


def test_box_at_gate_ignores_a_carton_that_fell_off_the_belt(belt) -> None:
    """Displacement along the belt says nothing about the other two axes.

    Measured: a carton knocked onto the floor 0.68 m to the side and 0.45 m
    below the deck still had an `along` inside the arrival tolerance, so it was
    returned as the box at the gate. The arm was sent to fetch it 1.08 m away
    and `pose_to` reported the target as outside the workspace - which is true,
    and names the arm rather than the belt that handed it a fallen box.
    """
    belt.load(2, box=(0.15, 0.15, 0.15))
    good, fallen = belt.boxes

    # Both are level with the stop along the direction of travel.
    at_gate = belt.length / 2.0 - 0.075
    good.position = np.array([at_gate, 0.0, 1.0 + 0.075])
    good.speed = 0.0
    fallen.position = np.array([at_gate, -1.10, 0.075])
    fallen.speed = 0.0

    assert belt.box_at_gate() is good

    # And with only the fallen one, nothing has arrived.
    belt.track([fallen])
    assert belt.box_at_gate() is None


def test_boxes_are_spawned_beside_the_belt_never_beneath_it(belt) -> None:
    """Parented under a kinematic belt, a box rides it instead of resting on it."""
    belt.load(2, box=(0.18, 0.13, 0.11))
    for spawn in belt.scene.spawned:
        assert not spawn["prim_path"].startswith(belt.belt_path + "/")


def test_a_box_longer_than_the_belt_is_refused(belt) -> None:
    with pytest.raises(ConveyorError, match="does not fit"):
        belt.load(1, box=(5.0, 0.13, 0.11))


def test_a_zero_sized_box_is_refused(belt) -> None:
    with pytest.raises(ConveyorError, match="positive"):
        belt.load(1, box=(0.18, 0.0, 0.11))


# ── The queue at the stop ────────────────────────────────────────────────────


def test_no_box_has_arrived_before_anything_moves(belt) -> None:
    assert belt.box_at_gate() is None
    assert belt.arrived() is False


def test_the_box_resting_against_the_stop_is_the_one_returned(belt) -> None:
    far = 2.0  # length / 2
    belt._boxes = [
        _FakeBody("/World/Box0", [far - 0.05, 0, 1.05], speed=0.0),
        _FakeBody("/World/Box1", [far - 0.90, 0, 1.05], speed=0.3),
    ]
    picked = belt.box_at_gate()
    assert picked is not None and picked.prim_path == "/World/Box0"


def test_an_arrived_box_is_found_though_its_centre_is_half_a_box_short(belt) -> None:
    """The live failure this fix comes from.

    Measured on the worker: a 3.2 m belt carrying 30 cm boxes brought the lead
    box to rest with its centre 0.15 m from the stop — exactly half a box, since
    it rests on its face and `position` is a centre. Measuring that gap against
    zero rejected it, so the belt conveyed perfectly and the arm was never told
    a box had arrived. The expected gap has to come from the box size.
    """
    belt.load(1, box=(0.30, 0.30, 0.30))          # sets box_size
    far = 2.0                                      # length / 2
    belt._boxes = [
        _FakeBody("/World/Box0", [far - 0.15, 0, 1.15], speed=0.0),
    ]
    picked = belt.box_at_gate()
    assert picked is not None and picked.prim_path == "/World/Box0"


def test_a_box_a_whole_box_further_back_has_not_arrived(belt) -> None:
    """The one queued behind it must not be mistaken for the one at the stop."""
    belt.load(1, box=(0.30, 0.30, 0.30))
    far = 2.0
    belt._boxes = [
        _FakeBody("/World/Box1", [far - 0.45, 0, 1.15], speed=0.0),
    ]
    assert belt.box_at_gate() is None


def test_a_box_still_being_pushed_is_not_ready_to_pick(belt) -> None:
    """Closing on a box that is still moving fails as if the gripper were bad."""
    far = 2.0
    belt._boxes = [_FakeBody("/World/Box0", [far - 0.05, 0, 1.05], speed=0.4)]
    assert belt.box_at_gate() is None
    assert belt.box_at_gate(max_speed=0.6) is not None


def test_a_box_still_far_up_the_belt_is_not_at_the_gate(belt) -> None:
    belt._boxes = [_FakeBody("/World/Box0", [0.0, 0, 1.05], speed=0.0)]
    assert belt.box_at_gate() is None


def test_the_nearest_settled_box_wins_when_several_have_stacked_up(belt) -> None:
    far = 2.0
    belt._boxes = [
        _FakeBody("/World/Box1", [far - 0.10, 0, 1.05], speed=0.0),
        _FakeBody("/World/Box0", [far - 0.02, 0, 1.05], speed=0.0),
    ]
    assert belt.box_at_gate().prim_path == "/World/Box0"


def test_an_unreadable_box_does_not_break_the_query(belt) -> None:
    class Dead:
        prim_path = "/World/Dead"

        @property
        def position(self):
            raise RuntimeError("stale handle")

        speed = 0.0

    far = 2.0
    belt._boxes = [Dead(), _FakeBody("/World/Box0", [far - 0.03, 0, 1.05], speed=0.0)]
    assert belt.box_at_gate().prim_path == "/World/Box0"


def test_a_belt_running_along_y_measures_arrival_along_y(monkeypatch) -> None:
    """Nothing may assume +X. The dot product against `direction` is the point."""
    monkeypatch.setattr(C, "_body_of", lambda path: path)
    monkeypatch.setattr(C, "drive_surface", lambda *a, **k: {"enabled": True})
    belt = Conveyor(
        "/World/Conveyor", direction=(0, 1, 0), speed=0.3, top_z=1.0,
        length=4.0, width=0.9, scene=_FakeScene(),
    )
    belt._origin = np.array([0.0, 0.0, 1.0])
    belt._boxes = [_FakeBody("/World/Box0", [0.0, 1.96, 1.05], speed=0.0)]
    assert belt.box_at_gate().prim_path == "/World/Box0"


# ── Reporting ────────────────────────────────────────────────────────────────


def test_describe_names_the_mechanism_it_actually_uses(belt) -> None:
    """An agent reading this must be able to tell a driven belt from scenery."""
    described = belt.describe()
    assert described["mechanism"] == "PhysxSurfaceVelocityAPI"
    assert described["direction"] == pytest.approx([1.0, 0.0, 0.0])
    assert described["gate_path"] == "/World/ConveyorGate"


# ── Measured asset constants ─────────────────────────────────────────────────


def test_the_deck_is_not_the_top_of_the_prop() -> None:
    """The 0.53 m that `_belt_surface` exists to avoid.

    Measured on ConveyorBelt_A09: the belt surface is at 1.781 m and the prop's
    bounding box tops out at 2.311 m on the gantry above it. Spawning boxes at
    the bbox top drops them half a metre onto the belt.
    """
    assert C.CONVEYOR_DECK_Z == pytest.approx(1.781)
    assert 2.311 - C.CONVEYOR_DECK_Z == pytest.approx(0.53, abs=0.001)


def test_the_belt_is_forty_millimetres_thick() -> None:
    assert C.CONVEYOR_BELT_THICKNESS == pytest.approx(0.040)


def test_the_pallet_decks_where_the_pattern_expects() -> None:
    assert C.PALLET_DECK_Z == pytest.approx(0.1425)
    assert C.PALLET_DECK_SIZE[0] == pytest.approx(1.2132)
    assert C.PALLET_DECK_SIZE[1] == pytest.approx(0.8023)


def test_belt_surface_prefers_a_child_named_belt(monkeypatch) -> None:
    """Both shipped variants measured name it `Belt`, and both put the body there."""

    class _Prim:
        def __init__(self, valid):
            self._valid = valid

        def IsValid(self):
            return self._valid

    class _Stage:
        def GetPrimAtPath(self, path):
            return _Prim(path in ("/World/Conveyor", "/World/Conveyor/Belt"))

    monkeypatch.setattr(C, "get_stage", lambda: _Stage())
    assert C._belt_surface("/World/Conveyor") == "/World/Conveyor/Belt"


def test_a_box_still_short_of_the_stop_is_not_reported_as_arrived(belt) -> None:
    """The corner-grab bug, from the live sim.

    A fixed 0.12 m tolerance accepted a 15 cm box while it was still 6.5 cm
    short of the stop and creeping. The pick then descended onto where the box
    had been, clipped its top edge, shoved it 3.4 cm, and the cup latched on a
    corner with the carton hanging off it. The tolerance has to scale with the
    box: a quarter of one, not a fixed distance.
    """
    belt.load(1, box=(0.15, 0.15, 0.15))
    far = 2.0
    # Arrived: centre half a box short of the stop.
    belt._boxes = [_FakeBody("/World/Box0", [far - 0.075, 0, 1.15], speed=0.0)]
    assert belt.box_at_gate() is not None

    # Still 6.5 cm out, which the old fixed tolerance accepted.
    belt._boxes = [_FakeBody("/World/Box0", [far - 0.14, 0, 1.15], speed=0.0)]
    assert belt.box_at_gate() is None


# ── Waking a sleeping load ───────────────────────────────────────────────────


def test_starting_a_belt_wakes_the_cartons_asleep_on_it(belt) -> None:
    """The failure this exists for cost a run, and looked like nothing at all.

    PhysX does not wake a body because the surface beneath it started moving;
    the docs say so, and NVIDIA's own conveyor node cycles its enable flag to
    work around it. A palletising cell halts the belt for a whole pick-and-
    place cycle - long enough for every carton to sleep - so on restart the
    belt reported `surfaceVelocityEnabled True` and `(0.2, 0, 0)` while three
    cartons sat at v=0.0 and the controller timed out waiting for one to
    arrive. Every observable said the belt was running.
    """
    cartons = [_FakeBody("/World/Box%d" % i, (i * 0.2, 0.0, 1.075))
               for i in range(3)]
    belt.track(cartons)

    belt.start()

    for carton in cartons:
        assert carton.velocity_writes, (
            "%s was never nudged, so PhysX will leave it asleep on a "
            "running belt" % carton.prim_path)
        linear, _ = carton.velocity_writes[-1]
        assert linear[0] > 0.0, "the nudge must push along the belt"
        assert linear[1] == pytest.approx(0.0)
        assert linear[2] == pytest.approx(0.0)


def test_the_wake_nudge_is_gentler_than_a_fast_belt(monkeypatch) -> None:
    """A 2 m/s belt must not launch its load to wake it."""
    monkeypatch.setattr(C, "_body_of", lambda path: path)
    monkeypatch.setattr(C, "drive_surface", lambda *a, **k: {"enabled": True})
    fast = Conveyor(
        "/World/Fast", direction=(1, 0, 0), speed=2.0, top_z=1.0,
        length=4.0, width=0.9, scene=_FakeScene(),
    )
    carton = _FakeBody("/World/Box0", (0.0, 0.0, 1.075))
    fast.track([carton])

    fast.start()

    linear, _ = carton.velocity_writes[-1]
    assert float(np.linalg.norm(linear)) <= C._WAKE_SPEED


def test_waking_survives_a_body_that_cannot_take_a_velocity(belt) -> None:
    """One unwritable carton must not stop the rest of the belt restarting."""

    class _Deaf(_FakeBody):
        def set_velocity(self, linear=None, angular=None):
            raise RuntimeError("mid-teleport")

    deaf = _Deaf("/World/Box0", (0.0, 0.0, 1.075))
    good = _FakeBody("/World/Box1", (0.2, 0.0, 1.075))
    belt.track([deaf, good])

    assert belt.wake_load() == 1
    assert good.velocity_writes


# ── Dressing: a real conveyor prop over the physics slab ────────────────────


class _FakeSpawnProp:
    """Records every call `dress()` makes, without touching the asset server."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, prop, *, prim_path, position, orientation=None, scene=None):
        self.calls.append({
            "prop": prop, "prim_path": prim_path,
            "position": np.asarray(position, dtype=float),
            "orientation": (None if orientation is None
                            else np.asarray(orientation, dtype=float)),
        })
        return {"key": prop, "extent": [2.0, 1.15, 1.166]}


@pytest.fixture
def dressable(monkeypatch, belt):
    """The `belt` fixture plus a spawn_prop double `dress()` can be checked
    against, and its own `_hide` short-circuited so hiding needs no stage."""
    fake = _FakeSpawnProp()
    monkeypatch.setattr(C, "spawn_prop", fake, raising=False)
    monkeypatch.setattr("simliverse_sim.props.spawn_prop", fake)
    monkeypatch.setattr(belt, "_hide", lambda path: True)
    return belt, fake


def test_dressing_is_rotated_to_the_belts_own_heading(dressable) -> None:
    """The bug this exists for: a dressing prop always faced world +X.

    `belt` (the fixture) runs along +X, so that bug was invisible on it -
    which is exactly how it shipped. A belt built running -X got a dressing
    section facing +X anyway, rendering off the far end of its own physics
    slab rather than on top of it.
    """
    belt, fake = dressable
    belt.direction = np.array([-1.0, 0.0, 0.0])

    belt.dress("conveyorbelt_a05")

    assert fake.calls, "dress() made no spawn_prop calls"
    for call in fake.calls:
        assert call["orientation"] is not None, (
            "a dressing prop with no orientation always faces world +X")
        assert call["orientation"][2] == pytest.approx(180.0)


def test_a_belt_along_plus_x_needs_no_rotation(dressable) -> None:
    belt, fake = dressable
    belt.dress("conveyorbelt_a05")
    assert fake.calls[0]["orientation"][2] == pytest.approx(0.0)


@pytest.mark.parametrize("heading,yaw", [
    ((1.0, 0.0, 0.0), 0.0),
    ((-1.0, 0.0, 0.0), 180.0),
    ((0.0, 1.0, 0.0), 90.0),
    ((0.0, -1.0, 0.0), -90.0),
])
def test_the_yaw_matches_the_belts_direction(dressable, heading, yaw) -> None:
    belt, fake = dressable
    belt.direction = np.asarray(heading, dtype=float)

    belt.dress("conveyorbelt_a05")

    assert fake.calls[0]["orientation"][2] == pytest.approx(yaw)


def test_a_belt_longer_than_one_section_gets_more_than_one(dressable) -> None:
    """One 2 m section over a longer belt used to leave the rest bare."""
    belt, fake = dressable
    belt.length = 6.4

    result = belt.dress("conveyorbelt_a05", section_length=2.0)

    assert result["sections"] == 4, "ceil(6.4 / 2.0) sections should be placed"
    assert len(fake.calls) == 4
    assert len(set(c["prim_path"] for c in fake.calls)) == 4, (
        "sections must not collide on the same prim path")


def test_a_belt_shorter_than_one_section_still_gets_one(dressable) -> None:
    belt, fake = dressable
    belt.length = 0.8

    result = belt.dress("conveyorbelt_a05", section_length=2.0)

    assert result["sections"] == 1
    assert len(fake.calls) == 1


def test_tiled_sections_walk_the_belts_full_length(dressable) -> None:
    """Placed edge to edge along the belt, not stacked on top of each other."""
    belt, fake = dressable
    belt.length = 6.4
    belt._origin = np.array([3.2, 0.0, 1.0])   # centre, belt spans 0.0..6.4

    belt.dress("conveyorbelt_a05", section_length=2.0)

    xs = sorted(c["position"][0] for c in fake.calls)
    assert xs[0] == pytest.approx(0.0, abs=1e-6)
    for a, b in zip(xs, xs[1:]):
        assert b - a == pytest.approx(2.0), (
            "sections must be spaced one section-length apart, not overlapping")


def test_the_slab_is_hidden_after_dressing(dressable) -> None:
    belt, fake = dressable
    hidden = []
    belt._hide = lambda path: hidden.append(path) or True

    belt.dress("conveyorbelt_a05")

    assert hidden == [belt.belt_path]


def test_the_result_reports_every_section_placed(dressable) -> None:
    belt, fake = dressable
    belt.length = 5.0

    result = belt.dress("conveyorbelt_a05", section_length=2.0)

    assert result["sections"] == 3
    assert len(result["prim_paths"]) == 3
    assert result["prop"] == "conveyorbelt_a05"
    assert result["deck"] == pytest.approx(C.DRESSING_DECK)


def test_a_single_section_keeps_the_undecorated_prim_path(dressable) -> None:
    """One section keeps `<belt>_Dressing`, matching what shipped before -
    a caller storing that exact path for a short belt should not break."""
    belt, fake = dressable
    belt.length = 1.0

    result = belt.dress("conveyorbelt_a05", section_length=2.0)

    assert result["prim_paths"] == [f"{belt.belt_path}_Dressing"]


# ── Dressing must not become the surface a carton actually rests on ─────────


def test_dressing_strips_the_physics_the_prop_shipped_with(monkeypatch, dressable) -> None:
    """The bug this exists for: `ConveyorBelt_A05` is indexed `physics:
    "dynamic"`. Its `Rollers` prim is a real, ungrounded rigid body dropped
    in the same space as the invisible physics slab, and a carton rests on
    it - not the slab - so driving the slab's surface velocity moves
    nothing. `enabled=True` at the right speed, and every carton sat still.
    """
    belt, fake = dressable
    stripped = []
    monkeypatch.setattr(C, "_strip_physics",
                        lambda scene, path: stripped.append(path) or 1)

    belt.dress("conveyorbelt_a05")

    assert stripped == [c["prim_path"] for c in fake.calls], (
        "every dressed section must have its physics stripped, in the "
        "order it was spawned")


def test_a_width_mismatch_against_the_belt_is_reported(dressable) -> None:
    """`ConveyorBelt_A05` dresses 1.15 m wide regardless of the belt's own
    configured width, which is exactly what put the visible conveyor wider
    than a fence crossing sized off the belt."""
    belt, fake = dressable
    belt.width = 0.4

    result = belt.dress("conveyorbelt_a05")

    assert result["width"] == pytest.approx(1.15)


def test_no_mismatch_reported_when_the_belt_is_already_wide_enough(dressable) -> None:
    belt, fake = dressable
    belt.width = 2.0

    result = belt.dress("conveyorbelt_a05")

    assert result["width"] == pytest.approx(1.15)


def test_strip_physics_does_not_raise_off_a_stage_with_no_such_prim() -> None:
    class _NoStage:
        def GetPrimAtPath(self, path):
            return None

    class _Scene:
        stage = _NoStage()

    assert C._strip_physics(_Scene(), "/World/Nowhere") == 0
