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

"""Belts that actually convey.

`spawn_prop` has referenced 50-odd `ConveyorBelt_A*` assets since the prop index
landed, and every one of them is scenery. The mesh has a collider, the boxes sit
on it, and nothing moves — there is no motor in a USD file. An agent asked for
"boxes coming down a conveyor" had three ways forward and all of them were bad:
set each box's velocity every frame from live calls (works until the timeline
stops, then the scene replays as a row of motionless boxes), teleport the boxes
along by writing poses (the thing `not_teleported` exists to catch), or animate
the belt mesh and watch the boxes ignore it. The honest option, saying "this
library cannot move a conveyor", was the one nobody took.

PhysX has had the right mechanism the whole time. `PhysxSurfaceVelocityAPI`
gives a collider a *surface* that moves while the body stays put — exactly what
a belt is. Contacting bodies are dragged by friction, so boxes accelerate up to
belt speed, decelerate against a stop, and pile up behind one another with real
contact forces. Nothing is scripted per frame and nothing is teleported, which
means the whole thing replays from Play with no controller involvement at all.

    from simliverse_sim import Conveyor

    belt = Conveyor.build(
        length=3.0, width=0.8,
        position=[0, 0, 0.55],          # z is the DECK, where a box rests
        speed=0.30, direction=(1, 0, 0),
    )
    boxes = belt.load(4, box=(0.18, 0.13, 0.11), mass=1.5)

    scene.play()
    scene.settle(6.0)
    ready = belt.box_at_gate()      # the one resting against the stop

**The gate is the point.** A belt with no stop delivers boxes to a moving target
and the pick has to be timed. A belt with a stop delivers them to one fixed
pose, over and over, which is what a real palletising cell does and what makes
the task reproducible. `build()` fits one by default.

**Boxes are placed once, not fed in over time.** Spawning prims mid-simulation
from a ScriptNode edits the stage while PhysX is stepping it, and a controller
that does it replays differently every run. `load()` places them along the belt
at authoring time and lets the belt carry them in — deterministic, and the queue
looks the same on every Play.

**GPU dynamics does not have to be turned off, whatever the forums say.**
IsaacLab discussion #3216 is the first thing anyone finds on this, and its
working recipe disables GPU dynamics and runs PhysX on the CPU
(`device="cpu"`, `use_fabric=False`, `enableGPUDynamics=False`). That was true
of the version it was written against and is **not** true of Isaac Sim 6.0.
Measured here, same belt and boxes, clean scene each time:

    enableGPUDynamics=True    boxes moved +1.05 / +1.25 m, arrived at the stop
    enableGPUDynamics=False   boxes moved +1.05 / +1.25 m, arrived at the stop

Identical to the centimetre. The advice is worth knowing about because
following it costs the whole point of a GPU — a cell that trains at CPU
throughput because of a limitation that has since been fixed. What the belt
does need is a **kinematic rigid body**, not a static one: a collider with no
RigidBodyAPI has no surface to move. `build()` and `from_prop()` both ensure
that, which is the part of #3216 that still applies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._compat import as_vec3, get_stage

logger = logging.getLogger(__name__)

#: Belt speeds outside this range are almost always a units mistake.
_SANE_SPEED = (0.0, 5.0)

#: Measured from the shipped USD, not read off a datasheet. `ConveyorBelt_A01`
#: and `ConveyorBelt_A09` agree on all of it, so it is a family convention
#: rather than a property of one asset:
#:
#:   deck top        1.781 m     the belt surface, where a box rests
#:   belt thickness  0.040 m     1.741 -> 1.781
#:   prop bbox top   2.311 m     the gantry. NOT the deck. 0.53 m higher.
#:   A09 belt        3.97 x 0.90 m, straight
#:   A01 belt        1.94 x 2.05 m, a 90-degree curve
#:
#: The gap between the third line and the first is the whole reason
#: `_belt_surface` exists.
CONVEYOR_DECK_Z = 1.781
CONVEYOR_BELT_THICKNESS = 0.040

#: Measured the same way. A standard pallet, decking at 0.1425 m — the number
#: `pallet_slots(origin=...)` wants for its Z, and a static collider with no
#: rigid body, so it stays where it is put.
PALLET_DECK_Z = 0.1425
PALLET_DECK_SIZE = (1.2132, 0.8023)


class ConveyorError(RuntimeError):
    """A belt could not be built, or cannot convey."""


def _unit(vector: Any, *, name: str = "direction") -> np.ndarray:
    """A unit vector in the XY plane. A belt that runs uphill is not a belt."""
    value = as_vec3(vector, name=name).astype(float)
    value[2] = 0.0
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ConveyorError(
            f"{name}={list(vector)} has no horizontal component. A belt needs a "
            f"direction of travel in the XY plane, e.g. (1, 0, 0)."
        )
    return value / norm


def _surface_velocity_api() -> Any:
    """`PhysxSurfaceVelocityAPI`, or an error naming what is missing.

    Raised rather than warned. A belt that silently fails to convey is the
    original bug in a new costume: the scene looks right, the boxes never
    arrive, and the reason is three layers down in a schema import.
    """
    try:
        from pxr import PhysxSchema
    except ImportError as exc:  # pragma: no cover - depends on the Isaac build
        raise ConveyorError(
            "PhysxSchema is not importable, so no conveyor can be driven. This "
            "library needs the PhysX USD schemas that ship with Isaac Sim."
        ) from exc

    api = getattr(PhysxSchema, "PhysxSurfaceVelocityAPI", None)
    if api is None:  # pragma: no cover - depends on the Isaac build
        raise ConveyorError(
            "PhysxSchema has no PhysxSurfaceVelocityAPI. Surface velocity is "
            "how a belt drags what sits on it; without it this library cannot "
            "move a conveyor and will not pretend to. Isaac Sim 4.0+ ships it."
        )
    return api


def drive_surface(prim_path: str, velocity: Any, *, enabled: bool = True) -> dict[str, Any]:
    """Give one collider a moving surface. The primitive the rest of this builds on.

    Applied to the *rigid body* prim, not the mesh under it — PhysX reads the
    attribute off the body and applies it to the shapes beneath. Pointing this
    at a mesh whose body is its parent applies cleanly and does nothing at all,
    which is why `Conveyor` resolves the body itself rather than trusting a path.
    """
    from pxr import Gf

    api = _surface_velocity_api()
    stage = get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ConveyorError(f"{prim_path} does not exist, so it cannot be driven.")

    vector = as_vec3(velocity, name="velocity").astype(float)
    speed = float(np.linalg.norm(vector))
    if not _SANE_SPEED[0] <= speed <= _SANE_SPEED[1]:
        logger.warning(
            "Belt speed %.2f m/s is outside the %.0f-%.0f m/s range real "
            "conveyors run at. Check the units — this is metres per second, "
            "not per minute.",
            speed, *_SANE_SPEED,
        )

    applied = api.Apply(prim)
    applied.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(*(float(v) for v in vector)))
    applied.CreateSurfaceVelocityEnabledAttr().Set(bool(enabled))
    # Local space would make the vector rotate with the belt. A conveyor's
    # direction of travel is a fact about the floor it is bolted to, so world
    # space is right and being explicit stops it inheriting a default.
    local = applied.GetPrim().GetAttribute("physxSurfaceVelocity:surfaceVelocityLocalSpace")
    if local:
        local.Set(False)

    return {"prim_path": prim_path, "velocity": vector.tolist(), "enabled": bool(enabled)}


def _body_of(prim_path: str) -> str:
    """The prim carrying RigidBodyAPI at or beneath `prim_path`."""
    from pxr import UsdPhysics

    stage = get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ConveyorError(f"{prim_path} does not exist.")
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        return prim_path
    for prim in _descendants(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
    return prim_path


def _descendants(root: Any) -> Any:
    """Depth-first walk of a prim and everything under it."""
    from pxr import Usd

    return iter(Usd.PrimRange(root))


class Conveyor:
    """A belt, the stop at the end of it, and the boxes queued on it.

    `Conveyor.build()` makes the belt out of a slab whose dimensions you chose,
    which is what you want when the numbers have to close against a robot's
    reach. `Conveyor.from_prop()` wraps one of the real `ConveyorBelt_A*`
    assets, measures it, and drives that instead — better looking, and the
    dimensions are whatever NVIDIA modelled rather than whatever you asked for.
    """

    def __init__(
        self,
        belt_path: str,
        *,
        direction: Any = (1.0, 0.0, 0.0),
        speed: float = 0.25,
        top_z: float = 0.0,
        length: float | None = None,
        width: float | None = None,
        gate_path: str | None = None,
        centre: Any = None,
        scene: Any = None,
    ) -> None:
        from .scene import Scene as _Scene

        self.scene = scene or _Scene.get()
        self.belt_path = belt_path
        self.body_path = _body_of(belt_path)
        self.direction = _unit(direction)
        self.speed = float(speed)
        self.top_z = float(top_z)
        self.length = length
        self.width = width
        self.gate_path = gate_path
        self._boxes: list[Any] = []
        self._driven = False
        # Centre of the belt deck and the across-travel axis. Set here so every
        # reader can rely on them; `build`/`from_prop` overwrite them with the
        # measured values rather than leaving later code to `getattr` a default.
        self._origin = (
            np.zeros(3) if centre is None
            else as_vec3(centre, name="centre").astype(float)
        )
        self._across = np.array([-self.direction[1], self.direction[0], 0.0])
        self.box_size: np.ndarray | None = None
        self.asset: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"<Conveyor {self.belt_path} speed={self.speed:.2f}m/s "
            f"boxes={len(self._boxes)} gate={'yes' if self.gate_path else 'no'}>"
        )

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        prim_path: str = "/World/Conveyor",
        *,
        length: float = 3.0,
        width: float = 0.8,
        position: Any = (0.0, 0.0, 0.0),
        direction: Any = (1.0, 0.0, 0.0),
        speed: float = 0.25,
        friction: float = 0.9,
        gate: bool = True,
        gate_height: float = 0.25,
        gate_thickness: float = 0.04,
        color: Any = (0.15, 0.16, 0.18),
        scene: Any = None,
    ) -> "Conveyor":
        """A belt of known dimensions, running along `direction`, with a stop.

        `position` is the centre of the belt *top*, so the deck sits at
        `position[2]` and you can put it at a robot's working height without
        arithmetic. That is deliberately unlike `spawn_rigid`, which places a
        body by its centre — the number anyone actually has to hand for a
        conveyor is the height of its surface.
        """
        from .scene import Scene as _Scene

        scene = scene or _Scene.get()
        heading = _unit(direction)
        centre = as_vec3(position, name="position").astype(float)
        deck = 0.06  # slab thickness; the belt is a deck, not a block

        if length <= 0 or width <= 0:
            raise ConveyorError(
                f"length={length} width={width}: a belt needs positive extents."
            )

        # `spawn_rigid` scales a Cube whose default size is 2.0, so a half-extent
        # is what goes in. Getting this wrong by the factor of two is the single
        # most common way a scene comes out at half the size it was specified at.
        across = np.array([-heading[1], heading[0], 0.0])
        yaw = float(np.degrees(np.arctan2(heading[1], heading[0])))

        scene.spawn_rigid(
            prim_path,
            shape="cube",
            scale=[length / 2.0, width / 2.0, deck / 2.0],
            position=[centre[0], centre[1], centre[2] - deck / 2.0],
            orientation=[0.0, 0.0, yaw],
            mass=0.0,
            friction=friction,
            restitution=0.0,
            static=True,
            color=color,
        )

        gate_path = None
        if gate:
            gate_path = _build_gate(
                scene, f"{prim_path}Gate", centre=centre, heading=heading,
                length=length, width=width, yaw=yaw,
                height=gate_height, thickness=gate_thickness,
            )

        belt = cls(
            prim_path,
            direction=heading,
            speed=speed,
            top_z=float(centre[2]),
            length=float(length),
            width=float(width),
            gate_path=gate_path,
            scene=scene,
        )
        belt._origin = centre.copy()
        belt._across = across
        belt.start()
        return belt

    @classmethod
    def from_prop(
        cls,
        query: str = "conveyor belt",
        *,
        prim_path: str = "/World/Conveyor",
        position: Any = (0.0, 0.0, 0.0),
        direction: Any = (1.0, 0.0, 0.0),
        speed: float = 0.25,
        friction: float = 0.9,
        gate: bool = True,
        gate_height: float = 0.35,
        gate_thickness: float = 0.04,
        scene: Any = None,
    ) -> "Conveyor":
        """Drive one of the real `ConveyorBelt_A*` assets.

        **The deck is measured off the `Belt` child, never off the prop.** Every
        one of these assets is a frame with a belt slung inside it, and the frame
        is much taller than the belt: on `ConveyorBelt_A09` the prop's bounding
        box tops out at 2.311 m on the gantry while the belt surface is at
        1.781 m. Taking the prop's bbox top for the deck height puts every box
        0.53 m in the air, and they arrive by falling. Both variants measured
        carry the same numbers — deck top 1.781 m, belt 0.04 m thick — and the
        `Belt` prim is already a kinematic rigid body with a collider, which is
        exactly what surface velocity needs.

        **Not all of the fifty are straight.** `ConveyorBelt_A01` is a 90-degree
        curve: its rollers sweep from y=+0.45 round to y=-1.57. A single
        world-space surface velocity drives a curve straight off its own side,
        so a belt whose footprint is not clearly longer than it is wide is
        reported rather than silently driven into a wall.
        """
        from .props import spawn_prop
        from .scene import Scene as _Scene

        scene = scene or _Scene.get()
        entry = spawn_prop(query, prim_path=prim_path, position=position, scene=scene)

        deck_path = _belt_surface(prim_path)
        bounds = _world_bounds(deck_path)
        if bounds is None:
            raise ConveyorError(
                f"{deck_path} has no measurable bounding box, so the deck height "
                f"is unknown. Use Conveyor.build() and state the dimensions."
            )
        low, high = bounds
        heading = _unit(direction)
        across_v = np.array([-heading[1], heading[0], 0.0])
        span = np.array(high) - np.array(low)
        along = float(abs(np.dot(span, heading)))
        across = float(abs(np.dot(span, across_v)))

        if along < 2.0 * across:
            logger.warning(
                "%s has a %.2f x %.2f m belt footprint, which is not the long "
                "thin shape a straight run has. Several of the ConveyorBelt_A* "
                "assets are 90-degree curves (A01 is one), and a curve cannot be "
                "driven by one world-space velocity — boxes leave over the side. "
                "Pick a straight variant such as conveyorbelt_a09, or build the "
                "belt with Conveyor.build().",
                entry["key"], along, across,
            )

        _force_kinematic(deck_path)
        scene.apply_physics_material(deck_path, friction=friction, restitution=0.0)

        belt = cls(
            prim_path,
            direction=heading,
            speed=speed,
            top_z=float(high[2]),
            length=along,
            width=across,
            gate_path=None,
            scene=scene,
        )
        # The belt surface is what gets driven and what boxes are laid on, so
        # every measurement below comes off it rather than off the wrapper.
        belt.body_path = deck_path
        belt._origin = np.array(
            [(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0, float(high[2])]
        )
        belt._across = across_v
        belt.asset = entry
        if gate:
            # The shipped belts have no stop, so without this boxes ride off
            # the far end and fall. Measured on A09: three boxes travelled the
            # full length and dropped 0.88 m onto the floor.
            belt.gate_path = _build_gate(
                scene, f"{prim_path}Gate",
                centre=belt._origin, heading=heading,
                length=along, width=across,
                yaw=float(np.degrees(np.arctan2(heading[1], heading[0]))),
                height=gate_height, thickness=gate_thickness,
            )
        belt.start()
        return belt

    # ── Driving ──────────────────────────────────────────────────────────────

    def start(self) -> dict[str, Any]:
        """Switch the belt on. Idempotent."""
        result = drive_surface(self.body_path, self.direction * self.speed, enabled=True)
        self._driven = True
        return result

    def halt(self) -> dict[str, Any]:
        """Switch the belt off. The boxes stay exactly where they are."""
        result = drive_surface(self.body_path, (0.0, 0.0, 0.0), enabled=False)
        self._driven = False
        return result

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Change belt speed while it runs."""
        self.speed = float(speed)
        return self.start()

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(
        self,
        count: int = 4,
        *,
        box: Any = (0.18, 0.13, 0.11),
        mass: float = 1.5,
        spacing: float | None = None,
        start_offset: float = 0.25,
        friction: float = 0.9,
        prefix: str = "Box",
        color: Any = (0.72, 0.55, 0.33),
    ) -> list[Any]:
        """Queue `count` boxes along the belt, furthest-along first.

        `box` is the full size in metres — width along travel, across, and
        height — not a half-extent and not a scale factor. Returned in the order
        they will reach the stop, which is the order a palletiser should pick
        them in.
        """
        from .objects import RigidObject  # noqa: F401  (documents the return type)

        size = as_vec3(box, name="box").astype(float)
        if np.any(size <= 0):
            raise ConveyorError(f"box={list(size)}: every dimension must be positive.")
        if self.length is not None and size[0] >= self.length:
            raise ConveyorError(
                f"A {size[0]:.2f} m box does not fit on a {self.length:.2f} m belt."
            )

        gap = float(spacing) if spacing is not None else float(size[0]) * 1.6
        origin = self._origin
        # Lay them out from the far (gate) end backwards, so box 0 is the one
        # that arrives first and the queue does not depend on `count`.
        far = (self.length or 0.0) / 2.0 - start_offset

        made = []
        for index in range(int(count)):
            along = far - index * gap
            # Dead centre of the belt, every time. A box that starts skewed
            # arrives skewed, and the pick orientation stops being one number.
            centre = origin[:2] + self.direction[:2] * along
            # Boxes live beside the belt, never under it. A dynamic body
            # parented beneath a kinematic one inherits the parent's transform,
            # so a box spawned into the belt's subtree rides it like paint
            # instead of resting on it — and surface velocity then does nothing
            # visible because there is no relative motion to generate friction.
            body = self.scene.spawn_rigid(
                f"/World/{prefix}{index}",
                shape="cube",
                scale=[size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
                position=[
                    float(centre[0]),
                    float(centre[1]),
                    float(self.top_z + size[2] / 2.0 + 0.002),
                ],
                mass=float(mass),
                friction=float(friction),
                restitution=0.0,
                color=color,
            )
            made.append(body)

        self._boxes = made
        self.box_size = size
        return made

    @property
    def boxes(self) -> list[Any]:
        """The boxes this belt is carrying, in arrival order."""
        return list(self._boxes)

    def track(self, objects: Any) -> list[Any]:
        """Watch boxes that are already on the stage.

        What a controller needs on the second Play. `load()` authors boxes and
        is a scene-building call; a controller must not author anything, but it
        does have to know which prims to watch — and on a replay those are the
        same prims, returned to their authored poses by the timeline stopping.
        """
        self._boxes = list(objects)
        return self._boxes

    @classmethod
    def attach(
        cls,
        prim_path: str,
        *,
        direction: Any,
        speed: float,
        top_z: float,
        length: float,
        width: float | None = None,
        centre: Any = None,
        gate_path: str | None = None,
        scene: Any = None,
    ) -> "Conveyor":
        """A handle on a belt that is already on the stage. Authors nothing.

        The counterpart to `Robot.attach`. A controller runs against a scene it
        did not build, and re-running `build()` from inside `compute()` would
        author a second belt over the first one on every Play.
        """
        return cls(
            prim_path,
            direction=direction,
            speed=speed,
            top_z=top_z,
            length=length,
            width=width,
            gate_path=gate_path,
            centre=centre,
            scene=scene,
        )

    # ── Reading the queue ────────────────────────────────────────────────────

    def box_at_gate(
        self,
        *,
        max_speed: float = 0.02,
        within: float = 0.12,
    ) -> Any | None:
        """The box resting against the stop, or None while none has settled.

        Two conditions, and both matter. `within` finds the box nearest the far
        end; `max_speed` refuses it until it has actually stopped moving. A pick
        commanded at the moment of contact closes on a box that is still being
        pushed, and the grasp fails in a way that looks like a gripper problem.

        **A box that has arrived is half its own length short of the stop**, and
        that is the whole subtlety here. Positions are centres; the box rests on
        its *face*. Measured on a 3.2 m belt with 30 cm boxes: the lead box came
        to rest with its centre 0.15 m from the gate — exactly half a box — and
        a `within` of 0.12 measured against zero rejected it, so the belt worked
        perfectly and the arm was never told anything had arrived. The expected
        gap now comes from the box size `load()` already knows, and `within` is
        the tolerance *around* it rather than an absolute distance.
        """
        if not self._boxes:
            return None
        origin = self._origin
        far = (self.length or 0.0) / 2.0
        expected = float(self.box_size[0]) / 2.0 if self.box_size is not None else 0.0

        best, best_error = None, None
        for body in self._boxes:
            try:
                position = np.asarray(body.position, dtype=float)
                speed = float(body.speed)
            except Exception:  # noqa: BLE001 - a despawned box is not an error
                continue
            along = float(np.dot(position[:2] - origin[:2], self.direction[:2]))
            error = abs((far - along) - expected)
            if error > within:
                continue
            if speed > max_speed:
                continue
            if best_error is None or error < best_error:
                best, best_error = body, error
        return best

    def arrived(self, **kwargs: Any) -> bool:
        """True when a box is settled against the stop and ready to pick."""
        return self.box_at_gate(**kwargs) is not None

    def describe(self) -> dict[str, Any]:
        """What this belt is and whether it is actually driving."""
        return {
            "belt_path": self.belt_path,
            "body_path": self.body_path,
            "gate_path": self.gate_path,
            "direction": self.direction.round(4).tolist(),
            "speed": self.speed,
            "running": self._driven,
            "top_z": self.top_z,
            "length": self.length,
            "width": self.width,
            "boxes": [b.prim_path for b in self._boxes],
            "mechanism": "PhysxSurfaceVelocityAPI",
        }


#: What the belt surface is called inside the shipped conveyor assets. Both
#: variants measured (A01, A09) name it exactly this, and in both it is the one
#: prim carrying RigidBodyAPI — the frame around it is a plain collider mesh.
_DECK_NAMES = ("Belt", "belt", "ConveyorBelt", "Surface")


def _belt_surface(prim_path: str) -> str:
    """The prim that is the moving surface, not the frame holding it up.

    Named children first, because that is what the shipped assets use and the
    name is unambiguous. Falling back to "the kinematic rigid body under here"
    covers a differently-authored asset; falling back to the wrapper itself is
    the last resort and is where the 0.53 m error came from, so it warns.
    """
    from pxr import UsdPhysics

    stage = get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ConveyorError(f"{prim_path} does not exist.")

    for name in _DECK_NAMES:
        candidate = stage.GetPrimAtPath(f"{prim_path}/{name}")
        if candidate.IsValid():
            return f"{prim_path}/{name}"

    for prim in _descendants(root):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())

    logger.warning(
        "%s has no child named any of %s and no rigid body beneath it, so the "
        "whole prop is being treated as the belt. If this asset is a frame with "
        "a belt inside it, the deck height is wrong by the height of the frame "
        "and boxes will be spawned in the air.",
        prim_path, ", ".join(_DECK_NAMES),
    )
    return prim_path


def _build_gate(
    scene: Any,
    gate_path: str,
    *,
    centre: Any,
    heading: Any,
    length: float,
    width: float,
    yaw: float,
    height: float = 0.25,
    thickness: float = 0.04,
) -> str:
    """The stop at the end of a belt, spanning its full width.

    Its inner face is what a box comes to rest against, which is what turns a
    moving target into one fixed pick pose. Shared by `build` and `from_prop`
    because a real conveyor asset needs a stop exactly as much as a slab does —
    the shipped belts have none, so boxes ride off the end and fall. Measured
    on `ConveyorBelt_A09`: three boxes travelled the length of the belt and
    dropped 0.88 m onto the floor, which looks like a physics bug and is only a
    missing piece of the cell.
    """
    centre = np.asarray(centre, dtype=float)
    heading = np.asarray(heading, dtype=float)
    far = centre[:2] + heading[:2] * (length / 2.0 + thickness / 2.0)
    scene.spawn_rigid(
        gate_path,
        shape="cube",
        scale=[thickness / 2.0, width / 2.0, height / 2.0],
        position=[float(far[0]), float(far[1]), float(centre[2] + height / 2.0)],
        orientation=[0.0, 0.0, float(yaw)],
        mass=0.0,
        friction=0.4,
        restitution=0.0,
        static=True,
        color=(0.55, 0.13, 0.13),
    )
    return gate_path


def _force_kinematic(prim_path: str) -> None:
    """Make a body immovable without removing its collider."""
    from pxr import UsdPhysics

    prim = get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ConveyorError(f"{prim_path} does not exist.")
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr().Set(True)


def _world_bounds(prim_path: str) -> tuple[Any, Any] | None:
    """World-space bounding box of a prim, or None if it has no extent."""
    from pxr import Usd, UsdGeom

    prim = get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    box = cache.ComputeWorldBound(prim)
    interval = box.ComputeAlignedRange()
    if interval.IsEmpty():
        return None
    return interval.GetMin(), interval.GetMax()
