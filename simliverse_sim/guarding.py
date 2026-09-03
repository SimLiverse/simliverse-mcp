"""Perimeter guarding: the fence, the gate, and the furniture around a cell.

A cell without guarding does not look like a cell. It also does not behave
like one: the thing an integrator is actually selling is a machine a person
can stand next to, and the fence line is what decides where the conveyor
enters, where the operator stands, and which side the cabinet goes on. Those
are layout decisions, and they are the ones a customer argues about.

Nothing here is in the Isaac prop index - all 169 entries were checked - so it
is authored from primitives, the same way `DeadPlate` and the belt rails are.
Panels are static colliders: a robot that swings into a fence should be
stopped by it, because a cell where the arm passes through the guarding is
answering a different question than the one it was built to answer.

The module is arranged around one recurring mistake in this codebase: these
things are positioned by their centre and are metres across, so the number
anyone reasons about - where the fence *line* is - is not the number the API
wants. Every entry point here takes the footprint.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("simliverse_sim.guarding")


class GuardingError(RuntimeError):
    """A fence that cannot be built as asked."""


#: Guard panels are mesh, not wall: tall enough to stop a reach-over, and set
#: clear of the floor so swarf and cable can pass under. Conventions rather
#: than measurements, which is why they are arguments.
PANEL_HEIGHT = 2.0
PANEL_GROUND_GAP = 0.10
POST_SIDE = 0.06

#: A person has to get in. Narrower than this is a hatch, not a gate.
MIN_GATE_WIDTH = 0.80

SIDES = ("north", "south", "east", "west")


def _sides(centre: np.ndarray, size: np.ndarray) -> dict[str, dict[str, Any]]:
    """The four fence lines of a rectangular footprint, by compass name.

    Returned as (axis, fixed coordinate, span) rather than corner pairs,
    because every caller here walks a line placing things along it and none of
    them want to do the corner arithmetic twice.
    """
    half = size / 2.0
    return {
        "north": {"axis": 0, "fixed": float(centre[1] + half[1]),
                  "span": (float(centre[0] - half[0]),
                           float(centre[0] + half[0]))},
        "south": {"axis": 0, "fixed": float(centre[1] - half[1]),
                  "span": (float(centre[0] - half[0]),
                           float(centre[0] + half[0]))},
        "east": {"axis": 1, "fixed": float(centre[0] + half[0]),
                 "span": (float(centre[1] - half[1]),
                          float(centre[1] + half[1]))},
        "west": {"axis": 1, "fixed": float(centre[0] - half[0]),
                 "span": (float(centre[1] - half[1]),
                          float(centre[1] + half[1]))},
    }


def _runs(span: tuple[float, float],
          openings: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Split a fence line into the stretches that actually get panelled.

    An opening is a gap in the guarding - a gate, or the slot a conveyor runs
    through. Subtracting them here rather than at spawn time is what stops a
    panel being authored across the belt: a fence panel through a conveyor is
    invisible from the usual camera angle and stops every carton, which reads
    as a conveyor fault for as long as it takes to walk the camera round.
    """
    lo, hi = span
    keep = [(lo, hi)]
    for start, end in sorted(openings):
        cut: list[tuple[float, float]] = []
        for a, b in keep:
            if end <= a or start >= b:
                cut.append((a, b))
                continue
            if a < start:
                cut.append((a, start))
            if end < b:
                cut.append((end, b))
        keep = cut
    return [(a, b) for a, b in keep if (b - a) > 1e-6]


def _translucent(scene: Any, prim_path: str, opacity: float) -> bool:
    """Make a panel see-through, so the guarding does not hide the cell.

    Guarding is mesh. Opaque panels are not a cosmetic problem: the render is
    the thing a layout is reviewed from, and a cell you cannot see inside has
    answered nothing. The first fenced render came back as a white box with a
    gap in it.

    Returns whether the attribute was authored, because `displayOpacity` is a
    hint that a renderer may ignore and claiming translucency that did not
    happen is worse than reporting that it did not.
    """
    try:
        from pxr import UsdGeom

        prim = scene.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayOpacityAttr().Set([float(opacity)])
        return True
    except Exception:  # noqa: BLE001 - a look is not worth failing a build for
        logger.debug("could not set opacity on %s", prim_path, exc_info=True)
        return False


class SafetyFence:
    """A rectangular perimeter of mesh panels, with openings where asked.

    `size` is the guarded area - the fence *line*, not a centre and a guess.
    A cell is specified by the floor it occupies, so that is what this takes.
    """

    def __init__(self, prim_path: str, *, centre: Any, size: Any,
                 height: float, openings: dict[str, list[tuple[float, float]]],
                 scene: Any = None) -> None:
        self.prim_path = prim_path
        self.centre = np.asarray(centre, dtype=float).reshape(2)
        self.size = np.asarray(size, dtype=float).reshape(2)
        self.height = float(height)
        self.openings = openings
        self.scene = scene
        self.panels: list[str] = []
        self.posts: list[str] = []

    def __repr__(self) -> str:
        return (f"SafetyFence({self.prim_path!r}, "
                f"{self.size[0]:.2f} x {self.size[1]:.2f} m, "
                f"{len(self.panels)} panels)")

    @classmethod
    def build(
        cls,
        prim_path: str = "/World/Fence",
        *,
        centre: Any = (0.0, 0.0),
        size: Any = (4.0, 4.0),
        height: float = PANEL_HEIGHT,
        gate: str | None = "south",
        gate_width: float = 1.0,
        gate_offset: float = 0.0,
        crossings: list[dict[str, Any]] | None = None,
        panel_max: float = 1.5,
        panel_opacity: float = 0.25,
        colour: Any = (0.82, 0.72, 0.10),
        scene: Any = None,
    ) -> "SafetyFence":
        """Fence the footprint, leaving a gate and any crossings open.

        `crossings` are gaps for the things that have to pass through the line
        - a conveyor, mostly. Each is `{"side": "east", "centre": 0.4,
        "width": 0.6}`, measured in metres along that side. Author them rather
        than letting a panel land on the belt: a panel through a conveyor is
        invisible from the usual camera angle and stops every carton.

        `panel_max` is the widest single panel. Real guarding is bought in
        fixed widths, and a 6 m run being several panels with posts between
        them is most of what makes a fence read as a fence rather than a wall.
        """
        from .scene import Scene

        scene = scene or Scene.get()
        centre = np.asarray(centre, dtype=float).reshape(2)
        size = np.asarray(size, dtype=float).reshape(2)

        if np.any(size <= 0):
            raise GuardingError(
                f"size={size.tolist()}: a guarded area needs positive extents.")
        if height <= 0:
            raise GuardingError(f"height={height}: a fence has positive height.")

        sides = _sides(centre, size)
        if gate is not None and gate not in sides:
            raise GuardingError(
                f"gate={gate!r}: expected one of {sorted(sides)}.")
        if gate is not None and gate_width < MIN_GATE_WIDTH:
            raise GuardingError(
                f"gate_width={gate_width}: a person has to get through it, and "
                f"{MIN_GATE_WIDTH} m is the least that is a gate rather than a "
                f"hatch.")

        openings: dict[str, list[tuple[float, float]]] = {k: [] for k in sides}
        if gate is not None:
            lo, hi = sides[gate]["span"]
            mid = (lo + hi) / 2.0 + float(gate_offset)
            opening = (mid - gate_width / 2.0, mid + gate_width / 2.0)
            if opening[0] < lo - 1e-9 or opening[1] > hi + 1e-9:
                raise GuardingError(
                    f"A {gate_width:.2f} m gate at offset {gate_offset:.2f} runs "
                    f"off the {gate} side, which spans {lo:.2f}..{hi:.2f}. That "
                    f"leaves a corner unguarded rather than a doorway.")
            openings[gate].append(opening)

        for crossing in (crossings or []):
            side = crossing["side"]
            if side not in sides:
                raise GuardingError(
                    f"crossing side={side!r}: expected one of {sorted(sides)}.")
            width = float(crossing["width"])
            if width <= 0:
                raise GuardingError(
                    f"crossing width={width}: a gap has positive width.")
            mid = float(crossing["centre"])
            openings[side].append((mid - width / 2.0, mid + width / 2.0))

        fence = cls(prim_path, centre=centre, size=size, height=height,
                    openings=openings, scene=scene)

        z = PANEL_GROUND_GAP + height / 2.0
        for name, line in sides.items():
            for index, (a, b) in enumerate(_runs(line["span"], openings[name])):
                count = max(1, int(np.ceil((b - a) / panel_max)))
                step = (b - a) / count
                for piece in range(count):
                    mid = a + piece * step + step / 2.0
                    path = f"{prim_path}_{name.capitalize()}{index}_{piece}"
                    along = step / 2.0
                    if line["axis"] == 0:
                        scale = [along, 0.01, height / 2.0]
                        pos = [mid, line["fixed"], z]
                    else:
                        scale = [0.01, along, height / 2.0]
                        pos = [line["fixed"], mid, z]
                    scene.spawn_rigid(
                        path, shape="cube", scale=scale, position=pos,
                        mass=0.0, static=True, friction=0.4, restitution=0.0,
                        color=(0.55, 0.60, 0.65),
                    )
                    _translucent(scene, path, panel_opacity)
                    fence.panels.append(path)

        # Posts at every corner and either side of every opening: that is where
        # real guarding is bolted down, and it is what the eye reads as a fence
        # rather than a floating pane of glass.
        half = size / 2.0
        stations: list[tuple[float, float]] = [
            (float(centre[0] - half[0]), float(centre[1] - half[1])),
            (float(centre[0] + half[0]), float(centre[1] - half[1])),
            (float(centre[0] + half[0]), float(centre[1] + half[1])),
            (float(centre[0] - half[0]), float(centre[1] + half[1])),
        ]
        for name, line in sides.items():
            for a, b in openings[name]:
                for edge in (a, b):
                    stations.append((edge, line["fixed"]) if line["axis"] == 0
                                    else (line["fixed"], edge))
        for index, (x, y) in enumerate(stations):
            path = f"{prim_path}_Post{index}"
            scene.spawn_rigid(
                path, shape="cube",
                scale=[POST_SIDE / 2.0, POST_SIDE / 2.0, height / 2.0],
                position=[float(x), float(y), z],
                mass=0.0, static=True, friction=0.4, restitution=0.0,
                color=colour,
            )
            fence.posts.append(path)

        return fence

    def contains(self, point: Any, *, margin: float = 0.0) -> bool:
        """Is this point inside the guarded area?

        `margin` shrinks the area before testing, which is how to ask the
        question that matters: not "is the robot inside the fence" but "is it
        far enough inside that it cannot reach through the mesh".
        """
        p = np.asarray(point, dtype=float).reshape(-1)[:2]
        half = self.size / 2.0 - float(margin)
        if np.any(half <= 0):
            return False
        return bool(np.all(np.abs(p - self.centre) <= half))

    def clearance(self, point: Any) -> float:
        """Distance from a point to the nearest fence line; negative outside.

        This is the number to check a reach envelope against. An arm whose
        envelope crosses the guarding will hit it, and finding that out from a
        float is faster than finding it out from a render.
        """
        p = np.asarray(point, dtype=float).reshape(-1)[:2]
        half = self.size / 2.0
        return float(np.min(half - np.abs(p - self.centre)))

    def fits(self, centre: Any, reach: float) -> dict[str, Any]:
        """Does an arm at `centre` with this `reach` stay inside the guarding?

        Reported rather than raised, because a robot that can touch the fence
        is a normal thing to build on purpose - guarding is set at the reach
        envelope, not beyond it. What is not normal is not knowing.
        """
        gap = self.clearance(centre)
        return {
            "inside": bool(gap >= 0.0),
            "clearance": round(gap, 4),
            "reach": float(reach),
            "touches_fence": bool(gap < float(reach)),
            "overhang": round(float(reach) - gap, 4),
        }

    def describe(self) -> dict[str, Any]:
        """Enough to rebuild it, and enough to reason about the layout."""
        return {
            "prim_path": self.prim_path,
            "centre": self.centre.tolist(),
            "size": self.size.tolist(),
            "height": self.height,
            "openings": {k: [list(o) for o in v]
                         for k, v in self.openings.items() if v},
            "panels": len(self.panels),
            "posts": len(self.posts),
        }


def spawn_cabinet(prim_path: str = "/World/Cabinet", *,
                  position: Any = (0.0, 0.0, 0.0),
                  size: Any = (0.6, 0.5, 1.0),
                  colour: Any = (0.85, 0.35, 0.05),
                  scene: Any = None) -> str:
    """The control cabinet. Positioned by its footprint on the floor.

    `position` is where it stands, not where its centre floats: a cabinet
    given a centre at z=0 is half underground, and that is the single most
    common way this kind of furniture gets placed wrong.
    """
    from .scene import Scene

    scene = scene or Scene.get()
    extent = np.asarray(size, dtype=float).reshape(3)
    at = np.asarray(position, dtype=float).reshape(3)
    scene.spawn_rigid(
        prim_path, shape="cube",
        scale=(extent / 2.0).tolist(),
        position=[float(at[0]), float(at[1]), float(at[2] + extent[2] / 2.0)],
        mass=0.0, static=True, friction=0.6, restitution=0.0, color=colour,
    )
    return prim_path


def spawn_beacon(prim_path: str = "/World/Beacon", *,
                 position: Any = (0.0, 0.0, 0.0),
                 height: float = 1.0,
                 colour: Any = (0.10, 0.25, 0.90),
                 scene: Any = None) -> str:
    """A stack light on a post. Stands on the floor at `position`."""
    from .scene import Scene

    scene = scene or Scene.get()
    at = np.asarray(position, dtype=float).reshape(3)
    scene.spawn_rigid(
        prim_path, shape="cylinder", radius=0.05, size=float(height),
        position=[float(at[0]), float(at[1]), float(at[2] + height / 2.0)],
        mass=0.0, static=True, friction=0.4, restitution=0.0, color=colour,
    )
    return prim_path


def spawn_operator_platform(prim_path: str = "/World/OperatorPlatform", *,
                            position: Any = (0.0, 0.0, 0.0),
                            size: Any = (1.2, 1.6),
                            thickness: float = 0.05,
                            colour: Any = (0.90, 0.80, 0.10),
                            scene: Any = None) -> str:
    """The marked-out standing area outside the gate.

    Sits on the floor rather than in it, so it reads as a platform in a render
    instead of a stain.
    """
    from .scene import Scene

    scene = scene or Scene.get()
    at = np.asarray(position, dtype=float).reshape(3)
    extent = np.asarray(size, dtype=float).reshape(2)
    scene.spawn_rigid(
        prim_path, shape="cube",
        scale=[float(extent[0] / 2.0), float(extent[1] / 2.0),
               float(thickness / 2.0)],
        position=[float(at[0]), float(at[1]), float(at[2] + thickness / 2.0)],
        mass=0.0, static=True, friction=0.8, restitution=0.0, color=colour,
    )
    return prim_path
