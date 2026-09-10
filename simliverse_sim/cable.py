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

"""A cable, hose or dress pack as a chain of rigid links.

Isaac Sim 6.0.1 has no cable. The asset called `deformabletube_tube` is a
static collision mesh; `PhysxDeformableBodyAPI` is gone from the schema, and
the replacement (`omni.physx.scripts.deformableUtils`, FEM tet meshes) needs
`enableGPUDynamics=True` and a scene rebuild - measured: a volume deformable
authored into the running palletising scene took the APIs and never cooked.

A chain of capsules on spherical joints needs none of that. Measured in the
live cell with GPU dynamics off: sixteen 12.5 cm links over a 2 m span, both
ends pinned, settled in 2 s to a 7.7 cm mid-span sag with no self-collision
and no drift. It hangs, swings, drapes over things and pulls on what it is
attached to, which is what a cable does in a cell.

    cable = Cable.build("/World/Dress", start=[0, 0, 1.5], end=[2, 0, 1.5],
                        slack=0.10, radius=0.01)
    cable.describe()   # ends, sag, link count, settled

Attach an end to a moving body by prim path - a robot flange, a cabinet - and
it follows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .scene import Scene

#: Link length that stays stable at 32 position iterations and reads as a
#: cable rather than a string of sausages.
#: Default per-link length. Short, because a rope is only as smooth as its
#: segments: at 0.125 m the links are long enough that each 60-deg bend shows a
#: visible kink and the chain reads as loose capsules, not a rope. At ~0.05 m
#: the curve is smooth and, with the caps overlapping (see `build`), continuous.
LINK_LENGTH = 0.05
#: Cone limit per joint. 60 degrees lets a chain fold back on itself over two
#: links without a joint reaching its stop and popping.
CONE_DEGREES = 60.0


def chain_layout(start: Any, end: Any, *, slack: float, links: int | None = None) -> dict:
    """Where the links go before physics: along a parabola with the slack.

    Laying a chain longer than the gap along the straight chord stacks the
    links into each other, and PhysX resolves that by throwing them apart. A
    parabola whose arc length matches the chain's length starts the cable
    where gravity is about to put it anyway.

    Returns centres, unit tangents and the per-link length.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    chord = float(np.linalg.norm(b - a))
    if chord <= 0.0:
        raise ValueError("start and end coincide; a cable needs two points")
    if slack < 0.0:
        raise ValueError("slack=%r: a cable cannot be shorter than its chord" % (slack,))
    length = chord * (1.0 + float(slack))
    count = int(links) if links else max(2, int(math.ceil(length / LINK_LENGTH)))
    link = length / count

    # Arc length of y = -4h t(1-t) over a chord d is about d + 8h^2/(3d).
    depth = math.sqrt(max(0.0, 3.0 * chord * (length - chord) / 8.0))
    axis = (b - a) / chord
    down = np.array([0.0, 0.0, -1.0])
    # A vertical cable has no "down" to sag toward; hang the bulge sideways.
    if abs(float(np.dot(axis, down))) > 0.99:
        down = np.array([1.0, 0.0, 0.0])
    down = down - axis * float(np.dot(down, axis))
    down /= np.linalg.norm(down)

    def point(t: float) -> np.ndarray:
        return a + axis * (chord * t) + down * (4.0 * depth * t * (1.0 - t))

    def tangent(t: float) -> np.ndarray:
        d = axis * chord + down * (4.0 * depth * (1.0 - 2.0 * t))
        return d / np.linalg.norm(d)

    centres, tangents = [], []
    for i in range(count):
        t = (i + 0.5) / count
        centres.append(point(t))
        tangents.append(tangent(t))
    return {
        "centres": np.asarray(centres),
        "tangents": np.asarray(tangents),
        "link": link,
        "length": length,
        "chord": chord,
        "depth": depth,
        "links": count,
    }


def align_z(direction: Any) -> list[float]:
    """Quaternion [w, x, y, z] turning +Z onto `direction`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if c > 1.0 - 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    if c < -1.0 + 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    axis = np.cross(z, d)
    s = math.sqrt((1.0 + c) * 2.0)
    return [s / 2.0, float(axis[0] / s), float(axis[1] / s), float(axis[2] / s)]


class Cable:
    """A built chain. `build` authors it; the rest measures it."""

    def __init__(self, prim_path: str, links: list[Any], *, layout: dict, start: Any, end: Any, scene: Scene) -> None:
        self.prim_path = prim_path
        self.links = links
        self.layout = layout
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.scene = scene

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        start: Any,
        end: Any,
        slack: float = 0.10,
        radius: float = 0.01,
        links: int | None = None,
        mass: float = 0.02,
        anchor_start: str | None = "",
        anchor_end: str | None = "",
        color: Any = (0.15, 0.15, 0.18),
        cone_degrees: float = CONE_DEGREES,
        scene: Scene | None = None,
    ) -> "Cable":
        """Author the chain between `start` and `end`.

        `anchor_start` / `anchor_end`: `""` pins that end to the world at the
        point given, a prim path pins it to that body (the point is taken in
        world space and converted), `None` leaves the end free.
        """
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        scene = scene or Scene.get()
        stage = scene.stage
        plan = chain_layout(start, end, slack=slack, links=links)
        link = float(plan["link"])
        # Overlap the capsules. If the cylinder is `link - 2*radius` the caps
        # just meet end to end, and any bend opens a gap between them; making the
        # cylinder the full `link` long makes each capsule's caps reach a radius
        # INTO its neighbours, so a bent rope stays a continuous tube with no
        # seams. Adjacent links do not collide (their shared joint filters it),
        # and the overlap never reaches the link beyond, so nothing self-traps.
        height = float(link)

        UsdGeom.Xform.Define(stage, prim_path)
        bodies = []
        for i, (centre, tangent) in enumerate(zip(plan["centres"], plan["tangents"])):
            path = "%s/link_%02d" % (prim_path, i)
            body = scene.spawn_rigid(
                path,
                shape="Capsule",
                radius=float(radius),
                size=height,
                position=[float(v) for v in centre],
                orientation=align_z(tangent),
                mass=float(mass),
                color=color,
                friction=0.6,
            )
            prim = stage.GetPrimAtPath(path)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr().Set(32)
            rb.CreateSolverVelocityIterationCountAttr().Set(4)
            rb.CreateAngularDampingAttr().Set(0.5)
            rb.CreateLinearDampingAttr().Set(0.1)
            bodies.append(body)

        half = Gf.Vec3f(0.0, 0.0, float(link / 2.0))
        limit = float(cone_degrees)
        for i in range(len(bodies) - 1):
            joint = UsdPhysics.SphericalJoint.Define(stage, "%s/joint_%02d" % (prim_path, i))
            joint.CreateBody0Rel().SetTargets([bodies[i].prim_path])
            joint.CreateBody1Rel().SetTargets([bodies[i + 1].prim_path])
            joint.CreateLocalPos0Attr().Set(half)
            joint.CreateLocalPos1Attr().Set(-half)
            joint.CreateConeAngle0LimitAttr().Set(limit)
            joint.CreateConeAngle1LimitAttr().Set(limit)
            joint.CreateCollisionEnabledAttr().Set(False)

        cls._anchor(stage, "%s/anchor_start" % prim_path, bodies[0].prim_path, -half, start, anchor_start)
        cls._anchor(stage, "%s/anchor_end" % prim_path, bodies[-1].prim_path, half, end, anchor_end)
        return cls(prim_path, bodies, layout=plan, start=start, end=end, scene=scene)

    @staticmethod
    def _anchor(stage: Any, path: str, link_path: str, local: Any, point: Any, anchor: str | None) -> None:
        """Pin one end: to the world, to a body, or not at all."""
        if anchor is None:
            return
        from pxr import Gf, UsdGeom, UsdPhysics

        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody1Rel().SetTargets([link_path])
        joint.CreateLocalPos1Attr().Set(local)
        world = Gf.Vec3d(*[float(v) for v in point])
        if anchor:
            prim = stage.GetPrimAtPath(anchor)
            if not prim or not prim.IsValid():
                raise ValueError("%s: no such prim to anchor the cable to" % anchor)
            joint.CreateBody0Rel().SetTargets([anchor])
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            local0 = xf.GetInverse().Transform(world)
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local0[0], local0[1], local0[2]))
        else:
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world[0], world[1], world[2]))
        joint.CreateCollisionEnabledAttr().Set(False)

    def centres(self) -> np.ndarray:
        return np.asarray([np.asarray(b.position, dtype=float) for b in self.links])

    def sag(self) -> float:
        """How far below the chord the lowest link centre hangs, in metres."""
        pts = self.centres()
        axis = self.end - self.start
        chord = float(np.linalg.norm(axis))
        axis = axis / chord
        rel = pts - self.start
        along = rel @ axis
        on_chord = self.start + np.outer(along, axis)
        drop = (on_chord - pts)[:, 2]
        return float(np.max(drop)) if drop.size else 0.0

    def speed(self) -> float:
        return float(max(float(b.speed) for b in self.links))

    def describe(self) -> dict:
        pts = self.centres()
        half = self.layout["link"] / 2.0
        return {
            "prim_path": self.prim_path,
            "links": len(self.links),
            "length": round(float(self.layout["length"]), 4),
            "chord": round(float(self.layout["chord"]), 4),
            "sag": round(self.sag(), 4),
            "speed": round(self.speed(), 4),
            "first_link": np.round(pts[0], 4).tolist(),
            "last_link": np.round(pts[-1], 4).tolist(),
            "end_gap": [
                round(float(np.linalg.norm(pts[0] - self.start)) - half, 4),
                round(float(np.linalg.norm(pts[-1] - self.end)) - half, 4),
            ],
        }


def verify_cable(cable: Cable, *, end_tolerance: float = 0.03, settled_speed: float = 0.02) -> dict:
    """Did it stay attached, settle, and hang rather than explode?

    `end_gap` is measured from each end link's centre to the anchor point,
    less half a link, so a chain that has torn off its anchor shows as a gap
    and not as a cable that happens to be somewhere else.
    """
    d = cable.describe()
    problems = []
    if d["end_gap"][0] > end_tolerance:
        problems.append("start end %.3f m from its anchor" % d["end_gap"][0])
    if d["end_gap"][1] > end_tolerance:
        problems.append("far end %.3f m from its anchor" % d["end_gap"][1])
    if d["speed"] > settled_speed:
        problems.append("still moving at %.3f m/s" % d["speed"])
    if d["sag"] < 0.0 or d["sag"] > d["length"]:
        problems.append("sag %.3f m is not a hanging cable" % d["sag"])
    return {"ok": not problems, "problems": problems, **d}
