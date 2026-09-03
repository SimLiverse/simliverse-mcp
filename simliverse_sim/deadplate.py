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

"""The dead plate: where a carton leaves the belt and waits to be picked.

A belt with a stop bolted to it delivers cartons to a fixed pose, which is what
made the first palletising cell reproducible. It is also not how the real thing
is built. In a real cell the belt *discharges* — the carton runs off the end,
drops a few centimetres onto a fixed plate, slides along it and comes to rest
against a mechanical stop. The plate has no motor; the carton is carried by its
own momentum and pushed by the ones behind it. That is why it is called dead.

Modelling it matters for more than looks. The drop and the slide are exactly
where a physics simulator says something an interpolating one cannot:

- a carton can land skewed, and the pick pose stops being one number
- it can bounce, and arrive with the queue in a different order
- it can jam against a side guide
- and how far it slides depends on friction and mass, not on geometry

None of those are visible in a kinematic tool, and all of them decide whether a
real cell runs at its quoted rate.

## The geometry, and why each number is what it is

    belt deck  ─────────────┐
                            │  drop
                            ▼
              ┌─────────────────────────────┐ ◀ stop
              │        dead plate           │
              └─────────────────────────────┘

**The drop is small.** 60 mm by default. A 150 mm carton dropped much further
tumbles rather than lands, and a tumbled carton has an orientation the pick has
to solve for rather than assume. Real dead plates sit just under the belt line
for the same reason.

**The plate is slippery.** Friction 0.15 against the belt's 0.9. A carton
arriving at 0.2 m/s onto a high-friction plate stops in a few centimetres and
never reaches the stop; the plate has to let it run. This is the one number
most likely to need tuning for a different carton mass.

**Side guides are not decoration.** Without them a carton that lands skewed
walks off the side over the next few cycles, and the failure appears later as
"the arm cannot reach the box" — which is true, and says nothing about why.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._compat import as_vec3

logger = logging.getLogger(__name__)

#: Belt-to-plate drop. Enough to be a real transfer, small enough to land flat.
DEFAULT_DROP = 0.06

#: A dead plate has to let a carton run. The belt is 0.9; this is not.
DEFAULT_FRICTION = 0.15


class DeadPlateError(RuntimeError):
    """A plate could not be built."""


class DeadPlate:
    """A fixed plate with a stop, fed by a belt discharging onto it."""

    def __init__(
        self,
        prim_path: str,
        *,
        deck_z: float,
        stop_x: float,
        length: float,
        width: float,
        centre_y: float,
        scene: Any = None,
    ) -> None:
        from .scene import Scene

        self.prim_path = prim_path
        self.scene = scene or Scene.get()
        self.deck_z = float(deck_z)
        self.stop_x = float(stop_x)
        self.length = float(length)
        self.width = float(width)
        self.centre_y = float(centre_y)
        self.box_size: np.ndarray | None = None
        self._boxes: list[Any] = []

    # ── Authoring ───────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        prim_path: str = "/World/DeadPlate",
        *,
        deck_z: float,
        stop_x: float,
        length: float = 0.5,
        width: float = 0.45,
        centre_y: float = 0.0,
        thickness: float = 0.03,
        stop_height: float = 0.18,
        guide_height: float = 0.10,
        friction: float = DEFAULT_FRICTION,
        colour: Any = (0.62, 0.64, 0.67),
        scene: Any = None,
    ) -> "DeadPlate":
        """Author the plate, its stop, and its side guides. All static.

        `deck_z` is the top surface a carton rests on, matching `Conveyor`'s
        convention — not the centre of the slab. Getting that wrong puts every
        carton half a plate underground, and the symptom is a pick that reaches
        into the floor.
        """
        from .scene import Scene

        scene = scene or Scene.get()
        if length <= 0 or width <= 0:
            raise DeadPlateError(
                f"length={length} width={width}: a plate needs positive extents."
            )

        plate = cls(prim_path, deck_z=deck_z, stop_x=stop_x, length=length,
                    width=width, centre_y=centre_y, scene=scene)

        centre_x = float(stop_x) - float(length) / 2.0
        # Static, so it never sags under a stack and never drifts. `spawn_rigid`
        # with static=True gives a kinematic body, which is what a collider
        # needs to be to take contact without moving.
        scene.spawn_rigid(
            prim_path,
            shape="cube",
            scale=[length / 2.0, width / 2.0, thickness / 2.0],
            position=[centre_x, float(centre_y), float(deck_z) - thickness / 2.0],
            static=True,
            friction=float(friction),
            restitution=0.0,
            color=colour,
        )

        # The stop. Its inner face is at `stop_x`, so a carton's own face rests
        # there and the pick pose is `stop_x - box/2` - the same arithmetic the
        # belt's gate uses, and the same place it is easy to be half a box out.
        scene.spawn_rigid(
            f"{prim_path}_Stop",
            shape="cube",
            scale=[0.02, width / 2.0, stop_height / 2.0],
            position=[float(stop_x) + 0.02, float(centre_y),
                      float(deck_z) + stop_height / 2.0],
            static=True,
            friction=0.5,
            restitution=0.0,
            color=(0.75, 0.25, 0.22),
        )

        # Side guides, so a carton that lands skewed is straightened rather than
        # walked off the edge over the following cycles.
        for side, sign in (("L", 1.0), ("R", -1.0)):
            scene.spawn_rigid(
                f"{prim_path}_Guide{side}",
                shape="cube",
                scale=[length / 2.0, 0.015, guide_height / 2.0],
                position=[centre_x, float(centre_y) + sign * (width / 2.0 + 0.015),
                          float(deck_z) + guide_height / 2.0],
                static=True,
                friction=0.2,
                restitution=0.0,
                color=colour,
            )
        return plate

    # ── Reading ─────────────────────────────────────────────────────────────

    def track(self, boxes: list[Any]) -> "DeadPlate":
        """Watch these cartons. They are spawned on the belt, not here."""
        self._boxes = list(boxes)
        return self

    def set_box_size(self, box: Any) -> "DeadPlate":
        self.box_size = as_vec3(box, name="box").astype(float)
        return self

    @property
    def boxes(self) -> list[Any]:
        return list(self._boxes)

    def box_at_stop(
        self,
        *,
        max_speed: float = 0.02,
        within: float | None = None,
    ) -> Any | None:
        """The carton resting against the stop, settled and on the plate.

        Deliberately the same shape of check as `Conveyor.box_at_gate`, and for
        the same reason: displacement toward the stop says nothing about the
        other two axes, and a carton that has bounced off the plate can still
        report an x inside the tolerance. It must be on the deck, between the
        guides, and stopped.
        """
        if not self._boxes:
            return None
        half_box = float(self.box_size[0]) / 2.0 if self.box_size is not None else 0.0
        if within is None:
            within = 0.5 * half_box if self.box_size is not None else 0.10

        rest_z = self.deck_z + (
            float(self.box_size[2]) / 2.0 if self.box_size is not None else 0.0
        )
        half_width = self.width / 2.0
        across_limit = half_width + (
            float(self.box_size[1]) / 2.0 if self.box_size is not None else 0.0
        )

        best, best_error = None, None
        for body in self._boxes:
            try:
                position = np.asarray(body.position, dtype=float)
                speed = float(body.speed)
            except Exception:  # noqa: BLE001 - a despawned carton is not an error
                continue
            if speed > max_speed:
                continue
            error = abs((self.stop_x - float(position[0])) - half_box)
            if error > within:
                continue
            if abs(float(position[1]) - self.centre_y) > across_limit:
                continue
            if self.box_size is not None and abs(
                float(position[2]) - rest_z
            ) > float(self.box_size[2]):
                continue
            if best_error is None or error < best_error:
                best, best_error = body, error
        return best

    def describe(self) -> dict:
        """Everything needed to rebuild this handle inside a controller."""
        return {
            "prim_path": self.prim_path,
            "deck_z": self.deck_z,
            "stop_x": self.stop_x,
            "length": self.length,
            "width": self.width,
            "centre_y": self.centre_y,
            "box_size": None if self.box_size is None
            else self.box_size.round(4).tolist(),
            "boxes": [b.prim_path for b in self._boxes],
        }

    @classmethod
    def from_description(cls, described: dict, *, scene: Any = None) -> "DeadPlate":
        """Rebuild a handle from `describe()` without authoring anything."""
        plate = cls(
            described["prim_path"],
            deck_z=described["deck_z"],
            stop_x=described["stop_x"],
            length=described["length"],
            width=described["width"],
            centre_y=described["centre_y"],
            scene=scene,
        )
        size = described.get("box_size")
        if size is not None:
            plate.box_size = np.asarray(size, dtype=float)
        return plate
