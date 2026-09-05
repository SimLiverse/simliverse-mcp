"""Read a layout sketch, and build the guarding somebody drew.

The dashboard's canvas is gridded in metres and emits its shapes as text, not
as an image: the coordinates are the user's own measurements taken off the
grid, so there is no vision model between what they drew and what gets built.
This module is the other end of that wire.

It exists because the layout numbers were the ones the agent guessed worst.
Reach envelopes and bounding boxes give a technically valid cell that nobody
would draw - guarding sized to the millimetre around a conveyor pointing
nowhere. A rectangle on a grid settles the fence line in one gesture, and a
rectangle is exactly what `SafetyFence.build` already wants: a centre and a
footprint.

The payload looks like this, header included:

    [LAYOUT SKETCH - plan view of the floor, all values in metres.
     ...]

    rect   "cell" centre (0.00, 0.00) 6.50 x 6.50 m (x -3.25..3.25, y -3.25..3.25)
    arrow  "infeed" (4.00, -0.40) -> (-1.00, -0.40) length 5.00 m heading 180.00 deg (-X)
    circle "pallet" centre (0.00, 0.75) radius 0.60 m

Isaac is Z-up, so the plan view is the XY plane and the numbers transfer
one-to-one. Nothing here rescales or reprojects anything, and that is the
whole point of the canvas being top-down rather than an overlay on the 3D
viewport, where a circle on screen is a cone in space.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger("simliverse_sim.sketch")


class SketchError(ValueError):
    """A sketch that cannot be read, or that does not describe a cell."""


_NUM = r"(-?\d+(?:\.\d+)?)"

_RECT = re.compile(
    r'rect\s+"(?P<label>[^"]*)"\s+centre\s+\(' + _NUM + r",\s*" + _NUM + r"\)\s+" + _NUM + r"\s*x\s*" + _NUM + r"\s*m"
)
_ARROW = re.compile(
    r'arrow\s+"(?P<label>[^"]*)"\s+\(' + _NUM + r",\s*" + _NUM + r"\)\s*->\s*\(" + _NUM + r",\s*" + _NUM + r"\)"
)
_CIRCLE = re.compile(
    r'circle\s+"(?P<label>[^"]*)"\s+centre\s+\(' + _NUM + r",\s*" + _NUM + r"\)\s+radius\s+" + _NUM + r"\s*m"
)

#: Words a person uses for the thing they want fenced. Checked before falling
#: back to "the biggest rectangle", because the biggest rectangle is only the
#: cell until someone draws a bigger floor around it.
FENCE_WORDS = ("fence", "cell", "guard", "enclosure", "perimeter", "cage", "safety")

#: Words for a thing that has to get through the fence line.
FEED_WORDS = ("conveyor", "belt", "infeed", "outfeed", "feed", "line")

#: Words for the person the gate exists to let in.
OPERATOR_WORDS = ("operator", "worker", "person", "human", "attendant")

#: Distinguishes "the caller did not ask" from "the caller asked for None",
#: which means no gate at all and has to stay reachable. A default of
#: `"south"` could never tell those apart from an explicit `gate="south"`,
#: which is exactly why the gate ignored an operator drawn on any other side:
#: there was no way to know a choice had not been made.
_AUTO_GATE = object()


def parse_sketch(text: str) -> dict[str, list[dict[str, Any]]]:
    """Pull the shapes out of a sketch payload.

    Unknown lines are ignored rather than refused - the payload carries a
    prose header by design, and a parser that chokes on it would make the
    header unusable for the thing it is there for.
    """
    if not text or not text.strip():
        raise SketchError("The sketch is empty, so there is nothing to build.")

    rects, arrows, circles = [], [], []
    for match in _RECT.finditer(text):
        cx, cy, w, h = (float(match.group(i)) for i in range(2, 6))
        rects.append(
            {
                "label": match.group("label"),
                "centre": (cx, cy),
                "size": (w, h),
                "area": w * h,
            }
        )
    for match in _ARROW.finditer(text):
        ax, ay, bx, by = (float(match.group(i)) for i in range(2, 6))
        arrows.append(
            {
                "label": match.group("label"),
                "from": (ax, ay),
                "to": (bx, by),
                "length": float(np.hypot(bx - ax, by - ay)),
            }
        )
    for match in _CIRCLE.finditer(text):
        cx, cy, r = (float(match.group(i)) for i in range(2, 5))
        circles.append({"label": match.group("label"), "centre": (cx, cy), "radius": r})

    return {"rects": rects, "arrows": arrows, "circles": circles}


def _labelled(shapes: list[dict[str, Any]], words: tuple[str, ...]):
    """Shapes whose label mentions any of `words`, in drawing order."""
    return [s for s in shapes if any(w in (s["label"] or "").lower() for w in words)]


def pick_footprint(rects: list[dict[str, Any]]) -> dict[str, Any]:
    """Which rectangle is the cell.

    A label wins over size, always. "The biggest rectangle" is a reasonable
    guess right up until someone draws the building around the cell, and then
    it silently fences the site instead - so the label is checked first and
    the fallback says so in the result.
    """
    if not rects:
        raise SketchError(
            "The sketch has no rectangle in it. A fence needs a footprint: "
            "draw the guarded area as a rectangle and label it."
        )

    named = _labelled(rects, FENCE_WORDS)
    if len(named) == 1:
        return dict(named[0], chosen_by="label")
    if len(named) > 1:
        biggest = max(named, key=lambda r: r["area"])
        logger.warning(
            "%d rectangles are labelled like a cell (%s); taking the largest.",
            len(named),
            ", ".join(repr(r["label"]) for r in named),
        )
        return dict(biggest, chosen_by="largest of several labelled")
    if len(rects) == 1:
        return dict(rects[0], chosen_by="the only rectangle")
    return dict(max(rects, key=lambda r: r["area"]), chosen_by="largest, unlabelled")


def _nearest_side(centre, size, point) -> str:
    """Which fence side sits nearest a point, by true distance to that side's
    own span - not by distance to the infinite line it lies on.

    The infinite-line version ties for a point sitting square off a corner of
    a square footprint, where the distance to each of the two neighbouring
    sides is equal - the same corner-tie failure `_crosses` had before it was
    rewritten to solve real segment intersections instead of asking "nearest
    line". Point-to-segment distance, clamped to each side's actual span,
    does not tie except exactly on a diagonal, which is a genuine ambiguity
    rather than an artefact of the method.
    """
    cx, cy = centre
    hw, hh = size[0] / 2.0, size[1] / 2.0
    px, py = float(point[0]), float(point[1])

    edges = {
        "south": ((cx - hw, cy - hh), (cx + hw, cy - hh)),
        "north": ((cx - hw, cy + hh), (cx + hw, cy + hh)),
        "west": ((cx - hw, cy - hh), (cx - hw, cy + hh)),
        "east": ((cx + hw, cy - hh), (cx + hw, cy + hh)),
    }
    best_side, best_dist = "south", None
    for side, (a, b) in edges.items():
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        span2 = dx * dx + dy * dy
        t = 0.0 if span2 == 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span2))
        qx, qy = ax + t * dx, ay + t * dy
        dist = float(np.hypot(px - qx, py - qy))
        if best_dist is None or dist < best_dist:
            best_side, best_dist = side, dist
    return best_side


def _crosses(fence_centre, fence_size, a, b):
    """Where a segment actually leaves the footprint, and through which side.

    Returns None when the arrow does not cross the boundary at all, which is
    the common case for an arrow drawn wholly inside the cell to mean travel
    direction rather than an entry point.

    This solves the real intersection rather than asking which fence line the
    outside end is nearest. That shortcut looks right and is not: for a point
    straight out from the north side of a square, the distances to the north,
    east and west lines are all equal, so it ties three ways and takes
    whichever was listed first. Every feed from the north or south came out
    as "east".
    """
    cx, cy = fence_centre
    hw, hh = fence_size[0] / 2.0, fence_size[1] / 2.0
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh

    def inside(p):
        return x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9

    if inside(a) == inside(b):
        return None

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay

    best = None
    edges = (
        ("west", x0, True),
        ("east", x1, True),
        ("south", y0, False),
        ("north", y1, False),
    )
    for side, fixed, vertical in edges:
        denom = dx if vertical else dy
        if abs(denom) < 1e-12:
            continue  # parallel: it never meets this edge
        t = ((fixed - ax) / denom) if vertical else ((fixed - ay) / denom)
        if not (-1e-9 <= t <= 1.0 + 1e-9):
            continue  # the crossing is off the segment
        px, py = ax + t * dx, ay + t * dy
        along = py if vertical else px
        lo, hi = (y0, y1) if vertical else (x0, x1)
        if not (lo - 1e-9 <= along <= hi + 1e-9):
            continue  # it misses this edge's span
        # Nearest crossing to the outside end is the one it enters through.
        rank = t if inside(b) else (1.0 - t)
        if best is None or rank < best[0]:
            best = (rank, {"side": side, "centre": float(along)})

    return None if best is None else best[1]


def fence_from_sketch(
    text: str,
    *,
    prim_path: str = "/World/Fence",
    gate: str | None | object = _AUTO_GATE,
    gate_width: float = 1.0,
    crossing_width: float = 0.7,
    scene: Any = None,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Build the guarding somebody drew, and say what was inferred.

    Returns the fence alongside the decisions taken, because a sketch is
    ambiguous by nature and silently picking one reading is how a drawing
    becomes a cell nobody recognises. `chosen_by` in particular is worth
    surfacing: "largest, unlabelled" means nobody said which rectangle was
    the cell and this guessed.

    An arrow that crosses the footprint becomes an opening in the line - that
    is what someone means by drawing a conveyor running in from outside. An
    arrow drawn wholly inside means travel direction and is left alone.

    `gate` left unset picks the side nearest an operator circle, if one was
    drawn - a fence with a gate that opens next to nobody is not a mistake
    the drawing made, it is one this function used to make on its behalf.
    Passing `gate=` explicitly, `None` included, always wins outright.
    """
    from .guarding import SafetyFence

    shapes = parse_sketch(text)
    footprint = pick_footprint(shapes["rects"])
    centre, size = footprint["centre"], footprint["size"]

    operator_spot = next(
        (c for c in shapes["circles"] if any(w in (c["label"] or "").lower() for w in OPERATOR_WORDS)), None
    )

    gate_side = gate
    gate_chosen_by = "explicit"
    if gate is _AUTO_GATE:
        if operator_spot is not None:
            gate_side = _nearest_side(centre, size, operator_spot["centre"])
            gate_chosen_by = "nearest the operator (%r)" % operator_spot["label"]
        else:
            gate_side = "south"
            gate_chosen_by = "default - no operator was drawn"

    crossings: list[dict[str, Any]] = []
    for arrow in shapes["arrows"]:
        hit = _crosses(centre, size, arrow["from"], arrow["to"])
        if hit is None:
            continue
        crossings.append({"side": hit["side"], "centre": hit["centre"], "width": crossing_width, "for": arrow["label"]})

    fence = SafetyFence.build(
        prim_path,
        centre=centre,
        size=size,
        gate=gate_side,
        gate_width=gate_width,
        crossings=[{k: c[k] for k in ("side", "centre", "width")} for c in crossings],
        scene=scene,
        **build_kwargs,
    )

    return {
        "fence": fence,
        "footprint": {
            "label": footprint["label"],
            "centre": list(centre),
            "size": list(size),
            "chosen_by": footprint["chosen_by"],
        },
        "gate": {"side": gate_side, "chosen_by": gate_chosen_by},
        "crossings": crossings,
        "ignored": {
            "rects": [r["label"] for r in shapes["rects"] if r is not footprint and r["label"] != footprint["label"]],
            "circles": [c["label"] for c in shapes["circles"] if c is not operator_spot],
        },
        "describe": fence.describe(),
    }


def zones_from_sketch(text: str) -> dict[str, Any]:
    """Everything else in the sketch, as placeable numbers.

    A circle is where something goes - "the pallet here" - and a rectangle
    that is not the cell is a footprint for something inside it. Returned
    rather than built, because what a circle *means* is the agent's call and
    guessing it in here would put layout decisions somewhere nobody can see.
    """
    shapes = parse_sketch(text)
    footprint = pick_footprint(shapes["rects"]) if shapes["rects"] else None
    return {
        "cell": None
        if footprint is None
        else {"label": footprint["label"], "centre": list(footprint["centre"]), "size": list(footprint["size"])},
        "spots": [{"label": c["label"], "centre": list(c["centre"]), "radius": c["radius"]} for c in shapes["circles"]],
        "areas": [
            {"label": r["label"], "centre": list(r["centre"]), "size": list(r["size"])}
            for r in shapes["rects"]
            if footprint is None or r["label"] != footprint["label"]
        ],
        "flows": [
            {"label": a["label"], "from": list(a["from"]), "to": list(a["to"]), "length": round(a["length"], 3)}
            for a in shapes["arrows"]
        ],
    }
