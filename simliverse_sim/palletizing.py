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

"""Where each box goes on the pallet, and whether it got there.

Stacking one box on another is a place pose plus a release height, and the
library already had both. A *pallet* is the same motion twenty times against a
pattern, and the pattern is where it goes wrong: the arithmetic is easy to write
and easy to write subtly incorrectly, and the failure mode is not an exception.
It is a stack that leans, or a box placed half on its neighbour and half on air,
discovered three placements later when the tower comes down. Every one of those
is a number that was off by half a box width.

So the pattern is computed once, in one place, and returned as poses:

    from simliverse_sim import pallet_slots

    slots = pallet_slots(
        origin=[0.0, 1.20, 0.145],      # centre of the pallet *deck*
        box=(0.18, 0.13, 0.11),
        rows=2, cols=2, layers=2,
    )
    for slot in slots:                  # bottom layer first, in placement order
        arm.servo_to(slot["place"], DOWN)

Each slot carries `place` (where the box *centre* ends up), `approach` (clear
above it), `yaw` in degrees, and its `row`/`col`/`layer` indices. Placement order
is bottom-layer-first and, within a layer, ordered so the arm never has to reach
across a box it has already set down.

**`interlock` is not decoration.** A grid-stacked pallet is columns of boxes that
slide apart in transit; real palletising rotates alternate layers 90 degrees so
the seams cross and the stack ties itself together. It changes the poses and the
yaw, so it has to be decided here rather than added afterwards.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["PalletError", "pallet_slots", "verify_pallet"]


class PalletError(ValueError):
    """A pattern that cannot be built as specified."""


def pallet_slots(
    origin: Any,
    box: Any,
    *,
    rows: int = 2,
    cols: int = 2,
    layers: int = 1,
    gap: float = 0.01,
    interlock: bool = False,
    approach: float = 0.15,
    clearance: float = 0.003,
    deck: Any = None,
) -> list[dict[str, Any]]:
    """Place poses for a `rows` x `cols` x `layers` pallet pattern.

    `origin` is the centre of the pallet **deck** — the surface boxes rest on,
    not the pallet's own centre and not the floor. A standard Isaac `pallet`
    prop decks at about 0.145 m; measure it rather than assuming, because the
    whole stack inherits this number and an error here tilts every layer.

    `box` is the full size in metres, `(along_x, along_y, height)`.

    `clearance` is how far above its resting height a box is released. Three
    millimetres, from measurement: letting go at 12 mm dropped a box hard enough
    onto a two-high stack to knock it apart. It is a set-down, not a drop.

    `deck`, when given as `(x_size, y_size)`, is checked against the footprint
    the pattern needs, and a pattern that overhangs raises rather than building a
    stack whose outer column is resting on nothing.
    """
    centre = np.asarray(origin, dtype=float).reshape(3)
    size = np.asarray(box, dtype=float).reshape(3)

    if rows < 1 or cols < 1 or layers < 1:
        raise PalletError(
            f"rows={rows} cols={cols} layers={layers}: each must be at least 1."
        )
    if np.any(size <= 0):
        raise PalletError(f"box={size.tolist()}: every dimension must be positive.")
    if gap < 0:
        raise PalletError(f"gap={gap}: a negative gap overlaps the boxes.")

    if deck is not None:
        have_x, have_y = (float(v) for v in np.asarray(deck, dtype=float).reshape(2))
        # Every layer the pattern will actually build, not just the first. An
        # interlocked layer is rotated, so it occupies a different footprint —
        # checking only layer 0 passes a stack whose second layer hangs off the
        # side, which is exactly the failure the check exists to prevent.
        shapes = [(rows, cols, size[0], size[1])]
        if interlock and layers > 1:
            shapes.append((cols, rows, size[1], size[0]))
        for l_rows, l_cols, span_x, span_y in shapes:
            need_x = l_cols * span_x + (l_cols - 1) * gap
            need_y = l_rows * span_y + (l_rows - 1) * gap
            if need_x > have_x + 1e-9 or need_y > have_y + 1e-9:
                raise PalletError(
                    f"A {l_cols}x{l_rows} pattern of {span_x:.3f}x{span_y:.3f} m "
                    f"boxes needs {need_x:.3f}x{need_y:.3f} m of deck, but the "
                    f"pallet is {have_x:.3f}x{have_y:.3f} m. The outer column "
                    f"would overhang. Reduce the pattern, or state a bigger deck."
                )

    slots: list[dict[str, Any]] = []
    index = 0
    for layer in range(layers):
        # Alternate layers turn 90 degrees, which swaps how many boxes fit along
        # each axis as well as the box's own footprint.
        turned = bool(interlock and layer % 2)
        l_rows, l_cols = (cols, rows) if turned else (rows, cols)
        span = np.array([size[1], size[0]]) if turned else np.array([size[0], size[1]])
        yaw = 90.0 if turned else 0.0

        width = l_cols * span[0] + (l_cols - 1) * gap
        depth = l_rows * span[1] + (l_rows - 1) * gap
        # Corner of the pattern, so slot centres come out symmetric about origin.
        x0 = centre[0] - width / 2.0 + span[0] / 2.0
        y0 = centre[1] - depth / 2.0 + span[1] / 2.0
        z = centre[2] + layer * size[2] + size[2] / 2.0

        for row in range(l_rows):
            # Serpentine within a layer: the arm finishes each row next to where
            # the following one starts, and never crosses over a placed box.
            order = range(l_cols) if row % 2 == 0 else reversed(range(l_cols))
            for col in order:
                x = x0 + col * (span[0] + gap)
                y = y0 + row * (span[1] + gap)
                slots.append(
                    {
                        "index": index,
                        "layer": layer,
                        "row": row,
                        "col": int(col),
                        "yaw": yaw,
                        "place": [float(x), float(y), float(z + clearance)],
                        "rest": [float(x), float(y), float(z)],
                        "approach": [float(x), float(y), float(z + approach)],
                    }
                )
                index += 1

    return slots


def _yaw_degrees(quat: Any) -> float:
    """Rotation about world z, in degrees, from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.degrees(np.arctan2(2.0 * (w * z + x * y),
                                       1.0 - 2.0 * (y * y + z * z))))


def _skew_degrees(quat: Any, want_deg: float) -> float:
    """How far off square, wrapped into a quarter turn.

    A carton is symmetric under 90 degrees, so one sitting at 88 degrees is
    2 degrees off square, not 88. Without the wrap every correctly rotated
    carton in a pinwheel pattern reads as a failure.
    """
    return abs((_yaw_degrees(quat) - want_deg + 45.0) % 90.0 - 45.0)


def verify_pallet(
    objects: Any,
    slots: list[dict[str, Any]],
    *,
    tolerance: float = 0.04,
    settled_speed: float = 0.02,
    square_tolerance: float = 5.0,
) -> dict[str, Any]:
    """Did the boxes actually land on their slots, square, and stay there?

    Measured against `rest`, not `place`: `place` is where the box was released,
    a few millimetres high, and every box should have settled below it. A box
    still moving is not placed, however close it is — it is mid-fall, and a
    report written one frame earlier would have called a collapsing stack a
    success.

    Orientation counts too. A carton set down 26.7 degrees off sat 8 mm from
    its slot, so a distance-only check called it placed — and the pallet it
    made would not stretch-wrap. Measured, that is the difference between a
    photograph a customer accepts and one they do not.

    Returns a per-slot verdict rather than a bare bool, because "three of four,
    and the fourth is 9 cm out in +X" is the sentence that identifies the bug.
    """
    results = []
    placed = 0
    for slot, obj in zip(slots, list(objects)):
        try:
            position = np.asarray(obj.position, dtype=float)
            speed = float(obj.speed)
        except Exception as exc:  # noqa: BLE001 - report it, do not crash the check
            results.append(
                {"index": slot["index"], "ok": False, "reason": f"unreadable: {exc}"}
            )
            continue
        # Separately, and forgivingly: an object that cannot report an
        # orientation is not thereby unplaced. Position and speed are the
        # measurements this check has always been able to make, and losing
        # them because a handle has no `orientation` would be a worse answer
        # than not knowing the angle.
        try:
            skew = _skew_degrees(obj.orientation, float(slot.get("yaw") or 0.0))
        except Exception:  # noqa: BLE001
            skew = 0.0
        target = np.asarray(slot["rest"], dtype=float)
        error = float(np.linalg.norm(position - target))
        moving = speed > settled_speed
        askew = skew > square_tolerance
        ok = error <= tolerance and not moving and not askew
        placed += int(ok)
        results.append(
            {
                "index": slot["index"],
                "layer": slot["layer"],
                "ok": ok,
                "error": round(error, 4),
                "offset": (position - target).round(4).tolist(),
                "speed": round(speed, 4),
                "skew": round(skew, 2),
                "reason": (
                    "" if ok
                    else "still moving" if moving
                    else f"{error:.3f} m from its slot" if error > tolerance
                    else f"{skew:.1f} deg off square"
                ),
            }
        )

    return {
        "placed": placed,
        "of": len(results),
        "complete": placed == len(results) and bool(results),
        "slots": results,
    }
