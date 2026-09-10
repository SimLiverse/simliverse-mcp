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

"""Editing the layout sketch - writing the drawing back.

The sketch is the source of truth for a cell. When a person asks the agent to
change it in words - "add a pallet at (1, 2)", "move the operator to the north
side", "draw the AMR's route round the racks" - the drawing has to change, not
just the 3D scene, or the two silently diverge and the next build starts from
a stale picture. This tool edits the sketch text and returns it re-emitted in
the dashboard's own format, so the dashboard can redraw it and the agent can
build from it. It is pure text: no simulator round-trip, no scene changes.
"""

import json
from typing import Callable

from mcp.server.fastmcp import FastMCP

from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:
    @mcp.tool("edit_sketch")
    def edit_sketch(sketch: str, ops: str) -> str:
        """Edit the user's layout sketch and return the updated sketch block.

        Use this whenever the user asks, in words, to change what they drew:
        add an object, move something, resize it, remove it, or draw a route.
        Apply the edit to the SKETCH and hand the returned block back, so the
        drawing is updated and stays the source of truth; then build the scene
        from the new sketch. Editing only the 3D scene leaves the drawing stale.

        `sketch` is the `[LAYOUT SKETCH ...]` block from the user's message (an
        empty string starts a fresh one). `ops` is a JSON list, applied in
        order, each one of:

          {"op":"add","kind":"circle","label":"pallet","centre":[1,2],"radius":0.6}
          {"op":"add","kind":"rect","label":"shelf","centre":[0,0],"size":[1.2,0.6]}
          {"op":"add","kind":"arrow","label":"infeed","from":[4,0],"to":[-1,0]}
          {"op":"add","kind":"path","label":"route","points":[[0,0],[2,0],[2,1]]}
          {"op":"move","label":"operator","centre":[0,3]}
          {"op":"resize","label":"cell","size":[8,6]}   (or "radius":0.8 for a circle)
          {"op":"remove","label":"shelf"}
          {"op":"relabel","label":"box","to":"pallet"}

        Edits are by label, the way a person names what they drew. Moving or
        removing a label that is not in the sketch is an error, not a silent
        second copy. Returns the full re-emitted sketch block, byte-compatible
        with what the dashboard itself emits, plus the shapes it now holds.
        Coordinates are metres on the floor (Isaac is Z-up), one-to-one.
        """
        from simliverse_sim.sketch import SketchError, parse_sketch
        from simliverse_sim.sketch import edit_sketch as _edit

        try:
            operations = json.loads(ops) if isinstance(ops, str) else ops
            if not isinstance(operations, list):
                raise SketchError("ops must be a JSON list of edit operations")
            updated = _edit(sketch or "", operations)
            shapes = parse_sketch(updated) if updated.strip() else {}
            return json.dumps(
                {
                    "status": "success",
                    "sketch": updated,
                    "shapes": {
                        k: len(v) for k, v in shapes.items()
                    },
                    "labels": sorted(
                        {str(s.get("label") or "") for kind in shapes.values() for s in kind}
                    ),
                },
                indent=2,
            )
        except (SketchError, KeyError, ValueError, TypeError) as exc:
            return json.dumps({"status": "error", "message": "%s: %s" % (type(exc).__name__, exc)}, indent=2)
