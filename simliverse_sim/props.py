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

"""Real props: conveyor belts, bins, pallets, and the YCB object set.

Asked to put a box on a conveyor belt, an agent used to have no way to find
either. `search_usd` calls a service our API key is not entitled to reach, and
nothing else walked the prop tree — so the honest-looking move was to build a
conveyor out of stretched cubes and call it done. Same fabrication as fitting a
gripper to an arm that has none: invented equipment, no decision point, nothing
in the report saying so.

The index is static and generated, never hand-written. Static because an agent
should not pay a network walk per call and because curated keywords match
"cardboard box" to `003_cracker_box` in a way a directory listing never will.
Generated because the one hand-written catalogue entry this project ever had
said `kuka_iiwa` and pointed at a KR210 asset — a typed index can invent an
asset that does not exist, and a walked one cannot. Regenerate with
`tools/generate_prop_index.py`.

**Every entry records whether it has physics, and that is the point.** Over half
of Isaac's prop library is visual-only: of 169 assets, 61 declare no collider
and no rigid body, and 11 more collide but cannot be moved. Spawned into a
manipulation task those fall through the floor or ignore the gripper, and
nothing anywhere says why. YCB ships every object twice for exactly this reason
— `Axis_Aligned` is render-only and `Axis_Aligned_Physics` is the usable one —
so the plain key always resolves to the more capable variant.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

from ._compat import add_reference, as_vec3, assets_root, get_stage

logger = logging.getLogger(__name__)

_INDEX_PATH = Path(__file__).parent / "data" / "props.json"

#: What you can do with a prop, worst to best.
PHYSICS_KINDS = ("visual", "static", "dynamic")


class PropNotFound(LookupError):
    """No prop matches. A real absence, not a service failure."""


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, Any]:
    try:
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prop index unreadable at %s: %s", _INDEX_PATH, exc)
        return {"assets_root": "", "count": 0, "props": {}}


def list_props(query: str | None = None, *, physics: str | None = None) -> list[dict[str, Any]]:
    """The prop catalogue, for an agent to choose from.

    `query` matches the key and the curated keywords, so "box", "bin" and
    "tote" all reach `small_klt`. `physics="dynamic"` restricts to props that
    can actually be picked up — worth doing for any manipulation task, because
    the majority of this library cannot.
    """
    words = [w for w in (query or "").lower().replace("-", " ").split() if w]
    scored: list[tuple[int, int, str, dict[str, Any]]] = []

    for entry in _index()["props"].values():
        if physics and entry["physics"] != physics:
            continue
        if not words:
            scored.append((0, -PHYSICS_KINDS.index(entry["physics"]), entry["key"], entry))
            continue

        haystack = f"{entry['key']} {' '.join(entry['keywords'])} {entry['category']}"
        # Scored, not all-or-nothing. Requiring every word meant "cardboard box"
        # matched nothing at all -- "cardboard" is in no keyword list -- and an
        # agent reading that empty result concludes no box exists and builds one
        # out of primitives. One unrecognised adjective should not hide the
        # asset the rest of the query clearly names.
        missed = [w for w in words if w not in haystack]
        hits = len(words) - len(missed)
        if not hits:
            continue
        # ...but scoring alone just trades a silent absence for a silent
        # approximation, which is no better. "wooden crate" reaches `small_klt`,
        # a *plastic* bin, on the strength of "crate"; "glass jar" reaches a lab
        # beaker on the strength of "glass". Handed back unlabelled, those get
        # spawned and reported as the thing that was asked for. So every result
        # carries what did not match, and callers can refuse to guess.
        annotated = dict(
            entry,
            match="exact" if not missed else "partial",
            unmatched=missed,
        )
        scored.append(
            (-hits, -PHYSICS_KINDS.index(entry["physics"]), entry["key"], annotated)
        )

    return [entry for *_rank, entry in sorted(scored, key=lambda row: row[:3])]


def find_prop(query: str, *, allow_partial: bool = False) -> dict[str, Any]:
    """The single best match, or raise naming what was searched for.

    Raises on a partial match by default, and that default is the point. The
    library has no wooden crate and no glass jar, but it has a plastic bin and a
    lab beaker, and both score well enough to come back first. Returned quietly,
    they get spawned and then reported as the thing that was asked for — which
    is a substitution nobody decided on, in a scene the user believes contains
    something else.

    So a partial match raises, saying what it found and what it could not
    account for. Pass `allow_partial=True` when the caller has already put the
    substitution to the user, or use `list_props(query)` to show the options.
    """
    exact = _index()["props"].get(query.strip().lower())
    if exact:
        return dict(exact, match="exact", unmatched=[])

    matches = list_props(query)
    if not matches:
        raise PropNotFound(
            f"No prop matches {query!r}. `list_props()` shows all "
            f"{_index().get('count', 0)}; if none of them is what was asked for, "
            f"say so rather than standing in a primitive — swapping the asset "
            f"changes what the scene is made of."
        )

    best = matches[0]
    if best["match"] == "partial" and not allow_partial:
        alternatives = ", ".join(m["key"] for m in matches[:4])
        raise PropNotFound(
            f"No prop matches all of {query!r}. The closest is {best['key']!r} "
            f"({best['path']}), which does not account for "
            f"{', '.join(repr(w) for w in best['unmatched'])}. Nearest: "
            f"{alternatives}. Substituting one of these changes what the scene "
            f"is made of, so it is the user's call — ask, or pass "
            f"allow_partial=True once they have chosen."
        )
    return best


def spawn_prop(
    query: str,
    *,
    allow_partial: bool = False,
    prim_path: str | None = None,
    position: Any = (0.0, 0.0, 0.0),
    scene: Any = None,
) -> dict[str, Any]:
    """Reference a real prop onto the stage. Returns its index entry plus the path.

    Warns when the chosen asset has no rigid body, because that is the failure
    that otherwise surfaces as a grasp which mysteriously never holds.
    """
    from .scene import Scene as _Scene

    entry = find_prop(query, allow_partial=allow_partial)
    scene = scene or _Scene.get()
    prim_path = prim_path or f"/World/{entry['key']}"

    add_reference(assets_root() + entry["path"], prim_path)

    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(get_stage().GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*as_vec3(position, name="position")))

    if entry["physics"] != "dynamic":
        logger.warning(
            "%s is a %s asset: it declares %s. It will not behave like an object "
            "a robot can pick up. `list_props(%r, physics='dynamic')` lists the "
            "ones that can.",
            entry["key"],
            entry["physics"],
            "no collider and no rigid body" if entry["physics"] == "visual"
            else "a collider but no rigid body",
            query,
        )

    return {**entry, "prim_path": prim_path}


def verify_index() -> dict[str, Any]:
    """Check the committed index against the asset server it was generated from.

    A static index is only safe while it is true. Isaac's asset root is
    versioned, so a release bump moves every path at once and the index would
    otherwise keep answering confidently with paths that no longer resolve —
    the same silent-staleness failure the generated index exists to avoid.
    """
    live = assets_root()
    recorded = _index().get("assets_root", "")
    ok = bool(live) and live == recorded
    return {
        "current": ok,
        "index_assets_root": recorded,
        "live_assets_root": live,
        "count": _index().get("count", 0),
        "stale_because": None if ok else (
            f"index was generated against {recorded or '<unknown>'} but this "
            f"session serves assets from {live}. Paths may not resolve; "
            f"regenerate with tools/generate_prop_index.py."
        ),
    }
