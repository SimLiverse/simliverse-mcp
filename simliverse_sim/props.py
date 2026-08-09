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

    # Clear the path first. Referencing over a prim that already exists keeps
    # the old prim's applied schemas, and props do not all carry physics in the
    # same place: `004_sugar_box` puts RigidBodyAPI on its default prim,
    # `basic_block` puts it on a child. Spawn the first, then respawn the second
    # at the same path, and the leftover API leaves a rigid body containing a
    # rigid body. PhysX does not allow that, so the object simply never falls —
    # measured, on a block that sat at z=0.30 above the floor with zero
    # velocity while reporting itself dynamic.
    stage = get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    add_reference(assets_root() + entry["path"], prim_path)

    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(get_stage().GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*as_vec3(position, name="position")))

    # KNOWN BUG — props referenced here do not fall. Reproducible in a clean
    # scene where `scene.spawn_rigid` drops a cube from 0.60 m and it rests at
    # 0.05 m. PhysX says why:
    #
    #   The rigid body at /World/Box_0 has a possibly invalid inertia tensor of
    #   {1.0, 1.0, 1.0} and a negative mass, small sphere approximated inertia
    #   was used. Either specify correct values in the mass properties, or add
    #   collider(s) to any subordinate prims.
    #
    # The prim this path creates carries PhysicsRigidBodyAPI, PhysxRigidBodyAPI
    # and PhysxContactReportAPI, none of which its source asset declares —
    # `basic_block.usd` puts RigidBodyAPI only on `/Root/Cube`. So it becomes a
    # rigid body with no collider of its own, hence no computable mass. That is
    # not one prop misbehaving: a single degenerate body stops dynamics for the
    # entire scene, and everything afterwards reports plausible numbers that
    # mean nothing.
    #
    # Ruled out by measurement, so as not to be retried:
    #   * stale prim paths — a never-used path behaves identically
    #   * constructing RigidObject on the wrapper — happens without it
    #   * applying MassAPI to prims tagged RigidBodyAPI — no effect, and
    #     applying it to the wrapper makes props fall through the floor to
    #     -45 m, which confirms the wrapper is wrongly a body rather than
    #     fixing anything
    #
    # Next step: bisect this function. Call `add_reference_to_stage` directly
    # and read the prim's applied schemas immediately, before anything else
    # touches it, to find what applies RigidBodyAPI to the wrapper.
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

    # Returned, not only logged. The physics warning above goes to a logger and
    # I missed my own — twice — because a log line is something you have to
    # think to read. An overlap is a fact about the scene that was just built,
    # so it comes back with the thing that built it.
    result = {**entry, "prim_path": prim_path}
    overlaps = _overlapping_robots(prim_path)
    if overlaps:
        result["overlaps"] = overlaps
        for hit in overlaps:
            logger.warning(
                "%s at %s intersects %s. The prop is %.2f m across and is placed "
                "by its centre, so its near edge sits behind the robot's base. "
                "PhysX reports invalid transforms on the arm links when this "
                "happens and the scene is unusable without saying so. Offset it "
                "by at least half its extent plus the base radius.",
                entry["key"], prim_path, hit["robot"],
                max(entry.get("extent") or [0.0]),
            )
    return result


def _world_bounds(prim_path: str) -> tuple[Any, Any] | None:
    """Axis-aligned world bounds of a prim, or None if it has no extent."""
    from pxr import Usd, UsdGeom

    prim = get_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return rng.GetMin(), rng.GetMax()


def _overlapping_robots(prim_path: str) -> list[dict[str, Any]]:
    """Robots whose base this prop has been placed on top of.

    A prop is positioned by its centre and most of them are large: a pallet is
    1.21 m long, so centring one 0.5 m in front of an arm puts its near edge at
    -0.055 — behind the robot, with the base inside the pallet. PhysX then
    reported invalid transforms on seven arm links and the scene was quietly
    unusable.

    Nothing about that was hard to catch. The extent is in the index and the
    robot's position is on the stage; the check is two axis-aligned boxes. It
    belongs here rather than in whoever is calling, because a rule that has to
    be remembered is a rule that gets skipped — this one was, by the author of
    the index it needed.
    """
    from pxr import UsdPhysics

    bounds = _world_bounds(prim_path)
    if bounds is None:
        return []
    low, high = bounds

    hits: list[dict[str, Any]] = []
    for prim in get_stage().Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        robot_path = str(prim.GetPath())
        if robot_path.startswith(prim_path):
            continue
        robot_bounds = _world_bounds(robot_path)
        if robot_bounds is None:
            continue
        rlow, rhigh = robot_bounds
        if all(low[i] <= rhigh[i] and high[i] >= rlow[i] for i in range(3)):
            hits.append({
                "robot": robot_path,
                "prop_bounds": [[round(float(v), 3) for v in low],
                                [round(float(v), 3) for v in high]],
                "robot_bounds": [[round(float(v), 3) for v in rlow],
                                 [round(float(v), 3) for v in rhigh]],
            })
    return hits


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
