# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
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

"""Robot creation and control command handlers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..adapters.base import IsaacAdapterBase
from .control import _ensure_library_on_path

# Hardcoded fallback — used only if live discovery fails.
# Keys are lowercase robot names, asset_path is relative to the assets root.
def _get_robot_library(adapter: IsaacAdapterBase) -> Dict[str, Dict[str, str]]:
    """The robot catalogue, from `simliverse_sim` — the only implementation.

    This used to walk the asset server itself, with its own cache and its own
    hardcoded fallback list, duplicating `simliverse_sim.robots.library`. Two
    walks meant two answers: this one listed `.thumbs` directories and
    `*.thumb.usd` previews as robots, because it filtered nothing, while the
    library filtered them out. An agent could therefore be offered a "robot"
    that `Robot.spawn()` would refuse to load.

    One implementation, one answer. The library also merges a seed catalogue
    that carries morphology and RMPflow config, so those surface here too —
    `cartesian_control` tells an agent whether `move_ee_to` will work before it
    spawns anything.
    """
    _ensure_library_on_path()
    from simliverse_sim.robots.library import discover_robots

    return {
        key: {
            "asset_path": asset.asset_path,
            "description": asset.description,
            "manufacturer": asset.manufacturer or asset.description.split(" ")[0],
            "morphology": asset.morphology.value,
            "cartesian_control": "yes" if asset.motion_config else "unknown",
        }
        for key, asset in discover_robots().items()
    }


def list_robots(
    adapter: IsaacAdapterBase,
    search: Optional[str] = None,
    manufacturer: Optional[str] = None,
) -> Dict[str, Any]:
    """List available robots, compactly unless the query is already narrow.

    Discovery finds ~200 robots on the 6.0 asset server. Returning all of them
    with a path, a description and a manufacturer each is several thousand
    tokens, resent on every subsequent turn because it lands in the transcript —
    for a question usually answered by one key.

    So: a narrow result comes back in full, a broad one comes back as an index
    of keys plus manufacturer counts, and the caller narrows. `search` matches
    the key, description or manufacturer.
    """
    library = _get_robot_library(adapter)

    def matches(key: str, spec: Dict[str, str]) -> bool:
        if manufacturer and manufacturer.lower() not in str(spec.get("manufacturer", "")).lower():
            return False
        if not search:
            return True
        needle = search.lower().replace("_", "").replace("-", "")
        haystack = " ".join(
            [key, str(spec.get("description", "")), str(spec.get("manufacturer", ""))]
        ).lower().replace("_", "").replace("-", "")
        return needle in haystack

    selected = {k: v for k, v in library.items() if matches(k, v)}

    if len(selected) <= _DETAIL_THRESHOLD:
        return {
            "status": "success",
            "robot_count": len(selected),
            "total_available": len(library),
            "robots": selected,
        }

    by_maker: Dict[str, int] = {}
    for spec in selected.values():
        maker = str(spec.get("manufacturer") or "unknown")
        by_maker[maker] = by_maker.get(maker, 0) + 1

    return {
        "status": "success",
        "robot_count": len(selected),
        "total_available": len(library),
        "keys": sorted(selected),
        "manufacturers": dict(sorted(by_maker.items(), key=lambda kv: -kv[1])),
        "hint": (
            "Too many to describe in full. Re-call with search= or "
            "manufacturer= to get asset paths and descriptions, e.g. "
            "list_available_robots(search='franka')."
        ),
    }


def refresh_robots(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    """Force re-scan the asset server for available robots."""
    _ensure_library_on_path()
    from simliverse_sim.robots.library import discover_robots

    discover_robots(refresh=True)
    library = _get_robot_library(adapter)
    return {
        "status": "success",
        "message": f"Refreshed robot library, found {len(library)} robots",
        "robot_count": len(library),
    }


def get_info(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        info = adapter.get_robot_joint_info(prim_path)
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def set_joints(
    adapter: IsaacAdapterBase,
    prim_path: Optional[str] = None,
    joint_positions: Optional[Sequence[float]] = None,
    joint_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    try:
        if not prim_path or joint_positions is None:
            return {"status": "error", "message": "prim_path and joint_positions are required"}
        adapter.set_joint_positions(prim_path, joint_positions, joint_indices)
        return {"status": "success", "message": f"Set joint positions on {prim_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_joints(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        positions = adapter.get_joint_positions(prim_path)
        return {"status": "success", "joint_positions": positions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _find_robot(adapter: IsaacAdapterBase, query: str) -> Optional[Dict[str, Any]]:
    """Find a robot by key, then by partial match on key/description/manufacturer."""
    library = _get_robot_library(adapter)
    q = query.lower().strip()

    if q in library:
        return {"key": q, **library[q]}

    matches = []
    for key, info in library.items():
        searchable = f"{key} {info.get('description', '')} {info.get('manufacturer', '')}".lower()
        if q in searchable:
            matches.append({"key": key, **info})

    if not matches:
        return None
    # Shortest key containing the query is the closest match.
    matches.sort(key=lambda m: len(m["key"]))
    return matches[0]


def create(
    adapter: IsaacAdapterBase,
    robot_type: str = "franka",
    position: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
    prim_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        match = _find_robot(adapter, robot_type)
        if not match:
            available = list(_get_robot_library(adapter).keys())[:20]
            return {
                "status": "error",
                "message": (
                    f"Robot '{robot_type}' not found. Try robots.list to see available "
                    f"robots. Some options: {available}"
                ),
            }

        asset_path = adapter.get_assets_root_path() + match["asset_path"]
        if prim_path is None:
            prim_path = f"/World/{name or match['key'].capitalize()}"
        adapter.add_reference_to_stage(asset_path, prim_path)
        if position:
            adapter.create_xform_prim(prim_path).set_world_pose(position=np.array(position))

        result = {
            "status": "success",
            "message": f"Created {match['description']} robot",
            "prim_path": prim_path,
            "robot_key": match["key"],
        }
        try:
            info = adapter.get_robot_joint_info(prim_path)
            result["joint_names"] = info.get("joint_names", [])
            result["num_dof"] = info.get("num_dof", 0)
        except Exception:
            pass
        try:
            # Zero stiffness and zero damping means the drive is off and the
            # joint will silently ignore every command. Report, never repair.
            warnings = adapter.get_joint_config(prim_path).get("warnings", [])
            if warnings:
                result["warnings"] = warnings
        except Exception:
            pass
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["robots.create"] = lambda **p: create(adapter, **p)
    registry["robots.list"] = lambda **p: list_robots(adapter, **p)
    registry["robots.refresh"] = lambda **p: refresh_robots(adapter, **p)
    registry["robots.get_info"] = lambda **p: get_info(adapter, **p)
    registry["robots.set_joints"] = lambda **p: set_joints(adapter, **p)
    registry["robots.get_joints"] = lambda **p: get_joints(adapter, **p)
