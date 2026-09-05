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

"""Scene management command handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..adapters.base import IsaacAdapterBase

_discovered_envs: Optional[Dict[str, Dict[str, str]]] = None


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["scene.get_info"] = lambda **p: get_info(adapter, **p)
    registry["scene.create_physics"] = lambda **p: create_physics(adapter, **p)
    registry["scene.clear"] = lambda **p: clear(adapter, **p)
    registry["scene.list_prims"] = lambda **p: list_prims(adapter, **p)
    registry["scene.get_prim_info"] = lambda **p: get_prim_info(adapter, **p)
    registry["scene.list_environments"] = lambda **p: list_environments(adapter, **p)
    registry["scene.load_environment"] = lambda **p: load_environment(adapter, **p)


def get_info(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        stage = adapter.get_stage()
        assets_root = adapter.get_assets_root_path()
        prim_count = len(list(stage.TraverseAll()))
        stage_path = stage.GetRootLayer().realPath
        return {
            "status": "success",
            "message": "pong",
            "assets_root_path": assets_root,
            "stage_path": stage_path,
            "prim_count": prim_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_physics(
    adapter: IsaacAdapterBase, gravity: Optional[Sequence[float]] = None, scene_name: str = "PhysicsScene"
) -> Dict[str, Any]:
    try:
        scene_path = adapter.create_physics_scene(gravity=gravity, scene_name=scene_name)
        # Create ground plane with collision so objects don't fall through
        floor_path = "/World/groundPlane"
        adapter.create_prim(floor_path, "Plane")
        from pxr import UsdPhysics

        stage = adapter.get_stage()
        gp = stage.GetPrimAtPath(floor_path)
        if gp.IsValid() and not gp.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(gp)
        return {"status": "success", "message": f"Physics scene created at {scene_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear(adapter: IsaacAdapterBase, keep_physics: bool = True) -> Dict[str, Any]:
    """Empty the stage while leaving it *buildable*.

    The first version of this deleted every root prim, `/World` included, and
    reported success -- after which the very next `spawn_robot` died on
    `'NoneType' object has no attribute 'create_articulation_view'`: the World
    singleton still held handles into the deleted physics scene, and nothing
    short of restarting Kit repaired it. "Cleared" has to mean "ready for the
    next cell", which is a different thing from "empty".

    So, in order:

    1. Stop the timeline. Deleting prims PhysX is mid-step over is a crash
       that names nothing the caller did.
    2. Rebind the viewport to the stock perspective camera. Deleting the
       camera a viewport is looking through fails quietly, which is how one
       stray camera survived every "successful" clear of this stage.
    3. Release our sensors, stop the Replicator orchestrator, and remove
       render products whose camera lives under `/World`. A live sensor
       re-creates its camera prim within five frames of deletion, and a
       Replicator graph does the same from in-memory state -- one camera
       survived every "successful" clear this way until its render product
       under `/Render` (which we keep) was found and removed too.
    4. Delete the *children* of `/World`, never `/World` itself, and keep the
       physics scene, its materials and the ground plane -- they are
       configuration, and handles into them stay live across cells.
    5. Delete stray root-level prims (robots authored at `/`), sparing Kit's
       own.

    `keep_physics=False` removes the physics configuration too; the next
    build must then recreate it (`simliverse_sim.Scene.configure_physics`
    does) before anything dynamic will work.
    """
    try:
        removed: list = []

        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass

        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = "/OmniverseKit_Persp"
        except Exception:
            pass

        try:
            from .sensors import release_all_sensors

            removed.extend(release_all_sensors())
        except Exception:
            pass

        try:
            import omni.replicator.core as rep

            rep.orchestrator.stop()
        except Exception:
            pass

        stage = adapter.get_stage()

        # Render products anchored to cameras we are about to delete. These
        # live under /Render, which survives the clear -- and while one
        # exists, Replicator re-authors its camera faster than a delete.
        try:
            for prim in list(stage.Traverse()):
                if prim.GetTypeName() != "RenderProduct":
                    continue
                rel = prim.GetRelationship("camera")
                targets = rel.GetTargets() if rel else []
                if any(str(t).startswith("/World/") for t in targets):
                    path = str(prim.GetPath())
                    stage.RemovePrim(path)
                    removed.append(path)
        except Exception:
            pass

        world_keep = {"PhysicsScene", "PhysicsMaterials", "GroundPlane"} if keep_physics else set()
        world = stage.GetPrimAtPath("/World")
        if world and world.IsValid():
            for child in list(world.GetChildren()):
                if child.GetName() in world_keep:
                    continue
                path = str(child.GetPath())
                if child.GetTypeName() == "OmniGraph":
                    # An action graph removed with a plain RemovePrim came
                    # back: /World/TaskGraph_x8 from one layout was still
                    # on the stage, nodes ticking, two layouts later. Kit's
                    # own delete command tells OmniGraph about it.
                    try:
                        import omni.kit.commands

                        omni.kit.commands.execute("DeletePrims", paths=[path], destructive=True)
                    except Exception:  # noqa: BLE001 -- fall through to the plain delete
                        pass
                if stage.GetPrimAtPath(path).IsValid():
                    adapter.delete_prim(path)
                removed.append(path)

        keep_paths = {
            "/World",
            "/OmniverseKit_Persp",
            "/OmniverseKit_Front",
            "/OmniverseKit_Top",
            "/OmniverseKit_Right",
            "/Render",
            "/Environment",
        }
        for child in list(stage.GetPseudoRoot().GetChildren()):
            path = str(child.GetPath())
            if path in keep_paths:
                continue
            if keep_physics and "Physics" in path:
                continue
            adapter.delete_prim(path)
            removed.append(path)

        return {"status": "success", "message": "Scene cleared", "removed": removed}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_prims(adapter: IsaacAdapterBase, root_path: str = "/", prim_type: Optional[str] = None) -> Dict[str, Any]:
    try:
        prims = adapter.list_prims(root_path=root_path, prim_type=prim_type)
        return {"status": "success", "prims": prims}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_prim_info(adapter: IsaacAdapterBase, prim_path: str = "/") -> Dict[str, Any]:
    try:
        info = adapter.get_prim_info(prim_path)
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_env_library(adapter: IsaacAdapterBase) -> Dict[str, Dict[str, str]]:
    global _discovered_envs
    if _discovered_envs is not None:
        return _discovered_envs
    try:
        envs = adapter.discover_environments()
        if envs:
            _discovered_envs = envs
            print(f"Discovered {len(envs)} environments from asset server")
            return _discovered_envs
    except Exception as e:
        print(f"Environment discovery failed: {e}")
    _discovered_envs = {}
    return _discovered_envs


def list_environments(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    library = _get_env_library(adapter)
    return {"status": "success", "environment_count": len(library), "environments": library}


def load_environment(
    adapter: IsaacAdapterBase, environment: Optional[str] = None, prim_path: str = "/Environment"
) -> Dict[str, Any]:
    try:
        if not environment:
            return {
                "status": "error",
                "message": "environment is required. Use scene.list_environments to see options.",
            }

        library = _get_env_library(adapter)
        q = environment.lower().strip()

        # Exact match
        match = library.get(q)

        # Fuzzy match
        if not match:
            for key, info in library.items():
                if q in key or q in info.get("description", "").lower():
                    match = info
                    break

        if not match:
            available = list(library.keys())[:15]
            return {"status": "error", "message": f"Environment '{environment}' not found. Options: {available}"}

        assets_root = adapter.get_assets_root_path()
        full_path = assets_root + match["asset_path"]
        adapter.load_environment(full_path, prim_path)
        return {"status": "success", "message": f"Loaded environment: {match['description']}", "prim_path": prim_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
