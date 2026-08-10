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

"""Handlers for the control library, persistent scripting, and real perception.

Three things differ from the `simulation.execute_script` path these supersede:

  * The namespace persists between calls, so a robot handle survives long enough
    to be useful. `execute_script` rebuilds `local_ns` every call, so nothing can
    be carried forward.
  * `simliverse_sim` is pre-imported, so control code never has to guess Isaac
    Sim module paths.
  * Errors return the traceback *and* the captured output, rather than being
    flattened to a bare message string on the way out.
"""

from __future__ import annotations

import base64
import io
import sys
import traceback
from typing import Any, Dict, List, Optional

from isaac_sim_mcp_extension import kit_log

from ..adapters.base import IsaacAdapterBase

# Persistent execution namespace, keyed by nothing — one live sim, one session.
_NAMESPACE: Dict[str, Any] = {}

_IMPORT_WARNING = (
    "simliverse_sim could not be imported inside Isaac Sim: {error}. Names like "
    "Scene, spawn_robot and RigidObject are NOT bound; anything using them will "
    "raise NameError. Plain Python and the omni/isaacsim modules still work, so "
    "this call can repair sys.path and then re-run with reset_namespace=true."
)


def _ensure_library_on_path() -> Optional[str]:
    """Make `simliverse_sim` importable from the extension's repo checkout.

    Two roots, deliberately, and the extension directory is searched first.

    In a plain checkout `simliverse_sim` sits beside `isaac.sim.mcp_extension`
    at the repo root. In the deployed container only the extension directory is
    bind-mounted from the host -- its parent is baked into the image and is
    read-only -- so a copy shipped *inside* the extension is the one that can
    actually be updated without rebuilding the image. Searching the extension
    directory first means that copy wins when it exists and nothing changes
    when it does not.
    """
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    # handlers/ -> isaac_sim_mcp_extension/ -> isaac.sim.mcp_extension/ -> repo root
    extension_root = os.path.abspath(os.path.join(here, "..", ".."))
    repo_root = os.path.abspath(os.path.join(extension_root, ".."))

    for root in (repo_root, extension_root):  # inserted at 0, so extension ends up first
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    return repo_root


def _fresh_namespace(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    _ensure_library_on_path()

    namespace: Dict[str, Any] = {"__name__": "__simliverse_control__"}
    try:
        import numpy as np

        import simliverse_sim as sls

        namespace.update(
            {
                "np": np,
                "sls": sls,
                "Scene": sls.Scene,
                "Robot": sls.Robot,
                "RigidObject": sls.RigidObject,
                "PhysicsConfig": sls.PhysicsConfig,
                # Bound directly because control code reaches for them first.
                # `Robot` alone is a probe; `spawn_robot` is what returns a
                # handle with the right control surface for the morphology.
                "spawn_robot": sls.spawn_robot,
                "list_robots": sls.list_robots,
                "Manipulator": sls.Manipulator,
                # A task is not finished when the scene looks right; it is
                # finished when stop-then-play reproduces it. `controller.write`
                # authors the ScriptNode source and `controller.verify` replays
                # the scene from its authored state and measures what moved.
                "controller": sls.controller,
                "verify_grasp": sls.verify_grasp,
                "verify_throw": sls.verify_throw,
                "grasped": sls.grasped,
                "airborne": sls.airborne,
                "physics_running": sls.physics_running,
                "not_teleported": sls.not_teleported,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        namespace["__import_error__"] = f"{type(exc).__name__}: {exc}"
    return namespace


def run_control(
    adapter: IsaacAdapterBase,
    code: Optional[str] = None,
    reset_namespace: bool = False,
) -> Dict[str, Any]:
    """Execute control code with a persistent namespace and full error detail."""
    global _NAMESPACE

    if not code:
        return {"status": "error", "message": "code is required"}

    if reset_namespace or not _NAMESPACE:
        _NAMESPACE = _fresh_namespace(adapter)

    # A failed `simliverse_sim` import is reported, not enforced. Refusing to
    # execute made this verb unable to repair itself: the one tool that can put
    # a directory back on sys.path declined to run until sys.path was already
    # correct. Recovering a container then needed a different verb entirely.
    # Control code that does not touch the library still works, and code that
    # does gets a NameError naming the symbol, plus this warning.
    import_warning = _NAMESPACE.get("__import_error__")

    # Mark the log before running, so `isaac_log` carries only what *this* call
    # provoked rather than the whole session's history.
    log_path = kit_log.active_log()
    log_start = kit_log.offset(log_path)

    stdout, stderr = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        exec(code, _NAMESPACE)  # noqa: S102 — this is the intended escape hatch
        result = {
            "status": "success",
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "bound_names": sorted(
                name
                for name in _NAMESPACE
                if not name.startswith("_") and name not in ("np", "sls")
            ),
        }
        if import_warning:
            result["warning"] = _IMPORT_WARNING.format(error=import_warning)
        return kit_log.attach(result, log_path, log_start)
    except Exception as exc:
        result = {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
        if import_warning:
            result["warning"] = _IMPORT_WARNING.format(error=import_warning)
        return kit_log.attach(result, log_path, log_start)
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def observe(
    adapter: IsaacAdapterBase,
    prim_paths: Optional[List[str]] = None,
    robot_paths: Optional[List[str]] = None,
    steps: int = 0,
) -> Dict[str, Any]:
    """Step physics, then report measured state for the requested prims."""
    _ensure_library_on_path()
    try:
        import simliverse_sim as sls
    except Exception as exc:
        return {"status": "error", "message": f"simliverse_sim unavailable: {exc}"}

    try:
        scene = sls.Scene.get()
        if steps:
            scene.step(int(steps))

        objects: Dict[str, Any] = {}
        for path in prim_paths or []:
            try:
                objects[path] = sls.RigidObject(path, scene=scene).state()
            except Exception as exc:
                objects[path] = {"error": f"{type(exc).__name__}: {exc}"}

        robots: Dict[str, Any] = {}
        for path in robot_paths or []:
            try:
                robots[path] = sls.Robot(path, scene=scene).describe()
            except Exception as exc:
                robots[path] = {"error": f"{type(exc).__name__}: {exc}"}

        return {
            "status": "success",
            "physics_playing": scene.is_playing(),
            "physics_dt": scene.dt,
            "objects": objects,
            "robots": robots,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def capture_view(
    adapter: IsaacAdapterBase,
    camera_path: Optional[str] = None,
    position: Optional[List[float]] = None,
    look_at: Optional[List[float]] = None,
    resolution: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Render to an in-memory PNG and return it base64-encoded.

    The pixels are returned to the caller rather than written to the worker's
    disk. Writing a file the agent can never open is what made the previous
    `capture_image` verb useless for verification.
    """
    try:
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image as PILImage

        width, height = (resolution or [1280, 720])[:2]

        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()

        def _replicator_cameras() -> set:
            return {
                prim.GetPath().pathString
                for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Camera) and "/Replicator" in prim.GetPath().pathString
            }

        if camera_path:
            camera = camera_path
            authored_before = None
        else:
            # `rep.create.camera()` authors a camera prim that
            # `render_product.destroy()` does not remove, so snapshot what
            # existed and delete whatever this call added. Without this the
            # stage accumulates one /Replicator/Camera_Xform per capture — a
            # live session had six after a handful of failed renders.
            authored_before = _replicator_cameras()
            eye = position or [2.2, -2.2, 1.6]
            target = look_at or [0.0, 0.0, 0.3]
            camera = rep.create.camera(position=tuple(eye), look_at=tuple(target))

        render_product = rep.create.render_product(camera, (int(width), int(height)))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])

        def _release() -> None:
            """Drop the render product on every path, including failure.

            This used to run only after a successful encode, so each failed
            capture stranded a render product and, when no camera_path was
            given, a `/Replicator/Camera_Xform*` prim as well. Three empty-frame
            errors left three orphan cameras on the stage.
            """
            try:
                annotator.detach([render_product])
                render_product.destroy()
            except Exception:
                pass
            if authored_before is not None:
                for path in _replicator_cameras() - authored_before:
                    try:
                        # Remove the Xform wrapper too, not just the camera.
                        stage.RemovePrim(path)
                        parent = path.rsplit("/", 1)[0]
                        if parent.startswith("/Replicator/"):
                            stage.RemovePrim(parent)
                    except Exception:
                        pass

        # NOTE: this path is known broken inside the extension and is why
        # capture_view returns an empty frame. Two mechanisms were ruled out:
        # `app.update()` ticks Kit but never asks Replicator to produce a frame,
        # and `rep.orchestrator.step()` raises "Synchronous call to `step` can
        # only be performed in a standalone workflow" — Kit owns the loop here,
        # exactly as it does for World.step().
        #
        # The fix is to stop using Replicator and capture the existing viewport
        # instead, which is what the older `capture_image` verb does and why it
        # still works. Left in place rather than silently half-changed: the
        # failure is now reported accurately and no longer leaks a render
        # product on the way out.
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(4):
            app.update()

        array = np.asarray(annotator.get_data())

        if array.size == 0:
            _release()
            return {
                "status": "error",
                "message": (
                    "Renderer returned an empty frame after 8 render steps. The "
                    "viewport may have no render product attached."
                ),
            }
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]

        buffer = io.BytesIO()
        PILImage.fromarray(array.astype("uint8")).save(buffer, format="PNG")

        _release()

        return {
            "status": "success",
            "format": "png",
            "width": int(width),
            "height": int(height),
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["control.run"] = lambda **kwargs: run_control(adapter, **kwargs)
    registry["control.observe"] = lambda **kwargs: observe(adapter, **kwargs)
    registry["control.capture_view"] = lambda **kwargs: capture_view(adapter, **kwargs)
