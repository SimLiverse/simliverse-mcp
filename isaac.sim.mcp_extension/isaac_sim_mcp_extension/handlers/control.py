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

from ..adapters.base import IsaacAdapterBase

# Persistent execution namespace, keyed by nothing — one live sim, one session.
_NAMESPACE: Dict[str, Any] = {}


def _ensure_library_on_path() -> Optional[str]:
    """Make `simliverse_sim` importable from the extension's repo checkout."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    # handlers/ -> isaac_sim_mcp_extension/ -> isaac.sim.mcp_extension/ -> repo root
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
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

    if "__import_error__" in _NAMESPACE:
        return {
            "status": "error",
            "message": (
                "simliverse_sim could not be imported inside Isaac Sim: "
                f"{_NAMESPACE['__import_error__']}. Check that the repo root is on "
                "sys.path and that the extension is loaded from a full checkout."
            ),
        }

    stdout, stderr = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        exec(code, _NAMESPACE)  # noqa: S102 — this is the intended escape hatch
        return {
            "status": "success",
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "bound_names": sorted(
                name
                for name in _NAMESPACE
                if not name.startswith("_") and name not in ("np", "sls")
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
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

        if camera_path:
            camera = camera_path
        else:
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
