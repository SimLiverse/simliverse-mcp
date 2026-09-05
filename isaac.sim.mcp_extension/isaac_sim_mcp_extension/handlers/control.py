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
                name for name in _NAMESPACE if not name.startswith("_") and name not in ("np", "sls")
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


def _aim_camera(stage: Any, path: str, eye: List[float], target: List[float]) -> str:
    """Author a camera at `eye` looking at `target`. Returns its prim path.

    Reuses whatever transform ops the prim already has. The obvious
    `ClearXformOpOrder()` then `AddTranslateOp()` raises "The xformOp
    'xformOp:translate' already exists in xformOpOrder" on any prim that has
    been transformed before -- the same failure that made `transform_object`
    unusable on /OmniverseKit_Persp.
    """
    import math

    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    xformable = UsdGeom.Xformable(camera.GetPrim())
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    translate = ops.get("xformOp:translate") or xformable.AddTranslateOp()
    rotate = ops.get("xformOp:rotateXYZ") or xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)

    dx, dy, dz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    # A USD camera looks down its own -Z. Deriving the rotation from the
    # eye->target vector means the caller never supplies Euler angles to point a
    # camera at a thing, which is the step they would get wrong.
    yaw = math.degrees(math.atan2(dy, dx)) - 90.0
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    translate.Set(Gf.Vec3d(*eye))
    rotate.Set(Gf.Vec3f(90.0 + pitch, 0.0, yaw))
    return path


def capture_view(
    adapter: IsaacAdapterBase,
    camera_path: Optional[str] = None,
    position: Optional[List[float]] = None,
    look_at: Optional[List[float]] = None,
    resolution: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Render to an in-memory PNG and return it base64-encoded.

    The pixels come back to the caller rather than being written to the worker's
    disk. A file the agent can never open is what made `capture_image` useless
    for verification: it reports `status: success` and a shape, and the agent
    still cannot see anything.

    ## Why this drives the viewport rather than a private render product

    Two earlier approaches both returned an empty frame, for the same reason --
    nothing had a render product attached:

    - **Replicator.** `app.update()` ticks Kit but never asks Replicator for a
      frame, and `rep.orchestrator.step()` raises "Synchronous call to `step`
      can only be performed in a standalone workflow". Kit owns the loop here.
    - **`isaacsim.sensors.camera.Camera`.** `get_rgba()` returns shape `[0]`
      even after `initialize()` and six ticks.

    The viewport is the one surface certainly rendering -- it is what the WebRTC
    stream shows -- so it is the one to read.

    ## Why it switches cameras instead of moving the current one

    Moving `/OmniverseKit_Persp` looks like it works and silently does not: the
    streaming client's camera manipulator re-asserts the pose every frame, so
    every capture came back as the operator's view no matter where the agent
    aimed. Three angles produced three byte-identical PNGs. Pointing the
    viewport at a camera the manipulator is not driving is what makes a
    different angle actually render -- verified as five distinct frames across
    user view, iso, top, side, and restored.

    The operator's stream does swing to the agent's viewpoint for the moment of
    the shot. That is the honest cost; the view is restored afterwards.
    """
    import base64 as _b64
    import ctypes
    import io as _io

    temp_camera: Optional[str] = None
    original_camera: Optional[str] = None
    viewport = None
    stage = None

    try:
        import omni.kit.app
        import omni.usd
        from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport
        from PIL import Image as PILImage

        viewport = get_active_viewport()
        if viewport is None:
            return {"status": "error", "message": "No active viewport to capture."}

        app = omni.kit.app.get_app()
        stage = omni.usd.get_context().get_stage()
        original_camera = str(viewport.camera_path)

        # Sweep any camera a previous capture left behind, before making a new
        # one. The `finally` below removes it on the way out, but cleanup that
        # only runs on exit is one missed path away from littering somebody's
        # stage -- and a stray prim in their outliner is a thing they have to
        # wonder about. Removing on entry means at most one can ever exist and
        # the next capture clears it, whatever happened last time.
        try:
            from pxr import Sdf as _Sdf

            _stale = stage.GetPrimAtPath("/World/__capture_view_cam")
            if _stale and _stale.IsValid() and original_camera != "/World/__capture_view_cam":
                stage.RemovePrim(_Sdf.Path("/World/__capture_view_cam"))
        except Exception:
            pass

        # `camera_path` says WHERE to put the camera prim; `position` and
        # `look_at` say where to aim it. Both together is the natural way to
        # ask for a named, reusable viewpoint -- and the old code took the
        # first branch and dropped the aim on the floor, pointing the viewport
        # at a prim that, first time round, did not exist yet. A dangling
        # camera renders nothing, so every capture died on the 40-frame
        # timeout with a message about the capture rather than the camera.
        # Silently ignoring an argument is the same failure that cost a whole
        # session of looking at the default view and believing it.
        if position or look_at:
            target_path = camera_path or "/World/__capture_view_cam"
            _aim_camera(
                stage,
                target_path,
                [float(v) for v in (position or [10.0, -10.0, 8.0])],
                [float(v) for v in (look_at or [0.0, 0.0, 0.0])],
            )
            if not camera_path:
                temp_camera = target_path
            viewport.camera_path = target_path
        elif camera_path:
            prim = stage.GetPrimAtPath(camera_path)
            if not (prim and prim.IsValid()):
                return {
                    "status": "error",
                    "message": (
                        f"No camera at {camera_path}. Either pass position and "
                        f"look_at to create one there, or name a camera that "
                        f"exists."
                    ),
                }
            viewport.camera_path = camera_path

        if camera_path or temp_camera:
            # Let the switch land before asking for a frame, or the shot is from
            # whichever camera the viewport had a moment ago.
            for _ in range(10):
                app.update()

        captured: Dict[str, Any] = {}

        def _on_capture(buffer: Any, buffer_size: int, width: int, height: int, fmt: Any) -> None:
            try:
                # A PyCapsule holding a raw pointer, valid only for the duration
                # of this callback -- so it is copied here, not stashed.
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                    ctypes.py_object,
                    ctypes.c_char_p,
                ]
                pointer = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                raw = bytes(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_byte * buffer_size)).contents)
                image = PILImage.frombytes("RGBA", (width, height), raw).convert("RGB")
                if resolution:
                    want = [int(v) for v in (list(resolution) + [0, 0])[:2]]
                    if want[0] and want[1] and tuple(want) != (width, height):
                        image = image.resize((want[0], want[1]))
                out = _io.BytesIO()
                image.save(out, format="PNG")
                captured["png"] = out.getvalue()
                captured["width"], captured["height"] = image.size
            except Exception as exc:  # noqa: BLE001
                captured["error"] = f"{type(exc).__name__}: {exc}"

        # Whether the renderer is producing frames at all, before waiting on
        # one. In the streaming headless app the render loop suspends when no
        # WebRTC client is attached: the viewport reports `updates_enabled`,
        # a plausible fps, and a frame counter that never moves -- measured
        # frozen at the same frame number across calls minutes apart the
        # moment the last browser tab closed. Every capture then times out,
        # and the old message blamed the capture rather than the cause.
        frame_before = None
        try:
            frame_before = viewport.frame_info.get("frame_number")
        except Exception:  # noqa: BLE001 -- diagnosis must not break capture
            pass

        capture_viewport_to_buffer(viewport, _on_capture)

        # The callback fires on a later frame, so the loop is the wait. 40 was
        # not enough for the first shot through a freshly created camera on a
        # cold worker -- RTX warms the new view before it will hand over a
        # frame. Waiting is cheap; a blind agent is not.
        for _ in range(240):
            app.update()
            if captured:
                break

        if "error" in captured:
            return {"status": "error", "message": captured["error"]}
        if "png" not in captured:
            frozen = False
            try:
                frozen = frame_before is not None and viewport.frame_info.get("frame_number") == frame_before
            except Exception:  # noqa: BLE001
                pass
            if frozen:
                return {
                    "status": "error",
                    "message": (
                        "The renderer is not producing frames: the frame "
                        "counter did not move across the wait. In the "
                        "streaming app this means no WebRTC client is "
                        "attached -- rendering suspends without a viewer. "
                        "Open the session's viewport in the dashboard (or "
                        "any stream client), then capture again."
                    ),
                }
            return {
                "status": "error",
                "message": "Viewport capture did not complete within 240 frames.",
            }

        return {
            "status": "success",
            "format": "png",
            "width": int(captured["width"]),
            "height": int(captured["height"]),
            "camera": str(viewport.camera_path),
            "image_base64": _b64.b64encode(captured["png"]).decode("ascii"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        # Give the operator their view back even on failure. Leaving somebody's
        # stream pointing wherever the agent last looked is worse than the
        # failure that got us here.
        # Two separate attempts, deliberately. Sharing one try meant a failure
        # restoring the operator's view skipped removing the camera, and the
        # stage kept a `__capture_view_cam` nobody put there. Each half of the
        # cleanup has to survive the other failing.
        try:
            if viewport is not None and original_camera:
                viewport.camera_path = original_camera
        except Exception:
            pass
        try:
            if stage is not None and temp_camera:
                from pxr import Sdf

                stage.RemovePrim(Sdf.Path(temp_camera))
        except Exception:
            pass


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["control.run"] = lambda **kwargs: run_control(adapter, **kwargs)
    registry["control.observe"] = lambda **kwargs: observe(adapter, **kwargs)
    registry["control.capture_view"] = lambda **kwargs: capture_view(adapter, **kwargs)
