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

"""Looking at the scene.

Every other module here measures. This one renders, because measurement does
not catch a cell that is numerically correct and visually absurd — a gripper
mounted ninety degrees off, a carton grabbed by its corner, a stage so dark it
reads as a crashed simulator. All three of those were found by a human looking
at the screen, and each cost a working session.

## Four routes, measured on a live L4 rather than reasoned about

The old path failed on every call and reported "Renderer returned an empty
frame", which describes a symptom shared by several unrelated causes. Rather
than guess, all four candidates were run against a booted simulator:

| route | result |
| --- | --- |
| `rgb` annotator, `capture_on_play(False)`, pump frames | `(0,)` — always empty |
| `rgb` annotator, `capture_on_play(True)`, timeline **playing** | `(720, 1280, 4)` |
| own `rep.create.render_product`, pump frames | `(0,)` — always empty |
| `capture_viewport_to_file`, pump frames | a real PNG in 9 frames |

The first row is the trap, and it is the opposite of what the documentation
leads you to. `set_capture_on_play(False)` is advice for a *standalone* script
that owns its loop and drives `rep.orchestrator.step()` itself. Inside a live
Kit session nothing else is driving capture, so turning it off guarantees the
render product never fills — permanently, silently, on every call.

## Why this uses the file route rather than the annotator

The annotator route works, but only with the timeline playing, and playing the
timeline **steps physics**. "Show me the scene" must not move the scene: a
capture that nudges a settled carton off a belt is worse than no capture.

`capture_viewport_to_file` needs neither. Measured from cold with the timeline
stopped, it produced a complete PNG after nine pumped frames and left
`is_playing` false. So the file route is the default, and the annotator route
stays as a fallback for a build where the utility is missing — restoring the
timeline state it found.

Both need the app pumped. Nothing here awaits, because `run_control` executes on
the event loop and a coroutine that blocks it cannot be advanced by it.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Any

from ._compat import update_app

logger = logging.getLogger(__name__)

#: Frames pumped while waiting for the capture to land. Measured at nine on an
#: L4; the rest is headroom for a colder cache or a heavier stage.
DEFAULT_SETTLE_FRAMES = 40

#: Under this many bytes a PNG is a flat fill, not a scene. Isaac's own agent
#: skills use file size as a blank-frame detector because it costs nothing and
#: catches the failure that looks most like success: a perfectly valid image of
#: an unlit or empty stage. Roughly: 80 KB is grey, 275 KB is a bare grid.
BLANK_PNG_BYTES = 90_000


class VisionUnavailable(RuntimeError):
    """Rendering could not be reached at all, with the reason attached."""


def _viewport():
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "omni.kit.viewport.utility is unavailable, so there is no viewport "
            "to capture. This is expected outside a running Isaac Sim."
        ) from exc
    viewport = get_active_viewport()
    if viewport is None:  # pragma: no cover - needs Kit
        raise VisionUnavailable("No active viewport.")
    return viewport


def png(
    *,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    path: str | None = None,
    encode: bool = False,
) -> dict:
    """Render the viewport to a PNG. Does not move the scene.

    Returns a report carrying `bytes` and `looks_blank`, so a caller that cannot
    see the image still learns whether there was anything in it. That matters
    more than it sounds: the failure being fixed here is an agent that believes
    it has looked.

    `encode=True` adds base64. Off by default — a 1280x720 frame is roughly a
    megabyte of text, and pushing that through a tool result costs far more than
    the verdict is worth when the answer is usually "yes, it rendered".
    """
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "omni.kit.viewport.utility.capture_viewport_to_file is unavailable."
        ) from exc

    viewport = _viewport()
    target = path or os.path.join(tempfile.gettempdir(), "simliverse_view.png")
    if os.path.exists(target):
        os.remove(target)

    capture_viewport_to_file(viewport, file_path=target)

    # Poll while pumping rather than pumping a fixed count and hoping. The
    # capture lands asynchronously several frames later, and the file appears
    # empty before it appears complete.
    landed = 0
    for frame in range(max(1, int(settle_frames))):
        update_app()
        if os.path.exists(target) and os.path.getsize(target) > 0:
            landed = frame + 1
            break

    if not landed:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            f"No frame was written after {settle_frames} pumped frames. The "
            "renderer is not producing images; check the viewport exists and "
            "the stage is not mid-load."
        )

    raw_size = os.path.getsize(target)
    report = {
        "width": int(viewport.resolution[0]),
        "height": int(viewport.resolution[1]),
        "bytes": raw_size,
        "looks_blank": raw_size < BLANK_PNG_BYTES,
        "path": target,
        "frames": landed,
    }
    if encode:
        with open(target, "rb") as handle:
            report["base64"] = base64.b64encode(handle.read()).decode("ascii")
    return report


def capture(*, settle_frames: int = DEFAULT_SETTLE_FRAMES) -> Any:
    """The rendered frame as a numpy array, `(H, W, 3)` uint8 RGB.

    Decoded from the PNG rather than read off an annotator, because the
    annotator route needs the timeline playing and this must not step physics.
    """
    report = png(settle_frames=settle_frames)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "Pillow/numpy unavailable, so the frame cannot be decoded."
        ) from exc
    return np.asarray(Image.open(report["path"]).convert("RGB"))


#: Named viewpoints for `views()`, as (eye, target) offsets from the subject.
#:
#: One camera is a keyhole. A cell that reads correctly from the front can have
#: a carton floating a centimetre off the pallet, an arm reaching through a
#: conveyor leg, or two boxes occupying the same space - none of which survive a
#: second angle. The top-down view is the one that matters most for layout, and
#: is the one NVIDIA's own spatial-reasoning guidance singles out, because
#: overlap and clearance are two-dimensional questions and a perspective view
#: answers them ambiguously.
STANDARD_VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # Straight down: footprints, spacing, overlap, reach.
    "top": ((0.0, 0.0, 3.2), (0.0, 0.0, 0.0)),
    # Along -Y, level-ish: heights, whether things rest on what they should.
    "front": ((0.0, -3.0, 1.1), (0.0, 0.0, 0.35)),
    # Three-quarter, the view a person would choose.
    "hero": ((2.2, -2.2, 1.7), (0.0, 0.0, 0.35)),
    # From the far side, so occlusion in one view is not occlusion in all.
    "back": ((-2.4, 2.0, 1.5), (0.0, 0.0, 0.35)),
}


def views(
    *,
    names: list[str] | None = None,
    centre: Any = (0.0, 0.0, 0.0),
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    directory: str = "/tmp",
    prefix: str = "view",
) -> dict:
    """Capture several viewpoints in one call, and restore the camera after.

    Returns `{name: report}` plus an `ok` list, so a caller that cannot see the
    images still learns which ones rendered and which came back blank.

    `centre` moves the whole rig, so this works on a cell that is not at the
    origin. The camera is put back where it started, because a capture that
    silently re-aims the user's viewport is a capture that gets switched off.
    """
    import numpy as np

    try:
        from isaacsim.core.rendering_manager import ViewportManager
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "isaacsim.core.rendering_manager is unavailable."
        ) from exc

    origin = np.asarray(centre, dtype=float).reshape(3)
    wanted = list(names) if names else list(STANDARD_VIEWS)
    unknown = [n for n in wanted if n not in STANDARD_VIEWS]
    if unknown:
        raise ValueError(
            f"Unknown view(s) {unknown}. Known: {sorted(STANDARD_VIEWS)}"
        )

    manager = ViewportManager()
    viewport = _viewport()
    camera = viewport.camera_path

    # Remember where the camera was, so this is a look and not a move.
    before = None
    try:
        from pxr import Usd, UsdGeom

        from ._compat import get_stage

        prim = get_stage().GetPrimAtPath(camera)
        if prim:
            before = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
    except Exception:  # pragma: no cover - restoring is best effort
        before = None

    out: dict = {"ok": [], "blank": [], "views": {}}
    try:
        for name in wanted:
            eye_offset, target_offset = STANDARD_VIEWS[name]
            manager.set_camera_view(
                camera,
                eye=(origin + np.asarray(eye_offset, dtype=float)).tolist(),
                target=(origin + np.asarray(target_offset, dtype=float)).tolist(),
            )
            report = png(
                settle_frames=settle_frames,
                path=f"{directory}/{prefix}_{name}.png",
            )
            out["views"][name] = report
            (out["blank"] if report["looks_blank"] else out["ok"]).append(name)
    finally:
        if before is not None:
            try:
                eye = before.ExtractTranslation()
                forward = before.TransformDir((0.0, 0.0, -1.0))
                manager.set_camera_view(
                    camera,
                    eye=[float(v) for v in eye],
                    target=[float(e) + float(f) for e, f in zip(eye, forward)],
                )
            except Exception:  # pragma: no cover
                logger.debug("Could not restore the camera after capturing views")

    return out


def look(*, settle_frames: int = DEFAULT_SETTLE_FRAMES) -> dict:
    """One call an agent can make to find out whether the scene looks right.

    Deliberately returns a verdict and not just pixels. `looks_blank` is the
    cheap check that catches an unlit stage, a camera pointing at nothing, and a
    renderer that has not produced a frame — three failures that each report
    themselves as something else.
    """
    try:
        report = png(settle_frames=settle_frames)
    except VisionUnavailable as exc:
        return {"ok": False, "reason": str(exc)}

    report["ok"] = True
    if report["looks_blank"]:
        report["hint"] = (
            "The frame encoded to very few bytes, which means a flat fill "
            "rather than a scene. Isaac's default stage is almost unlit: add a "
            "dome light before concluding the geometry is missing."
        )
    return report
