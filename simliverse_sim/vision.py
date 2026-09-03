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

## Why the obvious call returns an empty frame

The existing capture path fails on every call, and the message it produces
("Renderer returned an empty frame") describes the symptom of two unrelated
causes:

**1. `capture_on_play` defaults to on.** A render product only fills while the
timeline is playing. Build a scene, capture without pressing play, and the
annotator returns `None` forever — not slowly, not intermittently, never.

**2. A synchronous `rep.orchestrator.step()` inside a Kit callback deadlocks.**
`step()` is correct in a standalone script that owns its loop. Inside a running
Kit extension — which is where `run_control` executes — the caller is already
*on* the event loop, so blocking to wait for frames prevents the frames.

So the fix is not a different capture API. It is pumping the application
directly and reading the annotator afterwards, which is what the shipped
`Camera` sensor does internally.

## What this does instead

Attach an `rgb` annotator to the viewport's *existing* render product, pump the
app for a fixed number of frames, read the buffer, detach. No second camera, no
resolution mismatch, no coroutine, and nothing that needs the caller to be
async.

The warm-up is not superstition. A render product yields nothing on the frame
it is attached, and an RTX image is still converging for several frames after
that. `settle_frames` is the knob; the default is the smallest number that gave
a stable image in NVIDIA's own tests, and it is cheap to raise.
"""

from __future__ import annotations

import base64
from typing import Any

from ._compat import update_app

#: Frames pumped after attaching before the buffer is read. Below about three
#: the annotator returns an empty array; the extra frames buy RTX convergence.
DEFAULT_SETTLE_FRAMES = 12

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


def capture(
    *,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    annotator: str = "rgb",
) -> Any:
    """Render the active viewport and return it as a numpy array.

    Returns `(H, W, 4)` uint8 RGBA. Raises `VisionUnavailable` with the reason
    rather than returning an empty array, because an empty array is what the
    old path returned and it is indistinguishable from a black scene.
    """
    try:
        import omni.replicator.core as rep
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable("omni.replicator.core is unavailable.") from exc

    viewport = _viewport()
    render_product = viewport.render_product_path
    if not render_product:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "The viewport has no render product, so there is nothing to read."
        )

    # Manual control. Left on the default, frames only arrive while the timeline
    # plays, and a scene that has been built but not played captures nothing.
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:  # pragma: no cover - older Kit
        pass

    annot = rep.AnnotatorRegistry.get_annotator(annotator)
    annot.attach(render_product)
    try:
        # Pump the app rather than stepping the orchestrator. `update_app` runs a
        # whole frame — render plus the SDG pipeline that fills the annotator —
        # and returns, so it works from a synchronous caller on the event loop.
        for _ in range(max(1, int(settle_frames))):
            update_app()
        data = annot.get_data(do_array_copy=True)
    finally:
        annot.detach()

    if data is None or getattr(data, "size", 0) == 0:  # pragma: no cover
        raise VisionUnavailable(
            "The annotator returned no data after "
            f"{settle_frames} frames. The usual cause is that nothing has been "
            "rendered yet: check the stage has a light and the camera is "
            "pointing at something."
        )
    return data


def png(
    *,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    path: str | None = None,
) -> dict:
    """Capture and encode a PNG. Returns a report, and writes a file if asked.

    The report carries `bytes` and `looks_blank` so a caller that cannot see the
    image still learns whether there was anything in it. That matters here more
    than it sounds: the whole failure mode being fixed is an agent that believes
    it has looked.
    """
    data = capture(settle_frames=settle_frames)
    height, width = int(data.shape[0]), int(data.shape[1])

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - needs Kit
        raise VisionUnavailable(
            "Pillow is unavailable, so the frame cannot be encoded."
        ) from exc

    import io

    image = Image.fromarray(data[:, :, :3])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()

    if path:
        with open(path, "wb") as handle:
            handle.write(raw)

    return {
        "width": width,
        "height": height,
        "bytes": len(raw),
        "looks_blank": len(raw) < BLANK_PNG_BYTES,
        "path": path,
        "base64": base64.b64encode(raw).decode("ascii") if path is None else None,
    }


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
