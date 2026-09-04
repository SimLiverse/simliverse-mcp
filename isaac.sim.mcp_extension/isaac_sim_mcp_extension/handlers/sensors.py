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

"""Sensor creation and data capture command handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..adapters.base import IsaacAdapterBase


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["sensors.create_camera"] = lambda **p: create_camera(adapter, **p)
    registry["sensors.capture_image"] = lambda **p: capture_image(adapter, **p)
    registry["sensors.create_lidar"] = lambda **p: create_lidar(adapter, **p)
    registry["sensors.get_point_cloud"] = lambda **p: get_point_cloud(adapter, **p)


#: Live sensors, by the prim path they own.
#:
#: Kept because a sensor outlives the call that made it and keeps re-authoring
#: its prim on every render tick. `create_camera` used to drop the reference on
#: the floor, which made the camera permanently undeletable: `delete_object`
#: removed the prim, the sensor put it back within five frames, and every tool
#: involved reported success. A session accumulated cameras nobody could clear.
_SENSORS: Dict[str, Any] = {}


def release_all_sensors() -> list:
    """Release every sensor we own. `scene.clear` calls this before deleting
    prims, because a live sensor re-creates its camera within five frames of
    the prim going -- which is how a "cleared" stage grew cameras back."""
    return [path for path in list(_SENSORS) if release_sensor(path)]


def release_sensor(prim_path: str) -> bool:
    """Shut down the sensor owning `prim_path`, if we made one.

    Called before removing a prim. Returns whether there was anything to
    release, so the caller can say so rather than guess.
    """
    sensor = _SENSORS.pop(prim_path, None)
    if sensor is None:
        return False
    # No single teardown verb is guaranteed across Isaac versions, so try the
    # ones that exist and ignore the rest. What matters is dropping our
    # reference and detaching anything that pulls frames.
    for method in ("detach_annotators", "remove_annotators", "post_reset"):
        try:
            getattr(sensor, method)()
        except Exception:
            pass
    del sensor
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    return True


def create_camera(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/Camera",
    position: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    resolution: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    try:
        res = tuple(resolution) if resolution else (1280, 720)
        # Replace rather than stack: creating a camera twice at the same path
        # otherwise leaves the first sensor alive and fighting the second.
        release_sensor(prim_path)
        _SENSORS[prim_path] = adapter.create_camera(prim_path, resolution=res)
        if position or rotation:
            adapter.set_prim_transform(prim_path, position=position, rotation=rotation)
        return {"status": "success", "message": f"Camera created at {prim_path}", "prim_path": prim_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def capture_image(
    adapter: IsaacAdapterBase, prim_path: str = "/World/Camera", output_path: Optional[str] = None
) -> Dict[str, Any]:
    try:
        image_data = adapter.capture_camera_image(prim_path)
        if output_path:
            from PIL import Image

            img = Image.fromarray(image_data)
            img.save(output_path)
            return {"status": "success", "message": f"Image saved to {output_path}", "output_path": output_path}
        return {
            "status": "success",
            "message": "Image captured",
            "shape": list(image_data.shape) if hasattr(image_data, "shape") else None,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_lidar(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/Lidar",
    position: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    config: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        adapter.create_lidar(prim_path, config=config)
        if position or rotation:
            adapter.set_prim_transform(prim_path, position=position, rotation=rotation)
        return {"status": "success", "message": f"Lidar created at {prim_path}", "prim_path": prim_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_point_cloud(adapter: IsaacAdapterBase, prim_path: str = "/World/Lidar") -> Dict[str, Any]:
    try:
        pc = adapter.get_lidar_point_cloud(prim_path)
        point_count = len(pc) if pc is not None else 0
        return {"status": "success", "message": f"Got {point_count} points", "point_count": point_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}
