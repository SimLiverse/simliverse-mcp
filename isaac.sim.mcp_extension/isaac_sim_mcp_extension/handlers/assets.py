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

"""Asset import and loading command handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader, USDSearch3d

from ..adapters.base import IsaacAdapterBase


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["assets.import_urdf"] = lambda **p: import_urdf(adapter, **p)
    registry["assets.load_usd"] = lambda **p: load_usd(adapter, **p)
    registry["assets.search_usd"] = lambda **p: search_usd(adapter, **p)
    registry["assets.generate_3d"] = lambda **p: generate_3d(adapter, **p)


def import_urdf(
    adapter: IsaacAdapterBase,
    urdf_path: Optional[str] = None,
    prim_path: str = "/World/robot",
    position: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    try:
        if not urdf_path:
            return {"status": "error", "message": "urdf_path is required"}
        _result = adapter.import_urdf(urdf_path, prim_path=prim_path)
        if position:
            adapter.set_prim_transform(prim_path, position=position)
        return {"status": "success", "message": f"Imported URDF from {urdf_path}", "prim_path": prim_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_usd(
    adapter: IsaacAdapterBase,
    usd_url: Optional[str] = None,
    prim_path: str = "/World/my_usd",
    position: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    try:
        if not usd_url:
            return {"status": "error", "message": "usd_url is required"}
        loader = USDLoader()
        result_path = loader.load_usd_from_url(url_path=usd_url, target_path=prim_path, location=position, scale=scale)

        # Verify the reference actually composed before claiming success.
        #
        # `AddReference` on a URL that 404s does not raise: USD authors the
        # reference arc, the layer fails to resolve, and you are left with a
        # prim of empty type and no children. The handler then reported
        # `status: success`, which is the worst answer available to an agent --
        # it proceeds believing a conveyor is on the stage and every later
        # decision is built on that. Six guessed conveyor paths all "loaded"
        # this way; all six were 404s, and the scene stayed empty.
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(result_path)
        if not prim or not prim.IsValid():
            return {
                "status": "error",
                "message": f"No prim at {result_path} after loading {usd_url}.",
            }

        type_name = str(prim.GetTypeName() or "")
        has_children = bool(list(prim.GetChildren()))
        if not type_name and not has_children:
            return {
                "status": "error",
                "message": (
                    f"{usd_url} did not resolve: the reference was authored but "
                    f"composed to an empty prim at {result_path}. The asset "
                    f"probably does not exist at that URL."
                ),
                "prim_path": result_path,
                "hint": "Check the path with a HEAD request, or use search_usd.",
            }

        return {
            "status": "success",
            "message": f"Loaded USD from {usd_url}",
            "prim_path": result_path,
            "type": type_name,
            "children": len(list(prim.GetChildren())),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_usd(
    adapter: IsaacAdapterBase,
    text_prompt: Optional[str] = None,
    target_path: str = "/World/my_usd",
    position: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    from ..usd import USDSearchNoResults, USDSearchUnavailable

    try:
        if not text_prompt:
            return {"status": "error", "message": "text_prompt is required"}
        searcher = USDSearch3d()
        url = searcher.search(text_prompt)
        loader = USDLoader()
        prim_path = loader.load_usd_from_url(url_path=url, target_path=target_path)
        return {
            "status": "success",
            "message": f"Found and loaded USD for '{text_prompt}'",
            "prim_path": prim_path,
            "url": url,
        }
    # Kept apart on purpose. "The service is down" and "no such asset exists"
    # lead to different correct behaviour -- report a blocked task, or put the
    # substitution to the user -- and this verb is the only place that still
    # knows which one happened. Flattening both to `str(e)` is what produced the
    # message "0" and left an agent with no way to tell them apart.
    except USDSearchNoResults as exc:
        return {
            "status": "not_found",
            "message": str(exc),
            "searched_for": text_prompt,
            "service_available": True,
        }
    except USDSearchUnavailable as exc:
        return {
            "status": "unavailable",
            "message": str(exc),
            "searched_for": text_prompt,
            "service_available": False,
        }
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


def generate_3d(
    adapter: IsaacAdapterBase,
    text_prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    position: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    try:
        if not text_prompt and not image_url:
            return {"status": "error", "message": "Either text_prompt or image_url is required"}
        beaver = Beaver3d()
        if image_url:
            task_id = beaver.generate_3d_from_image(image_url)
        else:
            task_id = beaver.generate_3d_from_text(text_prompt)

        def on_complete(task_id, status, result_path):
            loader = USDLoader()
            loader.load_usd_model(task_id=task_id)
            try:
                loader.load_texture_and_create_material(task_id=task_id)
                loader.bind_texture_to_model()
            except Exception:
                pass
            if position or scale:
                loader.transform(position=position or (0, 0, 50), scale=scale or (10, 10, 10))

        from omni.kit.async_engine import run_coroutine

        run_coroutine(beaver.monitor_task_status_async(task_id, on_complete_callback=on_complete))
        return {"status": "success", "message": "3D generation started", "task_id": task_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}
