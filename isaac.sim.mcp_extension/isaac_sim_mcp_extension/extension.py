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

"""Isaac Sim MCP Extension — slim entry point.

Routes incoming socket commands to handler modules via a registry.
"""

from __future__ import annotations

import gc
import traceback
from typing import Any, Dict

import carb
import omni.ext
import omni.usd

from .adapters import get_adapter
from .handlers import register_all_handlers
from .socket_server import SocketServer


class MCPExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.ext_id = None
        self._settings = carb.settings.get_settings()
        self._registry: Dict[str, Any] = {}
        self._adapter = None
        self._server: SocketServer | None = None

    def on_startup(self, ext_id: str) -> None:
        print("trigger  on_startup for: ", ext_id)
        self.ext_id = ext_id
        self._put_scene_library_on_path()
        port = self._settings.get("/exts/isaac.sim.mcp/server.port") or 8766
        host = self._settings.get("/exts/isaac.sim.mcp/server.host") or "localhost"

        self._adapter = get_adapter()
        register_all_handlers(self._registry, self._adapter)
        print(f"Registered {len(self._registry)} command handlers")

        self._server = SocketServer(host, port, self._execute_command)
        self._server.start()

    @staticmethod
    def _put_scene_library_on_path() -> None:
        """Make `simliverse_sim` importable from a cold boot.

        The `--ext-folder` mechanism puts the *extension* on the path, not the
        repository it sits in -- and `simliverse_sim` lives at the repo root.
        Every `execute_script` that says `from simliverse_sim import ...`
        therefore fails on a fresh worker with "No module named
        'simliverse_sim'", while working perfectly in any session where some
        earlier script happened to insert the path by hand. That is the worst
        kind of works-on-the-warm-box bug, and this is its one fix: walk up
        from this file to the directory that actually contains the library.
        """
        import sys
        from pathlib import Path

        for parent in Path(__file__).resolve().parents:
            if (parent / "simliverse_sim" / "__init__.py").is_file():
                root = str(parent)
                if root not in sys.path:
                    sys.path.insert(0, root)
                    print(f"simliverse_sim on sys.path from {root}")
                return
        print("WARNING: simliverse_sim not found above the extension; "
              "execute_script imports of it will fail")

    def on_shutdown(self) -> None:
        print("trigger  on_shutdown for: ", self.ext_id)
        if self._server:
            self._server.stop()
        self._registry.clear()
        gc.collect()

    # ── Command routing ────────────────────────────────────────────────────────

    def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        cmd_type = command.get("type", "")
        params = command.get("params", {})
        handler = self._registry.get(cmd_type)
        if handler:
            try:
                result = handler(**params)
                if result and result.get("status") == "success":
                    return {"status": "success", "result": result}
                # Preserve the whole error payload. Collapsing it to `message`
                # discarded the traceback, stdout and stderr that a caller needs
                # to actually fix the failing code.
                error: Dict[str, Any] = {"status": "error", "message": "No result"}
                if result:
                    error = {**result, "status": "error"}
                    error.setdefault("message", "Unknown error")
                return error
            except Exception as e:
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
        known = ", ".join(sorted(self._registry)) or "none"
        return {
            "status": "error",
            "message": f"Unknown command: {cmd_type}. Registered commands: {known}",
        }
