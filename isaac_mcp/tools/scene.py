# MIT License
# Copyright (c) 2026 SimLiverse

"""Scene management MCP tools — SimLiverse minimal set."""

import json
from typing import TYPE_CHECKING, Callable

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:

    @mcp.tool("get_scene_info")
    def get_scene_info() -> str:
        """Get current Isaac Sim scene state: stage path, asset root, prim count,
        and a list of all top-level prims with their types and world positions.

        Call this AFTER execute_script to verify that objects were added correctly,
        or at the start of a session to understand what is already in the scene.
        """
        try:
            conn = get_connection()
            result = conn.send_command("scene.get_info")
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("reset_scene")
    def reset_scene(keep_physics: bool = False) -> str:
        """Clear the entire Isaac Sim scene and reset physics to initial state.

        Use this to start fresh before building a new simulation environment.
        All prims, robots, and objects will be removed.

        Args:
            keep_physics: If True, keep the physics scene prim (ground plane, gravity).
                          Default False clears everything.
        """
        try:
            conn = get_connection()
            result = conn.send_command("scene.clear", {"keep_physics": keep_physics})
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
