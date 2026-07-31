# MIT License
# Copyright (c) 2026 SimLiverse

"""Robot inspection MCP tools — SimLiverse minimal set."""

import json
from typing import TYPE_CHECKING, Callable

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:

    @mcp.tool("get_joint_states")
    def get_joint_states(prim_path: str) -> str:
        """Get joint positions, velocities, and limits for a robot articulation.

        Use this to inspect joint telemetry after stepping or executing control scripts.

        Args:
            prim_path: USD path to the robot articulation root (e.g. "/World/Franka").
        """
        try:
            conn = get_connection()
            result = conn.send_command("robots.get_joint_positions", {"prim_path": prim_path})
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
