# MIT License
# Copyright (c) 2026 SimLiverse

"""Simulation control MCP tools — SimLiverse minimal set."""

import json
from typing import TYPE_CHECKING, Callable, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:

    @mcp.tool("execute_script")
    def execute_script(code: str, cwd: Optional[str] = None) -> str:
        """Execute arbitrary Python code in the live Isaac Sim stage runtime.

        Use this to spawn robots, move objects, configure joints, set up cameras,
        or perform any simulation actions. Before writing code, use the NVIDIA RAG
        tools (get_isaac_sim_instructions, search_isaac_sim_code_examples) to verify
        the correct Isaac Sim Python API syntax.

        Args:
            code: Python code string to execute.
            cwd: Optional working directory to add to sys.path before execution.
        """
        try:
            conn = get_connection()
            params = {"code": code}
            if cwd is not None:
                params["cwd"] = cwd
            result = conn.send_command("simulation.execute_script", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("set_simulation_state")
    def set_simulation_state(action: str, num_steps: int = 1) -> str:
        """Control the physics simulation playback state.

        Args:
            action: State command. Must be one of: "play", "pause", "stop", or "step".
            num_steps: If action is "step", number of frames to step forward (default 1).
        """
        try:
            conn = get_connection()
            action_lower = action.lower().strip()
            if action_lower == "play":
                result = conn.send_command("simulation.play")
            elif action_lower == "pause":
                result = conn.send_command("simulation.pause")
            elif action_lower == "stop":
                result = conn.send_command("simulation.stop")
            elif action_lower == "step":
                result = conn.send_command("simulation.step", {"num_steps": num_steps})
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid action '{action}'. Must be 'play', 'pause', 'stop', or 'step'."
                })
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
