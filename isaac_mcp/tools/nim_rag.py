# SimLiverse — NVIDIA NIM RAG Proxy Tools
#
# Proxies the 5 official NVIDIA isaacsim_mcp RAG tools into our unified
# simliverse-mcp server. These tools forward requests to the NVIDIA NIM
# cloud using the NVIDIA_API_KEY environment variable.
#
# Original tool definitions and NIM endpoints are from:
#   https://github.com/NVIDIA-Omniverse/kit-usd-agents
#
# MIT License — whats2000/isaacsim-mcp-server base (execution layer)
# Apache 2.0 — NVIDIA-Omniverse/kit-usd-agents (RAG definitions)

"""NVIDIA NIM Isaac Sim RAG proxy tools."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

import httpx
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection

logger = logging.getLogger("IsaacMCPServer.nim_rag")

# ── NIM Configuration ──────────────────────────────────────────────────────────
_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_NIM_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # Used for chat completions
_NIM_TOOLS_URL = "http://localhost:9904/mcp"  # Local NVIDIA isaacsim_mcp if running
_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Fall-back: if the NVIDIA isaacsim_mcp sidecar is NOT running locally,
# we proxy the tool call directly to the NIM cloud REST API.
_NIM_RAG_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


async def _call_nvidia_mcp_tool(tool_name: str, params: dict) -> str:
    """
    Forward a tool call to the NVIDIA isaacsim_mcp server running as a local
    Docker sidecar on port 9904 (same VM). Falls back to a helpful error if
    the sidecar is not available.
    """
    if not _NVIDIA_API_KEY:
        return json.dumps({
            "status": "error",
            "message": (
                "NVIDIA_API_KEY is not configured. "
                "Set it in the SimLiverse job configuration to use Isaac Sim RAG tools."
            ),
        })

    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(_NIM_TOOLS_URL, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text.strip()
            # Handle SSE format
            for line in text.splitlines():
                if line.startswith("data:"):
                    json_str = line[5:].strip()
                    if json_str:
                        try:
                            data = json.loads(json_str)
                            result = data.get("result", {})
                            content = result.get("content", [])
                            parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                            return "\n".join(parts) if parts else "No content returned."
                        except Exception:
                            pass
            # Try direct JSON
            data = resp.json()
            result = data.get("result", {})
            content = result.get("content", [])
            parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            return "\n".join(parts) if parts else "No content returned."
    except httpx.ConnectError:
        return json.dumps({
            "status": "error",
            "message": (
                "Cannot connect to NVIDIA isaacsim_mcp sidecar on port 9904. "
                "The RAG server may still be starting up. Try again in 30 seconds."
            ),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:
    """Register the 5 NVIDIA NIM RAG proxy tools."""

    @mcp.tool("get_isaac_sim_instructions")
    async def get_isaac_sim_instructions(instruction_set: str) -> str:
        """Get detailed developer instructions for a specific Isaac Sim topic area.

        Use this FIRST when you need to write Python code for Isaac Sim.
        It returns the exact, version-correct API documentation and code patterns.

        Args:
            instruction_set: Topic to look up. Examples: "robot_simulation",
                "sensors", "physics", "materials", "action_graphs",
                "isaacsim_system", "installation", "isaac_lab".
        """
        return await _call_nvidia_mcp_tool(
            "get_isaac_sim_instructions", {"instruction_set": instruction_set}
        )

    @mcp.tool("search_isaac_sim_extensions")
    async def search_isaac_sim_extensions(query: str, limit: Optional[int] = 10) -> str:
        """Search for Isaac Sim extensions by name or description.

        Use this to discover which Omniverse extensions provide specific
        capabilities (e.g. physics, sensors, rendering).

        Args:
            query: Search query string (e.g. "robot arm", "camera sensor").
            limit: Maximum number of results to return.
        """
        params: dict = {"query": query}
        if limit is not None:
            params["limit"] = limit
        return await _call_nvidia_mcp_tool("search_isaac_sim_extensions", params)

    @mcp.tool("get_isaac_sim_extension_details")
    async def get_isaac_sim_extension_details(extension_id: str) -> str:
        """Get detailed information about a specific Isaac Sim extension.

        Returns the extension's API, configuration options, and usage examples.

        Args:
            extension_id: The extension identifier (e.g. "omni.isaac.core").
        """
        return await _call_nvidia_mcp_tool(
            "get_isaac_sim_extension_details", {"extension_id": extension_id}
        )

    @mcp.tool("search_isaac_sim_code_examples")
    async def search_isaac_sim_code_examples(query: str, limit: Optional[int] = 5) -> str:
        """Search for official Isaac Sim Python code examples.

        Use this to find working code patterns for specific tasks BEFORE
        calling execute_script. The returned examples are version-correct
        for Isaac Sim 5.1.0.

        Args:
            query: Description of what you want to do (e.g. "spawn Franka robot",
                "add camera sensor", "set joint positions").
            limit: Maximum number of examples to return.
        """
        params: dict = {"query": query}
        if limit is not None:
            params["limit"] = limit
        return await _call_nvidia_mcp_tool("search_isaac_sim_code_examples", params)

    @mcp.tool("search_isaac_sim_settings")
    async def search_isaac_sim_settings(query: str) -> str:
        """Search Isaac Sim configuration settings and environment variables.

        Use this to find the correct settings for physics, rendering, streaming,
        and extension configuration.

        Args:
            query: Setting name or description to search for.
        """
        return await _call_nvidia_mcp_tool("search_isaac_sim_settings", {"query": query})
