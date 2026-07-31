# MIT License
# Copyright (c) 2026 SimLiverse

"""SimLiverse MCP Server — Entry Point.

Unified 10-Tool Server (5 Execution + 5 NVIDIA NIM RAG Proxy Tools).
Runs as an HTTP Streamable MCP server on port 9905 inside the worker VM.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from mcp.server.fastmcp import FastMCP

from isaac_mcp.connection import get_isaac_connection, reset_isaac_connection
from isaac_mcp.tools import register_all_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SimLiverseMCPServer")

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "9905"))


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    try:
        logger.info(f"SimLiverse MCP server starting on {MCP_HOST}:{MCP_PORT}")
        try:
            get_isaac_connection()
            logger.info("Successfully connected to Isaac Sim extension socket on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Isaac Sim extension socket on startup: {e}")
        yield {}
    finally:
        reset_isaac_connection()
        logger.info("SimLiverse MCP server shut down")


_INSTRUCTIONS = """\
SimLiverse Physical AI Copilot — Unified Isaac Sim Interface.

## Available Capabilities (10 Tools):

1. **RAG Knowledge Base (NVIDIA NIM)**:
   - `get_isaac_sim_instructions`: Read official developer instruction guides.
   - `search_isaac_sim_code_examples`: Search version-correct Python code examples.
   - `search_isaac_sim_extensions`: Discover available Isaac Sim extensions.
   - `get_isaac_sim_extension_details`: Get extension API details.
   - `search_isaac_sim_settings`: Search physics & renderer settings.

2. **Execution & Control**:
   - `execute_script`: Run Python code inside the live Isaac Sim runtime stage.
   - `set_simulation_state`: Control physics playback (play, pause, stop, step).
   - `reset_scene`: Clear the USD stage and reset physics to initial state.

3. **Telemetry & Feedback**:
   - `get_scene_info`: Inspect USD stage prims, asset root, and scene count.
   - `get_joint_states`: Read robot joint positions, velocities, and limits.

## Recommended Workflow:
1. Call `get_isaac_sim_instructions` or `search_isaac_sim_code_examples` to lookup exact code syntax.
2. Call `execute_script` to run the code in Isaac Sim.
3. Call `get_scene_info` or `get_joint_states` to verify the execution result.
"""

mcp = FastMCP(
    "SimLiverseMCP",
    instructions=_INSTRUCTIONS,
    lifespan=server_lifespan,
    host=MCP_HOST,
    port=MCP_PORT,
)

register_all_tools(mcp, get_isaac_connection)


def main():
    logger.info(f"Starting SimLiverse FastMCP HTTP server on {MCP_HOST}:{MCP_PORT}...")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
