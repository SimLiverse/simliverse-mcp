# SimLiverse MCP Server (`simliverse-mcp`)

The unified Model Context Protocol (MCP) server for the **SimLiverse Physical AI Platform**.

It provides a minimal, clean **10-tool interface** to LLM Copilots:
- **5 Knowledge/RAG tools** (NVIDIA NIM cloud) — look up Isaac Sim Python API documentation and code examples.
- **5 Execution/Telemetry tools** (Isaac Sim TCP socket on port 8766) — execute code, play/pause physics, inspect USD scene state, and read joint telemetry.

## Available Tools (10 Total)

### 📚 Knowledge & RAG (NVIDIA NIM)
1. `get_isaac_sim_instructions`: Read official Isaac Sim developer guide topics.
2. `search_isaac_sim_code_examples`: Find working Python code snippets for Isaac Sim 5.1.0.
3. `search_isaac_sim_extensions`: Discover Omniverse extensions by capability.
4. `get_isaac_sim_extension_details`: Inspect extension API specifications.
5. `search_isaac_sim_settings`: Search physics engine and rendering config keys.

### 🤖 Execution & Control (Isaac Sim Extension via TCP :8766)
6. `execute_script`: Run Python code inside the live Isaac Sim stage runtime.
7. `set_simulation_state`: Play, pause, stop, or step physics frames.
8. `reset_scene`: Clear the USD stage and reset to initial state.

### 👁️ Telemetry & Sensing
9. `get_scene_info`: Inspect stage prims, positions, asset root, and prim counts.
10. `get_joint_states`: Read joint positions, velocities, and limits for any robot prim.

## Architecture

```
SimLiverse Control Plane API (mcp_bridge.py)
        │
        │ HTTP Streamable MCP (port 9905)
        ▼
simliverse-mcp container (this repo)
  ├── 5 RAG tools  → NVIDIA NIM Cloud (api.nvidia.com)
  └── 5 Exec tools → TCP socket (localhost:8766)
                         │
                         ▼
        isaac.sim.mcp_extension (Omniverse Extension)
                         │
                         ▼
               Live Isaac Sim Stage
```

## Running Locally

```bash
pip install -e .
export NVIDIA_API_KEY="nvapi-..."
python -m isaac_mcp.server
```

## Docker Deployment

```bash
docker build -t simliverse-mcp:latest .
docker run -d --name sl-mcp --network host -e NVIDIA_API_KEY="nvapi-..." simliverse-mcp:latest
```

## License
MIT License (forked from `whats2000/isaacsim-mcp-server` & `omni-mcp/isaac-sim-mcp`).
