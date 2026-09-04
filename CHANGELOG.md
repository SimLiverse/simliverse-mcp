# Changelog

All notable changes to the isaacsim-mcp-server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - feat/conveyor-palletizing

### Fixed
- **`clear_scene` leaves a buildable stage.** The old handler deleted every root prim, `/World` included, and reported success; the next `spawn_robot` then died on `'NoneType' object has no attribute 'create_articulation_view'` because the World singleton and `SimulationManager` still held handles into the deleted physics scene. `clear` now stops the timeline, rebinds the viewport, releases sensors, stops the Replicator orchestrator, removes render products whose camera lives under `/World` (one such product kept re-authoring a "deleted" camera for an entire session), deletes the *children* of `/World`, and keeps the physics scene, materials and ground plane. `keep_physics` now defaults `True`.
- **`get_world` heals a stage broken by the old clear**: its rebuild path purges `SimulationManager`'s expired `PhysxSceneAPI` handles, which re-defining the prim at the same path does not refresh.
- **`Conveyor.from_prop` places the deck centre at `position`.** The shipped assets author their origin wherever the artist left it (A09: the discharge end), so placing by origin put a 4 m belt two metres from where it was asked for.
- **`capture_view` distinguishes a slow capture from a renderer producing no frames.** In the streaming app rendering suspends when no WebRTC client is attached; the error now says so and says what to do, instead of a generic 40-frame timeout.
- Scoped the conveyor "start after Play or the drive is dropped" docstring to Isaac Sim 6.0 — the failure sequence was re-run on a 5.1 worker and did not reproduce.

### Fixed (palletizing, measured on a sketch-built KR210 cell)
- **The suction cup mounts along tool Z** -- the axis every down-orientation this library commands points at the floor. `_approach_axis` projected a 3.7 cm lateral wrist offset and mounted the cup sideways: 18 cm beside the flange at flange height, above a box 18 cm below, eight seal retries finding nothing. `approach_axis="auto"` keeps the old measurement.
- **`rebind_suction` reads the true tip back from the prim.** `attach_suction_gripper` stamps `simliverse:tip_offset` (standoff + cup); measuring the cup cylinder alone missed the standoff, the disagreement the code had documented as "descends that much too low" without fixing.
- **`Manipulator.downward_orientation(target)`**: flange down with the yaw facing the reach. The fixed `[0, 1, 0, 0]` quat pins the tool's yaw; on a cell drawn on the other side of the arm the servo plateaued 0.18 m short of a reachable target (position-only servo reached it in 53 ticks).
- **`controllers/kuka_palletizing.py` adds the cup's length to every contact height** and reads the belt from its stamp; `demo/kuka_palletizing.py` defaults to the real low belt (A08) instead of a primitive slab.
- **Isaac 5.x suction**: `create_surface_gripper` is not exported there; the shim goes straight to `robot_schema.CreateSurfaceGripper` with the same child-prim layout.

### Added
- **The extension puts `simliverse_sim` on `sys.path` at startup.** `--ext-folder` paths the extension, not the repository around it, so on a cold worker every `execute_script` import of the library failed with "No module named 'simliverse_sim'" -- while working in any session where an earlier script had inserted the path by hand. Found by restarting the container mid-session.
- **`Scene.ensure_light()`**: a dome light if the stage has no light of any kind, checked by type so an already-lit stage is left alone. A cold headless stage renders every correct scene as shapes on black, and the numbers never say "dark".
- **Belts stamp their geometry and drive on their own prim** (`simliverse:conveyor`, JSON, the `describe()` record), and `Conveyor.attach(path)` with no other arguments reads it back — a session that did not build a belt can take a handle on it, the same move as `simliverse:motion_config` on robots.
- `release_all_sensors()` in the sensors handler, used by `clear_scene`.
- `capture_view` switches the viewport to a temporary camera and restores it, so capturing never moves the operator's view; sensors are registered on creation and released on delete.

## [0.5.2] - 2026-04-07

### Fixed
- Code style: apply ruff formatting to v5 adapter, graphs handler, and scene handler

## [0.5.1] - 2026-04-06

### Added
- **`edit_action_graph` tool**: Modify attribute values and add connections on existing Action Graphs. Uses `og.Controller.set()` for ScriptNode `usePath`/`scriptPath` attributes (matching the pattern from `omni.graph.scriptnode` official tests). Auto-resets `state:omni_initialized` when script content or path changes to force ScriptNode reload
- **`script_file` parameter on `create_action_graph`**: One-step convenience for the common OnPlaybackTick → ScriptNode workflow. Automatically creates nodes, wires connections, and attaches the script file — replaces the previous two-step create + edit pattern
- **`prim_path` parameter on `create_robot`**: Explicit USD prim path control (e.g. `/World/Franka`) instead of name-based path derivation. Solves the common issue where robots are created at `/{Name}` but scripts expect `/World/{Name}`
- ScriptNode workflow documentation in MCP server instructions covering one-step (`script_file`) and two-step (`create` + `edit`) patterns, script reload via `edit_action_graph`, and `setup(db)`/`compute(db)` function requirements

### Changed
- `create_action_graph` docstring updated with `script_file` example and inline/file-based usage patterns
- `create_robot` docstring updated with `prim_path` parameter documentation
- Tool count updated to 42 across 9 categories

## [0.5.0] - 2026-04-06

### Added
- **`create_action_graph` tool**: Build OmniGraph Action Graphs programmatically (nodes, connections, attribute values) via `og.Controller.edit()` — no more raw `execute_script` calls for OnPlaybackTick → ScriptNode wiring
- **Drive config warnings**: `get_joint_config` and `create_robot` now return a `warnings` array when any joint has `stiffness=0` and `damping=0` (e.g. FR3 `finger_joint2` broken drive)
- **Dimensional data in responses**: `create_object` now returns `actual_size` [x, y, z] in meters and `bounding_box` (min/max world-space corners)
- **Prim size inspection**: `get_prim_info` returns `actual_size` for geometric prims (Cube, Sphere, Cylinder, Cone, Capsule)
- **Inline joint info**: `create_robot` now returns `joint_names` and `num_dof` in the response, eliminating the need for a follow-up `get_robot_info` call
- **Joint limits**: `get_robot_info` now returns `joint_limits` with type (revolute/prismatic), lower/upper limits, and units per joint
- **Comprehensive server instructions**: MCP `instructions` field now includes workflow guidance for scene setup, debug loop (step-and-observe), controller development, and tool priority
- `get_prim_actual_size` adapter method for computing prim dimensions from USD geometry attributes and scale

### Changed
- **Tool docstrings rewritten** with workflow guidance:
  - `step_simulation` promoted as the primary debug tool with typical debug loop example
  - `execute_script` reframed as escape hatch with explicit list of preferred alternatives
  - `reload_script` positioned as the controller loading workflow
  - `get_joint_config`, `get_physics_state`, `get_isaac_logs` marked as diagnostic tools with when-to-call guidance
  - `set_joint_positions`, `get_joint_positions` now document units (radians/meters)
  - `create_object` documents default primitive sizes and scale behavior
- Replaced `asset_creation_strategy` prompt with inline `instructions` covering MCP vs Script/Action Graph scope
- Updated package name and version in extension.toml
- Added new application icon and social badge image

### Fixed
- **Ground plane collision**: `create_physics_scene` now applies `UsdPhysics.CollisionAPI` to the ground plane — objects no longer fall through the floor
- **Stale `.pyc` in `reload_script`**: Dev script now clears bytecode cache before `importlib.reload()` for both extension and user modules, preventing stale code from loading
- **Orphaned subscriptions**: `reload_script` exec() mode now cleans up subscriptions from previous runs before re-executing
- Dev hot-reload script: bypass pybind11 `__setattr__` on `omni.ext.IExt` subclasses using `__dict__` assignment
- Dev hot-reload script: use `isinstance(obj, MCPExtension)` instead of fragile `hasattr` checks that matched wrong objects
- Dev hot-reload script: clear stale `.pyc` files before `importlib.reload()` to ensure fresh source is loaded
- Use `Usd.TimeCode.Default()` instead of non-existent `Gf.TimeCode(0)` in `get_prim_actual_size`
- World-space (not local-space) transform for bounding box computation
- Cylinder/Cone axis attribute respected when computing dimensions

## [0.4.1] - 2026-04-02

### Changed
- Added MCP registry metadata (`server.json`) for marketplace listing
- Fixed demo GIF URL in README to use absolute GitHub raw URL

## [0.4.0] - 2026-04-02

### Added
- **Observability tools**: `get_simulation_state`, `get_physics_state`, `get_joint_config`, `get_isaac_logs`, `reload_script`
- **Step-and-observe**: `observe` parameters on `step_simulation` for combined stepping and inspection (issue #8)
- `cwd` parameter and stdout/stderr capture for `execute_script`
- Franka pick-and-place demo scene and USD file
- Development wrapper for MCP server with hot-reloading support
- Environment discovery and loading tools
- Dynamic robot discovery from Isaac Sim asset server
- PyPI packaging via `pyproject.toml` — installable with `pip install isaacsim-mcp-server`
- Tag-triggered PyPI publish and GitHub Release CD pipeline
- Smithery registry manifest
- CI lint and format checks on PRs (ruff)
- Desktop launcher instructions and scripts
- Documentation for running multiple Isaac Sim instances with MCP

### Changed
- **Renamed package** from `isaac-sim-mcp` to `isaacsim-mcp-server` across all references
- Complete modular architecture rewrite:
  - Extracted `IsaacConnection` into dedicated connection module
  - Added adapter layer with base ABC and v5 implementation
  - Split into 8 handler modules with 31+ command handlers
  - Split into 8 MCP tool modules with 31+ tools
  - Rewrote `server.py` as slim entry point using modular tools
  - Rewrote `extension.py` as slim registry-based command router
  - Extracted socket server from `extension.py`
- Added type hints across all handler, adapter, and connection modules
- Migrated all imports from `omni.isaac.*` to `isaacsim.*` for Isaac Sim 5.1.0 compatibility
- Refreshed project documentation to reflect the current Isaac Sim `5.1.0`-focused architecture
- Reworked the README with a clearer quickstart, architecture overview, and example prompting workflow
- Updated build scripts to use installed `isaacsim-mcp-server` CLI
- Added MIT License to all source files; updated copyright headers for fork continuation
- Now documents `39` MCP tools across `8` categories

### Fixed
- Correct argument order in `set_channel_enabled` (issue #2 bug 1)
- Use PhysX velocity API for accurate runtime readings (issue #2 bug 2)
- Read runtime joint targets from articulation controller (issue #2 bug 3)
- Flatten `execute_script` and `reload_script` response structure (issue #2 bug 4)
- Use `add_message_consumer` API for Isaac Sim 5.1 log listener
- Compare log level enum by value for Isaac Sim 5.1 compatibility
- Use USD `RigidBodyAPI` velocity attrs instead of missing PhysX methods
- Initialize `SingleArticulation` before accessing controller APIs
- `scene.clear` now removes all user prims including root-level ones
- Fix transform precision conflict and URDF file validation
- Remove dead code and fix adapter bypass in handlers

### Tests
- Added 43 integration tests for all tool categories
- Updated structural tests for new observability methods

## [0.3.0] - 2025-04-22

### Added
- USD asset search integration with `search_3d_usd_by_text` tool
- Ability to search and load pre-existing 3D models from USD libraries
- Support for custom positioning and scaling of USD models
- Direct model transformation capabilities with the improved `transform` tool
- Enhanced scene management with multi-object placement

### Improved
- Scene object manipulation with precise positioning controls
- Asset loading performance and reliability
- Error handling for model search and placement
- Integration with existing physics scene management

### Technical Details
- Advanced USD model retrieval system
- Optimized asset loading pipeline
- Position and scale customization for USD models
- Better compatibility with Isaac Sim's native USD handling

## [0.2.1] - 2025-04-15

### Added
- Beaver3D integration for 3D model generation from text prompts and images
- Asynchronous model loading with asyncio support
- Task caching system to prevent duplicate model generation
- New MCP tools:
  - `generate_3d_from_text_or_image` for AI-powered 3D asset creation
  - `transform` for manipulating generated 3D models in the scene
- Texture and material binding for generated 3D models

### Improved
- Asynchronous command execution with `run_coroutine`
- Error handling and reporting for 3D generation tasks
- Performance optimizations for model loading

### Technical Details
- Integration with Beaver3D API for 3D generation
- Task monitoring with callback support
- Position and scale customization for generated models

## [0.1.0] - 2025-04-02

### Added
- Initial implementation of Isaac Sim MCP Extension
- Natural language control interface for Isaac Sim through MCP framework
- Core robot manipulation capabilities:
  - Dynamic placement and positioning of robots (Franka, G1, Go1, Jetbot)
  - Robot movement controls with position updates
  - Multi-robot grid creation (3x3 arrangement support)
- Advanced simulation features:
  - Quadruped robot walking simulation with waypoint navigation
  - Physics-based interactions between robots and environment
  - Custom lighting controls for better scene visualization
- Environment enrichment:
  - Various obstacle types: boxes, spheres, cylinders, cones
  - Wall creation for maze-like environments
  - Dynamic obstacle placement with customizable properties
- Development tools:
  - MCP server integration with Cursor AI
  - Debug interface accessible via local web server
  - Connection status verification with `get_scene_info`
- Documentation:
  - Installation instructions
  - Example prompts for common simulation scenarios
  - Configuration guidelines

### Technical Details
- Extension server running on localhost:8766
- Compatible with NVIDIA Isaac Sim 4.2.0
- Support for Python 3.9+
- MIT License for open development 
