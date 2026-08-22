# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this repo is

`simliverse-mcp` is an experimental Model Context Protocol (MCP) server that exposes NVIDIA
Isaac Sim as a set of tool-callable actions (scene setup, robot/object creation, sensors,
lighting, materials, action graphs, simulation stepping, etc.). The tool implementations live
in `isaac_mcp/tools/`, the MCP entry point is `isaac_mcp/server.py`, the transport/connection
layer is `isaac_mcp/connection.py`, and the Isaac Sim-side extension that the server talks to
is `isaac.sim.mcp_extension/`.

**This repo is currently under experiment.** It is not a finished, stable product. The
longer-term goal is an AI agent for SimLiverse that includes a sub-agent for managing Isaac
Sim; this repo is the test bed for building the Isaac Sim MCP server half of that. Expect
churn in tool names, arguments, and behavior.

## Relationship to upstream and to simliverse-core

- This repo is a **fork of** the external upstream project
  [`whats2000/isaacsim-mcp-server`](https://github.com/whats2000/isaacsim-mcp-server). General
  contribution norms (dev environment setup, PR process, code style, project layout) are
  inherited from that upstream project — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for those
  details rather than duplicating them here.
- This server is consumed by **simliverse-core**, specifically
  `api/app/services/mcp_bridge.py`, which resolves the private VPC IP of a job's worker VM and
  talks to this server's `/mcp` endpoint using MCP JSON-RPC 2.0. The integration is documented
  (as of this writing) in `simliverse-core/docs/adr/010-isaacsim-copilot-mcp-bridge.md`
  (relative path in the sibling `simliverse-core` repo — not guaranteed to be readable from
  here, check if it exists before relying on it).
- **The agent harness in simliverse-core that drives this MCP server currently doesn't work
  well at all.** simliverse-core's own `ARCHITECTURE.md` (§9) explicitly flags that "the MCP
  tool-call contract between this bridge and simliverse-mcp is unstable" and recommends
  investigating that contract before expanding further. Anyone picking up work here should
  expect to dig into that integration rather than assume it's solid.

## Networking constraint (important, security-relevant)

The server listens via `mcp.run(transport="streamable-http")` on
`MCP_PORT` (default `9905`, see `isaac_mcp/server.py`). Connectivity from simliverse-core is
meant to be **VPC-internal only** — port 9905 must never be exposed publicly. Do not add
config, docs, or deployment scripts that expose this port to the public internet.

## Version adapters

`adapters/` splits three ways, and the split is what keeps the two supported Isaac Sim
versions readable:

- **`base.py`** — the abstract contract. Handler code never imports `isaacsim.*` directly; it
  goes through this interface.
- **`common.py`** — the implementations that do not vary by version, because they reach Isaac
  through APIs that did not move between 5.1 and 6.0 (`pxr`, `omni.usd`, `omni.kit.commands`)
  or through no Isaac API at all.
- **`v5.py` / `v6.py`** — only what genuinely differs, which is mostly the
  `isaacsim.core.*` → `isaacsim.core.experimental.*` move and the timeline and articulation
  rewrites.

A method belongs in `common.py` when v5 and v6 would implement it identically. Copying it into
both instead is how this directory previously grew to hold eighteen duplicated methods.

## Loose experimental files at repo root

One file sits at the repo root, outside the installed package (`isaac_mcp/`) and outside
`tests/`. It is an active, in-progress experiment — **not stable structure, not part of the
public API, and may be moved, rewritten, or deleted without notice**:

- **`test_material_binding.py`** — despite the `test_` prefix, this is **not** a real pytest
  test (no assertions, lives outside `tests/`). It's a throwaway USD API exploration script
  that pokes at `UsdShade.MaterialBindingAPI` (constructor vs. `.Apply()`) to figure out the
  correct binding pattern, presumably to inform `isaac_mcp/tools/materials.py`.

Do not treat it as a reference for "how this codebase does things" — it's a scratch/experiment
artifact that happened to get committed.

## Tooling actually in use

- **Package/dependency management**: `pyproject.toml` (hatchling build backend) + `uv`
  (`uv.lock` is checked in but excluded from agent context via `.claudeignore` — large and
  low-signal for reading, don't edit it by hand).
- **Linting/formatting**: [Ruff](https://docs.astral.sh/ruff/), configured in `ruff.toml`
  (`target-version = "py310"`, `line-length = 120`, `E501` ignored, double-quote strings).
- **Pre-commit**: `.pre-commit-config.yaml` runs `ruff` (lint, `--fix`) and `ruff-format` on
  every commit. Both configs were reviewed as part of writing this file and look correct for
  the project as-is — no changes made.
- **License headers**: MIT-style headers are added via `add_license_headers.py`
  (`LICENSE_HEADER.py` is the template).
- Other repo docs: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`,
  `LICENSE`, plus `docs/superpowers/{plans,specs}`, `server.json`, `smithery.yaml`.

## Future step (not to build yet)

Once the MCP tool-call contract between this server and simliverse-core's bridge stabilizes,
it would make sense to add a docs-freshness CI check here (mirroring the pattern used in
simliverse-core) so this file and related docs can't silently drift from the actual tool
surface. Given the contract is explicitly called out as unstable right now, do **not** build
that check yet — it would encode a contract that doesn't exist. Revisit once the integration
work referenced above has landed.
