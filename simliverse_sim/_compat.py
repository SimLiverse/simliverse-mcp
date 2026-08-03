"""
Isaac Sim import compatibility.

Isaac Sim 5.x and 6.x both expose the same concepts under `isaacsim.*`, but a few
symbols moved. Every import the rest of the package needs is resolved here once,
so control code never has to guess a module path — the single most common way
generated scripts fail.

Never import from `omni.isaac.core.*`: it is removed in 6.0 and raises
ImportError.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np


class IsaacSimUnavailable(RuntimeError):
    """Raised when the package is imported outside a running Isaac Sim."""


@functools.lru_cache(maxsize=1)
def isaac_version() -> int:
    """Major version of the running Isaac Sim (5 or 6)."""
    try:
        import isaacsim.core.version as version_mod

        raw = version_mod.get_version()
        text = raw[0] if isinstance(raw, (tuple, list)) else str(raw)
        return int(str(text).split(".", 1)[0])
    except Exception:
        return 5


def get_stage() -> Any:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise IsaacSimUnavailable("No USD stage — is Isaac Sim running?")
    return stage


def get_world(physics_dt: float = 1.0 / 60.0) -> Any:
    """Return the singleton `World`, creating and initializing it if needed."""
    from isaacsim.core.api import World

    world = World.instance()
    if world is None:
        world = World(physics_dt=physics_dt, stage_units_in_meters=1.0)
    try:
        world.initialize_physics()
    except Exception:
        # Already initialized — harmless, and the common case on re-entry.
        pass
    return world


def get_timeline() -> Any:
    import omni.timeline

    return omni.timeline.get_timeline_interface()


def single_articulation(prim_path: str, name: str | None = None) -> Any:
    from isaacsim.core.prims import SingleArticulation

    return SingleArticulation(prim_path=prim_path, name=name or prim_path.rsplit("/", 1)[-1])


def articulation_action(**kwargs: Any) -> Any:
    from isaacsim.core.utils.types import ArticulationAction

    return ArticulationAction(**kwargs)


def motion_generation() -> Any:
    """The `isaacsim.robot_motion.motion_generation` module.

    This is what makes Cartesian control possible — without it the only lever is
    raw joint angles, which no language model can solve inverse kinematics for.
    """
    import isaacsim.robot_motion.motion_generation as mg

    return mg


def assets_root() -> str:
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if not root:
        raise IsaacSimUnavailable("Isaac Sim asset root is not reachable.")
    return root


def add_reference(usd_path: str, prim_path: str) -> Any:
    from isaacsim.core.utils.stage import add_reference_to_stage

    return add_reference_to_stage(usd_path, prim_path)


def as_vec3(value: Any, *, name: str = "value") -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have 3 components, got {arr.size}: {value!r}")
    return arr
