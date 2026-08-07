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
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("simliverse_sim._compat")


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


def _physics_ready(_world: Any = None) -> bool:
    """Is the physics simulation view live?

    Deliberately NOT `world._physics_context`. That attribute is `None` on Isaac
    Sim 6.0 even for a freshly constructed, perfectly healthy `World` — it
    belongs to the older stepping path — so checking it reports every session as
    broken. `SimulationManager.get_physics_sim_view()` is what 6.0 actually
    populates, and what articulation and rigid-body views read through.
    """
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except ImportError:  # older layout
        return True
    return SimulationManager.get_physics_sim_view() is not None


def get_world(physics_dt: float = 1.0 / 60.0) -> Any:
    """Return the singleton `World`, with physics genuinely initialized.

    The previous version wrapped `initialize_physics()` in a bare `except: pass`
    on the theory that a second call is harmless. It is — but so is every other
    failure under a bare except, and a genuine one left `_physics_context` as
    None and handed back a `World` that raised on the next `step()`. That killed
    a live grasp run: eight failures across all three agent tiers, each reported
    as an AttributeError on `_step` with nothing pointing here.

    Re-entry is now handled by asking whether physics is ready rather than by
    calling and discarding the answer.
    """
    from isaacsim.core.api import World

    world = World.instance()
    if world is None:
        world = World(physics_dt=physics_dt, stage_units_in_meters=1.0)
    elif _physics_ready(world):
        return world

    try:
        world.initialize_physics()
    except Exception as exc:
        # A singleton left behind by a previous run can hold a stage that no
        # longer exists, and it cannot be repaired in place. Drop it and build a
        # fresh one — the alternative is every later call failing on a corpse.
        logger.warning("World.initialize_physics() failed (%s); rebuilding", exc)
        try:
            World.clear_instance()
        except Exception:  # noqa: BLE001 — best effort; the retry is what matters
            logger.debug("World.clear_instance() failed", exc_info=True)
        world = World(physics_dt=physics_dt, stage_units_in_meters=1.0)
        world.initialize_physics()

    if not _physics_ready(world):
        # A warning, not an error. The sim view is only populated once physics
        # has actually stepped, so it is legitimately absent on a scene that has
        # just been built — and raising here would make `Scene.get()` fail for
        # every caller that only wants to author prims or step PhysX directly,
        # neither of which needs it. Articulation code checks for itself.
        logger.warning(
            "No physics sim view yet; articulation reads will fail until the "
            "scene has been played and stepped at least once."
        )
    return world


def get_physx() -> Any:
    """The PhysX interface, which is how an extension steps physics itself."""
    import omni.physx

    return omni.physx.get_physx_interface()


def update_app() -> None:
    """Advance Kit one frame — rendering only, never physics."""
    import omni.kit.app

    omni.kit.app.get_app().update()


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


def as_quat(value: Any, *, name: str = "orientation") -> np.ndarray:
    """Normalise any orientation to a plain float `[w, x, y, z]` array.

    Isaac Sim's numpy backend calls `.astype(np.float32)` on whatever it is
    handed, which fails on `Gf.Quatf` with
    `TypeError: float() argument must be a string or a real number, not 'Quatf'`
    — several frames below the call, naming nothing the caller wrote. A `Gf`
    quaternion is the obvious type to reach for from pxr, so this accepts it,
    along with a 4-tuple and a 3-tuple of euler angles in degrees.
    """
    if value is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Gf.Quatf / Gf.Quatd — real part and imaginary vector, not indexable.
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return np.array(
            [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])],
            dtype=float,
        )

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 4:
        return array
    if array.size == 3:
        # Euler degrees, XYZ order — how a human describes "tilt it 20 degrees".
        roll, pitch, yaw = np.radians(array)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    raise ValueError(
        f"{name} must be a quaternion [w,x,y,z], euler degrees [x,y,z], or a "
        f"Gf.Quat; got {value!r}"
    )
