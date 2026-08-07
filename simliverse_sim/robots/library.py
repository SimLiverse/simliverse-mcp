"""
Robot catalogue and the spawn factory.

`spawn_robot` loads an asset and returns the handle whose control surface matches
the body — a `Manipulator` for an arm, a `WheeledRobot` for a rover, a `Humanoid`
for a humanoid. The class is chosen from the articulation's actual joint
structure after loading, so a robot that is not in the catalogue below still gets
the right controller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .._compat import add_reference, as_vec3, assets_root, get_stage
from .base import Morphology, Robot

logger = logging.getLogger("simliverse_sim.robots.library")


@dataclass(frozen=True)
class RobotAsset:
    key: str
    asset_path: str
    morphology: Morphology
    # RMPflow / Lula configuration name, where Isaac ships one. Without it,
    # Cartesian control is unavailable and joint control is the fallback.
    motion_config: str | None = None
    description: str = ""


# Asset paths are the 6.0 layout, where NVIDIA regrouped every robot under a
# vendor directory: Robots/Franka/franka.usd became
# Robots/FrankaRobotics/FrankaPanda/franka.usd. Nine of these seventeen entries
# still pointed at the old flat layout and 404'd — including the Franka, which
# is the robot every manipulation task reaches for first. Verified against the
# live asset server, not guessed.
CATALOGUE: dict[str, RobotAsset] = {
    # ── Manipulators ──────────────────────────────────────────────────────────
    "franka": RobotAsset(
        "franka", "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", Morphology.MANIPULATOR,
        "Franka", "7-DOF arm with a parallel gripper.",
    ),
    "fr3": RobotAsset(
        "fr3", "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd", Morphology.MANIPULATOR,
        "FR3", "Franka Research 3 — 7-DOF arm with a parallel gripper.",
    ),
    "ur10": RobotAsset(
        "ur10", "/Isaac/Robots/UniversalRobots/ur10/ur10.usd", Morphology.MANIPULATOR,
        "UR10", "6-DOF arm, no gripper by default.",
    ),
    "ur5": RobotAsset(
        "ur5", "/Isaac/Robots/UniversalRobots/ur5/ur5.usd", Morphology.MANIPULATOR,
        "UR5", "6-DOF arm, no gripper by default.",
    ),
    "kuka_iiwa": RobotAsset(
        "kuka_iiwa", "/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd", Morphology.MANIPULATOR,
        "Kuka_iiwa7", "7-DOF arm.",
    ),
    "kinova_gen3": RobotAsset(
        "kinova_gen3", "/Isaac/Robots/Kinova/Gen3/gen3n7_instanceable.usd", Morphology.MANIPULATOR,
        "Kinova_Gen3", "7-DOF arm.",
    ),
    # ── Dexterous hands ───────────────────────────────────────────────────────
    "allegro_hand": RobotAsset(
        "allegro_hand", "/Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
        Morphology.DEXTEROUS_HAND, None, "16-DOF four-finger hand.",
    ),
    "shadow_hand": RobotAsset(
        "shadow_hand", "/Isaac/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd",
        Morphology.DEXTEROUS_HAND, None, "24-DOF five-finger hand.",
    ),
    # ── Wheeled ───────────────────────────────────────────────────────────────
    "carter": RobotAsset(
        "carter", "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd", Morphology.WHEELED,
        None, "Differential-drive research AMR.",
    ),
    "jetbot": RobotAsset(
        "jetbot", "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd", Morphology.WHEELED,
        None, "Small two-wheel differential-drive robot.",
    ),
    "kaya": RobotAsset(
        "kaya", "/Isaac/Robots/NVIDIA/Kaya/kaya.usd", Morphology.WHEELED,
        None, "Three-wheel holonomic robot.",
    ),
    # ── Quadrupeds ────────────────────────────────────────────────────────────
    "anymal_c": RobotAsset(
        "anymal_c", "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    "unitree_go2": RobotAsset(
        "unitree_go2", "/Isaac/Robots/Unitree/Go2/go2.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    "spot": RobotAsset(
        "spot", "/Isaac/Robots/BostonDynamics/spot/spot.usd", Morphology.QUADRUPED,
        None, "12-DOF quadruped. Locomotion needs a trained policy.",
    ),
    # ── Humanoids ─────────────────────────────────────────────────────────────
    "unitree_h1": RobotAsset(
        "unitree_h1", "/Isaac/Robots/Unitree/H1/h1.usd", Morphology.HUMANOID,
        None, "Full-size humanoid. Locomotion needs a trained policy.",
    ),
    "unitree_g1": RobotAsset(
        "unitree_g1", "/Isaac/Robots/Unitree/G1/g1.usd", Morphology.HUMANOID,
        None, "Compact humanoid. Locomotion needs a trained policy.",
    ),
    # ── Aerial ────────────────────────────────────────────────────────────────
    "quadcopter": RobotAsset(
        "quadcopter", "/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd", Morphology.AERIAL,
        None, "Small quadcopter, thrust-controlled.",
    ),
}


_CONTROLLERS: dict[Morphology, str] = {
    Morphology.MANIPULATOR: "Manipulator",
    Morphology.DEXTEROUS_HAND: "DexterousHand",
    Morphology.WHEELED: "WheeledRobot",
    Morphology.MOBILE_MANIPULATOR: "MobileManipulator",
    Morphology.QUADRUPED: "LeggedRobot",
    Morphology.HUMANOID: "Humanoid",
    Morphology.AERIAL: "AerialRobot",
}


def _controller_class(morphology: Morphology) -> type[Robot]:
    from . import aerial, legged, manipulator, mobile

    return {
        Morphology.MANIPULATOR: manipulator.Manipulator,
        Morphology.DEXTEROUS_HAND: manipulator.DexterousHand,
        Morphology.WHEELED: mobile.WheeledRobot,
        Morphology.MOBILE_MANIPULATOR: mobile.MobileManipulator,
        Morphology.QUADRUPED: legged.LeggedRobot,
        Morphology.HUMANOID: legged.Humanoid,
        Morphology.AERIAL: aerial.AerialRobot,
    }.get(morphology, Robot)


def resolve(robot_type: str) -> RobotAsset:
    """Look up a catalogue entry, tolerating partial names."""
    key = robot_type.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CATALOGUE:
        return CATALOGUE[key]
    matches = [k for k in CATALOGUE if key in k or k in key]
    if matches:
        return CATALOGUE[sorted(matches, key=len)[0]]
    raise ValueError(
        f"Unknown robot {robot_type!r}. Known robots: {', '.join(sorted(CATALOGUE))}. "
        f"For anything else, load the USD yourself and call Robot.attach(prim_path)."
    )


def specialize(probe: Robot, **kwargs: Any) -> Robot:
    """Re-wrap a generic `Robot` as the subclass matching its actual structure."""
    from .base import classify_morphology

    morphology = classify_morphology(probe.joint_names, probe.groups)
    controller = _controller_class(morphology)
    if controller is Robot:
        logger.info(
            "No specialised controller for %s (morphology=%s); joint-level control only.",
            probe.prim_path,
            morphology.value,
        )
        return probe
    return controller(probe.prim_path, scene=probe.scene, **kwargs)


def spawn_robot(
    robot_type: str,
    *,
    prim_path: str | None = None,
    position: Any = (0.0, 0.0, 0.0),
    scene: Any = None,
    **kwargs: Any,
) -> Robot:
    """Load a robot and return a handle with the right control surface."""
    from ..scene import Scene as _Scene
    from .base import classify_morphology

    asset = resolve(robot_type)
    prim_path = prim_path or f"/World/{asset.key}"
    scene = scene or _Scene.get()

    add_reference(assets_root() + asset.asset_path, prim_path)

    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(get_stage().GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*as_vec3(position, name="position")))

    if asset.morphology is Morphology.AERIAL:
        from .aerial import AerialRobot

        return AerialRobot(prim_path, scene=scene, **kwargs)

    # Load first, then classify from the real joint set — the catalogue's
    # morphology is a hint, but the articulation is the ground truth.
    probe = Robot(prim_path, scene=scene)
    morphology = classify_morphology(probe.joint_names, probe.groups)
    controller = _controller_class(morphology)

    if controller is Robot:
        return probe
    if morphology in (Morphology.MANIPULATOR, Morphology.DEXTEROUS_HAND):
        kwargs.setdefault("rmp_config", asset.motion_config)
        if morphology is Morphology.DEXTEROUS_HAND:
            kwargs.pop("rmp_config", None)
    return controller(prim_path, scene=scene, **kwargs)


def list_robots() -> list[dict[str, str]]:
    """The catalogue, for an agent to choose from."""
    return [
        {
            "key": asset.key,
            "morphology": asset.morphology.value,
            "cartesian_control": "yes" if asset.motion_config else "no",
            "description": asset.description,
        }
        for asset in sorted(CATALOGUE.values(), key=lambda a: (a.morphology.value, a.key))
    ]
