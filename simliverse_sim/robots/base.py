"""
Robot base: articulation access, joint naming, drive health, and morphology
classification.

Everything common to every robot type lives here. Morphology-specific control —
Cartesian reaching, driving, walking, flying — lives in the subclasses, because
"move the end effector" is meaningless for a quadcopter and "set wheel speeds"
is meaningless for an arm.

Classification is structural: it reads the joint set and articulation topology
rather than looking for "spot" or "h1" in the prim path. Path-string matching is
what the previous `inspect_robot` did, and it misclassifies any robot someone
named differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import articulation_action, as_vec3, get_stage, single_articulation

if TYPE_CHECKING:
    from ..scene import Scene

logger = logging.getLogger("simliverse_sim.robots.base")


class Morphology(str, Enum):
    MANIPULATOR = "manipulator"
    DEXTEROUS_HAND = "dexterous_hand"
    WHEELED = "wheeled"
    QUADRUPED = "quadruped"
    HUMANOID = "humanoid"
    AERIAL = "aerial"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    UNKNOWN = "unknown"


# Joint-name tokens per functional group. Matching is on joint names because
# those come from the robot's own URDF/USD and are far more stable than the prim
# path a user happened to spawn it at.
GRIPPER_TOKENS = (
    "finger", "gripper", "knuckle", "jaw", "grip_", "claw",
    # Per-finger naming used by dexterous hands (Allegro, Shadow, LEAP).
    "thumb", "index", "middle", "ring", "pinky", "little", "forefinger",
    # Shadow Hand's abbreviated joints: FFJ1, MFJ2, RFJ3, LFJ4, THJ5.
    "ffj", "mfj", "rfj", "lfj", "thj",
)
WHEEL_TOKENS = ("wheel", "caster", "roller")
STEER_TOKENS = ("steer", "steering")
LEG_TOKENS = ("hip", "thigh", "calf", "knee", "ankle", "foot", "shank", "haa", "hfe", "kfe")
ARM_TOKENS = ("shoulder", "elbow", "wrist", "upper_arm", "forearm", "arm_")
TORSO_TOKENS = ("torso", "waist", "spine", "pelvis", "trunk")
HEAD_TOKENS = ("head", "neck")
ROTOR_TOKENS = ("rotor", "prop", "propeller", "motor_", "thruster")


def _matching(names: list[str], tokens: tuple[str, ...]) -> list[int]:
    return [i for i, name in enumerate(names) if any(t in name.lower() for t in tokens)]


@dataclass
class JointGroups:
    """Functional decomposition of an articulation's degrees of freedom."""

    gripper: list[int]
    wheels: list[int]
    steering: list[int]
    legs: list[int]
    arms: list[int]
    torso: list[int]
    head: list[int]
    rotors: list[int]
    other: list[int]

    @classmethod
    def classify(cls, joint_names: list[str]) -> "JointGroups":
        steering = _matching(joint_names, STEER_TOKENS)
        # "steering" contains "ring", which is also a finger name. Steering wins:
        # a joint named steer-something is never a finger.
        gripper = [i for i in _matching(joint_names, GRIPPER_TOKENS) if i not in steering]
        wheels = _matching(joint_names, WHEEL_TOKENS)
        legs = _matching(joint_names, LEG_TOKENS)
        arms = _matching(joint_names, ARM_TOKENS)
        torso = _matching(joint_names, TORSO_TOKENS)
        head = _matching(joint_names, HEAD_TOKENS)
        rotors = _matching(joint_names, ROTOR_TOKENS)

        claimed = set(gripper + wheels + steering + legs + arms + torso + head + rotors)
        other = [i for i in range(len(joint_names)) if i not in claimed]
        return cls(gripper, wheels, steering, legs, arms, torso, head, rotors, other)

    def to_dict(self, joint_names: list[str]) -> dict[str, list[str]]:
        return {
            group: [joint_names[i] for i in indices]
            for group, indices in (
                ("gripper", self.gripper),
                ("wheels", self.wheels),
                ("steering", self.steering),
                ("legs", self.legs),
                ("arms", self.arms),
                ("torso", self.torso),
                ("head", self.head),
                ("rotors", self.rotors),
                ("other", self.other),
            )
            if indices
        }


def classify_morphology(joint_names: list[str], groups: JointGroups) -> Morphology:
    """Infer what kind of robot this is from its joint structure."""
    dof = len(joint_names)

    if groups.rotors and len(groups.rotors) >= 3:
        return Morphology.AERIAL

    has_legs = len(groups.legs) >= 8      # four limbs x >= 2 joints
    has_arms = len(groups.arms) >= 4
    has_wheels = len(groups.wheels) >= 2

    if has_legs and has_arms:
        return Morphology.HUMANOID
    if has_legs:
        # A biped without arms is still humanoid-shaped for control purposes.
        return Morphology.QUADRUPED if len(groups.legs) >= 12 else Morphology.HUMANOID
    if has_wheels and has_arms:
        return Morphology.MOBILE_MANIPULATOR
    if has_wheels:
        return Morphology.WHEELED

    # A standalone hand: many finger joints, no arm chain to carry it.
    if len(groups.gripper) >= 6 and not has_arms:
        return Morphology.DEXTEROUS_HAND
    if has_arms or dof >= 5:
        return Morphology.MANIPULATOR
    return Morphology.UNKNOWN


class Robot:
    """Base handle for any articulated robot.

    Instantiating `Robot` directly gives you joint-level access that works for
    every morphology. Use `spawn()` or `attach()` to get the morphology-specific
    subclass with the control methods that make sense for that body.
    """

    morphology: Morphology = Morphology.UNKNOWN

    def __init__(self, prim_path: str, *, scene: "Scene | None" = None) -> None:
        from ..scene import Scene as _Scene

        self.prim_path = prim_path
        self.scene = scene or _Scene.get()

        self._articulation = single_articulation(prim_path)
        self.scene.play()
        # An articulation cannot initialize before physics has ticked; doing this
        # here removes a whole class of "the robot ignores my commands" reports.
        self.scene.step(4)
        self._articulation.initialize()

        self.groups = JointGroups.classify(self.joint_names)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.prim_path} dof={self.dof}>"

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def attach(prim_path: str, *, scene: "Scene | None" = None, **kwargs: Any) -> "Robot":
        """Wrap an existing robot on the stage in the right subclass for its body."""
        from .library import specialize

        probe = Robot(prim_path, scene=scene)
        return specialize(probe, **kwargs)

    @staticmethod
    def spawn(
        robot_type: str,
        *,
        prim_path: str | None = None,
        position: Any = (0.0, 0.0, 0.0),
        scene: "Scene | None" = None,
        **kwargs: Any,
    ) -> "Robot":
        """Load a robot from the Isaac asset library and return a typed handle."""
        from .library import spawn_robot

        return spawn_robot(
            robot_type, prim_path=prim_path, position=position, scene=scene, **kwargs
        )

    # ── Joint state ───────────────────────────────────────────────────────────

    @property
    def dof(self) -> int:
        return int(self._articulation.num_dof)

    @property
    def joint_names(self) -> list[str]:
        return list(self._articulation.dof_names)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.asarray(self._articulation.get_joint_positions(), dtype=float)

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self._articulation.get_joint_velocities(), dtype=float)

    @property
    def joint_limits(self) -> list[tuple[float | None, float | None]]:
        try:
            return [
                (float(low), float(high))
                for low, high in np.asarray(self._articulation.get_dof_limits())
            ]
        except Exception:
            return [(None, None)] * self.dof

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"No joint {name!r} on {self.prim_path}. Joints: {self.joint_names}"
            ) from exc

    def set_joint_positions(
        self, targets: Any, *, indices: list[int] | None = None, settle_steps: int = 0
    ) -> None:
        """Command joint position targets, optionally for a subset of joints."""
        current = self.joint_positions.copy()
        values = np.asarray(targets, dtype=float).reshape(-1)
        if indices is None:
            if values.size != self.dof:
                raise ValueError(f"Expected {self.dof} values, got {values.size}")
            current = values
        else:
            if values.size != len(indices):
                raise ValueError(f"Expected {len(indices)} values, got {values.size}")
            for slot, value in zip(indices, values):
                current[slot] = value
        self._controller().apply_action(articulation_action(joint_positions=current))
        if settle_steps:
            self.scene.step(settle_steps)

    def set_joint_velocities(
        self, targets: Any, *, indices: list[int] | None = None, settle_steps: int = 0
    ) -> None:
        current = np.zeros(self.dof)
        values = np.asarray(targets, dtype=float).reshape(-1)
        if indices is None:
            current = values
        else:
            for slot, value in zip(indices, values):
                current[slot] = value
        self._controller().apply_action(articulation_action(joint_velocities=current))
        if settle_steps:
            self.scene.step(settle_steps)

    def _controller(self) -> Any:
        return self._articulation.get_articulation_controller()

    # ── Base pose ─────────────────────────────────────────────────────────────

    @property
    def base_position(self) -> np.ndarray:
        """World position of the articulation root."""
        try:
            position, _ = self._articulation.get_world_pose()
            return np.asarray(position, dtype=float)
        except Exception:
            from pxr import UsdGeom

            matrix = UsdGeom.Xformable(
                get_stage().GetPrimAtPath(self.prim_path)
            ).ComputeLocalToWorldTransform(0)
            return np.asarray(matrix.ExtractTranslation(), dtype=float)

    @property
    def base_orientation(self) -> np.ndarray:
        _, quaternion = self._articulation.get_world_pose()
        return np.asarray(quaternion, dtype=float)

    def set_base_pose(self, position: Any = None, orientation: Any = None) -> None:
        self._articulation.set_world_pose(
            position=as_vec3(position, name="position") if position is not None else None,
            orientation=orientation,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def drive_health(self) -> list[dict[str, Any]]:
        """Joints whose drives will not track a position target.

        Stiffness and damping both zero means the drive is off; the joint will
        silently ignore every command. That presents as "the robot doesn't move".
        """
        from pxr import Usd, UsdPhysics

        problems: list[dict[str, Any]] = []
        root = get_stage().GetPrimAtPath(self.prim_path)
        for prim in Usd.PrimRange(root):
            for kind in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, kind)
                if not drive:
                    continue
                stiffness = drive.GetStiffnessAttr().Get() or 0.0
                damping = drive.GetDampingAttr().Get() or 0.0
                if stiffness == 0.0 and damping == 0.0:
                    problems.append(
                        {
                            "joint": str(prim.GetPath()),
                            "stiffness": stiffness,
                            "damping": damping,
                            "problem": "drive disabled — will not track position targets",
                        }
                    )
        return problems

    def repair_drives(self) -> list[str]:
        """Enable joint drives that are switched off. **Explicit request only.**

        Never call this to make a task succeed. It modifies the robot: after it
        runs, the simulation no longer models the asset that was loaded, and
        anything measured or trained afterwards describes a robot that does not
        exist outside this session. On a sim-to-real platform that is a worse
        outcome than the task failing, because the failure is silent and shows up
        later as hardware that "doesn't match sim".

        A misconfigured asset is a finding to report. `drive_health()` finds
        them; motion code refuses and says so. This exists for when the user has
        seen that report and decided they want the robot changed.

        Isaac Sim's Franka FR3 is the common case: `fr3_finger_joint1` has
        stiffness 60000 / damping 6000 and `fr3_finger_joint2` has 0 / 0, because
        the asset expects joint2 to mimic joint1 and no mimic joint is
        configured.

        Gains are copied from the healthiest sibling drive on the same robot, so
        a correctly authored asset is left untouched.
        """
        from pxr import Usd, UsdPhysics

        root = get_stage().GetPrimAtPath(self.prim_path)
        drives: list[tuple[str, Any, float, float, float]] = []
        for prim in Usd.PrimRange(root):
            for kind in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, kind)
                if not drive:
                    continue
                drives.append(
                    (
                        str(prim.GetPath()),
                        drive,
                        float(drive.GetStiffnessAttr().Get() or 0.0),
                        float(drive.GetDampingAttr().Get() or 0.0),
                        float(drive.GetMaxForceAttr().Get() or 0.0),
                    )
                )

        broken = [d for d in drives if d[2] == 0.0 and d[3] == 0.0]
        if not broken:
            return []

        stiffness = max((d[2] for d in drives), default=0.0)
        damping = max((d[3] for d in drives), default=0.0)
        # A finite maxForce from a sibling beats the `inf` an unconfigured drive
        # carries, which would let a repaired finger crush what it is holding.
        finite = [d[4] for d in drives if 0.0 < d[4] < float("inf")]
        max_force = min(finite) if finite else 0.0

        if stiffness <= 0.0:
            # Every drive on the robot is off, so there is nothing to copy from.
            # These are the Franka FR3 factory finger values and a sane default
            # for a parallel jaw.
            stiffness, damping, max_force = 60_000.0, 6_000.0, 100.0

        repaired: list[str] = []
        for path, drive, _, _, force in broken:
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping or 0.1 * stiffness)
            if max_force > 0.0 and not 0.0 < force < float("inf"):
                drive.GetMaxForceAttr().Set(max_force)
            repaired.append(path)

        logger.warning(
            "Enabled %d disabled joint drive(s) on %s (stiffness=%s damping=%s): %s",
            len(repaired),
            self.prim_path,
            stiffness,
            damping,
            ", ".join(repaired),
        )
        return repaired

    def links(self) -> list[str]:
        """Every link prim under this robot — useful for finding a TCP or foot."""
        from pxr import Usd, UsdPhysics

        root = get_stage().GetPrimAtPath(self.prim_path)
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]

    def touching(self, other_path_prefix: str) -> bool:
        """True when any link of this robot is in contact with the given prim."""
        from ..objects import RigidObject

        try:
            return any(
                body.startswith(self.prim_path)
                for body in RigidObject(other_path_prefix, scene=self.scene).contact_bodies()
            )
        except Exception:
            return False

    # ── Description ───────────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """Everything an agent needs to know about this robot, in one call."""
        return {
            "prim_path": self.prim_path,
            "morphology": self.morphology.value,
            "controller": type(self).__name__,
            "dof": self.dof,
            "joint_names": self.joint_names,
            "joint_positions": self.joint_positions.round(4).tolist(),
            "joint_groups": self.groups.to_dict(self.joint_names),
            "base_position": self.base_position.round(4).tolist(),
            "drive_problems": self.drive_health(),
            "capabilities": self.capabilities(),
        }

    def capabilities(self) -> list[str]:
        """Control methods available on this handle, so an agent need not guess."""
        skip = {"attach", "spawn", "capabilities", "describe"}
        return sorted(
            name
            for name in dir(self)
            if not name.startswith("_")
            and name not in skip
            and callable(getattr(type(self), name, None))
        )
