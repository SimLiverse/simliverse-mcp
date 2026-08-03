"""
Wheeled robots and mobile manipulators.

A wheeled base has no end effector, so `move_ee_to` is meaningless for it. The
right primitive is "drive at this linear and angular velocity", plus a blocking
`drive_to` that closes the loop on the base pose.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import as_vec3
from .base import Morphology, Robot

if TYPE_CHECKING:
    from ..objects import RigidObject

logger = logging.getLogger("simliverse_sim.robots.mobile")


class NavigationError(RuntimeError):
    pass


class WheeledRobot(Robot):
    """A differential- or skid-steer wheeled base."""

    morphology = Morphology.WHEELED

    def __init__(
        self,
        prim_path: str,
        *,
        scene: Any = None,
        wheel_radius: float = 0.06,
        wheel_base: float = 0.4,
    ) -> None:
        super().__init__(prim_path, scene=scene)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.wheel_indices = self.groups.wheels
        if not self.wheel_indices:
            raise NavigationError(
                f"{prim_path} has no joints that look like wheels. Joints: {self.joint_names}"
            )

    # ── Velocity control ──────────────────────────────────────────────────────

    def set_wheel_velocities(self, velocities: Any, *, settle_steps: int = 0) -> None:
        """Command each wheel joint directly, in rad/s."""
        self.set_joint_velocities(
            velocities, indices=self.wheel_indices, settle_steps=settle_steps
        )

    def drive(self, linear: float = 0.0, angular: float = 0.0, *, steps: int = 0) -> None:
        """Drive at `linear` m/s forward and `angular` rad/s yaw.

        Wheels are split left/right by the sign of their y offset from the base,
        so this works without hardcoding a joint order per robot model.
        """
        left, right = self._split_wheels()
        v_left = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        v_right = (linear + angular * self.wheel_base / 2.0) / self.wheel_radius

        targets = np.zeros(self.dof)
        for index in left:
            targets[index] = v_left
        for index in right:
            targets[index] = v_right

        self._controller_apply(targets)
        if steps:
            self.scene.step(steps)

    def _controller_apply(self, targets: np.ndarray) -> None:
        from .._compat import articulation_action

        self._controller().apply_action(articulation_action(joint_velocities=targets))

    def _split_wheels(self) -> tuple[list[int], list[int]]:
        """Partition wheel joints into left and right by link position."""
        from pxr import UsdGeom

        from .._compat import get_stage

        names = self.joint_names
        left, right = [], []
        stage = get_stage()

        for index in self.wheel_indices:
            name = names[index].lower()
            side: float | None = None
            if "left" in name or name.endswith("_l"):
                side = 1.0
            elif "right" in name or name.endswith("_r"):
                side = -1.0
            else:
                # Fall back to geometry: find a link whose name matches and read
                # its lateral offset from the base.
                for link in self.links():
                    if name.replace("_joint", "") in link.lower():
                        matrix = UsdGeom.Xformable(
                            stage.GetPrimAtPath(link)
                        ).ComputeLocalToWorldTransform(0)
                        side = float(matrix.ExtractTranslation()[1]) - self.base_position[1]
                        break
            (left if (side or 0.0) >= 0 else right).append(index)

        if not left or not right:
            # Undifferentiable naming — split evenly and warn via the exception
            # path only if the caller actually tries to turn.
            midpoint = len(self.wheel_indices) // 2
            left = self.wheel_indices[:midpoint]
            right = self.wheel_indices[midpoint:]
        return left, right

    def stop(self, *, settle_steps: int = 10) -> None:
        self.set_wheel_velocities([0.0] * len(self.wheel_indices), settle_steps=settle_steps)

    # ── Closed-loop navigation ────────────────────────────────────────────────

    def drive_to(
        self,
        position: Any,
        *,
        tolerance: float = 0.15,
        max_speed: float = 0.6,
        max_steps: int = 3000,
        raise_on_fail: bool = True,
    ) -> bool:
        """Drive the base to a world-space XY position. Blocks until arrival.

        A simple turn-then-go controller: it is not a path planner and will drive
        into obstacles. For anything with clutter, plan waypoints and call this
        for each leg.
        """
        target = as_vec3(position, name="position")[:2]
        self.scene.play()

        for _ in range(max_steps):
            current = self.base_position[:2]
            offset = target - current
            distance = float(np.linalg.norm(offset))
            if distance < tolerance:
                self.stop()
                return True

            heading = self._heading()
            desired = float(np.arctan2(offset[1], offset[0]))
            error = float(np.arctan2(np.sin(desired - heading), np.cos(desired - heading)))

            # Turn in place when badly misaligned, otherwise arc toward the goal.
            if abs(error) > 0.4:
                self.drive(linear=0.0, angular=float(np.clip(2.0 * error, -1.5, 1.5)), steps=1)
            else:
                self.drive(
                    linear=float(min(max_speed, distance)),
                    angular=float(np.clip(1.5 * error, -1.0, 1.0)),
                    steps=1,
                )

        self.stop()
        if raise_on_fail:
            raise NavigationError(
                f"Did not reach {target.round(3).tolist()} within {max_steps} steps. "
                f"The base may be stuck against an obstacle, or the wheel drives may "
                f"be disabled — check describe()['drive_problems']."
            )
        return False

    def _heading(self) -> float:
        """Base yaw in radians, from the root orientation quaternion (w, x, y, z)."""
        w, x, y, z = self.base_orientation
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        left, right = self._split_wheels()
        names = self.joint_names
        info["drive"] = {
            "wheel_radius": self.wheel_radius,
            "wheel_base": self.wheel_base,
            "left_wheels": [names[i] for i in left],
            "right_wheels": [names[i] for i in right],
            "heading_rad": round(self._heading(), 4),
        }
        return info


class MobileManipulator(WheeledRobot):
    """A wheeled base carrying an arm.

    Composes both control surfaces: `drive_to` positions the base, then the
    manipulator methods reach from wherever it ended up. The arm's workspace
    moves with the base, so drive first and reach second.
    """

    morphology = Morphology.MOBILE_MANIPULATOR

    def __init__(self, prim_path: str, *, scene: Any = None, **kwargs: Any) -> None:
        super().__init__(prim_path, scene=scene, **{
            k: v for k, v in kwargs.items() if k in ("wheel_radius", "wheel_base")
        })
        from .manipulator import Gripper

        self.gripper = Gripper(self, self.groups.gripper)
        self.arm_joint_indices = self.groups.arms

    def set_arm_pose(self, positions: Any, *, settle_steps: int = 30) -> None:
        self.set_joint_positions(
            positions, indices=self.arm_joint_indices, settle_steps=settle_steps
        )

    def is_grasping(self, obj: "RigidObject", *, min_contacts: int = 1) -> bool:
        touching = [b for b in obj.contact_bodies() if b.startswith(self.prim_path)]
        return len(touching) >= min_contacts

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        names = self.joint_names
        info["arm"] = {
            "joint_names": [names[i] for i in self.arm_joint_indices],
            "gripper_joints": self.gripper.joint_names,
        }
        info["note"] = (
            "Cartesian arm control needs an RMPflow config for this robot; if one "
            "exists, attach it with Manipulator(prim_path, rmp_config=...). "
            "Otherwise drive the arm joints directly with set_arm_pose()."
        )
        return info
