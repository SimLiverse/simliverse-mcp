"""
Legged robots: quadrupeds and humanoids.

Locomotion for a legged robot is a learned policy in practice — Isaac Lab ships
trained gait controllers, and hand-authoring one is not something an agent should
attempt mid-task. What this layer provides is honest about that boundary:

  * joint-level and per-limb control that always works,
  * standing, posture holding, and balance measurement,
  * a hook to drive a trained locomotion policy when one is available.

The failure mode this avoids is an agent cheerfully "walking" a quadruped by
teleporting its base, which looks fine in a render and is physically meaningless.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._compat import as_vec3
from .base import Morphology, Robot

logger = logging.getLogger("simliverse_sim.robots.legged")


class LocomotionError(RuntimeError):
    pass


class LeggedRobot(Robot):
    """Common behaviour for anything that stands on legs."""

    morphology = Morphology.QUADRUPED

    def __init__(self, prim_path: str, *, scene: Any = None) -> None:
        super().__init__(prim_path, scene=scene)
        self.leg_indices = self.groups.legs
        self.limbs = self._group_limbs()
        self._policy: Any = None
        self._stand_pose: np.ndarray | None = None

    def _group_limbs(self) -> dict[str, list[int]]:
        """Group leg joints into limbs by the standard FL/FR/RL/RR-style prefix."""
        names = self.joint_names
        limbs: dict[str, list[int]] = {}
        for index in self.leg_indices:
            name = names[index].upper()
            key = next(
                (token for token in ("FL", "FR", "RL", "RR", "HL", "HR", "LF", "RF", "LH", "RH")
                 if token in name),
                "left" if "LEFT" in name else "right" if "RIGHT" in name else "unknown",
            )
            limbs.setdefault(key, []).append(index)
        return limbs

    # ── Posture ───────────────────────────────────────────────────────────────

    def capture_stand_pose(self) -> np.ndarray:
        """Record the current joint pose as the nominal standing posture."""
        self._stand_pose = self.joint_positions.copy()
        return self._stand_pose

    def stand(self, *, settle_steps: int = 120) -> bool:
        """Hold the nominal standing pose and report whether the robot stayed up.

        With no recorded pose, the robot's authored default joint state is used.
        Returns the result of a balance check rather than assuming success.
        """
        pose = self._stand_pose if self._stand_pose is not None else self.joint_positions.copy()
        self.set_joint_positions(pose, settle_steps=settle_steps)
        return self.is_upright()

    def set_limb_pose(self, limb: str, positions: Any, *, settle_steps: int = 20) -> None:
        if limb not in self.limbs:
            raise ValueError(f"No limb {limb!r}. Known limbs: {sorted(self.limbs)}")
        indices = self.limbs[limb]
        self.set_joint_positions(positions, indices=indices, settle_steps=settle_steps)

    # ── Balance ───────────────────────────────────────────────────────────────

    def is_upright(self, *, max_tilt_deg: float = 45.0) -> bool:
        """True when the base is within `max_tilt_deg` of level.

        This is the check that distinguishes "the robot is standing" from "the
        robot fell over and the scene still rendered".
        """
        return self.tilt_degrees() <= max_tilt_deg

    def tilt_degrees(self) -> float:
        """Angle between the base's up axis and world up."""
        w, x, y, z = self.base_orientation
        # Third column of the rotation matrix = the body's local +Z in world space.
        up = np.array(
            [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
        )
        cosine = float(np.clip(np.dot(up, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def base_height(self) -> float:
        return float(self.base_position[2])

    # ── Locomotion policy ─────────────────────────────────────────────────────

    def attach_policy(self, policy: Any) -> None:
        """Attach a trained locomotion policy.

        `policy` must be a callable taking an observation array and returning
        joint targets — the shape Isaac Lab's exported policies use.
        """
        self._policy = policy

    def walk(self, velocity_command: Any = (0.5, 0.0, 0.0), *, steps: int = 200) -> dict[str, Any]:
        """Drive the attached locomotion policy for `steps` physics steps.

        Requires a policy. There is deliberately no fallback that fakes walking
        by moving the base — a gait that was not produced by the legs is not
        locomotion, and reporting it as such would make every downstream
        verification meaningless.
        """
        if self._policy is None:
            raise LocomotionError(
                "No locomotion policy attached. Legged locomotion needs a trained "
                "controller — load one from Isaac Lab and call attach_policy(). "
                "For posture and manipulation tasks you can use stand() and "
                "set_limb_pose() without a policy."
            )

        command = as_vec3(velocity_command, name="velocity_command")
        start = self.base_position.copy()
        self.scene.play()

        for _ in range(steps):
            observation = self._observation(command)
            self.set_joint_positions(self._policy(observation), settle_steps=0)
            self.scene.step(1)

        displacement = self.base_position - start
        return {
            "commanded_velocity": command.tolist(),
            "displacement": displacement.round(4).tolist(),
            "distance": round(float(np.linalg.norm(displacement[:2])), 4),
            "upright": self.is_upright(),
            "tilt_degrees": round(self.tilt_degrees(), 2),
            "base_height": round(self.base_height(), 4),
        }

    def _observation(self, command: np.ndarray) -> np.ndarray:
        """Standard locomotion observation: base state, joint state, command."""
        return np.concatenate(
            [
                self.base_orientation,
                self.joint_positions,
                self.joint_velocities,
                command,
            ]
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        names = self.joint_names
        info["locomotion"] = {
            "limbs": {limb: [names[i] for i in idx] for limb, idx in self.limbs.items()},
            "policy_attached": self._policy is not None,
            "upright": self.is_upright(),
            "tilt_degrees": round(self.tilt_degrees(), 2),
            "base_height": round(self.base_height(), 4),
        }
        if self._policy is None:
            info["locomotion"]["note"] = (
                "No locomotion policy attached — walk() will raise. stand(), "
                "set_limb_pose() and joint-level control work regardless."
            )
        return info


class Humanoid(LeggedRobot):
    """A humanoid: legs for locomotion, arms for manipulation, plus a torso."""

    morphology = Morphology.HUMANOID

    def __init__(self, prim_path: str, *, scene: Any = None) -> None:
        super().__init__(prim_path, scene=scene)
        from .manipulator import Gripper

        self.arm_indices = self.groups.arms
        self.torso_indices = self.groups.torso
        self.head_indices = self.groups.head
        self.hands = Gripper(self, self.groups.gripper)

    def set_arm_pose(self, positions: Any, *, settle_steps: int = 30) -> None:
        self.set_joint_positions(positions, indices=self.arm_indices, settle_steps=settle_steps)

    def set_torso_pose(self, positions: Any, *, settle_steps: int = 30) -> None:
        if not self.torso_indices:
            raise ValueError(f"{self.prim_path} has no torso joints.")
        self.set_joint_positions(positions, indices=self.torso_indices, settle_steps=settle_steps)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        names = self.joint_names
        info["upper_body"] = {
            "arm_joints": [names[i] for i in self.arm_indices],
            "torso_joints": [names[i] for i in self.torso_indices],
            "head_joints": [names[i] for i in self.head_indices],
            "hand_joints": self.hands.joint_names,
        }
        return info
