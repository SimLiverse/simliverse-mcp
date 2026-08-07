"""
Manipulators: arms with an end effector, and standalone dexterous hands.

Cartesian control comes from Lula/RMPflow, so control code names a pose in world
space rather than seven joint angles. That substitution is the whole reason
manipulation became expressible — see ADR 012 §1.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._compat import as_vec3, motion_generation
from .base import Morphology, Robot

if TYPE_CHECKING:
    from ..objects import RigidObject

logger = logging.getLogger("simliverse_sim.robots.manipulator")


class MotionError(RuntimeError):
    """A motion could not be completed — unreachable, blocked, or timed out."""


@dataclass
class MotionResult:
    reached: bool
    steps: int
    final_error: float
    target: list[float]

    def __bool__(self) -> bool:
        return self.reached


class Gripper:
    """Any set of finger joints that can open and close together.

    Covers a two-finger parallel jaw and a multi-finger hand alike: closing is
    "drive every finger joint toward closed until contact stops it", which is the
    same operation either way.
    """

    def __init__(self, robot: "Robot", joint_indices: list[int]) -> None:
        self._robot = robot
        self.joint_indices = joint_indices
        self._open_value: float | None = None
        self._closed_value: float | None = None

    def __repr__(self) -> str:
        return f"<Gripper {len(self.joint_indices)} joints: {self.joint_names}>"

    @property
    def exists(self) -> bool:
        return bool(self.joint_indices)

    @property
    def joint_names(self) -> list[str]:
        names = self._robot.joint_names
        return [names[i] for i in self.joint_indices]

    def _limits(self) -> tuple[float, float]:
        if self._open_value is None or self._closed_value is None:
            limits = self._robot.joint_limits
            lows = [limits[i][0] for i in self.joint_indices if limits[i][0] is not None]
            highs = [limits[i][1] for i in self.joint_indices if limits[i][1] is not None]
            # Prismatic fingers open at the upper limit; revolute finger joints
            # on a dexterous hand usually curl toward the upper limit instead.
            self._closed_value = float(max(lows)) if lows else 0.0
            self._open_value = float(min(highs)) if highs else 0.04
        return self._open_value, self._closed_value

    @property
    def open_width(self) -> float:
        return self._limits()[0]

    def _assert_can_grip(self) -> None:
        """Refuse to command fingers whose drives cannot exert force.

        A drive with stiffness and damping both zero is a PD controller with no
        gains: it produces no force however it is commanded. Isaac Sim's Franka
        FR3 ships that way on `fr3_finger_joint2`, expecting a mimic joint that
        this build does not configure.

        This reports rather than repairs. Writing gains onto the asset would make
        the grasp succeed while quietly changing the robot being simulated, and a
        policy trained against a gripper we silently modified does not transfer
        to the real one — the failure would surface as bad hardware, long after
        anyone could connect it to this. Changing a robot's dynamics is the
        user's call, not a side effect of `close()`.

        `Robot.repair_drives()` does it, when someone asks for it.
        """
        names = set(self.joint_names)
        disabled = [
            problem["joint"]
            for problem in self._robot.drive_health()
            if problem["joint"].rsplit("/", 1)[-1] in names
        ]
        if not disabled:
            return
        raise MotionError(
            f"{self._robot.prim_path} cannot grip: the drive on "
            f"{', '.join(disabled)} is disabled (stiffness and damping both 0), "
            f"so those fingers exert no force and will be pushed open by contact. "
            f"This is how the asset ships — it is not something this run broke. "
            f"Report it rather than working around it. If the user wants the "
            f"robot changed, `robot.repair_drives()` enables the drives, but that "
            f"alters the dynamics being simulated and must be their decision."
        )

    def set_position(self, value: float, *, settle_steps: int = 30) -> None:
        if not self.exists:
            raise MotionError(f"{self._robot.prim_path} has no gripper joints.")
        self._assert_can_grip()
        self._robot.set_joint_positions(
            [float(value)] * len(self.joint_indices),
            indices=self.joint_indices,
            settle_steps=settle_steps,
        )

    def open(self, *, settle_steps: int = 30) -> None:
        self.set_position(self._limits()[0], settle_steps=settle_steps)

    def close(self, *, settle_steps: int = 45) -> None:
        """Drive the fingers closed.

        Commanding fully-closed against a solid object is intentional: the drive
        pushes until contact stops it, and that residual push is the normal force
        a friction grasp depends on.
        """
        self.set_position(self._limits()[1], settle_steps=settle_steps)

    @property
    def position(self) -> float:
        positions = self._robot.joint_positions
        return float(np.mean([positions[i] for i in self.joint_indices]))


class Manipulator(Robot):
    """A robot arm with an end effector."""

    morphology = Morphology.MANIPULATOR

    def __init__(
        self,
        prim_path: str,
        *,
        scene: Any = None,
        rmp_config: str | None = None,
        end_effector_frame: str | None = None,
    ) -> None:
        super().__init__(prim_path, scene=scene)
        self._rmp_config_name = rmp_config
        self._end_effector_frame = end_effector_frame
        self._rmpflow: Any = None
        self._policy: Any = None
        self._ik: Any = None
        self.gripper = Gripper(self, self.groups.gripper)

    @property
    def arm_joint_indices(self) -> list[int]:
        finger = set(self.groups.gripper)
        return [i for i in range(self.dof) if i not in finger]

    # ── Motion policy ─────────────────────────────────────────────────────────

    def _ensure_motion_policy(self) -> None:
        if self._policy is not None:
            return
        mg = motion_generation()
        loader = mg.interface_config_loader

        name = self._rmp_config_name
        if name is None:
            supported = loader.get_supported_robot_policy_pairs()

            # Match on the robot's own joint names first, then the prim path.
            #
            # Path matching alone is what ADR 012 rejected for morphology
            # classification, and it fails here for the same reason: a Franka
            # Panda spawned at /World/Panda has no supported name inside "panda",
            # so Cartesian control silently became unavailable on a robot that
            # RMPflow fully supports. Joint names come from the asset's own
            # URDF/USD and survive whatever the user called the prim.
            joints = " ".join(self.joint_names).lower().replace("_", "")
            leaf = self.prim_path.rsplit("/", 1)[-1].lower().replace("_", "")

            # Asset joint prefixes whose names differ from the RMPflow config.
            aliases = {"panda": "Franka", "fr3": "FR3"}
            for token, config in aliases.items():
                if token in joints and config in supported:
                    name = config
                    break

            if name is None:
                for candidate in supported:
                    key = candidate.lower().replace("_", "")
                    if key in joints or key in leaf:
                        name = candidate
                        break
            if name is None:
                raise MotionError(
                    f"No RMPflow configuration matches {self.prim_path} "
                    f"(joints: {self.joint_names[:3]}...). Supported "
                    f"robots: {sorted(supported)}. Pass rmp_config= explicitly, or "
                    f"drive the joints directly with set_joint_positions."
                )
            self._rmp_config_name = name

        config = loader.load_supported_motion_policy_config(name, "RMPflow")
        self._rmpflow = mg.RmpFlow(**config)
        self._policy = mg.ArticulationMotionPolicy(
            robot_articulation=self._articulation,
            motion_policy=self._rmpflow,
            default_physics_dt=self.scene.dt,
        )
        frame = self._end_effector_frame or config.get("end_effector_frame_name")
        self._ik = mg.ArticulationKinematicsSolver(
            robot_articulation=self._articulation,
            kinematics_solver=mg.LulaKinematicsSolver(
                **loader.load_supported_lula_kinematics_solver_config(name)
            ),
            end_effector_frame_name=frame,
        )
        self._end_effector_frame = frame

    @property
    def ee_position(self) -> np.ndarray:
        self._ensure_motion_policy()
        position, _ = self._ik.compute_end_effector_pose()
        return np.asarray(position, dtype=float)

    @property
    def ee_orientation(self) -> np.ndarray:
        self._ensure_motion_policy()
        _, rotation = self._ik.compute_end_effector_pose()
        return np.asarray(rotation, dtype=float)

    def move_ee_to(
        self,
        position: Any,
        orientation: Any = None,
        *,
        tolerance: float = 0.005,
        max_steps: int = 600,
        hold_steps: int = 3,
        raise_on_fail: bool = True,
    ) -> MotionResult:
        """Drive the end effector to a world-space pose. Blocks until converged.

        Convergence requires staying inside `tolerance` for `hold_steps`
        consecutive steps, so flying through the target does not count as
        arriving.
        """
        self._ensure_motion_policy()
        target = as_vec3(position, name="position")
        self._rmpflow.set_end_effector_target(
            target_position=target,
            target_orientation=np.asarray(orientation) if orientation is not None else None,
        )

        self.scene.play()
        controller = self._controller()
        settled, error = 0, float("inf")

        for step in range(max_steps):
            self._rmpflow.update_world()
            controller.apply_action(self._policy.get_next_articulation_action())
            self.scene.step(1)

            error = float(np.linalg.norm(self.ee_position - target))
            settled = settled + 1 if error < tolerance else 0
            if settled >= hold_steps:
                return MotionResult(True, step + 1, error, target.tolist())

        result = MotionResult(False, max_steps, error, target.tolist())
        if raise_on_fail:
            raise MotionError(
                f"End effector did not reach {target.round(3).tolist()} within "
                f"{max_steps} steps (final error {error:.4f} m > {tolerance} m). The "
                f"target is likely outside the workspace or blocked by a collision."
            )
        return result

    def move_ee_by(self, delta: Any, **kwargs: Any) -> MotionResult:
        return self.move_ee_to(self.ee_position + as_vec3(delta, name="delta"), **kwargs)

    # ── Grasping ──────────────────────────────────────────────────────────────

    def is_grasping(self, obj: "RigidObject", *, min_contacts: int = 1) -> bool:
        """True when the object is genuinely held — measured from contact reports."""
        touching = [b for b in obj.contact_bodies() if b.startswith(self.prim_path)]
        return len(touching) >= min_contacts

    def grasp(
        self,
        obj: "RigidObject",
        *,
        approach_height: float = 0.12,
        grasp_offset: Any = (0.0, 0.0, 0.0),
        orientation: Any = None,
        verify_steps: int = 60,
    ) -> bool:
        """Approach, close on, and verify a grasp.

        Returns True only if the object is still held after `verify_steps` under
        gravity. Nothing here teleports the object into the hand — that shortcut
        is why the previous skills never generalised.
        """
        target = obj.position + as_vec3(grasp_offset, name="grasp_offset")
        self.gripper.open()
        self.move_ee_to(target + np.array([0.0, 0.0, approach_height]), orientation)
        self.move_ee_to(target, orientation)
        self.gripper.close()
        self.scene.step(verify_steps)
        return self.is_grasping(obj)

    def release(self, *, settle_steps: int = 20) -> None:
        self.gripper.open(settle_steps=settle_steps)

    def throw(
        self,
        obj: "RigidObject",
        *,
        direction: Any = (1.0, 0.0, 1.0),
        speed: float = 2.5,
        windup: float = 0.25,
        release_fraction: float = 0.6,
        observe_steps: int = 120,
    ) -> dict[str, Any]:
        """Throw a held object. The arm accelerates and releases mid-swing.

        Nothing sets the object's velocity directly — it leaves with whatever
        momentum the hand actually transferred, so the result is a real ballistic
        trajectory that can be measured and verified.
        """
        if not self.is_grasping(obj):
            raise MotionError(
                "Cannot throw: the object is not currently grasped. Call grasp() "
                "first and confirm it returned True."
            )

        vector = as_vec3(direction, name="direction")
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-6:
            raise ValueError("direction must be a non-zero vector")
        unit = vector / magnitude

        start = self.ee_position
        self.move_ee_to(start - unit * windup, raise_on_fail=False)

        sweep = float(speed) * 0.35 + windup
        end = start + unit * sweep
        release_at = start - unit * windup + unit * (sweep * release_fraction)

        self._ensure_motion_policy()
        self._rmpflow.set_end_effector_target(target_position=end)
        controller = self._controller()

        released, release_speed = False, 0.0
        previous = self.ee_position

        for _ in range(400):
            self._rmpflow.update_world()
            controller.apply_action(self._policy.get_next_articulation_action())
            self.scene.step(1)

            current = self.ee_position
            hand_speed = float(np.linalg.norm(current - previous)) / self.scene.dt
            previous = current

            if float(np.dot(current - release_at, unit)) >= 0.0:
                self.gripper.set_position(self.gripper.open_width, settle_steps=0)
                released, release_speed = True, hand_speed
                break

        if not released:
            self.release()
            raise MotionError(
                "The arm never reached the release point — the throw arc is likely "
                "outside the workspace. Try a shorter windup or a direction closer "
                "to the robot's reach."
            )

        apex, trajectory = -float("inf"), []
        for step in range(observe_steps):
            self.scene.step(1)
            position = obj.position
            apex = max(apex, float(position[2]))
            if step % 10 == 0:
                trajectory.append(position.round(4).tolist())

        final = obj.position
        return {
            "released": True,
            "release_hand_speed": round(release_speed, 3),
            "object_speed_after_release": round(obj.speed, 3),
            "apex_height": round(apex, 4),
            "landing_position": final.round(4).tolist(),
            "horizontal_distance": round(float(np.linalg.norm((final - start)[:2])), 4),
            "still_held": self.is_grasping(obj),
            "trajectory": trajectory,
        }

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["gripper"] = {
            "present": self.gripper.exists,
            "joint_names": self.gripper.joint_names,
            "open_width": self.gripper.open_width if self.gripper.exists else None,
            "current_position": self.gripper.position if self.gripper.exists else None,
        }
        info["end_effector_frame"] = self._end_effector_frame
        try:
            info["ee_position"] = self.ee_position.round(4).tolist()
        except Exception as exc:
            info["ee_position"] = None
            info["cartesian_control"] = f"unavailable: {exc}"
        return info


class DexterousHand(Robot):
    """A standalone multi-finger hand with no arm to carry it.

    Control is per-finger rather than a single open/close width, but the coarse
    grasp primitive still works: curl every finger until contact stops it.
    """

    morphology = Morphology.DEXTEROUS_HAND

    def __init__(self, prim_path: str, *, scene: Any = None) -> None:
        super().__init__(prim_path, scene=scene)
        finger_indices = self.groups.gripper or list(range(self.dof))
        self.gripper = Gripper(self, finger_indices)
        self.fingers = self._group_fingers(finger_indices)

    def _group_fingers(self, indices: list[int]) -> dict[str, list[int]]:
        """Group joints into fingers by the common prefix of their names."""
        names = self.joint_names
        fingers: dict[str, list[int]] = {}
        for index in indices:
            name = names[index].lower()
            key = next(
                (token for token in ("thumb", "index", "middle", "ring", "little", "pinky")
                 if token in name),
                name.split("_")[0],
            )
            fingers.setdefault(key, []).append(index)
        return fingers

    def close_finger(self, finger: str, value: float, *, settle_steps: int = 20) -> None:
        if finger not in self.fingers:
            raise ValueError(f"No finger {finger!r}. Known: {sorted(self.fingers)}")
        indices = self.fingers[finger]
        self.set_joint_positions([value] * len(indices), indices=indices, settle_steps=settle_steps)

    def open(self, *, settle_steps: int = 30) -> None:
        self.gripper.open(settle_steps=settle_steps)

    def close(self, *, settle_steps: int = 45) -> None:
        self.gripper.close(settle_steps=settle_steps)

    def is_grasping(self, obj: "RigidObject", *, min_contacts: int = 2) -> bool:
        """A multi-finger hand should be touching an object at more than one point."""
        touching = [b for b in obj.contact_bodies() if b.startswith(self.prim_path)]
        return len(touching) >= min_contacts

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        names = self.joint_names
        info["fingers"] = {
            finger: [names[i] for i in indices] for finger, indices in self.fingers.items()
        }
        return info
