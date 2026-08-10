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

from .._compat import as_quat, as_vec3, get_stage, motion_generation
from .base import Morphology, Robot

if TYPE_CHECKING:
    from ..objects import RigidObject

logger = logging.getLogger("simliverse_sim.robots.manipulator")


def _same_orientation(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(a, b))


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


class SuctionGripper:
    """A surface (suction) gripper — grips by contact, not by squeezing.

    Deliberately the same shape as `Gripper`: `open()`, `close()`, and a way to
    ask what is held. Control code should not have to branch on which kind of
    end effector it has.

    Suction is worth reaching for on stacking and pick-and-place. A friction
    pinch depends on finger drive gains, contact patches and material friction
    all being right at once — and when any of them is not, the failure is a
    silent slip that looks exactly like bad IK. Suction reports whether it
    latched, which turns that class of failure into a fact you can read.
    """

    # Isaac's action convention: 1.0 closes (grips), -1.0 opens (releases).
    CLOSE = 1.0
    OPEN = -1.0

    def __init__(
        self,
        prim_path: str,
        *,
        scene: Any = None,
        max_grip_distance: float = 0.02,
        coaxial_force_limit: float = 200.0,
        shear_force_limit: float = 200.0,
        retry_interval: float = 0.05,
    ) -> None:
        from ..scene import Scene as _Scene

        self.prim_path = prim_path
        self.scene = scene or _Scene.get()
        self._settings = dict(
            max_grip_distance=max_grip_distance,
            coaxial_force_limit=coaxial_force_limit,
            shear_force_limit=shear_force_limit,
            retry_interval=retry_interval,
        )
        self._view: Any = None

    def __repr__(self) -> str:
        return f"<SuctionGripper {self.prim_path} holding={self.gripped_objects}>"

    @classmethod
    def create(
        cls,
        parent_prim_path: str,
        *,
        scene: Any = None,
        offset: Any = (0.0, 0.0, -0.02),
        forward_axis: str = "Z",
        clearance_offset: float = 0.01,
        **kwargs: Any,
    ) -> "SuctionGripper":
        """Author a surface gripper on `parent_prim_path` and wrap it.

        Two things here are not obvious and are not optional, both from the
        Isaac Sim 6.0 surface-gripper documentation:

        An attachment point is a **D6 joint**, not an Xform. The gripper casts a
        ray from the joint's world position along its `isaac:forwardAxis` and
        latches onto the first rigid body within `maxGripDistance`. A plain Xform
        target leaves the gripper with nothing to cast from, so it sits in
        "Closing" forever and never reports an error. The joint must have Body 0
        set to the mounting body, be enabled, and be excluded from the
        articulation.

        The timeline must be **stopped** while this is authored. Physics entities
        created mid-play are never registered, and the gripper then ignores every
        action — measured: status stayed 0 when authored during play, and moved
        to 1 (Closing) when authored stopped and then played.

        KNOWN INCOMPLETE: with all of the above correct, the gripper reaches
        Closing but has not been observed to latch. See the class docstring.
        """
        from isaacsim.robot.surface_gripper import create_surface_gripper
        from pxr import Gf, UsdPhysics
        from usd.schema.isaac import robot_schema

        from ..scene import Scene as _Scene

        scene = scene or _Scene.get()
        if scene.is_playing():
            logger.warning(
                "Authoring a surface gripper while the timeline is playing; it "
                "will not register with physics. Call scene.stop() first."
            )

        attach_path = f"{parent_prim_path}/SuctionAttachPoint"
        joint = UsdPhysics.Joint.Define(scene.stage, attach_path)
        joint.CreateBody0Rel().SetTargets([parent_prim_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*as_vec3(offset, name="offset")))
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateExcludeFromArticulationAttr().Set(True)

        joint_prim = joint.GetPrim()
        robot_schema.ApplyAttachmentPointAPI(joint_prim)
        joint_prim.GetAttribute("isaac:forwardAxis").Set(forward_axis)
        joint_prim.GetAttribute("isaac:clearanceOffset").Set(float(clearance_offset))

        prim = create_surface_gripper(scene.stage, parent_prim_path)
        prim.GetRelationship("isaac:attachmentPoints").SetTargets([attach_path])

        gripper = cls(prim.GetPath().pathString, scene=scene, **kwargs)
        # Author the schema attributes directly as well as through the view: the
        # view sets them per-instance, the prim carries them across a stop/play.
        for attr, value in (
            ("isaac:maxGripDistance", gripper._settings["max_grip_distance"]),
            ("isaac:coaxialForceLimit", gripper._settings["coaxial_force_limit"]),
            ("isaac:shearForceLimit", gripper._settings["shear_force_limit"]),
            ("isaac:retryInterval", gripper._settings["retry_interval"]),
        ):
            prim.GetAttribute(attr).Set(float(value))
        return gripper

    def _ensure_view(self) -> Any:
        """Build the GripperView lazily.

        It reads physics state, so it cannot be constructed before the scene has
        played and stepped — the same constraint articulations have.
        """
        if self._view is not None:
            return self._view
        import numpy as np
        from isaacsim.robot.surface_gripper import GripperView

        self.scene.play()
        self.scene.step(2)
        self._view = GripperView(
            paths=self.prim_path,
            max_grip_distance=np.array([self._settings["max_grip_distance"]]),
            coaxial_force_limit=np.array([self._settings["coaxial_force_limit"]]),
            shear_force_limit=np.array([self._settings["shear_force_limit"]]),
            retry_interval=np.array([self._settings["retry_interval"]]),
        )
        return self._view

    def _act(self, value: float, settle_steps: int) -> None:
        self._ensure_view().apply_gripper_action([value])
        if settle_steps:
            self.scene.step(settle_steps)

    def close(self, *, settle_steps: int = 30) -> None:
        """Engage suction. Latches onto whatever is within `max_grip_distance`."""
        self._act(self.CLOSE, settle_steps)

    def open(self, *, settle_steps: int = 15) -> None:
        """Release."""
        self._act(self.OPEN, settle_steps)

    @property
    def gripped_objects(self) -> list[str]:
        """Prim paths currently held. Empty means nothing latched."""
        try:
            return list(self._ensure_view().get_gripped_objects()[0] or [])
        except Exception:
            logger.debug("get_gripped_objects failed", exc_info=True)
            return []

    @property
    def status(self) -> str:
        try:
            return str(self._ensure_view().get_surface_gripper_status()[0])
        except Exception:
            return "unknown"

    @property
    def holding(self) -> bool:
        return bool(self.gripped_objects)

    def is_holding(self, prim_path: str) -> bool:
        return any(prim_path in held for held in self.gripped_objects)


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
        self._obstacles: dict[str, Any] = {}
        self._servo_target: Any = None
        self._servo_orientation: Any = None
        self._servo_settled = 0
        self._servo_error = float("inf")
        self.gripper = Gripper(self, self.groups.gripper)

    def attach_suction_gripper(
        self, parent_prim_path: str | None = None, **kwargs: Any
    ) -> "SuctionGripper":
        """Fit a suction gripper to this arm and use it as the end effector.

        Defaults to authoring it under the arm's own prim. The finger `gripper`
        stays available, so an arm can have both and control code chooses.
        """
        self.suction = SuctionGripper.create(
            parent_prim_path or self.prim_path, scene=self.scene, **kwargs
        )
        return self.suction

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
        self._sync_base_pose()

    def _sync_base_pose(self) -> None:
        """Tell RMPflow and Lula where the robot actually stands.

        Both solve in the robot's *base* frame. Until they are given the base
        pose they assume it is the world origin, which silently turns every
        world-space target and every `ee_position` reading into a base-frame
        quantity. A robot at the origin is unaffected — which is exactly why
        this went unnoticed — but a second arm spawned at y=0.8 never moves at
        all, and reports its end effector near the origin while doing so.

        Re-read every call rather than cached once: a manipulator on a mobile
        base moves, and a stale base pose is the same bug with extra steps.
        """
        try:
            position = self.base_position
            orientation = self.base_orientation
        except Exception:
            logger.debug("Could not read base pose for %s", self.prim_path, exc_info=True)
            return
        kinematics = None
        if self._ik is not None:
            getter = getattr(self._ik, "get_kinematics_solver", None)
            kinematics = getter() if getter else getattr(self._ik, "_kinematics_solver", None)
        for solver in (self._rmpflow, kinematics):
            setter = getattr(solver, "set_robot_base_pose", None)
            if setter is not None:
                setter(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float))

    @property
    def ee_position(self) -> np.ndarray:
        """World-space end-effector position."""
        self._ensure_motion_policy()
        self._sync_base_pose()
        position, _ = self._ik.compute_end_effector_pose()
        return np.asarray(position, dtype=float)

    @property
    def ee_orientation(self) -> np.ndarray:
        self._ensure_motion_policy()
        self._sync_base_pose()
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
        target = as_vec3(position, name="position")
        self.scene.play()

        for step in range(max_steps):
            reached = self.servo_to(
                target, orientation, tolerance=tolerance, hold=hold_steps
            )
            self.scene.step(1)
            if reached:
                return MotionResult(True, step + 1, self._servo_error, target.tolist())

        error = self._servo_error
        result = MotionResult(False, max_steps, error, target.tolist())
        if raise_on_fail:
            raise MotionError(
                f"End effector did not reach {target.round(3).tolist()} within "
                f"{max_steps} steps (final error {error:.4f} m > {tolerance} m). The "
                f"target is likely outside the workspace or blocked by a collision."
            )
        return result

    def servo_to(
        self,
        position: Any,
        orientation: Any = None,
        *,
        tolerance: float = 0.005,
        hold: int = 3,
    ) -> bool:
        """Advance the arm one control tick toward a Cartesian target.

        Does NOT step physics, and does not block. Returns True once the end
        effector has stayed inside `tolerance` for `hold` consecutive ticks.

        This is the form a controller needs. `move_ee_to` steps physics itself,
        which is correct when driving the sim from outside but wrong inside a
        ScriptNode or any OnPlaybackTick callback, where the timeline owns
        stepping -- stepping from within the callback either deadlocks or
        double-advances the world. Calling the same target repeatedly is the
        intended usage; the target is only re-issued to RMPflow when it changes,
        so convergence state survives across ticks.
        """
        self._ensure_motion_policy()
        self._sync_base_pose()

        target = as_vec3(position, name="position")
        rotation = as_quat(orientation) if orientation is not None else None
        changed = (
            self._servo_target is None
            or not np.allclose(self._servo_target, target)
            or not _same_orientation(self._servo_orientation, rotation)
        )
        if changed:
            self._servo_target = target
            self._servo_orientation = rotation
            self._servo_settled = 0
            self._rmpflow.set_end_effector_target(
                target_position=target, target_orientation=rotation
            )

        self._rmpflow.update_world()
        self._controller().apply_action(self._policy.get_next_articulation_action())

        self._servo_error = float(np.linalg.norm(self.ee_position - target))
        self._servo_settled = self._servo_settled + 1 if self._servo_error < tolerance else 0
        return self._servo_settled >= hold

    def move_ee_by(self, delta: Any, **kwargs: Any) -> MotionResult:
        return self.move_ee_to(self.ee_position + as_vec3(delta, name="delta"), **kwargs)

    # ── Obstacles ─────────────────────────────────────────────────────────────

    _OBSTACLE_WRAPPERS = {
        "Cube": "VisualCuboid",
        "Sphere": "VisualSphere",
        "Cylinder": "VisualCylinder",
        "Capsule": "VisualCapsule",
        "Cone": "VisualCone",
    }

    def add_obstacle(self, target: Any, *, static: bool = False) -> bool:
        """Register a body RMPflow must plan around.

        RMPflow avoids obstacles, but only ones it has been told about — an
        empty obstacle set means the arm plans straight through the scene. It
        still *reaches* its target, so this failure looks like success right up
        until the elbow sweeps a finished stack off the table.

        `target` may be a `RigidObject`, a prim path, or an already-wrapped
        core-API object. Pass `static=True` for anything that will not move
        (a table, a wall); a static obstacle is baked once instead of re-read
        every step.

        Note the object being *manipulated* should not be registered — the arm
        has to touch that one.
        """
        self._ensure_motion_policy()

        obstacle = target
        if hasattr(target, "prim_path") and not hasattr(target, "geom"):
            obstacle = self._wrap_obstacle(target.prim_path)
        elif isinstance(target, str):
            obstacle = self._wrap_obstacle(target)

        added = bool(self._rmpflow.add_obstacle(obstacle, static=static))
        if added:
            self._obstacles[getattr(obstacle, "prim_path", str(target))] = obstacle
        return added

    def _wrap_obstacle(self, prim_path: str) -> Any:
        """Wrap an existing prim in the core-API type RMPflow expects."""
        import isaacsim.core.api.objects as core_objects

        prim = get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"No prim at {prim_path!r} to use as an obstacle")

        kind = prim.GetTypeName()
        wrapper = self._OBSTACLE_WRAPPERS.get(str(kind))
        if wrapper is None:
            raise ValueError(
                f"{prim_path} is a {kind}, which Lula cannot represent as an "
                f"obstacle. Supported: {sorted(self._OBSTACLE_WRAPPERS)}. Wrap the "
                f"region in a cuboid of your own and register that instead."
            )
        # Visual* rather than Dynamic*: this binds to the prim already on the
        # stage for collision queries and must not add a second rigid body to it.
        return getattr(core_objects, wrapper)(
            prim_path=prim_path, name=f"obstacle_{prim_path.strip('/').replace('/', '_')}"
        )

    def remove_obstacle(self, target: Any) -> bool:
        path = getattr(target, "prim_path", target)
        obstacle = self._obstacles.pop(path, None)
        if obstacle is None:
            return False
        self._rmpflow.remove_obstacle(obstacle)
        return True

    def clear_obstacles(self) -> None:
        for obstacle in list(self._obstacles.values()):
            try:
                self._rmpflow.remove_obstacle(obstacle)
            except Exception:  # noqa: BLE001 — the registry is what must end up empty
                logger.debug("Could not remove obstacle", exc_info=True)
        self._obstacles.clear()

    def obstacles(self) -> list[str]:
        return sorted(self._obstacles)

    # ── Grasping ──────────────────────────────────────────────────────────────

    def is_grasping(
        self, obj: "RigidObject", *, min_contacts: int = 1, min_force: float = 0.05
    ) -> bool:
        """True when the object is genuinely held — measured from contact reports.

        `min_force` (newtons) is what separates holding from touching. A closed
        Franka hand reports three contacts with a grasped cube, and one of them
        is the palm at ~0 N: it is along for the ride, not carrying the object.
        Counting it would make "the object brushed the hand on its way past"
        indistinguishable from a grasp.
        """
        touching = {
            c["body"]
            for c in obj.contacts()
            if c["body"].startswith(self.prim_path) and c["force"] >= min_force
        }
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
