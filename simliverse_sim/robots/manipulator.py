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

from .._compat import articulation_action, as_quat, as_vec3, get_stage, motion_generation
from .base import Morphology, Robot, StaleArticulation

if TYPE_CHECKING:
    from ..objects import RigidObject

logger = logging.getLogger("simliverse_sim.robots.manipulator")


def _same_orientation(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(a, b))


_EXTENT_REPORTED: set[str] = set()


def _repair_extent(prim: Any) -> bool:
    """Make a primitive's `extent` agree with its own geometry.

    `extent` is local-space bounds. For a `UsdGeom.Cube` of `size` S it is
    ±S/2, and the transform's scale is applied on top. Some authoring paths
    write it with the scale already baked in, and then the scale is applied a
    second time — a 6 cm x 6 cm x 35 cm post reports a bound of 2 mm x 2 mm x
    6 cm, its size multiplied by its own scale twice.

    Nothing about that is visible. The prim renders and collides at its true
    size, because rendering and PhysX use the geometry; only *bounds queries*
    are wrong. The motion planner is a bounds query. So the arm was routed
    neatly around a sliver a thirtieth of the post's width while the links swept
    through the real one — visible only by watching, which is how it was
    actually caught.

    Returns True when something was repaired.
    """
    from pxr import UsdGeom

    if prim.GetTypeName() != "Cube":
        return False
    cube = UsdGeom.Cube(prim)
    size_attr, extent_attr = cube.GetSizeAttr(), cube.GetExtentAttr()
    if not size_attr or not extent_attr or extent_attr.Get() is None:
        return False

    half = float(size_attr.Get() or 2.0) / 2.0
    current = np.asarray([[c[i] for i in range(3)] for c in extent_attr.Get()], dtype=float)
    expected = np.array([[-half] * 3, [half] * 3], dtype=float)
    if np.allclose(current, expected, atol=1e-4):
        return False

    # Loud once per prim, quiet thereafter. Wrapping an obstacle re-corrupts
    # it, so a task that registers the same body each run repeats this forever —
    # and a warning that always fires is one nobody reads. The repair happens
    # either way; only the reporting is throttled.
    path = str(prim.GetPath())
    detail = (
        "%s has an extent of %s where its size implies %s — bounds queries see it "
        "at the wrong size, and the motion planner is a bounds query. Repairing it."
    )
    args = (path, current[1].round(4).tolist(), expected[1].round(4).tolist())
    if path in _EXTENT_REPORTED:
        logger.debug(detail, *args)
    else:
        _EXTENT_REPORTED.add(path)
        logger.warning(detail, *args)
    extent_attr.Set([tuple(expected[0]), tuple(expected[1])])
    return True


def _slerp(start: Any, end: Any, fraction: float) -> np.ndarray:
    """Shortest-arc interpolation between two (w, x, y, z) quaternions."""
    a = np.asarray(start, dtype=float).reshape(4)
    b = np.asarray(end, dtype=float).reshape(4)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    dot = float(a @ b)
    if dot < 0.0:          # take the short way round
        b, dot = -b, -dot
    if dot > 0.9995:       # nearly parallel: lerp is exact enough and stable
        result = a + (b - a) * fraction
        return result / (np.linalg.norm(result) or 1.0)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    return (a * float(np.sin((1.0 - fraction) * theta)) / sin_theta
            + b * float(np.sin(fraction * theta)) / sin_theta)


def _angle_between(rotation: Any, quaternion: Any) -> float:
    """Degrees between an achieved 3x3 rotation and a requested (w,x,y,z) quaternion."""
    achieved = np.asarray(rotation, dtype=float).reshape(3, 3)
    w, x, y, z = [float(v) for v in np.asarray(quaternion, dtype=float).reshape(4)]
    wanted = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    trace = float(np.trace(achieved.T @ wanted))
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def _quaternion_from_matrix(rotation: Any) -> np.ndarray:
    """A 3x3 rotation matrix as a (w, x, y, z) quaternion.

    Shepperd's method: pick the largest of the four possible divisors rather
    than always using w, which loses all precision near a 180 degree rotation -
    exactly the case for a tool pointed straight down from a frame that rests
    pointing up.
    """
    m = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * scale,
                (m[2][1] - m[1][2]) / scale,
                (m[0][2] - m[2][0]) / scale,
                (m[1][0] - m[0][1]) / scale]
    else:
        i = int(np.argmax([m[0][0], m[1][1], m[2][2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2.0
        quat = [0.0, 0.0, 0.0, 0.0]
        quat[0] = (m[k][j] - m[j][k]) / scale
        quat[i + 1] = 0.25 * scale
        quat[j + 1] = (m[j][i] + m[i][j]) / scale
        quat[k + 1] = (m[k][i] + m[i][k]) / scale
    result = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(result))
    return result / norm if norm else np.array([1.0, 0.0, 0.0, 0.0])


class MotionError(RuntimeError):
    """A motion could not be completed — unreachable, blocked, or timed out."""


@dataclass
class MotionResult:
    reached: bool
    steps: int
    final_error: float
    target: list[float]
    # Degrees between the requested and achieved tool orientation. Only
    # meaningful when an orientation was asked for; 0.0 otherwise.
    angle_error: float = 0.0

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
            if not lows or not highs:
                # Say so once, loudly. Guessing 0.0/0.04 is right for a Panda and
                # wrong for anything else, and the way it is wrong is a grasp that
                # closes, appears to hold, and drops the object — which reads as a
                # control problem for as long as anyone is willing to look.
                logger.warning(
                    "%s: gripper joints %s declare no travel limits, so open() and "
                    "close() are falling back to 0.04 and 0.0 m. If this gripper's "
                    "real travel differs, grasps will fail in a way that looks like "
                    "bad control. This is the asset, not the controller — see "
                    "describe()['asset_problems'].",
                    self._robot.prim_path,
                    self.joint_names,
                )
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

    def open(self, *, settle_steps: int = 0) -> None:
        """Command the fingers open. Does not step physics by default.

        Non-blocking is the safe default because the dangerous caller is the
        silent one. A controller runs inside the simulator's own step
        callback, so stepping from there is re-entrant, and it does not raise
        — it quietly desynchronises the run, which then diverges from the same
        controller replayed headless. Pass `settle_steps` explicitly when you
        are driving the sim from outside and want to wait.
        """
        self.set_position(self._limits()[0], settle_steps=settle_steps)

    def close(self, *, settle_steps: int = 0) -> None:
        """Drive the fingers closed. Does not step physics by default.

        Commanding fully-closed against a solid object is intentional: the drive
        pushes until contact stops it, and that residual push is the normal force
        a friction grasp depends on. Closing takes real time either way — a
        controller waits by staying in its state for a number of ticks, not by
        stepping.
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

    Working as of the joint recipe in `create` - see there for what the three
    non-obvious requirements are and what each one looks like when missed.
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
        coaxial_force_limit: float = 10000.0,
        shear_force_limit: float = 10000.0,
        retry_interval: float = 1.0,
    ) -> None:
        from ..scene import Scene as _Scene

        self.prim_path = prim_path
        self.scene = scene or _Scene.get()
        self.approach_axis = "Z"
        self.cup_path: str | None = None
        # How far the cup tip sits beyond the mounting frame. Tool poses are
        # commanded for the flange, so a pick has to allow for it.
        self.tip_offset = 0.0
        self._settings = dict(
            max_grip_distance=max_grip_distance,
            coaxial_force_limit=coaxial_force_limit,
            shear_force_limit=shear_force_limit,
            retry_interval=retry_interval,
        )
        self._view: Any = None

    def __repr__(self) -> str:
        return f"<SuctionGripper {self.prim_path} holding={self.gripped_objects}>"

    # The rotations that carry the joint's local +Z onto a chosen body axis.
    # Isaac's own assets author `isaac:forwardAxis = "Z"` and rotate the joint
    # frame rather than naming a different axis, which also settles the question
    # of which frame the token is read in: the joint's.
    _Z_ONTO = {
        "X": (0.70710678, 0.0, 0.70710678, 0.0),
        "-X": (0.70710678, 0.0, -0.70710678, 0.0),
        "Y": (0.70710678, -0.70710678, 0.0, 0.0),
        "-Y": (0.70710678, 0.70710678, 0.0, 0.0),
        "Z": (1.0, 0.0, 0.0, 0.0),
        "-Z": (0.0, 1.0, 0.0, 0.0),
    }

    @classmethod
    def create(
        cls,
        parent_prim_path: str,
        *,
        scene: Any = None,
        approach_axis: str = "Z",
        cup_parent: str | None = None,
        offset: float = 0.0,
        cup_radius: float = 0.03,
        cup_length: float = 0.02,
        cup_mass: float = 0.2,
        clearance_offset: float = 0.008,
        max_grip_distance: float = 0.02,
        **kwargs: Any,
    ) -> "SuctionGripper":
        """Author a suction cup on `parent_prim_path` and wrap it.

        `approach_axis` names the axis of the *mounting body* that points out of
        the cup - "X" for a flange whose tool points along its own +X. It is not
        the `isaac:forwardAxis` token; that is always "Z" here, with the joint
        frame rotated to suit, which is how Isaac's own suction assets are built.

        The cup is a **rigid body of its own**, held on the flange by a fixed
        joint, and authored as a **sibling of the robot** rather than a child of
        the flange. Both parts of that matter.

        Parenting a rigid body under an articulation link is quietly fatal:
        physics writes each body's world pose into its local transform, the
        parent link's transform is then applied on top, and the error compounds
        every frame. Measured, the cup's world bounds reached z = 47,491 m - and
        since the gripper casts its ray from the cup, it was searching for
        something to grip 47 km above the table while reporting a healthy Open.
        Nothing about it looked wrong from the gripper's own status.

        That the cup is a separate body at all is also not decoration. An attachment joint authored
        with `body1` empty is a joint to the *world*: PhysX anchors the flange to
        a fixed world frame, and on an arm whose asset declares no joint limits
        the wrist simply spun - measured at -90 rad on joint_a5 within a second
        of pressing play, with the arm reporting poses near the world origin.
        Isaac's own working example (`SurfaceGripper_gantry.usda`) puts a cup
        body on body0 and the mount on body1, and that is what is done here.

        The rest of the recipe is taken from the same file, because reproducing
        it from the documentation did not work and the difference was not
        guessable:

        * The mount is **compliant, not rigid**. `transZ` travels from 0 to the
          grip distance and the rotational axes give +-3 degrees, sprung back by
          PD drives. That travel is how the cup closes the last millimetres onto
          a surface it is not perfectly square to.
        * `transX`/`transY` are locked with **low > high** (1 and -1), USD's
          idiom for a locked axis. It is not interchangeable with a zero-width
          range: authored as `low = high = 0` the gripper stopped acknowledging
          actions at all - status never left Open, with no error anywhere.
        * The timeline must be **stopped** while this is authored. Physics
          entities created mid-play are never registered, and the gripper then
          ignores every action.
        """
        from isaacsim.robot.surface_gripper import create_surface_gripper
        from pxr import Gf, UsdGeom, UsdPhysics
        from usd.schema.isaac import robot_schema

        from ..scene import Scene as _Scene

        scene = scene or _Scene.get()
        if scene.is_playing():
            # Stop, do not merely complain. A gripper authored while the
            # timeline runs is never registered by the surface-gripper plugin,
            # which then logs "Gripper not found" once per frame while every
            # Python-side call keeps reporting a healthy Open status. Nothing in
            # the returned object reveals it. And the caller usually has not done
            # anything wrong: `Robot.attach` starts physics to build its
            # articulation view, so an arm handle obtained the obvious way leaves
            # the timeline playing before this is ever reached.
            logger.info(
                "Stopping the timeline to author the surface gripper on %s; "
                "physics does not register entities created mid-play.",
                parent_prim_path,
            )
            scene.stop()

        axis = str(approach_axis).upper()
        if axis not in cls._Z_ONTO:
            raise ValueError(
                f"approach_axis must be one of {sorted(cls._Z_ONTO)}, not {approach_axis!r}"
            )
        rot = Gf.Quatf(*cls._Z_ONTO[axis])
        direction = np.zeros(3)
        direction["XYZ".index(axis[-1])] = -1.0 if axis.startswith("-") else 1.0
        mount = direction * (float(offset) + float(cup_length) / 2.0)

        # Where the cup sits in the world right now, so it can be authored
        # outside the articulation and still start in the right place.
        link_world = UsdGeom.Xformable(
            scene.stage.GetPrimAtPath(parent_prim_path)
        ).ComputeLocalToWorldTransform(0)
        local = Gf.Matrix4d(1.0)
        local.SetRotateOnly(Gf.Quatd(rot))
        local.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in mount]))
        cup_world = local * link_world

        parent = cup_parent or parent_prim_path.rsplit("/", 1)[0]
        stem = parent_prim_path.strip("/").replace("/", "_")
        cup_path = f"{parent}/{stem}_SuctionCup"
        cup = UsdGeom.Cylinder.Define(scene.stage, cup_path)
        cup.CreateRadiusAttr(float(cup_radius))
        cup.CreateHeightAttr(float(cup_length))
        cup.CreateAxisAttr("Z")
        cup.CreateExtentAttr([
            (-cup_radius, -cup_radius, -cup_length / 2.0),
            (cup_radius, cup_radius, cup_length / 2.0),
        ])
        cup_prim = cup.GetPrim()
        xform = UsdGeom.Xformable(cup_prim)
        xform.AddTranslateOp().Set(cup_world.ExtractTranslation())
        xform.AddOrientOp().Set(Gf.Quatf(cup_world.ExtractRotationQuat()))
        UsdPhysics.CollisionAPI.Apply(cup_prim)
        UsdPhysics.RigidBodyAPI.Apply(cup_prim)
        UsdPhysics.MassAPI.Apply(cup_prim).CreateMassAttr(float(cup_mass))

        # Two joints, and the split matters. A fixed joint carries the cup on the
        # flange; the attachment joint - the one the gripper casts from - is left
        # with `body1` empty for the gripper to fill in when it latches.
        #
        # Wiring the attachment joint straight to the flange instead (body1 set)
        # leaves nothing holding the cup: measured, it drifted 0.29 m away inside
        # a second while the gripper reported a perfectly healthy status.
        mount_path = f"{cup_path}_Mount"
        fixed = UsdPhysics.FixedJoint.Define(scene.stage, mount_path)
        fixed.CreateBody0Rel().SetTargets([parent_prim_path])
        fixed.CreateBody1Rel().SetTargets([cup_path])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in mount]))
        fixed.CreateLocalRot0Attr().Set(rot)
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fixed.CreateExcludeFromArticulationAttr().Set(True)
        fixed.CreateCollisionEnabledAttr().Set(False)

        attach_path = f"{cup_path}_AttachPoint"
        joint = UsdPhysics.Joint.Define(scene.stage, attach_path)
        joint.CreateBody0Rel().SetTargets([cup_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        # KNOWN DEFECT, left enabled on purpose: this is the only setting that
        # grips, and it is also the one that wobbles.
        #
        # An attachment point holds nothing at rest, so `body1` is empty — and
        # PhysX reads an empty body as *the world*. Enabled, this becomes a joint
        # anchoring the cup to a fixed world pose, with live drives, while the
        # mount joint drags the same cup along with the arm. The two fight, and
        # on an arm left to sag under gravity the cup-to-flange gap swings
        # between 0.008 m and 0.180 m instead of holding at its authored 0.010 m.
        #
        # Four configurations were measured on a UR10, over 200 steps of
        # playback (idle spread / does it latch):
        #
        #   body1 empty, enabled, drives on   0.1717   yes   <- this one
        #   body1 empty, disabled             0.0000   no
        #   body1 = flange, enabled           0.0000   no
        #   body1 empty, enabled, drives 0    0.0031   no
        #
        # Only the first grips. The gripper needs `body1` free to fill in with
        # whatever it latches onto, and needs the joint live to do it.
        #
        # Toggling `physics:jointEnabled` around `close()` looks like the answer
        # and is not: PhysX drops the write, which Kit reports as
        # "PxConstraint::setFlag() not allowed while simulation is running. Call
        # will be ignored." A real fix has to change the flag while the timeline
        # is paused, or go through Isaac's gripper API rather than USD.
        joint.CreateJointEnabledAttr().Set(True)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)

        for name, low, high in (
            ("transX", 1.0, -1.0),          # locked: low > high
            ("transY", 1.0, -1.0),          # locked: low > high
            ("transZ", 0.0, float(max_grip_distance)),
            ("rotX", -3.0, 3.0),
            ("rotY", -3.0, 3.0),
            ("rotZ", -3.0, 3.0),
        ):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), name)
            limit.CreateLowAttr().Set(low)
            limit.CreateHighAttr().Set(high)
        for name, stiffness, damping in (
            ("rotX", 100.0, 0.0), ("rotY", 100.0, 0.0),
            ("rotZ", 10000.0, 0.0), ("transZ", 5000.0, 100.0),
        ):
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), name)
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(damping)

        joint_prim = joint.GetPrim()
        robot_schema.ApplyAttachmentPointAPI(joint_prim)
        joint_prim.GetAttribute("isaac:forwardAxis").Set("Z")
        joint_prim.GetAttribute("isaac:clearanceOffset").Set(float(clearance_offset))

        prim = create_surface_gripper(scene.stage, cup_path)
        prim.GetRelationship("isaac:attachmentPoints").SetTargets([attach_path])

        gripper = cls(
            prim.GetPath().pathString, scene=scene,
            max_grip_distance=float(max_grip_distance), **kwargs,
        )
        gripper.approach_axis = axis
        gripper.cup_path = cup_path
        gripper.tip_offset = float(offset) + float(cup_length)
        # Author the schema attributes as well as setting them through the view:
        # the view sets them per-instance, the prim carries them across stop/play.
        for attr, value in (
            ("isaac:maxGripDistance", gripper._settings["max_grip_distance"]),
            ("isaac:coaxialForceLimit", gripper._settings["coaxial_force_limit"]),
            ("isaac:shearForceLimit", gripper._settings["shear_force_limit"]),
            ("isaac:retryInterval", gripper._settings["retry_interval"]),
        ):
            attribute = prim.GetAttribute(attr)
            if attribute:
                attribute.Set(float(value))
        return gripper

    def _ensure_view(self) -> Any:
        """Build the GripperView lazily.

        It reads physics state, so it cannot be constructed before the scene has
        played — the same constraint articulations have.

        Only starts physics if it is not already running, and never steps
        otherwise. It used to step unconditionally, which made the first
        `close()` a blocking call however it was invoked: from a controller's
        `compute` that is re-entrant stepping, and `_refuse_reentrant_step` threw
        on the first grasp of an otherwise working task. A non-blocking API has
        to be non-blocking on the call that builds its lazy state too.
        """
        if self._view is not None:
            return self._view
        from isaacsim.robot.surface_gripper import GripperView

        if not self.scene.is_playing():
            self.scene.play()
            self.scene.step(2)
        self._view = GripperView(paths=self.prim_path)
        # The view must be told its properties explicitly. Passing them to the
        # constructor was not enough - the gripper acknowledged actions and then
        # reported Closing forever, which reads exactly like a control problem.
        self._view.set_surface_gripper_properties(
            max_grip_distance=[self._settings["max_grip_distance"]],
            coaxial_force_limit=[self._settings["coaxial_force_limit"]],
            shear_force_limit=[self._settings["shear_force_limit"]],
            retry_interval=[self._settings["retry_interval"]],
        )
        return self._view

    def _act(self, value: float, settle_steps: int) -> None:
        self._ensure_view().apply_gripper_action([value])
        if settle_steps:
            self.scene.step(settle_steps)

    def close(self, *, settle_steps: int = 30) -> None:
        """Engage suction. Latches onto whatever is within `max_grip_distance`.

        Gating the attachment joint around this call — enabling it here and
        disabling it in `open()` — is the obvious cure for the cup wobble
        documented in `create`, and it does not work. PhysX ignores a constraint
        flag written while the simulation is running, silently, so the joint is
        still disabled when the gripper looks for something to hold and the
        status never leaves Closing. Measured, not assumed.
        """
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
        # Two structures on purpose. `_obstacle_paths` is what the caller said
        # the arm must avoid, and is backend-agnostic; `_obstacles` is what the
        # reactive policy managed to represent, which is a strict subset — Lula
        # holds no cylinders or cones, while the planner's world does.
        self._obstacle_paths: set[str] = set()
        self._obstacles: dict[str, Any] = {}
        # Built once per prim and kept, because building one mutates the prim.
        self._obstacle_wrappers: dict[str, Any] = {}
        self._planner: Any = None
        self._plan: Any = None
        self._plan_time = 0.0
        self._plan_clock = 0.0
        self._plan_command: Any = None
        self._plan_lag = 0.0
        self._servo_target: Any = None
        self._servo_orientation: Any = None
        self._servo_settled = 0
        self._servo_error = float("inf")
        self._pose_command: Any = None
        self._pose_solution: Any = None
        self._pose_goal: Any = None
        self._pose_orientation: Any = None
        self._pose_from: Any = None
        self._pose_from_quat: Any = None
        self._pose_seed: Any = None
        self._solver_index_map: Any = None
        self._pose_ramp = 0
        self._pose_phase = 0
        self.gripper = Gripper(self, self.groups.gripper)

    def attach_suction_gripper(
        self, parent_prim_path: str | None = None, **kwargs: Any
    ) -> "SuctionGripper":
        """Fit a suction gripper to this arm and use it as the end effector.

        Mounts on the **tool flange** and casts out along whichever of that
        link's own axes points away from the arm. Both are worked out from the
        robot rather than asked for, because the previous default - the arm's
        root prim - put the ray origin at the robot's base, aimed along the base
        frame, where it could never reach anything the arm was holding out in
        front of it. That produced a gripper that armed, searched and reported
        Closing indefinitely.

        The finger `gripper` stays available, so an arm can have both and
        control code chooses.
        """
        if parent_prim_path is None:
            parent_prim_path = self._tool_link()
            kwargs.setdefault("approach_axis", self._approach_axis(parent_prim_path))
            # Outside the robot's own hierarchy - see `SuctionGripper.create`.
            kwargs.setdefault("cup_parent", self.prim_path.rsplit("/", 1)[0])
        self.suction = SuctionGripper.create(
            parent_prim_path, scene=self.scene, **kwargs
        )
        return self.suction

    def remove_suction_gripper(self) -> None:
        """Delete a cup authored by `attach_suction_gripper`, if there is one."""
        gripper = getattr(self, "suction", None)
        if gripper is None or not gripper.cup_path:
            return
        stage = get_stage()
        for path in (gripper.prim_path, f"{gripper.cup_path}_AttachPoint",
                     f"{gripper.cup_path}_Mount", gripper.cup_path):
            stage.RemovePrim(path)
        self.suction = None

    def _tool_link(self) -> str:
        """The link the cup mounts on: the solver's end-effector frame."""
        try:
            self._ensure_motion_policy()
        except MotionError:
            logger.debug("No motion policy; falling back to the last link", exc_info=True)
        links = [str(link) for link in self.links()]
        frame = self._end_effector_frame
        if frame:
            for link in links:
                if link.rsplit("/", 1)[-1] == frame:
                    return link
        if not links:
            raise MotionError(
                f"{self.prim_path} reports no links, so there is nothing to mount "
                f"a gripper on."
            )
        return links[-1]

    def _approach_axis(self, tool_link: str) -> str:
        """Which of `tool_link`'s own axes points away from the arm.

        Measured, not assumed: the world-space step from the previous link to the
        tool is the direction the arm reaches, and projecting it onto the tool's
        own axes says which one that is. A KR210 flange answers +X, a Franka hand
        answers +Z, and neither has to be written down anywhere.

        Falls back to "Z" when the two links are coincident, which is the
        convention for a tool frame with no offset.
        """
        from pxr import UsdGeom

        links = [str(link) for link in self.links()]
        try:
            previous = links[links.index(tool_link) - 1] if links.index(tool_link) else None
        except ValueError:
            previous = links[-2] if len(links) > 1 else None
        if previous is None:
            return "Z"

        stage = get_stage()

        def world(path: str) -> Any:
            return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)

        tool = world(tool_link)
        tip = np.asarray(tool.ExtractTranslation(), dtype=float)

        # Walk back to the last link that is not sitting on top of the tool.
        #
        # A pure tool frame carries no offset from the link it hangs off, so the
        # immediate step is zero and says nothing. This used to answer "Z" in
        # that case, which is measurably wrong on a UR10: `ee_link` sits exactly
        # on `wrist_3_link`, and the last real offset — `wrist_2_link` to
        # `wrist_3_link` — lies along the tool's own X with a projection of
        # 1.000. Z is ninety degrees off, and the only symptom is a suction cup
        # mounted across the flange instead of along it.
        index = links.index(tool_link) if tool_link in links else len(links) - 1
        step = np.zeros(3)
        for candidate in reversed(links[:index]):
            step = tip - np.asarray(world(candidate).ExtractTranslation(), dtype=float)
            if float(np.linalg.norm(step)) >= 1e-6:
                break

        if float(np.linalg.norm(step)) < 1e-6:
            return "Z"
        step /= float(np.linalg.norm(step))

        rotation = tool.ExtractRotationMatrix()
        best, sign = "Z", 1.0
        score = -1.0
        for index, name in enumerate("XYZ"):
            axis = np.asarray([rotation[index][i] for i in range(3)], dtype=float)
            projection = float(axis @ step)
            if abs(projection) > score:
                score, best, sign = abs(projection), name, 1.0 if projection > 0 else -1.0
        return best if sign > 0 else f"-{best}"

    @property
    def arm_joint_indices(self) -> list[int]:
        finger = set(self.groups.gripper)
        return [i for i in range(self.dof) if i not in finger]

    # ── Motion policy ─────────────────────────────────────────────────────────

    def _recorded_motion_config(self) -> str | None:
        """The RMPflow config `spawn` stored on the prim, if it did.

        The catalogue knows which config an asset needs; `attach` does not have
        the catalogue. Storing the answer on the prim means the robot carries it,
        and a handle built any way at all can find it.
        """
        try:
            attr = get_stage().GetPrimAtPath(self.prim_path).GetAttribute(
                "simliverse:motion_config"
            )
            value = attr.Get() if attr and attr.IsValid() else None
            return str(value) if value else None
        except Exception:  # noqa: BLE001 — absence is the normal case
            return None

    def _ensure_motion_policy(self) -> None:
        if self._policy is not None:
            return
        mg = motion_generation()
        loader = mg.interface_config_loader

        name = self._rmp_config_name
        if name is None:
            name = self._recorded_motion_config()
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

    def _arm_base_pose(self) -> tuple[Any, Any]:
        """World pose of the link the solver treats as the robot's base.

        Not the articulation root. On a fixed arm the two are the same link and
        the distinction never shows; on a mobile base they are not, and the
        difference is exactly how far the robot has driven.

        Measured on a Ridgeback-Franka: the articulation root is `world`, which
        sits *before* the base's prismatic joints and therefore never moves,
        while the arm's mount travels with the base. Feeding the root pose to
        Lula told it the arm was at the origin while the arm was at x = 0.886,
        so a target at x = 1.90 was solved as 1.90 in a frame already 0.886 out
        — 2.79 in world, unreachable, reported as "did not reach".

        Read through the physics view rather than USD. Physics results are not
        written back to USD for every articulation: the same link that reads
        0.886 from the physics view reports a constant 0.308 in USD, before or
        after an explicit transform sync, which is also why such a base looks
        stationary in the viewport while it is driving.
        """
        link = self._arm_base_link()
        if link is not None:
            try:
                position, orientation = link.get_world_pose()
                return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)
            except Exception:
                logger.debug("Could not read the arm base link pose", exc_info=True)
        return self.base_position, self.base_orientation

    def _arm_base_link(self) -> Any:
        """The physics handle for the arm's mounting link, found once."""
        if getattr(self, "_arm_base_view", "unset") != "unset":
            return self._arm_base_view

        self._arm_base_view = None
        candidates = [str(l) for l in self.links()]
        # `*_link0` is the convention for an arm's root frame (panda_link0,
        # ur_link0); `base_link` is the usual fallback for the body an arm is
        # bolted to. Anything else and the articulation root is as good a guess
        # as any.
        chosen = next((c for c in candidates if c.rsplit("/", 1)[-1].endswith("link0")), None)
        if chosen is None:
            chosen = next((c for c in candidates if c.rsplit("/", 1)[-1] == "base_link"), None)
        if chosen is None:
            return None

        try:
            from isaacsim.core.prims import SingleRigidPrim

            view = SingleRigidPrim(prim_path=chosen)
            view.initialize()
            self._arm_base_view = view
            logger.info("Solving in the frame of %s", chosen)
        except Exception:
            logger.debug("Could not bind the arm base link %s", chosen, exc_info=True)
        return self._arm_base_view

    def _require_solvable(self) -> None:
        """Refuse to solve when the answer could only be nonsense.

        Two states produce a plausible-looking `MotionResult` that means nothing,
        and both were found the hard way — by reading Isaac's own log after the
        library had already reported a number.

        **A stopped timeline de-initialises the articulation.** A handle bound
        before `scene.stop()` keeps `is_valid == True`, so nothing looks wrong,
        but `handles_initialized` goes False and Lula cannot read joint
        positions. Isaac logs it:

            [Warning] [articulation_subset] Attempting to access an
                      uninitialized robot Articulation.
            [Error]   [articulation_kinematics_solver] Attempted to compute
                      inverse kinematics for an uninitialized robot
                      Articulation. Cannot get joint positions

        and the solve comes back as a pose error of `inf` at 180 degrees, for
        every target, including ones well inside the workspace. Read as data
        that is a workspace boundary; it is a stale handle.

        **No solver was ever built.** A UR10 whose end-effector frame does not
        resolve leaves `_ik` as None, and every `pose_to` on it fails while
        looking like a motion problem.
        """
        if self._ik is None:
            raise MotionError(
                f"{self.prim_path}: no inverse-kinematics solver. The end-effector "
                f"frame did not resolve for this robot, so `pose_to` and "
                f"`command_pose` cannot be used on it. `move_ee_to` and `servo_to` "
                f"go through RMPflow and may still work; check "
                f"describe()['end_effector_frame'] and pass `rmp_config=` if the "
                f"motion configuration was not matched."
            )
        if getattr(self._articulation, "handles_initialized", True) is False:
            raise MotionError(
                f"{self.prim_path}: this handle was bound before the timeline "
                f"stopped and its articulation is no longer initialised, so the "
                f"solver cannot read joint positions. Every solve from here "
                f"returns an infinite error at 180 degrees regardless of the "
                f"target. Play the timeline and re-bind with Robot.attach()."
            )

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
            position, orientation = self._arm_base_pose()
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
        return self._require_pose(position, "position")

    def _require_pose(self, value: Any, kind: str) -> np.ndarray:
        """Lula's forward kinematics, or an error saying why there is none.

        Asked for the tool pose on a de-initialised articulation, Lula logs

            [Error] [articulation_kinematics_solver] Attempted to compute
                    forward kinematics for an uninitialized robot Articulation.
                    Cannot get joint positions

        and returns a scalar `nan`. `np.asarray` turns that into a
        zero-dimensional array, so `robot.ee_position` answers `nan` — printable,
        indexable-looking, and false. It is the same shape of bug as the joint
        read in `base._read_joint_state`, one layer up, and it has to be caught
        here as well: this path never touches that one.
        """
        pose = np.asarray(value, dtype=float)
        if pose.ndim >= 1 and pose.size >= 3 and bool(np.all(np.isfinite(pose))):
            return pose
        raise StaleArticulation(
            f"{self.prim_path}: cannot compute the end-effector {kind}. Lula got "
            f"nothing back from the articulation, which returns as `nan` rather "
            f"than an error. The handle is no longer backed by a live physics "
            f"view — play the timeline and re-bind with Robot.attach(). "
            f"describe() reports the same thing without raising."
        )

    @property
    def ee_orientation(self) -> np.ndarray:
        """End-effector orientation as a quaternion (w, x, y, z).

        Lula hands back a 3x3 rotation matrix here, and this property used to
        return it unchanged - while `move_ee_to(orientation=...)` takes a
        quaternion. Reading the current orientation and passing it straight back
        as a target therefore raised deep inside numpy, and any code that got as
        far as indexing it silently read a matrix row as a quaternion.
        """
        return _quaternion_from_matrix(self.ee_rotation)

    @property
    def ee_rotation(self) -> np.ndarray:
        """End-effector orientation as a 3x3 rotation matrix.

        Column `i` is the tool frame's own axis `i` expressed in world - so
        `ee_rotation[:, 0]` is where the flange's +X is pointing, which is what
        you want when checking that a suction cup is square to a surface.
        """
        self._ensure_motion_policy()
        self._sync_base_pose()
        _, rotation = self._ik.compute_end_effector_pose()
        return self._require_pose(rotation, "orientation")

    # -- Orientation-exact posing -------------------------------------------
    #
    # RMPflow treats orientation as a soft objective, so `move_ee_to` will put
    # the tool in the right place pointing somewhere convenient - measured at 11
    # degrees off vertical when straight down was asked for, and it stayed there
    # however long it ran. A suction cup that is 11 degrees off its surface does
    # not seal, so a task that needs the tool square to something needs a
    # different mechanism: solve the pose with IK, then command joints.
    #
    # The joints alone are not enough either. Commanding Lula's solution once
    # leaves this arm 0.21 rad short on the wrist under its own weight - the same
    # 11 degrees. One joint-space correction removes it (0.0003 rad, 0.04
    # degrees), so the loop below is a correction loop, not a single shot.
    #
    # Split into command / refine / measure rather than one blocking call so a
    # controller can drive it one transition per tick. `pose_to` is the blocking
    # convenience for scripts and demos.

    def _solver_indices(self) -> list[int]:
        """Articulation joint indices for the joints Lula solves, in Lula's order.

        Lula reports its own joint list and its own ordering, and there is no
        guarantee either matches the articulation's. Mapping by name is the only
        thing that survives an asset whose joints are declared in a different
        order from its kinematics file.
        """
        if self._solver_index_map is not None:
            return self._solver_index_map
        names = list(self.joint_names)
        solver_names = list(self._ik.get_kinematics_solver().get_joint_names())
        missing = [n for n in solver_names if n not in names]
        if missing:
            raise MotionError(
                f"{self.prim_path}: the kinematics solver names joints {missing} "
                f"that this articulation does not have ({names}). The motion "
                f"configuration does not match this robot."
            )
        self._solver_index_map = [names.index(n) for n in solver_names]
        return self._solver_index_map

    def command_pose(
        self,
        position: Any,
        orientation: Any = None,
        *,
        ramp: int = 0,
        raise_on_fail: bool = True,
    ) -> bool:
        """Solve for a tool pose and write the joint targets. Does not step.

        Returns False (or raises) when Lula finds no solution, which is a real
        answer: the pose is out of reach, or the orientation is unreachable at
        that point. Do not retry it with a nudged target and call that success.

        `ramp` splits the move into that many waypoints **in space**, issued one
        per `advance_pose()`, each solved for separately.

        Interpolating the joint vector instead is the obvious thing and it does
        not work: a straight line between two joint vectors bows in Cartesian
        space, and on an arm this size the bow is large. Commanded straight to a
        pose 25 cm above a box, the tool reached it correctly and hit the box on
        the way - the tool travels an arc, and the arc went through the table.

        Each waypoint is solved **warm-started from the previous solution**. That
        is not an optimisation. Solved independently, neighbouring waypoints come
        back in different IK branches - the same pose, the elbow on the other
        side - and the arm snaps between them hard enough to throw a 5 kg box
        28 m. Warm-started, consecutive solutions differ by about 0.07 rad for a
        10 cm step, and the arm simply travels.
        """
        self._require_solvable()
        self._ensure_motion_policy()
        self._sync_base_pose()
        target = as_vec3(position, name="position")
        rotation = as_quat(orientation) if orientation is not None else None

        action, solved = self._ik.compute_inverse_kinematics(
            np.asarray(target, dtype=float),
            np.asarray(rotation, dtype=float) if rotation is not None else None,
        )
        if not solved:
            self._pose_command = None
            if raise_on_fail:
                raise MotionError(
                    f"{self.prim_path}: no inverse-kinematics solution for position "
                    f"{np.round(target, 4).tolist()}"
                    + (f" with orientation {np.round(rotation, 4).tolist()}"
                       if rotation is not None else "")
                    + ". The pose is out of reach or the orientation cannot be held "
                      "there. This is a property of the arm, not of this run."
                )
            return False

        self._pose_solution = np.asarray(action.joint_positions, dtype=float)
        self._pose_command = self._pose_solution.copy()
        self._pose_goal = np.asarray(target, dtype=float)
        self._pose_orientation = rotation

        # Solving the final pose first is the reachability check: a ramp that
        # walks most of the way and then fails is worse than not starting.
        self._pose_from = self.ee_position.copy()
        self._pose_from_quat = self.ee_orientation
        current = np.asarray(self.joint_positions, dtype=float)
        self._pose_seed = np.asarray(
            [current[i] for i in self._solver_indices()], dtype=float
        )
        self._pose_ramp = max(0, int(ramp))
        self._pose_phase = 0
        if self._pose_ramp:
            self.advance_pose()
        else:
            self._apply_pose_command()
        return True

    def advance_pose(self) -> bool:
        """Issue the next increment of a ramped move. True once the target is out.

        One transition per call, so a controller can drive a ramped approach from
        `compute` without ever stepping physics itself.
        """
        if self._pose_command is None:
            raise MotionError(
                f"{self.prim_path}: advance_pose() before any command_pose()."
            )
        if not self._pose_ramp:
            return True
        self._pose_phase = min(self._pose_phase + 1, self._pose_ramp)
        if self._pose_phase >= self._pose_ramp:
            self._pose_command = self._pose_solution.copy()
            self._pose_ramp = 0
            self._apply_pose_command()
            return True

        fraction = self._pose_phase / float(self._pose_ramp)
        waypoint = self._pose_from + (self._pose_goal - self._pose_from) * fraction
        rotation = None
        if self._pose_orientation is not None:
            rotation = _slerp(self._pose_from_quat, self._pose_orientation, fraction)

        self._sync_base_pose()
        solution, solved = self._ik.get_kinematics_solver().compute_inverse_kinematics(
            self._end_effector_frame,
            np.asarray(waypoint, dtype=float),
            np.asarray(rotation, dtype=float) if rotation is not None else None,
            warm_start=self._pose_seed,
        )
        if solved:
            self._pose_seed = np.asarray(solution, dtype=float)
            self.set_joint_positions(self._pose_seed, indices=self._solver_indices())
        else:
            # A waypoint on a straight line between two reachable poses can still
            # be unreachable. Say so and hold - do not silently take the joint
            # shortcut, which is the arc this ramp exists to avoid.
            logger.warning(
                "%s: no IK solution for waypoint %s on the way to %s; holding. "
                "The straight path between these poses leaves the workspace.",
                self.prim_path, np.round(waypoint, 3).tolist(),
                np.round(self._pose_goal, 3).tolist(),
            )
        return False

    def _apply_pose_command(self) -> None:
        command = self._pose_command
        indices = self.arm_joint_indices
        if command.size == self.dof:
            self.set_joint_positions(command)
        else:
            self.set_joint_positions(command[: len(indices)], indices=indices)

    def refine_pose(self) -> float:
        """Add one joint-space correction. Returns the error it corrected, in rad.

        Feed-forward, not feedback on the goal: the command is pushed past the
        target by however far the arm fell short, so gravity droop is cancelled
        rather than merely measured.
        """
        if self._pose_command is None:
            raise MotionError(
                f"{self.prim_path}: refine_pose() before any command_pose()."
            )
        achieved = np.asarray(self.joint_positions, dtype=float)
        size = self._pose_solution.size
        error = self._pose_solution - achieved[:size]
        self._pose_command = self._pose_command + error
        self._apply_pose_command()
        return float(np.max(np.abs(error))) if error.size else 0.0

    def pose_error(self) -> dict[str, float]:
        """How far the tool is from the last commanded pose, as measured."""
        if self._pose_command is None:
            raise MotionError(
                f"{self.prim_path}: pose_error() before any command_pose()."
            )
        achieved = np.asarray(self.joint_positions, dtype=float)
        size = self._pose_solution.size
        joint = float(np.max(np.abs(self._pose_solution - achieved[:size])))
        position = float(np.linalg.norm(self.ee_position - self._pose_goal))
        angle = 0.0
        if self._pose_orientation is not None:
            angle = _angle_between(self.ee_rotation, self._pose_orientation)
        return {"position": position, "angle": angle, "joint": joint}

    def pose_to(
        self,
        position: Any,
        orientation: Any = None,
        *,
        tolerance: float = 0.005,
        angle_tolerance: float = 2.0,
        corrections: int = 4,
        settle_steps: int = 120,
        ramp: int = 16,
        ramp_steps: int = 24,
        raise_on_fail: bool = True,
    ) -> MotionResult:
        """Drive the tool to a pose, holding orientation. Blocks; steps physics.

        Use this over `move_ee_to` whenever the tool's direction matters - a
        suction cup on a surface, an insertion, a pour. `move_ee_to` is a
        reactive policy and will not hold an orientation; this will, or will say
        it could not.
        """
        if not self.command_pose(position, orientation, ramp=ramp,
                                 raise_on_fail=raise_on_fail):
            return MotionResult(False, 0, float("inf"), list(as_vec3(position)), 180.0)

        steps = 0
        while not self.advance_pose():
            self.scene.step(ramp_steps)
            steps += ramp_steps
        for attempt in range(max(1, corrections) + 1):
            self.scene.step(settle_steps)
            steps += settle_steps
            error = self.pose_error()
            if error["position"] <= tolerance and error["angle"] <= angle_tolerance:
                return MotionResult(True, steps, error["position"],
                                    self._pose_goal.tolist(), error["angle"])
            if attempt < corrections:
                self.refine_pose()

        error = self.pose_error()
        if raise_on_fail:
            raise MotionError(
                f"{self.prim_path}: pose not held after {corrections} corrections - "
                f"{error['position']:.4f} m and {error['angle']:.1f} deg from target "
                f"{np.round(self._pose_goal, 4).tolist()}. The solution exists (IK "
                f"found it); the drives are not tracking it."
            )
        return MotionResult(False, steps, error["position"],
                            self._pose_goal.tolist(), error["angle"])

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

        # The closest approach over the whole run, not just the last sample. A
        # tool hovering on the tolerance boundary dips in and out, so the error
        # at the moment the loop gives up is an arbitrary point in that
        # oscillation — and reporting it produced the message
        # "final error 0.0049 m > 0.005 m", which is not an inequality that can
        # hold. Measured on a UR10: the arm was arriving every few ticks and
        # never staying for three, and the message sent me looking for a
        # workspace problem that did not exist.
        best = float("inf")

        for step in range(max_steps):
            reached = self.servo_to(
                target, orientation, tolerance=tolerance, hold=hold_steps
            )
            self.scene.step(1)
            best = min(best, self._servo_error)
            if reached:
                return MotionResult(True, step + 1, self._servo_error, target.tolist())

        error = self._servo_error
        result = MotionResult(False, max_steps, best, target.tolist())
        if raise_on_fail:
            if best <= tolerance:
                # It got there. It would not settle, which is a different fault
                # with a different fix: damp the approach or relax `hold_steps`,
                # rather than go looking for an obstacle.
                raise MotionError(
                    f"End effector reached {target.round(3).tolist()} (closest "
                    f"approach {best:.4f} m, inside the {tolerance} m tolerance) but "
                    f"never held it for {hold_steps} consecutive steps in "
                    f"{max_steps}. It is oscillating around the target, not blocked: "
                    f"raise `tolerance`, lower `hold_steps`, or damp the approach."
                )
            raise MotionError(
                f"End effector did not reach {target.round(3).tolist()} within "
                f"{max_steps} steps (closest approach {best:.4f} m, last "
                f"{error:.4f} m, tolerance {tolerance} m)."
                + self._why_it_could_not_reach(orientation)
            )
        return result

    def _why_it_could_not_reach(self, orientation: Any) -> str:
        """Name the likely cause instead of offering the same two guesses.

        "outside the workspace or blocked by a collision" was the whole
        explanation, and for the failure that actually keeps happening it is
        wrong in a way that ends runs. Measured: a Franka asked for
        `[0.45, 0.2, 0.135]` pointing down came back 0.095 m short, repeatedly,
        for a target comfortably inside its envelope. The joint vector said why
        -- joint 6 sat at 3.724 rad against a 3.752 limit. The wrist winds up
        over a sequence of solves, and once it is against the stop RMPflow can
        only satisfy the orientation by trading position away.

        An agent read the old message, concluded "a real physical/kinematic
        limit", and abandoned the object. Homing the arm and repeating the same
        request solved it to 0.009 m at 0.0 degrees. So when joints are pinned,
        say so first: the fix is a starting pose, not a different target.
        """
        pinned = self._joints_against_limits()
        if pinned and orientation is not None:
            return (
                f" Joints are at their limits: {', '.join(pinned)}. With an "
                f"orientation demanded as well, the solver can only satisfy it by "
                f"driving into those stops, so it gives up position instead — "
                f"which is what this error looks like. The target is probably "
                f"reachable: send the arm to a neutral pose first "
                f"(`set_joint_positions(home)`) and repeat this call. Retry "
                f"without `orientation` to confirm before assuming the workspace "
                f"is the problem."
            )
        if pinned:
            return (
                f" Joints are at their limits: {', '.join(pinned)}. Home the arm "
                f"and repeat before concluding the target is unreachable."
            )
        return " The target is likely outside the workspace or blocked by a collision."

    def _joints_against_limits(self, margin: float = 0.05) -> list[str]:
        """Arm joints sitting within `margin` radians of a travel limit."""
        try:
            positions = self.joint_positions
            limits = self.joint_limits
            names = self.joint_names
        except Exception:
            return []

        pinned = []
        finger = set(getattr(getattr(self, "gripper", None), "joint_indices", []) or [])
        for index, (low, high) in enumerate(limits):
            if index in finger or index >= len(positions) or low is None or high is None:
                continue
            if high - low <= 0:          # `low > high` is USD for "locked"
                continue
            value = float(positions[index])
            if value - low < margin or high - value < margin:
                pinned.append(f"{names[index]}={value:.3f} (limits {low:.3f}..{high:.3f})")
        return pinned

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

    # ── Planned motion ────────────────────────────────────────────────────────

    def planner(self) -> Any:
        """The global planner for this arm, built on first use.

        Separate from `servo_to` on purpose. The reactive policy and the planner
        answer different questions — "which way do I move now" versus "is there
        a route at all" — and a task normally wants both: plan the transfer,
        servo the last centimetres onto the object.
        """
        from .planning import CuMotionPlanner

        if self._planner is None:
            self._planner = CuMotionPlanner(
                self._planner_robot_name(),
                self.joint_names,
                obstacles=lambda: sorted(self._obstacle_paths),
            )
        return self._planner

    def _planner_robot_name(self) -> str:
        """Which cuMotion configuration describes this arm.

        Taken from the catalogue entry the robot was spawned from where there is
        one, so a second arm needs a registry line rather than a code change.
        Falls back to the prim's own name, which is right often enough to be
        worth trying and produces a listing of valid names when it is not.
        """
        for source in (getattr(self, "_robot_type", None), getattr(self, "asset_name", None)):
            if source:
                return str(source).lower()
        return self.prim_path.rstrip("/").rsplit("/", 1)[-1].lower()

    def plan_to(self, position: Any, orientation: Any = None) -> Any:
        """Plan a collision-free route to a Cartesian target. Does not move.

        Returns a `MotionPlan`; drive it with `follow` one tick at a time inside
        a controller, or `move_along` while exploring. Raises `NoPathFound` when
        there is no route, which is a real answer and not a reason to fall back
        to reactive control silently — RMPflow would drive at the same target
        and stall against whatever the planner just told you is in the way.
        """
        planner = self.planner()
        planner.set_base_pose(self.base_position, self.base_orientation)
        q_initial = planner.joint_subset(self.joint_positions, self.joint_names)
        return planner.plan_to_pose(
            q_initial,
            as_vec3(position, name="position"),
            as_quat(orientation) if orientation is not None else None,
        )

    def follow(
        self,
        plan: Any,
        *,
        restart: bool = False,
        lag_tolerance: float = 0.6,
        max_step: float | None = None,
    ) -> bool:
        """Advance one control tick along a plan. Returns True when it is done.

        Same contract as `servo_to` — one tick, no blocking, no stepping — so a
        controller state that plans instead of servoing keeps exactly the same
        shape.

        Two things keep the *tracked* motion close to the *planned* one, which
        is the whole safety argument for planning at all. A plan that is
        collision-free on paper buys nothing if the arm takes a shortcut across
        it.

          * **The clock advances by one physics step per call.** That is
            exact, because the graph is triggered by the physics step and
            `move_along` steps once per call — but only because of that. It was
            briefly read from the timeline instead, to be safe against a caller
            ticking at the render rate, and that broke every use outside the
            graph: `Scene.step` drives PhysX directly and does not advance the
            timeline, so elapsed time read as zero and the arm never moved at
            all. The lag gate below is what actually makes a wrong tick rate
            safe; the clock just has to agree with whoever is stepping.
          * **The clock stalls while the arm is behind.** If the measured joints
            are more than `lag_tolerance` radians from what was last commanded,
            time does not advance this tick. The arm is allowed to catch up
            rather than being handed a target further along a curve it has not
            reached — standard practice for trajectory following on hardware,
            and the reason a slow tick makes the motion slower instead of
            straighter.

        An arm that can never catch up will never finish; that is what the
        calling state's own timeout is for, and it is the right failure to have
        rather than a collision.
        """
        if restart or self._plan is not plan:
            self._plan = plan
            self._plan_time = 0.0

        index = {name: i for i, name in enumerate(self.joint_names)}
        indices = np.asarray([index[n] for n in plan.joint_names], dtype=int)
        measured = np.asarray(self.joint_positions, dtype=float)[indices]

        elapsed = float(getattr(self.scene, "dt", None) or 1.0 / 60.0)

        # Lag is measured against the plan, never against the last command.
        # Comparing with the command is self-defeating once the command is
        # clamped: the arm is always near the clamped target by construction, so
        # the clock advances anyway, the plan runs away, and the final sample
        # drags the arm to the end pose in a straight line through everything
        # the route went around. Which is precisely the collision this exists to
        # prevent, arrived at by a different road.
        sample, velocities = plan.sample(self._plan_time)
        self._plan_lag = float(np.max(np.abs(measured - sample))) if sample.size else 0.0
        if self._plan_lag <= lag_tolerance:
            self._plan_time += elapsed
            sample, velocities = plan.sample(self._plan_time)

        # Never command a pose further than `max_step` from where the arm is.
        # A plan is collision-free along its path, not along the chord between
        # two samples of it, and a slow tick samples it coarsely.
        scale = 1.0
        if max_step is not None:
            step = sample - measured
            distance = float(np.max(np.abs(step))) if step.size else 0.0
            if distance > max_step:
                scale = max_step / distance
        self._plan_command = measured + (sample - measured) * scale

        self._controller().apply_action(
            articulation_action(
                joint_positions=self._plan_command,
                joint_velocities=np.asarray(velocities, dtype=float) * scale,
                joint_indices=indices,
            )
        )
        return self._plan_time >= plan.duration and self._plan_lag <= lag_tolerance

    # ── Obstacles ─────────────────────────────────────────────────────────────

    # Measured against Lula, not taken from the Visual* class list. Every one of
    # the five core primitives has a Visual* wrapper and constructs happily, but
    # Lula's world model only understands three of them — `add_obstacle` returns
    # False for a cylinder or a cone and logs nothing above debug. Listing them
    # here meant a registered pillar was silently not an obstacle, and the arm
    # swept it off the table on the first traverse.
    _OBSTACLE_WRAPPERS = {
        "Cube": "VisualCuboid",
        "Sphere": "VisualSphere",
        "Capsule": "VisualCapsule",
    }

    def add_obstacle(self, target: Any, *, static: bool = False, reactive: bool = True) -> bool:
        """Register a body the arm must not hit.

        An empty obstacle set means the arm moves straight through the scene. It
        still *reaches* its target, so that failure looks like success right up
        until the elbow sweeps a finished stack off the table.

        `target` may be a `RigidObject`, a prim path, or an already-wrapped
        core-API object. Pass `static=True` for anything that will not move
        (a table, a wall); a static obstacle is baked once instead of re-read
        every step.

        Note the object being *manipulated* should not be registered — the arm
        has to touch that one. The same goes for a surface it must place *onto*:
        a registered obstacle is somewhere the tool will not be taken, so
        registering the table makes putting anything down impossible.

        The two backends do not see the same world. The planner holds cuboids,
        spheres, capsules and planes; Lula holds the first three. Where only the
        planner can represent something, the path is still recorded and a
        warning says the reactive policy will not avoid it —
        `unavoidable_by_servo()` lists those, so a caller can check rather than
        discover it by collision.

        Where *neither* can represent it this raises, because the alternative is
        worse than a rejection: a cylinder handed to the planner does not fail
        politely, it makes every later plan fail for as long as it stays
        registered.

        `reactive=False` registers it for planning only. Worth reaching for
        whenever gross motion is planned and the policy is left to do short
        local moves: RMPflow's repulsion does not stop at the obstacle, and a
        post 23 cm away was measured pulling a descent 1.4 cm off-centre —
        enough to land the fingers beside a 4 cm cube and push it away. An
        obstacle the tool never goes near costs accuracy and buys nothing.
        Registering the *surface being placed onto* this way is the same idea:
        the planner routes over it, and the final descent is not fought.

        Returns True when the reactive policy accepted it too.
        """
        from .planning import UNREPRESENTABLE_TYPES

        path = target if isinstance(target, str) else getattr(target, "prim_path", str(target))
        prim = get_stage().GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"No prim at {path!r} to use as an obstacle")

        kind = str(prim.GetTypeName())
        if kind in UNREPRESENTABLE_TYPES:
            raise ValueError(
                f"{path} is a {kind}, which neither the planner nor the reactive "
                f"policy can represent — both hold cuboids, spheres and capsules, "
                f"and the planner adds planes.\n\n"
                f"Cover the same volume with a capsule or a cuboid and register "
                f"that instead. The proxy needs a collider and a pose; it does not "
                f"need to be visible, and the real {kind} can stay as it is."
            )

        _repair_extent(prim)
        self._obstacle_paths.add(path)
        # A changed obstacle set invalidates the planner's world; it is rebound
        # on the next plan rather than here, so adding several costs one rebind.
        self._plan = None

        if not reactive:
            logger.info("%s registered for planning only", path)
            return False

        try:
            self._ensure_motion_policy()
            obstacle = self._wrap_obstacle(path)
            accepted = bool(self._rmpflow.add_obstacle(obstacle, static=static))
        except Exception as exc:
            logger.warning(
                "%s is registered as an obstacle for planning, but the reactive "
                "policy cannot represent it (%s). servo_to will drive straight "
                "through it; use plan_to for motions that pass near it.",
                path,
                exc,
            )
            return False

        if not accepted:
            logger.warning(
                "%s is registered as an obstacle for planning, but Lula would not "
                "accept it, so servo_to will not avoid it.",
                path,
            )
            return False

        self._obstacles[path] = obstacle
        return True

    def unavoidable_by_servo(self) -> list[str]:
        """Registered obstacles the reactive policy cannot see.

        These are avoided by `plan_to` and ignored by `servo_to`. A non-empty
        list means gross motion has to be planned rather than servoed.
        """
        return sorted(self._obstacle_paths - set(self._obstacles))

    def _wrap_obstacle(self, prim_path: str) -> Any:
        """Wrap an existing prim in the core-API type RMPflow expects.

        Wrappers are cached per prim and reused. Building one has a side effect —
        it rewrites the prim's `extent` with the transform's scale baked in — so
        a task that takes an obstacle out of the reactive set and puts it back
        each time it picks or places re-corrupts and re-repairs the same prim
        dozens of times per run. That showed up as a wall of repair warnings,
        and it leaves a window in which any bounds query, the motion planner's
        world included, reads the obstacle at a fraction of its size.
        """
        import isaacsim.core.api.objects as core_objects

        cached = self._obstacle_wrappers.get(prim_path)
        if cached is not None:
            return cached

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
        wrapped = getattr(core_objects, wrapper)(
            prim_path=prim_path, name=f"obstacle_{prim_path.strip('/').replace('/', '_')}"
        )
        # Binding rewrites `extent` with the transform's scale already applied,
        # and the transform then applies it again. Measured on a bare cube:
        #
        #   before   extent ±1                    world size 0.06 x 0.06 x 0.35
        #   after    extent ±(0.03, 0.03, 0.175)  world size 0.0018 x 0.0018 x 0.0613
        #
        # So the act of registering something as an obstacle is what makes every
        # bounds query see it at a fraction of its size — including the motion
        # planner's collision world, which is why the tool cleared a sliver while
        # the arm's links went through the real post. Repair it here, at the
        # point of damage: repairing before this call, which is what the first
        # attempt did, is undone by this call one line later.
        _repair_extent(prim)
        self._obstacle_wrappers[prim_path] = wrapped
        return wrapped

    def remove_obstacle(self, target: Any) -> bool:
        """Stop avoiding a body, in both backends.

        Needed before touching something that was previously being avoided —
        and before placing onto it, which is the same thing from the tool's
        point of view.
        """
        path = getattr(target, "prim_path", target)
        known = path in self._obstacle_paths
        self._obstacle_paths.discard(path)
        self._plan = None

        obstacle = self._obstacles.pop(path, None)
        if obstacle is not None:
            try:
                self._rmpflow.remove_obstacle(obstacle)
            except Exception:  # noqa: BLE001 — the registry is what must end up right
                logger.debug("Could not remove obstacle %s from RMPflow", path, exc_info=True)
        return known

    def clear_obstacles(self) -> None:
        for obstacle in list(self._obstacles.values()):
            try:
                self._rmpflow.remove_obstacle(obstacle)
            except Exception:  # noqa: BLE001 — the registry is what must end up empty
                logger.debug("Could not remove obstacle", exc_info=True)
        self._obstacles.clear()
        self._obstacle_paths.clear()
        self._plan = None

    def obstacles(self) -> list[str]:
        """Everything the arm has been told to avoid, whichever backend sees it."""
        return sorted(self._obstacle_paths)

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
        self.gripper.open(settle_steps=30)
        self.move_ee_to(target + np.array([0.0, 0.0, approach_height]), orientation)
        self.move_ee_to(target, orientation)
        self.gripper.close(settle_steps=45)
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

        `speed` is the hand speed to aim for, in m/s, and it is a request rather
        than a promise: `release_hand_speed` reports what the arm achieved and
        `speed_shortfall` how far short it fell. Check them. A swing that asks
        for more than the arm can deliver is the normal case, not an error.
        """
        if not self.is_grasping(obj):
            raise MotionError(
                "Cannot throw: the object is not currently grasped. Call grasp() "
                "first and confirm it returned True."
            )
        if observe_steps < 0:
            raise ValueError("observe_steps must be >= 0")

        vector = as_vec3(direction, name="direction")
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-6:
            raise ValueError("direction must be a non-zero vector")
        unit = vector / magnitude

        # How many consecutive decelerating steps mean the swing is done
        # accelerating, and how many steps to let the fingers actually open.
        stall_steps, spinup_steps, settle_steps = 6, 8, 4

        start = self.ee_position
        self.move_ee_to(start - unit * windup, raise_on_fail=False)

        sweep = float(speed) * 0.35 + windup
        end = start + unit * sweep
        release_at = start - unit * windup + unit * (sweep * release_fraction)

        self._ensure_motion_policy()
        self._rmpflow.set_end_effector_target(target_position=end)
        controller = self._controller()

        released, reason = False, ""
        release_speed = peak_speed = 0.0
        decelerating = 0
        previous = self.ee_position

        for step in range(400):
            self._rmpflow.update_world()
            controller.apply_action(self._policy.get_next_articulation_action())
            self.scene.step(1)

            current = self.ee_position
            hand_speed = float(np.linalg.norm(current - previous)) / self.scene.dt
            previous = current

            decelerating = decelerating + 1 if hand_speed < peak_speed * 0.85 else 0
            peak_speed = max(peak_speed, hand_speed)

            if hand_speed >= float(speed):
                reason = "reached the requested hand speed"
            elif float(np.dot(current - release_at, unit)) >= 0.0:
                reason = "passed the geometric release point"
            elif step >= spinup_steps and decelerating >= stall_steps:
                # The swing is already slowing down. Carrying on to the
                # geometric release point means letting go at a crawl, which is
                # what `sweep = speed * 0.35` produces whenever the requested
                # speed puts `end` outside the arm's reach — a Franka reaches
                # about 0.85 m and the default `speed` alone asks for 1.1 m of
                # travel. RMPflow damps hard against the workspace boundary, so
                # the arm arrived at the release point at 0.15 m/s and the
                # "throw" was a drop. Releasing at the peak is the honest
                # reading of a swing that has run out of room.
                reason = (
                    "the swing stopped accelerating before the release point — "
                    "the arc runs out of workspace"
                )
            else:
                continue

            release_speed = hand_speed
            released = True
            break

        if not released:
            self.release()
            raise MotionError(
                "The arm never reached the release point — the throw arc is likely "
                "outside the workspace. Try a shorter windup or a direction closer "
                "to the robot's reach."
            )

        # Opening the gripper is a command to the finger joints, not an event.
        # Until physics steps, the fingers are still closed and the object is
        # still pinched -- which is how this used to return `released: True`
        # next to `still_held: True`, and with `observe_steps=0` never stepped
        # at all.
        self.gripper.set_position(self.gripper.open_width, settle_steps=0)
        for _ in range(settle_steps):
            self.scene.step(1)
        still_held = self.is_grasping(obj)

        release_position = obj.position
        apex = float(release_position[2])
        trajectory = [release_position.round(4).tolist()]
        landed_at = None
        previous_height, falling = float(release_position[2]), False

        for step in range(observe_steps):
            self.scene.step(1)
            position = obj.position
            height = float(position[2])
            apex = max(apex, height)

            # Touchdown is where the fall stops, and it is worth finding: the
            # distance after it is the object rolling, which is not throwing.
            if landed_at is None:
                if height < previous_height - 1e-5:
                    falling = True
                elif falling:
                    landed_at = position
            previous_height = height

            if step % 10 == 0:
                trajectory.append(position.round(4).tolist())

        final = obj.position
        flight = (
            round(float(np.linalg.norm((landed_at - release_position)[:2])), 4)
            if landed_at is not None
            else None
        )
        total = float(np.linalg.norm((final - release_position)[:2]))
        return {
            "released": not still_held,
            "release_reason": reason,
            "requested_speed": float(speed),
            "release_hand_speed": round(release_speed, 3),
            "peak_hand_speed": round(peak_speed, 3),
            "speed_shortfall": round(max(0.0, float(speed) - release_speed), 3),
            "object_speed_after_release": round(obj.speed, 3),
            "apex_height": round(apex, 4),
            "landing_position": final.round(4).tolist(),
            "flight_distance": flight,
            "rolled_distance": round(total - flight, 4) if flight is not None else None,
            "horizontal_distance": round(float(np.linalg.norm((final - start)[:2])), 4),
            "still_held": still_held,
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
