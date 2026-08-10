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

from .._compat import articulation_action, as_quat, as_vec3, get_stage, single_articulation

if TYPE_CHECKING:
    from ..scene import Scene

logger = logging.getLogger("simliverse_sim.robots.base")


class StaleArticulation(RuntimeError):
    """This handle is no longer backed by a live physics view.

    Its own type rather than a bare RuntimeError because it is the one failure
    a caller can actually recover from without changing anything else: play the
    timeline, `Robot.attach()` again, carry on. Everything else about the scene
    is still true.
    """


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


def classify_morphology(
    joint_names: list[str],
    groups: JointGroups,
    link_names: list[str] | None = None,
) -> Morphology:
    """Infer what kind of robot this is from its joint structure.

    Link names are consulted as well, because a joint set alone is often not
    descriptive enough. Isaac's quadcopter names its joints `m1_joint`..`m4_joint`
    — matching no rotor token — while its links are `m1_prop`..`m4_prop`, which
    say plainly what the robot is. Classified from joints alone it came back
    UNKNOWN, so `attach` returned a bare handle with no `fly_to`, `hover` or
    `altitude`, while `spawn` returned a working AerialRobot because it reads the
    morphology from the catalogue instead. The same robot answered differently
    depending on how you got hold of it, and every controller uses `attach`.
    """
    dof = len(joint_names)
    links = [n.rsplit("/", 1)[-1] for n in (link_names or [])]

    rotor_links = [n for n in links if any(t in n.lower() for t in ROTOR_TOKENS)]
    if len(groups.rotors) >= 3 or len(rotor_links) >= 3:
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
        # An articulation cannot initialize before physics has ticked, and
        # building a handle on a stopped timeline is the common way to get a
        # robot that ignores every command. But this runs inside controllers
        # too, where the timeline is already playing and physics is mid-step:
        # play/step from there is re-entrant and silently desynchronises the
        # run. When it is already playing there is nothing to arrange, so do
        # nothing — the condition that makes the wait necessary is exactly the
        # condition that makes it safe.
        if not self._timeline_playing():
            self.scene.play()
            self.scene.step(4)
        self._articulation.initialize()

        self._root_body: Any = None
        self._pose_source = "unknown"
        self.groups = JointGroups.classify(self.joint_names)

    @staticmethod
    def _timeline_playing() -> bool:
        from .._compat import get_timeline

        try:
            return bool(get_timeline().is_playing())
        except Exception:
            logger.debug("Could not read timeline state", exc_info=True)
            return False

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
        return self._joint_state("get_joint_positions", "positions")

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._joint_state("get_joint_velocities", "velocities")

    def _read_joint_state(self, method: str) -> np.ndarray | None:
        """One attempt at a joint-state read. `None` means "do not trust this".

        The failure worth catching is not the exception — it is the read that
        *succeeds* and means nothing. When the physics simulation view has gone,
        Isaac does not raise: it logs

            [Warning] [articulation] Physics Simulation View is not created yet
                      in order to use get_joint_positions

        and hands back `None`. `np.asarray(None, dtype=float)` is `array(nan)`,
        a zero-dimensional array, which then propagates as a plausible-looking
        answer until something indexes it. Downstream that surfaced as
        `ee_position` returning a bare `nan`, and as

            IndexError: too many indices for array: array is 0-dimensional

        raised from `Gripper.position` — three frames deep in this library,
        naming nothing about the actual cause. An agent reading that spent ten
        calls checking whether it was holding the API wrong.

        A real reading is one value per degree of freedom, and all of them
        finite. Anything else is the simulator telling us it cannot answer.
        """
        try:
            value = np.asarray(getattr(self._articulation, method)(), dtype=float)
        except Exception:
            logger.debug("%s failed on %s", method, self.prim_path, exc_info=True)
            return None

        if value.ndim != 1 or value.size == 0:
            return None
        # `dof` reads the articulation too, so a dead handle can report 0 here.
        # Compare only when it gives an answer, rather than turning one unusable
        # reading into a second.
        try:
            expected = int(self._articulation.num_dof)
        except Exception:
            expected = 0
        if expected and value.size != expected:
            return None
        if not bool(np.all(np.isfinite(value))):
            return None
        return value

    def _joint_state(self, method: str, kind: str) -> np.ndarray:
        """Joint state, rebuilding the articulation view if it has gone stale.

        The tensor backend can refuse a view that was built across a timeline
        cycle — "Failed to get DOF positions from backend", raised from inside
        the physics API, naming neither the robot nor the reason. It is the same
        failure that made a quadruped's `stand()` unusable while the identical
        articulation, bound freshly, answered immediately.

        So the view is rebuilt once and the read retried. If the rebuilt view
        cannot answer either, the problem is real rather than stale, and saying
        so is the whole point — see `_read_joint_state` for what "cannot answer"
        looked like before it was checked.
        """
        value = self._read_joint_state(method)
        if value is not None:
            return value

        logger.debug("Rebuilding the articulation view for %s", self.prim_path)
        try:
            self._articulation = single_articulation(self.prim_path)
            self._articulation.initialize()
        except Exception as exc:
            raise StaleArticulation(self._stale_message(kind, str(exc))) from exc

        value = self._read_joint_state(method)
        if value is None:
            raise StaleArticulation(self._stale_message(kind, None))
        return value

    def _stale_message(self, kind: str, rebuild_error: str | None) -> str:
        detail = (
            f" Re-binding it failed too: {rebuild_error}."
            if rebuild_error
            else " Re-binding it produced a reading that was still unusable."
        )
        return (
            f"{self.prim_path}: cannot read joint {kind}. The articulation is no "
            f"longer backed by a live physics view — Isaac returns nothing rather "
            f"than raising, which reads as `nan` if it is not checked.{detail} "
            f"This normally means the timeline stopped, or the simulation view "
            f"was torn down under a running session. Play the timeline and "
            f"re-bind with Robot.attach(); every reading taken from this handle "
            f"until then is meaningless."
        )

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

    def _root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the articulation root, from whichever source can answer.

        Three sources, tried in order of how much they can be trusted, because
        each of them has been observed failing on a different robot:

        1. **The articulation view.** Correct when it works, and it does not
           always: a quadruped spawned through the library raises "Failed to get
           root link transforms from backend" from inside the tensor API, while
           the identical articulation referenced by hand answers fine. The view
           built during the spawn's timeline cycling is the broken one.
        2. **The root link as a rigid body.** The physics pose of the same body,
           by a different route, and it answered every time the view above did
           not.
        3. **USD.** Last, because physics results are not written back to USD
           for every articulation — a Ridgeback reports a constant 0.308 for a
           link that physics has at 1.44, before or after a transform sync.

        `base_orientation` used to have no fallback at all, so a robot whose
        view was unusable could not report which way up it was, and every
        legged check — `is_upright`, `tilt_degrees`, `describe` — raised from
        four frames down with a message about tensor backends.
        """
        try:
            position, quaternion = self._articulation.get_world_pose()
            self._pose_source = "articulation"
            return np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
        except Exception:
            logger.debug("Articulation view has no root pose for %s", self.prim_path,
                         exc_info=True)

        try:
            from .._compat import articulation_root

            if self._root_body is None:
                from isaacsim.core.prims import SingleRigidPrim

                self._root_body = SingleRigidPrim(prim_path=articulation_root(self.prim_path))
                self._root_body.initialize()
            position, quaternion = self._root_body.get_world_pose()
            self._pose_source = "root body"
            return np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
        except Exception:
            logger.debug("Root body has no pose for %s", self.prim_path, exc_info=True)

        # Neither physics source could answer, so this is the authored pose:
        # what the scene was built with, not where the robot is. Recorded so
        # `asset_problems` can say so — a humanoid reporting exactly its spawn
        # height and exactly zero tilt after three seconds of settling is not
        # standing still, it is not being simulated.
        self._pose_source = "usd (authored)"
        from pxr import Gf, UsdGeom

        matrix = UsdGeom.Xformable(
            get_stage().GetPrimAtPath(self.prim_path)
        ).ComputeLocalToWorldTransform(0)
        rotation = Gf.Transform(matrix).GetRotation().GetQuat()
        imaginary = rotation.GetImaginary()
        return (
            np.asarray(matrix.ExtractTranslation(), dtype=float),
            np.asarray(
                [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float
            ),
        )

    @property
    def base_position(self) -> np.ndarray:
        """World position of the articulation root."""
        return self._root_pose()[0]

    @property
    def base_orientation(self) -> np.ndarray:
        return self._root_pose()[1]

    def set_base_pose(self, position: Any = None, orientation: Any = None) -> None:
        self._articulation.set_world_pose(
            position=as_vec3(position, name="position") if position is not None else None,
            orientation=as_quat(orientation) if orientation is not None else None,
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
        """Everything an agent needs to know about this robot, in one call.

        `describe()` is the call you make *because* something is wrong, so it
        must not be the call that fails. It used to raise `IndexError` from
        three frames down whenever the physics view had gone, which is the one
        moment it was most needed and the one moment it said nothing.

        A dead handle now returns a diagnosis in the same shape as the rest of
        this dictionary. Static facts — the prim path, the morphology, the
        controller class — still come back, because they remain true.
        """
        try:
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
                "asset_problems": self.asset_problems(),
                "capabilities": self.capabilities(),
            }
        except StaleArticulation as exc:
            return {
                "prim_path": self.prim_path,
                "morphology": self.morphology.value,
                "controller": type(self).__name__,
                "usable": False,
                "problem": str(exc),
                "recover": (
                    "scene.play(), then Robot.attach(prim_path) for a fresh handle. "
                    "Do not interpret any number read from this one."
                ),
            }

    def asset_problems(self) -> list[dict[str, str]]:
        """Defects in the robot asset itself, and what each one prevents.

        Reported, never repaired or worked around. A robot that cannot do
        something should say so plainly, in the first call an agent makes, so
        the answer is "this task is not possible with this asset and here is
        why" rather than a controller that has been bent around a broken
        gripper and works for nothing else.

        Editing the asset would be worse than failing: a policy trained against
        a robot we silently modified does not transfer to the real one, and the
        failure resurfaces as bad hardware long after anyone can connect it to
        this. Changing the robot is the user's call.
        """
        problems: list[dict[str, str]] = []
        limits = self.joint_limits
        names = self.joint_names

        # An arm with no end effector at all. This has to come first, and it has
        # to be here rather than left to the caller's judgement: the check below
        # is for a gripper whose *limits* are broken, and it iterates over the
        # finger joints — so a robot with zero finger joints passes it silently.
        # The worse defect was invisible while the milder one was reported.
        #
        # It reads as an ordinary arm otherwise. A UR10 answers `hasattr(arm,
        # "gripper")` with True and hands back a Gripper holding an empty joint
        # list, whose open() and close() then succeed by doing nothing. That is
        # the shape of the failure worth naming: not a crash, a no-op that
        # everything downstream treats as a working hand.
        end_effector = getattr(self, "gripper", None)
        if end_effector is not None and not getattr(end_effector, "joint_indices", None):
            problems.append({
                "issue": "this arm has no end effector",
                "detail": (
                    "the articulation declares no finger or jaw joints, so there is "
                    "nothing on the flange that can hold an object"
                ),
                "consequence": (
                    "open() and close() raise MotionError, and grasp() cannot form "
                    "a grasp. Any task that involves picking something up needs a "
                    "gripper fitted, or a different arm. Fitting one changes the "
                    "robot, so it is the user's decision — ask rather than "
                    "authoring one."
                ),
            })

        gripper = getattr(end_effector, "joint_indices", None) or []
        unbounded = [names[i] for i in gripper if limits[i][0] is None or limits[i][1] is None]
        if unbounded:
            problems.append({
                "issue": "gripper joints have no travel limits",
                "detail": ", ".join(unbounded),
                "consequence": (
                    "open() and close() have nothing to open or close *to*, so "
                    "they fall back to 0.0 and 0.04 m. If this gripper's real "
                    "travel differs, it will close past the object or stop short "
                    "of it, and a grasp will look like it formed and then drop."
                ),
            })

        self.base_position  # refresh `_pose_source`
        if self._pose_source == "usd (authored)":
            problems.append({
                "issue": "physics cannot see this robot",
                "detail": (
                    "neither the articulation view nor the root body would report a "
                    "pose, so positions come from the authored USD transform"
                ),
                "consequence": (
                    "Every pose reading is the value the scene was built with, not "
                    "where the robot is: it will report itself upright and at its "
                    "spawn height forever, including while it falls over. Joint "
                    "commands will not be tracked either. The articulation is not "
                    "registered with PhysX."
                ),
            })
        problems.extend(self._inertia_problems())
        problems.extend(self._pose_feedback_problems())
        return problems

    def _inertia_problems(self) -> list[dict[str, str]]:
        """Links PhysX had to invent an inertia tensor for.

        Isaac says so, once, at Info level and never again:

            The rigid body at /World/Arm/ee_link has a possibly invalid inertia
            tensor of {1.0, 1.0, 1.0}, small sphere approximated inertia was
            used. Either specify correct values...

        Note the mismatch: {1.0, 1.0, 1.0} is the tensor PhysX *substituted*,
        not the one the asset declares. A UR10's `ee_link` actually carries
        `diagonalInertia = (0, 0, 0)` with a mass of 1e-4, which is what a
        converter writes for a frame it has no inertia for. Checking for the
        number in the log message finds nothing; checking for the zero finds it.

        The link is then simulated with dynamics nobody chose, and on a wrist
        that carries a tool that shows up as a joint that will not settle.

        This is reported and never repaired: correcting an asset's inertia
        changes the dynamics being simulated, and a policy trained against a
        robot we quietly edited does not transfer.
        """
        try:
            from pxr import UsdPhysics
        except Exception:  # noqa: BLE001 — no USD, no check
            return []

        stage = get_stage()
        suspect: list[str] = []
        for link in self.links():
            prim = stage.GetPrimAtPath(str(link))
            if not prim or not prim.IsValid():
                continue
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            attr = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr()
            value = attr.Get() if attr else None
            if value is not None and all(abs(float(v)) < 1e-12 for v in value):
                suspect.append(str(link))

        if not suspect:
            return []
        return [{
            "issue": "links carry a placeholder inertia tensor",
            "detail": ", ".join(suspect),
            "consequence": (
                "PhysX reports these as possibly invalid and substitutes a small "
                "sphere approximation, so they are simulated with dynamics the "
                "asset never specified. Expect joints that oscillate or refuse to "
                "settle, worst on a wrist carrying a tool. Fixing it means editing "
                "the asset, which changes what is being simulated — the user's "
                "call, not a repair to make in passing."
            ),
        }]

    def _pose_feedback_problems(self) -> list[dict[str, str]]:
        """Links whose USD transform disagrees with where physics has them.

        Isaac writes physics results back to USD for most articulations and not
        for all of them. When it does not, the viewport shows a robot standing
        still while it drives, and every measurement taken from a link transform
        is wrong by however far it has actually moved — silently, because a
        stale number looks exactly like a real one.

        Only detectable once the robot has moved. At its authored pose the two
        sources agree by construction, so an empty result here means "nothing to
        see yet", not "these transforms are trustworthy". Call it again after
        the robot has driven somewhere if the answer matters.
        """
        try:
            from pxr import UsdGeom

            from isaacsim.core.prims import SingleRigidPrim

            from .._compat import get_stage

            stage = get_stage()
            worst_name, worst = None, 0.0
            for link in list(self.links())[:12]:
                path = str(link)
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                usd = np.asarray(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation(),
                    dtype=float,
                )
                view = SingleRigidPrim(prim_path=path)
                view.initialize()
                physics = np.asarray(view.get_world_pose()[0], dtype=float)
                gap = float(np.linalg.norm(physics - usd))
                if gap > worst:
                    worst_name, worst = path, gap
            if worst > 0.02:
                return [{
                    "issue": "link transforms are not written back from physics",
                    "detail": f"{worst_name} is {worst:.3f} m from where physics has it",
                    "consequence": (
                        "The viewport will show this robot in the wrong place, and "
                        "anything measured from a link transform — distance "
                        "travelled, where the base is — reads the stale value. "
                        "Read poses through the physics view instead."
                    ),
                }]
        except Exception:
            logger.debug("Could not compare USD and physics poses", exc_info=True)
        return []

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
