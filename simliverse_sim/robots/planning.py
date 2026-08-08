"""
Global motion planning, as a complement to the reactive policy — not a
replacement for it.

RMPflow is a local policy: it recomputes a direction to move every tick and has
no plan, so it cannot reason about a route. Two consequences are measurable
rather than theoretical, and both were hit on a two-cube transfer task:

  * **Local minima.** Crossing laterally over a 35 cm post with 11 cm clearance
    stalls the tool 18 cm from its target, pushed up and back by the obstacle's
    repulsive field, and it stays there forever. The identical target from a
    different starting pose converges in 3.5 s. The target is reachable; the
    *route* is what the policy cannot find.
  * **Field bias.** That same post pulled a descent 1.4 cm off-centre from 23 cm
    away — enough that the fingers landed beside a 4 cm cube and shoved it
    instead of gripping it. A repulsive field does not stop at the obstacle.

A planner has neither problem: it searches the configuration space once and
returns a route, so an obstacle 23 cm away is simply not on the path.

What it is *not* good at is the last centimetre. Insertion, contact, and
anything that has to react to what it touches still belong to the reactive
policy. The intended division is: plan the transfer, servo the approach.

Nothing here is required. `available()` reports whether the backend is present,
and every entry point degrades to a clear error naming `servo_to` rather than a
traceback from inside a GPU library.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = logging.getLogger("simliverse_sim.planning")

# cuMotion ships robot configurations as directories; the name is the directory.
# Enumerated at call time rather than hardcoded, so the error message lists what
# this install actually has instead of what this file was written against.
_CONFIG_SUBDIR = "robot_configurations"
_EXTENSION = "isaacsim.robot_motion.cumotion"


# Measured by reading which `CumotionWorldInterface.add_*` methods are actually
# implemented, rather than which ones exist — `add_cylinders` and `add_cones`
# are declared, typed `NoReturn`, and raise. Lula's world is a strict subset of
# this. Nothing represents a cylinder or a cone, so those have to be screened
# before they reach either backend: a cylinder passed to the planner does not
# fail politely, it makes every subsequent plan fail while it stays registered.
PLANNER_OBSTACLE_TYPES = frozenset({"Cube", "Sphere", "Capsule", "Plane"})
SERVO_OBSTACLE_TYPES = frozenset({"Cube", "Sphere", "Capsule"})
UNREPRESENTABLE_TYPES = frozenset({"Cylinder", "Cone"})


class PlanningUnavailable(RuntimeError):
    """No planner is available for this robot, or none is installed."""


class NoPathFound(RuntimeError):
    """The planner ran and could not find a collision-free route."""


def _to_numpy(value: Any) -> np.ndarray:
    """Warp arrays do not survive `np.asarray`; they carry `.numpy()`."""
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=float)
    return np.asarray(value, dtype=float)


class MotionPlan:
    """A time-parameterised, collision-free joint trajectory.

    Deliberately a value, not a controller: it holds no articulation and steps
    nothing, so it can be planned once and followed by whoever owns the control
    loop — a blocking helper while exploring, a ScriptNode one tick at a time.
    """

    def __init__(self, trajectory: Any, joint_names: Sequence[str], waypoints: np.ndarray) -> None:
        self._trajectory = trajectory
        self.joint_names = list(joint_names)
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.duration = float(trajectory.duration)

    def __repr__(self) -> str:
        return (
            f"<MotionPlan {self.duration:.2f}s, {len(self.waypoints)} waypoints, "
            f"{len(self.joint_names)} joints>"
        )

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions and velocities at time `t`, clamped to the duration."""
        state = self._trajectory.get_target_state(float(np.clip(t, 0.0, self.duration)))
        return _to_numpy(state.joints.positions), _to_numpy(state.joints.velocities)


def available() -> bool:
    """Whether a planning backend is installed and importable."""
    try:
        import isaacsim.robot_motion.cumotion  # noqa: F401
        import isaacsim.robot_motion.experimental.motion_generation  # noqa: F401
    except Exception:
        logger.debug("cuMotion is not importable", exc_info=True)
        return False
    return True


def supported_robots() -> list[str]:
    """Robot names this install ships cuMotion configurations for."""
    try:
        import os
        import pathlib

        import isaacsim.robot_motion.cumotion as cu_mg

        # Walk up from the package to the extension root, rather than asking the
        # extension manager: the manager's lookup needs a versioned extension id
        # that differs between installs, and getting it wrong returns an empty
        # list, which reads as "no robots supported" instead of "lookup failed".
        root = pathlib.Path(cu_mg.__file__).parents[3] / _CONFIG_SUBDIR
        return sorted(d for d in os.listdir(root) if (root / d).is_dir())
    except Exception:
        logger.debug("Could not enumerate cuMotion robot configurations", exc_info=True)
        return []


class CuMotionPlanner:
    """A cuMotion planner bound to one articulation.

    Holds the three pieces cuMotion needs — a robot configuration, a world it
    can collision-check against, and the planner itself — and keeps the world in
    step with a caller-supplied obstacle set.

    `obstacles` is a callable rather than a list because the set changes during
    a task: the thing being picked up must not be an obstacle while it is being
    picked up. Reading it per plan means the caller's `add_obstacle` /
    `remove_obstacle` calls are simply respected, with no second registry to
    keep in sync.
    """

    #: Metres of clearance demanded around every obstacle, over and above its
    #: geometry. A plan is collision-free for the *planned* joint path, and the
    #: arm does not follow that path exactly — PD drives lag, and lag is largest
    #: exactly where the trajectory is fastest. Without a margin the tracked
    #: path can clip an obstacle the plan cleared by millimetres, which presents
    #: as "the planner drove into the thing it was avoiding".
    DEFAULT_SAFETY_MARGIN = 0.02

    def __init__(
        self,
        robot_name: str,
        joint_names: Sequence[str],
        *,
        obstacles: Callable[[], Sequence[str]] | None = None,
        tool_frame: str | None = None,
        safety_margin: float | None = None,
    ) -> None:
        if not available():
            raise PlanningUnavailable(
                "cuMotion is not available in this Isaac Sim install, so there is "
                "no global planner. Reactive control is still available through "
                "servo_to / move_ee_to — but it has no plan, so it can stall in "
                "front of an obstacle rather than route around it."
            )
        self.robot_name = robot_name
        self.joint_names = list(joint_names)
        self._obstacles = obstacles or (lambda: [])
        self._tool_frame = tool_frame
        self.safety_margin = (
            self.DEFAULT_SAFETY_MARGIN if safety_margin is None else float(safety_margin)
        )

        self._robot = self._load_robot(robot_name)
        # cuMotion plans its own joint set, which is the arm without the
        # gripper. Everything crossing this boundary is mapped by name.
        self.planned_joints = list(self._robot.controlled_joint_names)

        self._binding: Any = None
        self._planner: Any = None
        self._generator: Any = None
        self._bound_obstacles: tuple[str, ...] = ()

    @staticmethod
    def _load_robot(robot_name: str) -> Any:
        import isaacsim.robot_motion.cumotion as cu_mg

        try:
            return cu_mg.load_cumotion_supported_robot(robot_name)
        except Exception as exc:
            known = supported_robots()
            raise PlanningUnavailable(
                f"cuMotion has no configuration for {robot_name!r}. This install "
                f"ships: {', '.join(known) if known else '(none found)'}.\n\n"
                f"A robot outside that list needs its own URDF and XRDF; until "
                f"then, use servo_to for reactive control."
            ) from exc

    # ── World ─────────────────────────────────────────────────────────────────

    def _ensure_world(self) -> None:
        """(Re)bind the planning world when the obstacle set has changed.

        `WorldBinding` reads geometry straight off the named prims, so obstacles
        are given as paths and the warp-array plumbing is not ours to get wrong.
        """
        import isaacsim.robot_motion.cumotion as cu_mg
        import isaacsim.robot_motion.experimental.motion_generation as mg

        wanted = tuple(sorted(self._obstacles()))
        if self._planner is not None and wanted == self._bound_obstacles:
            return

        strategy = mg.ObstacleStrategy()
        try:
            strategy.set_default_safety_tolerance(self.safety_margin)
        except Exception:
            logger.debug("Could not set the planner safety margin", exc_info=True)

        binding = mg.WorldBinding(
            world_interface=cu_mg.CumotionWorldInterface(),
            obstacle_strategy=strategy,
            tracked_prims=list(wanted),
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        try:
            binding.initialize()
        except RuntimeError as exc:
            # The binding reads collision geometry, so a purely visual prim is
            # not something it can avoid. Worth naming: a proxy authored to mark
            # a keep-out region is exactly the case, and the raw message does not
            # say what to do about it.
            raise PlanningUnavailable(
                f"The planning world could not be built: {exc}\n\n"
                f"Every obstacle needs collision geometry — the planner reads "
                f"colliders, not renderable shapes. Apply UsdPhysics.CollisionAPI "
                f"to the prim, or register something that already has one."
            ) from exc

        kwargs: dict[str, Any] = {
            "cumotion_robot": self._robot,
            "cumotion_world_interface": binding.get_world_interface(),
        }
        if self._tool_frame:
            kwargs["tool_frame"] = self._tool_frame

        self._binding = binding
        self._planner = cu_mg.GraphBasedMotionPlanner(**kwargs)
        self._generator = cu_mg.TrajectoryGenerator(
            cumotion_robot=self._robot, robot_joint_space=self.planned_joints
        )
        self._bound_obstacles = wanted
        logger.info(
            "cuMotion world bound with %d obstacle(s): %s",
            len(wanted),
            ", ".join(wanted) or "(none)",
        )

    def set_base_pose(self, position: Any, orientation: Any) -> None:
        """Tell the planner where the robot's base is in world coordinates.

        cuMotion plans in the base frame while targets are given in world
        coordinates, so a robot anywhere other than the origin plans into the
        wrong place entirely if this is skipped — the same class of bug that
        made Cartesian control silently base-relative before.
        """
        self._ensure_world()
        try:
            import warp as wp

            world = self._binding.get_world_interface()
            world.update_world_to_robot_root_transforms(
                (
                    wp.array(np.asarray([position], dtype=np.float32), dtype=wp.vec3),
                    wp.array(np.asarray([orientation], dtype=np.float32), dtype=wp.vec4),
                )
            )
        except Exception:
            logger.debug("Could not set the planner's base transform", exc_info=True)

    # ── Planning ──────────────────────────────────────────────────────────────

    def joint_subset(self, values: Sequence[float], names: Sequence[str]) -> np.ndarray:
        """Pick this planner's joints out of a full articulation state, by name."""
        index = {name: i for i, name in enumerate(names)}
        missing = [n for n in self.planned_joints if n not in index]
        if missing:
            raise PlanningUnavailable(
                f"The articulation does not have the joints cuMotion plans for: "
                f"{', '.join(missing)}. Its joints are: {', '.join(names)}."
            )
        return np.asarray([values[index[n]] for n in self.planned_joints], dtype=float)

    def plan_to_pose(
        self,
        q_initial: Sequence[float],
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
    ) -> MotionPlan:
        """Plan a collision-free route to a world-frame pose.

        Raises `NoPathFound` rather than returning None: a planner that found
        nothing is a fact the caller has to handle, and returning a falsy value
        for it is how "the arm did not move" becomes a silent success.
        """
        self._ensure_world()
        q_initial = np.asarray(q_initial, dtype=float)
        position = np.asarray(position, dtype=float)

        if orientation is None:
            path = self._planner.plan_to_translation_target(
                q_initial=q_initial, translation_target=position
            )
        else:
            path = self._planner.plan_to_pose_target(
                q_initial=q_initial,
                position=position,
                orientation=np.asarray(orientation, dtype=float),
            )

        if path is None:
            raise NoPathFound(
                f"No collision-free route to {np.round(position, 4).tolist()}. "
                f"The target may be unreachable, inside an obstacle, or walled "
                f"off by one. Obstacles considered: "
                f"{', '.join(self._bound_obstacles) or '(none)'}."
            )

        waypoints = _to_numpy(path.get_waypoints())
        trajectory = self._generator.generate_trajectory_from_cspace_waypoints(waypoints)
        if trajectory is None:
            raise NoPathFound(
                "A route was found but could not be turned into a trajectory. "
                "This usually means the waypoints violate a joint or velocity "
                "limit."
            )
        return MotionPlan(trajectory, self.planned_joints, waypoints)
