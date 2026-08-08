"""
Path generation and following for wheeled bases.

`drive_to` is a turn-then-go loop: it pivots on the spot, drives, pivots again.
That is fine for one hop and poor for a route — it stops dead at every waypoint,
and a controller written on top of it overshoots the last one because nothing is
decelerating toward it. Isaac Sim already ships better, and this wraps it rather
than reinventing it a third time:

  * `quintic_polynomials_planner` turns a pair of poses into a smooth trajectory
    honouring acceleration and jerk limits, and reports the speed along it.
  * `stanley_control` tracks a reference path, steering by cross-track error
    rather than by aiming at the next point.

What neither of them does is decide the route. They take a path and make it
smooth and followable; they have no world and no obstacles. Getting past
something still means supplying waypoints that go around it — from a real
global planner, from the scene's geometry, or by hand. Saying otherwise would
repeat the mistake of assuming the tool with "planner" in its name plans.

Both live in `extsDeprecated` in Isaac Sim 6, the same status Lula has for arms,
so this is deliberately a thin adapter over a stable-but-legacy API.

There are two routes here and they do not compose into one pipeline — pick one:

    PathFollower    quintic -> path -> stanley_control -> DifferentialController
                    Continuous, and NOT YET WORKING. Two measured faults: the
                    quintic trajectory starts from rest, so its first samples
                    share a position and the heading derived from them is
                    atan2 of noise (yaw[1] came out at 2.53 rad where the leg
                    runs at 0.73); and `calc_target_index` picks the nearest
                    point, so the index stops advancing once the base leaves
                    the path. Prefer PoseDriver until both are handled.

    PoseDriver      goal pose -> WheelBasePoseController -> DifferentialController
                    Closed-loop point-to-point, turn-then-go internally, and it
                    stops at every goal it is handed. Right for a single hop.

`DifferentialController` is the floor of both, and that matters: it is the same
unicycle model `drive()` implements by hand, except it carries the speed limits.
Anything that ends in `drive()` instead is unlimited by accident.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("simliverse_sim.navigation")


class NavigationUnavailable(RuntimeError):
    """The wheeled path-following stack is not importable."""


def available() -> bool:
    try:
        import isaacsim.robot.wheeled_robots.controllers  # noqa: F401
    except Exception:
        logger.debug("wheeled_robots controllers unavailable", exc_info=True)
        return False
    return True


class PathPlan:
    """A smooth reference path with a speed profile.

    A value, like `MotionPlan` for arms: it holds no robot and drives nothing,
    so it can be planned once and followed by whoever owns the control loop.
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float],
                 yaws: Sequence[float], speeds: Sequence[float]) -> None:
        self.xs = [float(v) for v in xs]
        self.ys = [float(v) for v in ys]
        self.yaws = [float(v) for v in yaws]
        self.speeds = [float(v) for v in speeds]

    def __len__(self) -> int:
        return len(self.xs)

    def __repr__(self) -> str:
        return f"<PathPlan {len(self)} points, {self.length:.2f} m>"

    @property
    def length(self) -> float:
        if len(self) < 2:
            return 0.0
        dx = np.diff(np.asarray(self.xs))
        dy = np.diff(np.asarray(self.ys))
        return float(np.sum(np.hypot(dx, dy)))

    @property
    def goal(self) -> tuple[float, float]:
        return (self.xs[-1], self.ys[-1])


def plan_path(
    waypoints: Sequence[Sequence[float]],
    *,
    start_yaw: float,
    cruise: float = 0.4,
    max_accel: float = 0.6,
    max_jerk: float = 1.0,
    dt: float = 0.1,
) -> PathPlan:
    """Smooth a list of XY waypoints into a followable path.

    The waypoints are the route and are not checked against anything — see the
    module docstring. Heading at each waypoint is taken from the direction of
    travel into the next one, which is what makes the joins smooth rather than
    a sequence of pivots.
    """
    if not available():
        raise NavigationUnavailable(
            "isaacsim.robot.wheeled_robots is not importable, so there is no "
            "path smoothing or tracking. drive(linear, angular) still works, "
            "and drive_to() will turn-then-go one waypoint at a time."
        )
    from isaacsim.robot.wheeled_robots.controllers import quintic_polynomials_planner

    points = [np.asarray(w, dtype=float).reshape(-1)[:2] for w in waypoints]
    if len(points) < 2:
        raise ValueError("plan_path needs at least a start and a goal")

    # Heading into each leg; the last one keeps the previous heading.
    headings = []
    for i in range(len(points) - 1):
        delta = points[i + 1] - points[i]
        headings.append(math.atan2(float(delta[1]), float(delta[0])))
    headings.append(headings[-1])
    headings[0] = start_yaw

    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    speeds: list[float] = []
    for i in range(len(points) - 1):
        # Zero speed only at the ends, so the base flows through the corners
        # instead of stopping at each one.
        sv = 0.0 if i == 0 else cruise
        gv = 0.0 if i == len(points) - 2 else cruise
        _, rx, ry, ryaw, rv, _, _ = quintic_polynomials_planner(
            float(points[i][0]), float(points[i][1]), headings[i], sv, 0.0,
            float(points[i + 1][0]), float(points[i + 1][1]), headings[i + 1], gv, 0.0,
            max_accel, max_jerk, dt,
        )
        # The first sample of each leg repeats the previous leg's last.
        start = 1 if xs else 0
        xs.extend(rx[start:])
        ys.extend(ry[start:])
        yaws.extend(ryaw[start:])
        speeds.extend(rv[start:])

    plan = PathPlan(xs, ys, yaws, speeds)
    logger.info("Planned a %.2f m path through %d waypoints (%d samples)",
                plan.length, len(points), len(plan))
    return plan


class PoseDriver:
    """Drive toward one goal pose, a tick at a time.

    Wraps `WheelBasePoseController`, which is the extension's own closed-loop
    answer to "go here": it consumes an open-loop wheel model and returns the
    wheel commands that close the loop. Worth preferring over a hand-rolled
    turn-then-go for a single hop — it carries the speed limits that a hand
    rolled one does not, and `DifferentialController` is the same unicycle model
    `drive()` implements by hand.

    For a route rather than a hop, use `plan_path` and `PathFollower`: this one
    still stops at every goal it is given.
    """

    def __init__(self, robot: Any, *, max_linear: float = 0.5,
                 max_angular: float = 1.0, position_tol: float = 0.1,
                 heading_tol: float = 0.05) -> None:
        if not available():
            raise NavigationUnavailable(
                "isaacsim.robot.wheeled_robots is not importable; use "
                "drive(linear, angular) directly."
            )
        from isaacsim.robot.wheeled_robots.controllers import (
            DifferentialController,
            WheelBasePoseController,
        )

        self.robot = robot
        self.position_tol = float(position_tol)
        self.heading_tol = float(heading_tol)
        self._open_loop = DifferentialController(
            name="open_loop",
            wheel_radius=float(robot.wheel_radius),
            wheel_base=float(robot.wheel_base),
            max_linear_speed=float(max_linear),
            max_angular_speed=float(max_angular),
        )
        self._controller = WheelBasePoseController(
            name="pose", open_loop_wheel_controller=self._open_loop, is_holonomic=False
        )

    def step(self, goal: Sequence[float]) -> bool:
        """One tick toward `goal` (x, y). True once inside the position tolerance."""
        position = np.asarray(self.robot.base_position, dtype=float)
        target = np.asarray(goal, dtype=float).reshape(-1)[:2]
        if float(np.linalg.norm(position[:2] - target)) <= self.position_tol:
            self.robot.drive(0.0, 0.0)
            return True

        action = self._controller.forward(
            start_position=position[:2],
            start_orientation=np.asarray(self.robot.base_orientation, dtype=float),
            goal_position=target,
            heading_tol=self.heading_tol,
            position_tol=self.position_tol,
        )
        self.robot.apply_wheel_action(action)
        return False


class PathFollower:
    """Tracks a `PathPlan` with Stanley steering, one tick at a time.

    Deliberately the same contract as `Manipulator.follow`: call it every frame
    with the same plan, and it returns True when the goal is reached. It steps
    nothing and blocks nothing, so it belongs inside a controller's `compute`.
    """

    def __init__(self, robot: Any, plan: PathPlan, *, tolerance: float = 0.12,
                 max_linear: float = 0.6, max_angular: float = 1.2,
                 min_linear: float = 0.12) -> None:
        if not available():
            raise NavigationUnavailable(
                "isaacsim.robot.wheeled_robots is not importable; there is no "
                "path tracking. drive(linear, angular) still works."
            )
        from isaacsim.robot.wheeled_robots.controllers import DifferentialController

        self.robot = robot
        self.plan = plan
        self.tolerance = float(tolerance)
        self.max_angular = float(max_angular)
        # A floor under the commanded speed, because the profile starts at
        # zero and the tracker indexes by nearest point. Taking the minimum
        # of the two at a standstill commands zero, so the base does not
        # move, so the nearest point stays put, so the speed stays zero —
        # a deadlock that looks exactly like a controller doing nothing.
        self.min_linear = float(min_linear)
        # The same unicycle model `drive()` implements by hand, but carrying the
        # speed limits that the hand-rolled version has no idea about. Tracking
        # and kinematics are separate jobs: Stanley says which way to point,
        # this says what that means for the wheels.
        self._kinematics = DifferentialController(
            name="path_follower",
            wheel_radius=float(robot.wheel_radius),
            wheel_base=float(robot.wheel_base),
            max_linear_speed=float(max_linear),
            max_angular_speed=float(max_angular),
        )
        self._index = 0
        self._arrived = False

    def _state(self) -> Any:
        from isaacsim.robot.wheeled_robots.controllers.stanley_control import State

        position = np.asarray(self.robot.base_position, dtype=float)
        w, x, y, z = [float(v) for v in self.robot.base_orientation]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Speed from the wheels, which a wheeled base always has. This guarded on
        # a `base_velocity` attribute that does not exist, so it passed zero on
        # every tick — and Stanley's cross-track term is atan2(k * error, v),
        # which at v = 0 saturates to +-pi/2. The steering was pinned hard over
        # on the sign of the error instead of scaling with it.
        wheels = np.asarray(self.robot.joint_velocities, dtype=float)[
            self.robot.wheel_indices
        ]
        speed = float(self.robot.wheel_radius * float(np.mean(wheels)))
        # A floor keeps that same atan2 well-conditioned at a standstill.
        speed = max(speed, 0.05)
        # `State` carries the wheel base: Stanley steers about the front axle,
        # so the geometry is part of the state rather than of the gains.
        return State(
            wheel_base=float(self.robot.wheel_base),
            x=float(position[0]),
            y=float(position[1]),
            yaw=yaw,
            v=speed,
        )

    def distance_to_goal(self) -> float:
        position = np.asarray(self.robot.base_position, dtype=float)[:2]
        return float(np.linalg.norm(position - np.asarray(self.plan.goal)))

    def step(self) -> bool:
        """Advance one control tick. True once the goal is within tolerance."""
        from isaacsim.robot.wheeled_robots.controllers import calc_target_index, stanley_control

        if self._arrived:
            self.robot.drive(0.0, 0.0)
            return True

        if self.distance_to_goal() <= self.tolerance:
            self._arrived = True
            self.robot.drive(0.0, 0.0)
            return True

        state = self._state()
        if self._index == 0:
            self._index, _ = calc_target_index(state, self.plan.xs, self.plan.ys)
        delta, self._index = stanley_control(
            state, self.plan.xs, self.plan.ys, self.plan.yaws, self._index
        )

        profile = self.plan.speeds[min(self._index, len(self.plan) - 1)]
        # Floor first, then taper toward the goal. Order matters: tapering a
        # zero is still zero.
        speed = max(float(profile), self.min_linear)
        speed = float(min(speed, max(self.min_linear, 0.9 * self.distance_to_goal())))

        # Stanley returns a steering angle for a bicycle; a differential base
        # takes a yaw rate. The conversion is geometry, not a tuning constant —
        # this was a hand-picked gain until it was written down properly.
        wheel_base = float(self.robot.wheel_base)
        angular = float(speed * math.tan(float(delta)) / wheel_base) if wheel_base else 0.0
        angular = float(np.clip(angular, -self.max_angular, self.max_angular))

        self.robot.apply_wheel_action(self._kinematics.forward([speed, angular]))
        return False
