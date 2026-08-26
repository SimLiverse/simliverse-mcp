# MIT License
#
# Copyright (c) 2026 SimLiverse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Record what a robot actually did, in the space a real controller speaks.

A recording of world poses is a picture of a run. A recording of *joint*
positions and velocities over time is a thing a controller can execute, which is
the difference between showing a customer a video and handing them something
their arm can run.

So this samples joint state, not prim transforms, and exports the shape every
vendor bridge already converts from -- ROS 2 `trajectory_msgs/JointTrajectory`:
joint names, and points of (positions, velocities, time_from_start).

It hangs off `Scene.step`, which is the only place every physics advance passes
through. That matters because an agent does not drive a robot one way: it
servos through RMPflow, follows a planned trajectory, and writes joint targets
directly, sometimes within a single task. A recorder that wrapped any one of
those would miss the others and produce a trajectory with holes in it.

What this is not
----------------
`violations()` checks the recorded trajectory against the limits the asset
declares, and a clean report means only that: no joint left its range and no
joint exceeded its declared maximum velocity *in simulation*. It is not a
safety certificate. Two things do not survive the trip to hardware on their own:

- **Contact.** Positions replayed open-loop reproduce free motion well and
  contact-rich motion badly. The instant a gripper closes on something, the
  result depends on friction, compliance and where the object really is, none
  of which the recording carries.
- **Timing.** This samples at the simulation's fixed dt. A real controller
  interpolates on its own clock, and a trajectory dense at 60 Hz may need
  resampling and re-timing before it is accepted.

Anyone putting one of these on real hardware validates it themselves, at
reduced speed, with an estop in reach. Saying so is part of the artifact:
`trajectory()` carries the caveats in its payload rather than in a README
nobody reads.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("simliverse_sim.recording")

SCHEMA = "simliverse.joint_trajectory/1"

CAVEATS = (
    "Recorded from simulation. Free motion replays open-loop; contact-rich "
    "phases (grasping, insertion, anything touching a surface) depend on "
    "friction, compliance and true object pose and do not transfer on their "
    "own. Sampled at the simulator's fixed timestep and may need re-timing for "
    "the target controller. Validate on hardware at reduced speed before "
    "trusting it."
)


class JointRecorder:
    """Samples a robot's joint state once per physics step.

    Used as a context manager it registers on entry and detaches on exit, so a
    recording cannot outlive the block that made it and quietly keep growing.
    """

    def __init__(self, robot: Any, *, scene: Any = None, every: int = 1, label: str = "") -> None:
        self.robot = robot
        self.scene = scene if scene is not None else robot.scene
        self.every = max(1, int(every))
        self.label = label or getattr(robot, "prim_path", "robot")

        self.joint_names: list[str] = list(robot.joint_names)
        self.times: list[float] = []
        self.positions: list[list[float]] = []
        self.velocities: list[list[float]] = []
        self._seen = 0
        self._t0: float | None = None
        self._attached = False

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "JointRecorder":
        if not self._attached:
            self.scene.add_step_listener(self._on_step)
            self._attached = True
        return self

    def stop(self) -> "JointRecorder":
        if self._attached:
            self.scene.remove_step_listener(self._on_step)
            self._attached = False
        return self

    def __enter__(self) -> "JointRecorder":
        return self.start()

    def __exit__(self, *_exc: Any) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        return (
            f"<JointRecorder {self.label}: {len(self.times)} points, "
            f"{self.duration:.2f}s, {len(self.joint_names)} joints>"
        )

    # -- sampling --------------------------------------------------------
    def _on_step(self, sim_time: float) -> None:
        self._seen += 1
        if self._seen % self.every:
            return
        if self._t0 is None:
            self._t0 = float(sim_time)

        self.times.append(float(sim_time) - self._t0)
        self.positions.append([float(v) for v in self.robot.joint_positions])
        try:
            self.velocities.append([float(v) for v in self.robot.joint_velocities])
        except Exception:  # noqa: BLE001 - velocity is optional, position is not
            self.velocities.append([0.0] * len(self.joint_names))

    @property
    def duration(self) -> float:
        return self.times[-1] if self.times else 0.0

    # -- what comes out --------------------------------------------------
    def trajectory(self) -> dict[str, Any]:
        """The recording as a `trajectory_msgs/JointTrajectory`-shaped payload.

        Field names follow the ROS 2 message so a bridge is a rename, not a
        reinterpretation. The extras -- `schema`, `caveats`, `violations` --
        travel with it on purpose: a trajectory that arrives without the
        conditions it was recorded under is the thing that gets someone hurt.
        """
        return {
            "schema": SCHEMA,
            "label": self.label,
            "joint_names": list(self.joint_names),
            "points": [
                {
                    "positions": positions,
                    "velocities": velocities,
                    "time_from_start": round(t, 6),
                }
                for t, positions, velocities in zip(
                    self.times, self.positions, self.velocities
                )
            ],
            "duration": round(self.duration, 6),
            "sample_hz": round(len(self.times) / self.duration, 3) if self.duration else 0.0,
            "violations": self.violations(),
            "caveats": CAVEATS,
        }

    def violations(self, *, position_tolerance: float = 1e-3) -> list[str]:
        """Where the recording leaves what the asset says the robot can do.

        Worth having only because `joint_limits` reports real numbers now. It
        returned `(None, None)` for every joint of every robot until recently,
        which would have made this a function that always said "looks fine".
        """
        if not self.positions:
            return []

        found: list[str] = []
        limits = list(getattr(self.robot, "joint_limits", []) or [])
        maxima = self._declared_max_velocities()
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)

        for index, name in enumerate(self.joint_names):
            if index < len(limits) and limits[index] and limits[index][0] is not None:
                low, high = limits[index]
                column = positions[:, index]
                if column.min() < low - position_tolerance:
                    found.append(
                        f"{name}: reaches {column.min():.4f}, below its lower limit {low:.4f}"
                    )
                if column.max() > high + position_tolerance:
                    found.append(
                        f"{name}: reaches {column.max():.4f}, above its upper limit {high:.4f}"
                    )
            ceiling = maxima.get(name)
            if ceiling:
                fastest = float(np.abs(velocities[:, index]).max())
                if fastest > ceiling:
                    found.append(
                        f"{name}: peaks at {fastest:.4f} rad/s, above its declared "
                        f"maximum {ceiling:.4f}"
                    )
        return found

    def _declared_max_velocities(self) -> dict[str, float]:
        try:
            properties = self.robot._articulation.dof_properties
            return {
                name: float(value)
                for name, value in zip(self.joint_names, properties["maxVelocity"])
                if float(value) > 0.0
            }
        except Exception:  # noqa: BLE001 - absent on some assets, not an error
            logger.debug("No declared joint velocity maxima for %s", self.label, exc_info=True)
            return {}

    def save(self, path: str) -> str:
        """Write the trajectory as JSON and return the path."""
        payload = self.trajectory()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        logger.info(
            "Wrote %d points (%.2fs) to %s%s",
            len(payload["points"]),
            payload["duration"],
            path,
            f" with {len(payload['violations'])} limit violations"
            if payload["violations"]
            else "",
        )
        return path


def replay(robot: Any, trajectory: dict[str, Any], *, scene: Any = None, speed: float = 1.0) -> None:
    """Drive `robot` back through a recorded trajectory, in joint space.

    Position replay, deliberately: it reproduces the commanded path, and where
    the run depended on contact the result will differ from the original. That
    difference is information -- it is the same gap the trajectory will meet on
    hardware -- so it is not smoothed over here.
    """
    scene = scene if scene is not None else robot.scene
    names = list(trajectory["joint_names"])
    indices = [robot.joint_names.index(name) for name in names]
    points = trajectory["points"]
    if not points:
        return

    scene.play()
    previous = 0.0
    for point in points:
        target = float(point["time_from_start"])
        steps = max(1, int(round((target - previous) / (scene.dt * max(speed, 1e-6)))))
        robot.set_joint_positions(point["positions"], indices=indices, settle_steps=steps)
        previous = target
