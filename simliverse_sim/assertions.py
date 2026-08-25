"""
Verification primitives.

An agent that cannot observe the outcome of an action cannot self-correct, which
caps its capability no matter how good the prompt is (ADR 012 §1.3). These are
the checks a verifier runs to turn "no exception was raised" into "the task
demonstrably happened".

Every check returns a `Check` rather than raising, so a verifier can report all
failures at once instead of stopping at the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .objects import RigidObject
    from .robots import Robot
    from .scene import Scene


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "measured": c.measured,
                }
                for c in self.checks
            ],
        }

    def __str__(self) -> str:
        header = "ALL CHECKS PASSED" if self.passed else f"{len(self.failures)} CHECK(S) FAILED"
        return "\n".join([header, *(str(c) for c in self.checks)])


# ── Individual checks ─────────────────────────────────────────────────────────


def grasped(
    robot: "Robot",
    obj: "RigidObject",
    *,
    hold_steps: int = 90,
    max_drift: float = 0.05,
) -> Check:
    """The object is held, and stays held, while gravity acts on it.

    Holding for a while is the part that matters. A gripper that has merely
    touched an object passes an instantaneous contact check and then drops it
    one step later.
    """
    start = obj.position.copy()
    if not robot.is_grasping(obj):
        return Check(
            "grasped",
            False,
            "No contact between the object and any robot link at the moment of check.",
            {"contacts": obj.contacts()},
        )

    robot.scene.step(hold_steps)

    still = robot.is_grasping(obj)
    drift = float(np.linalg.norm(obj.position - start))
    if not still:
        return Check(
            "grasped",
            False,
            f"The object was released during the {hold_steps}-step hold — the grip slipped.",
            {"drift": round(drift, 4), "position": obj.position.round(4).tolist()},
        )
    if drift > max_drift:
        return Check(
            "grasped",
            False,
            f"The object moved {drift:.3f} m while supposedly held (limit {max_drift} m) — "
            f"it is sliding through the fingers.",
            {"drift": round(drift, 4)},
        )
    return Check(
        "grasped",
        True,
        f"Held for {hold_steps} steps with {drift:.4f} m drift.",
        {
            "drift": round(drift, 4),
            "contact_force": round(obj.total_contact_force(), 4),
            "contacts": obj.contacts(),
        },
    )


def not_teleported(
    obj: "RigidObject",
    previous_position: Any,
    *,
    max_jump: float = 0.30,
) -> Check:
    """The object moved continuously rather than being snapped into place.

    This exists because the skill this library replaces achieved its "grasp" by
    teleporting the ball into the tool centre point (ADR 012 §1.2). Any pipeline
    that regresses to that shortcut should fail loudly.
    """
    jump = float(np.linalg.norm(obj.position - np.asarray(previous_position, dtype=float)))
    return Check(
        "not_teleported",
        jump <= max_jump,
        (
            f"Object moved {jump:.3f} m since the previous observation."
            if jump <= max_jump
            else f"Object jumped {jump:.3f} m in one observation window (limit {max_jump} m) — "
            f"this looks like a teleport, not physical motion."
        ),
        {"displacement": round(jump, 4)},
    )


def airborne(obj: "RigidObject", *, min_speed: float = 0.5) -> Check:
    """The object is in free flight — moving, and touching nothing."""
    contacts = obj.contacts()
    speed = obj.speed
    ok = not contacts and speed >= min_speed
    return Check(
        "airborne",
        ok,
        (
            f"In free flight at {speed:.2f} m/s."
            if ok
            else f"Not airborne: speed {speed:.2f} m/s (need ≥ {min_speed}), "
            f"{len(contacts)} contact(s)."
        ),
        {"speed": round(speed, 3), "contacts": contacts},
    )


def upright(robot: "Robot", *, max_tilt_deg: float = 45.0, min_height: float | None = None) -> Check:
    """A legged robot is standing rather than lying on its side.

    Without this, "the quadruped walked" is indistinguishable from "the quadruped
    fell over and slid", since both change the base position.
    """
    tilt = getattr(robot, "tilt_degrees", None)
    if tilt is None:
        return Check("upright", False, f"{type(robot).__name__} does not report tilt.")

    angle = tilt()
    height = robot.base_position[2]
    ok = angle <= max_tilt_deg and (min_height is None or height >= min_height)
    return Check(
        "upright",
        ok,
        f"Base tilted {angle:.1f}° at {height:.3f} m."
        + ("" if ok else f" Limit is {max_tilt_deg}°"
           + (f" and {min_height} m." if min_height is not None else ".")),
        {"tilt_degrees": round(angle, 2), "base_height": round(float(height), 4)},
    )


def reached_position(
    robot: "Robot", target: Any, *, tolerance: float = 0.25, planar: bool = True
) -> Check:
    """A mobile robot actually arrived where it was sent."""
    goal = np.asarray(target, dtype=float).reshape(-1)
    current = robot.base_position
    if planar:
        goal, current = goal[:2], current[:2]
    error = float(np.linalg.norm(current - goal[: current.size]))
    return Check(
        "reached_position",
        error <= tolerance,
        f"Base is {error:.3f} m from the target (tolerance {tolerance} m).",
        {"error": round(error, 4), "position": robot.base_position.round(4).tolist()},
    )


def moved_under_own_power(
    robot: "Robot", start_position: Any, *, min_distance: float = 0.1
) -> Check:
    """The robot travelled, and its joints actually moved to do it.

    Guards against a base whose transform was set directly — displacement with
    motionless joints is teleportation, not locomotion.
    """
    displacement = float(
        np.linalg.norm(robot.base_position[:2] - np.asarray(start_position, dtype=float)[:2])
    )
    joint_motion = float(np.max(np.abs(robot.joint_velocities))) if robot.dof else 0.0

    if displacement < min_distance:
        return Check(
            "moved_under_own_power",
            False,
            f"Base moved only {displacement:.3f} m (needed {min_distance} m).",
            {"displacement": round(displacement, 4)},
        )
    if joint_motion < 1e-3:
        return Check(
            "moved_under_own_power",
            False,
            f"Base moved {displacement:.3f} m but no joint is moving — this looks "
            f"like the base was repositioned directly rather than driven.",
            {"displacement": round(displacement, 4), "max_joint_velocity": joint_motion},
        )
    return Check(
        "moved_under_own_power",
        True,
        f"Travelled {displacement:.3f} m with joints in motion.",
        {"displacement": round(displacement, 4), "max_joint_velocity": round(joint_motion, 4)},
    )


def reached_height(obj: "RigidObject", apex: float, *, minimum: float) -> Check:
    return Check(
        "reached_height",
        apex >= minimum,
        f"Apex {apex:.3f} m against a {minimum:.3f} m minimum.",
        {"apex": round(apex, 4), "minimum": minimum},
    )


def travelled(distance: float, *, minimum: float) -> Check:
    return Check(
        "travelled",
        distance >= minimum,
        f"Horizontal distance {distance:.3f} m against a {minimum:.3f} m minimum.",
        {"distance": round(distance, 4), "minimum": minimum},
    )


def physics_running(scene: "Scene") -> Check:
    """Physics is actually advancing.

    A stopped timeline makes every subsequent observation meaningless — objects
    do not fall, contacts never form, and a task can look "successful" purely
    because nothing moved.
    """
    if not scene.is_playing():
        return Check("physics_running", False, "The timeline is stopped — nothing is simulating.")
    marker = _drop_probe(scene)
    return Check(
        "physics_running",
        marker > 1e-4,
        f"Probe body fell {marker:.5f} m over 10 steps."
        if marker > 1e-4
        else "A free body did not fall over 10 steps — physics is not advancing.",
        {"probe_fall": round(marker, 6)},
    )


# Deliberately outside `/World`. Everything the agent lists, searches or clears
# is scoped to `/World`, so a probe living here is invisible to the scene the
# task is about and cannot be mistaken for part of it.
_PROBE_PATH = "/PhysicsProbe"
_PROBE_HEIGHT = 50.0


def _drop_probe(scene: "Scene") -> float:
    """Drop a body from a known height and measure how far it falls.

    The probe is created once and then reused forever. It is never removed, and
    that is the entire point of this function's shape.

    Removing it is what the first version did, and `RemovePrim` on a body PhysX
    has registered tears down the physics tensor view. Every articulation in the
    scene is de-registered with it: joint drives stop being serviced, a closed
    gripper relaxes, and whatever the robot was holding falls on the floor.

    So `verify_grasp` — which calls `physics_running` first — destroyed the grasp
    it had been asked to verify, then truthfully reported the object was on the
    ground. A verifier that breaks the thing it measures is worse than no
    verifier, because its answer looks like evidence.

    Reusing one body also costs less than spawning per call, and leaves the
    scene's prim count stable across verifications — which matters when the
    thing under test is "did the agent build the right scene".
    """
    probe = _existing_probe(scene)
    if probe is None:
        probe = scene.spawn_rigid(
            _PROBE_PATH,
            shape="Sphere",
            radius=0.01,
            position=(0.0, 0.0, _PROBE_HEIGHT),
            mass=0.01,
        )
    else:
        # Lift it back up rather than letting it accumulate falls. Writing a pose
        # to a body nothing else touches is safe; removing one is not.
        probe.set_pose(position=(0.0, 0.0, _PROBE_HEIGHT))
        # And drop it from rest. A reused body keeps the speed it reached last
        # time, which would make each successive probe report a longer fall than
        # the one before and turn a fixed threshold into a moving one.
        probe.set_velocity(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))

    start = probe.position[2]
    scene.step(10)
    return float(start - probe.position[2])


def _existing_probe(scene: "Scene") -> Any:
    """The probe from a previous call, if it is still on the stage."""
    from .objects import RigidObject

    try:
        if not scene.stage.GetPrimAtPath(_PROBE_PATH).IsValid():
            return None
        return RigidObject(_PROBE_PATH)
    except Exception:
        return None


# ── Composite suites ──────────────────────────────────────────────────────────


def verify_grasp(robot: "Robot", obj: "RigidObject", previous_position: Any = None) -> Report:
    report = Report()
    report.add(physics_running(robot.scene))
    if previous_position is not None:
        report.add(not_teleported(obj, previous_position))
    report.add(grasped(robot, obj))
    return report


def verify_navigation(
    robot: "Robot",
    target: Any,
    start_position: Any,
    *,
    tolerance: float = 0.25,
    require_upright: bool = False,
) -> Report:
    """Confirm a mobile or legged robot actually drove/walked to a target."""
    report = Report()
    report.add(physics_running(robot.scene))
    report.add(moved_under_own_power(robot, start_position))
    report.add(reached_position(robot, target, tolerance=tolerance))
    if require_upright:
        report.add(upright(robot))
    return report


def verify_throw(
    obj: "RigidObject",
    throw_result: dict[str, Any],
    *,
    min_apex: float = 0.0,
    min_distance: float = 0.20,
) -> Report:
    report = Report()
    report.add(
        Check(
            "released",
            not throw_result.get("still_held", True),
            "The gripper opened and the object left the hand."
            if not throw_result.get("still_held", True)
            else "The object is still in the gripper — it was never released.",
            {"still_held": throw_result.get("still_held")},
        )
    )
    report.add(reached_height(obj, throw_result.get("apex_height", 0.0), minimum=min_apex))
    report.add(travelled(throw_result.get("horizontal_distance", 0.0), minimum=min_distance))
    return report
