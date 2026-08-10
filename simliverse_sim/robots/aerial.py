"""
Aerial robots: quadcopters and other multirotors.

A multirotor in Isaac Sim is usually not an articulation you position-control —
it is a rigid body you apply thrust and torque to. So the control surface here is
force-based, with a blocking `fly_to` that closes the loop on the body pose.

As with legged locomotion, there is no fallback that fakes flight by teleporting
the body. A drone that "flew" because someone set its transform has not
demonstrated anything.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._compat import as_vec3, get_stage
from .base import Morphology, Robot

logger = logging.getLogger("simliverse_sim.robots.aerial")


class FlightError(RuntimeError):
    pass


class AerialRobot(Robot):
    """A multirotor controlled by body-frame thrust and torque.

    Uses a simple PD position-and-attitude controller. It is adequate for
    "get to this waypoint and hover", not for aggressive or aerobatic flight.
    """

    morphology = Morphology.AERIAL

    def __init__(
        self,
        prim_path: str,
        *,
        scene: Any = None,
        mass: float | None = None,
        body_path: str | None = None,
    ) -> None:
        try:
            super().__init__(prim_path, scene=scene)
            self._articulated = True
        except Exception:
            # Many drone assets are a single rigid body with no articulation.
            from ..scene import Scene as _Scene

            self.prim_path = prim_path
            self.scene = scene or _Scene.get()
            self.groups = None  # type: ignore[assignment]
            self._articulated = False
            self.scene.play()

        self.body_path = body_path or prim_path
        self._body: Any = None
        self._mass = mass

    # ── Body access ───────────────────────────────────────────────────────────

    def _rigid_body(self) -> Any:
        if self._body is None:
            from isaacsim.core.prims import SingleRigidPrim

            self._body = SingleRigidPrim(prim_path=self.body_path)
            try:
                self._body.initialize()
            except Exception:
                logger.debug("Rigid body init deferred for %s", self.body_path, exc_info=True)
        return self._body

    @property
    def position(self) -> np.ndarray:
        pos, _ = self._rigid_body().get_world_pose()
        return np.asarray(pos, dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self._rigid_body().get_linear_velocity(), dtype=float)

    @property
    def mass(self) -> float:
        if self._mass is None:
            from pxr import UsdPhysics

            api = UsdPhysics.MassAPI(get_stage().GetPrimAtPath(self.body_path))
            attr = api.GetMassAttr()
            self._mass = float(attr.Get()) if attr and attr.Get() else 1.0
        return self._mass

    def altitude(self) -> float:
        return float(self.position[2])

    # ── Force control ─────────────────────────────────────────────────────────

    def apply_thrust(self, force: Any, *, torque: Any = None) -> None:
        """Apply a world-frame force (and optional torque) to the body for one step."""
        body = self._rigid_body()
        vector = as_vec3(force, name="force")
        try:
            body.apply_forces_and_torques_at_pos(
                forces=vector.reshape(1, 3),
                torques=as_vec3(torque, name="torque").reshape(1, 3)
                if torque is not None
                else None,
                is_global=True,
            )
        except Exception as exc:
            raise FlightError(
                f"Could not apply force to {self.body_path}: {exc}. The prim must "
                f"have RigidBodyAPI applied and physics must be running."
            ) from exc

    def hover(self, *, steps: int = 60) -> bool:
        """Hold altitude against gravity. Returns whether it stayed roughly put."""
        start = self.position.copy()
        target = start.copy()
        for _ in range(steps):
            self._position_control_step(target)
            self.scene.step(1)
        drift = float(np.linalg.norm(self.position - start))
        return drift < 0.5

    def fly_to(
        self,
        position: Any,
        *,
        tolerance: float = 0.25,
        max_steps: int = 3000,
        raise_on_fail: bool = True,
    ) -> bool:
        """Fly to a world-space position and hold there. Blocks until arrival."""
        target = as_vec3(position, name="position")
        self.scene.play()

        settled = 0
        for _ in range(max_steps):
            self._position_control_step(target)
            self.scene.step(1)
            if float(np.linalg.norm(self.position - target)) < tolerance:
                settled += 1
                if settled >= 20:
                    return True
            else:
                settled = 0

        if raise_on_fail:
            error = float(np.linalg.norm(self.position - target))
            raise FlightError(
                f"Did not reach {target.round(3).tolist()} within {max_steps} steps "
                f"(final error {error:.3f} m). The controller may be under-powered "
                f"for this mass, or the drone may be obstructed."
            )
        return False

    def _position_control_step(
        self, target: np.ndarray, *, kp: float = 8.0, kd: float = 5.0
    ) -> None:
        """One PD step: gravity compensation plus position and damping terms."""
        gravity = 9.81
        error = target - self.position
        command = kp * error - kd * self.velocity
        # Feed-forward the weight so the drone holds altitude at zero error.
        command[2] += gravity
        self.apply_thrust(command * self.mass)

    def describe(self) -> dict[str, Any]:
        if self._articulated:
            info = super().describe()
        else:
            info = {
                "prim_path": self.prim_path,
                "morphology": self.morphology.value,
                "controller": type(self).__name__,
                "note": "Single rigid body — no articulation, controlled by thrust.",
            }
        info["flight"] = {
            "body_path": self.body_path,
            "mass": round(self.mass, 4),
            "position": self.position.round(4).tolist(),
            "velocity": self.velocity.round(4).tolist(),
            "altitude": round(self.altitude(), 4),
        }
        return info
