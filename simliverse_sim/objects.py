"""
Rigid bodies: pose, velocity, and real contact state.

Contact reporting is the piece the MCP verb layer never had — `get_physics_state`
returns a hardcoded empty contact list (ADR 012 §1.3), which makes it impossible
to prove a grasp actually holds. Here contacts come from the PhysX contact-report
API, so `is_grasped_by` is a measurement rather than a guess.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import as_vec3, get_stage

if TYPE_CHECKING:
    from .scene import Scene

logger = logging.getLogger("simliverse_sim.objects")


class RigidObject:
    """A dynamic rigid body on the stage."""

    def __init__(self, prim_path: str, scene: "Scene | None" = None) -> None:
        self.prim_path = prim_path
        self._scene = scene
        self._view: Any = None
        if not self.prim.IsValid():
            raise ValueError(f"No prim at {prim_path!r}")
        self._enable_contact_reporting()

    def __repr__(self) -> str:
        p = self.position
        return f"<RigidObject {self.prim_path} at [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]>"

    # ── Prim access ───────────────────────────────────────────────────────────

    @property
    def prim(self) -> Any:
        return get_stage().GetPrimAtPath(self.prim_path)

    def _rigid_view(self) -> Any:
        if self._view is None:
            from isaacsim.core.prims import SingleRigidPrim

            self._view = SingleRigidPrim(prim_path=self.prim_path)
            try:
                self._view.initialize()
            except Exception:
                logger.debug("RigidPrim init deferred for %s", self.prim_path, exc_info=True)
        return self._view

    def _enable_contact_reporting(self, threshold: float = 0.0) -> None:
        """Opt this body into PhysX contact reports.

        Without this API applied, PhysX does not publish contacts for the prim
        and every contact query comes back empty — which is exactly the failure
        the old hardcoded `"contacts": []` masked.
        """
        try:
            from pxr import PhysxSchema

            api = PhysxSchema.PhysxContactReportAPI.Apply(self.prim)
            api.CreateThresholdAttr().Set(float(threshold))
        except Exception:
            logger.debug("Could not enable contact reporting on %s", self.prim_path, exc_info=True)

    # ── Pose and velocity ─────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """World-space position.

        World-space, not local — the MCP `get_prim_info` verb documents
        world-space and returns the local translation (ADR 012 §1.5).
        """
        try:
            pos, _ = self._rigid_view().get_world_pose()
            return np.asarray(pos, dtype=float)
        except Exception:
            from pxr import UsdGeom

            xform = UsdGeom.Xformable(self.prim)
            matrix = xform.ComputeLocalToWorldTransform(0)
            return np.asarray(matrix.ExtractTranslation(), dtype=float)

    @property
    def orientation(self) -> np.ndarray:
        """World-space orientation as a (w, x, y, z) quaternion."""
        _, quat = self._rigid_view().get_world_pose()
        return np.asarray(quat, dtype=float)

    @property
    def linear_velocity(self) -> np.ndarray:
        try:
            return np.asarray(self._rigid_view().get_linear_velocity(), dtype=float)
        except Exception:
            return np.zeros(3)

    @property
    def angular_velocity(self) -> np.ndarray:
        try:
            return np.asarray(self._rigid_view().get_angular_velocity(), dtype=float)
        except Exception:
            return np.zeros(3)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.linear_velocity))

    @property
    def mass(self) -> float:
        from pxr import UsdPhysics

        api = UsdPhysics.MassAPI(self.prim)
        attr = api.GetMassAttr()
        return float(attr.Get()) if attr and attr.Get() is not None else 0.0

    def set_pose(self, position: Any = None, orientation: Any = None) -> None:
        """Teleport the body.

        Use this for *initial placement only*. Teleporting an object into a
        gripper is not a grasp — that shortcut is precisely why the previous
        skills never generalised (ADR 012 §1.2).
        """
        view = self._rigid_view()
        pos = as_vec3(position, name="position") if position is not None else None
        view.set_world_pose(position=pos, orientation=orientation)

    def set_velocity(self, linear: Any = None, angular: Any = None) -> None:
        view = self._rigid_view()
        if linear is not None:
            view.set_linear_velocity(as_vec3(linear, name="linear"))
        if angular is not None:
            view.set_angular_velocity(as_vec3(angular, name="angular"))

    # ── Contacts ──────────────────────────────────────────────────────────────

    def contacts(self) -> list[dict[str, Any]]:
        """Current contacts on this body: [{"body": path, "force": float}, ...]."""
        try:
            from omni.physx import get_physx_simulation_interface

            raw = get_physx_simulation_interface().get_contact_report()
        except Exception:
            logger.debug("Contact report unavailable", exc_info=True)
            return []

        out: list[dict[str, Any]] = []
        headers = raw[0] if isinstance(raw, tuple) and raw else raw
        for header in headers or []:
            actor0 = str(getattr(header, "actor0", ""))
            actor1 = str(getattr(header, "actor1", ""))
            if self.prim_path not in (actor0, actor1):
                continue
            other = actor1 if actor0 == self.prim_path else actor0
            impulse = float(getattr(header, "total_normal_impulse", 0.0) or 0.0)
            out.append({"body": other, "force": impulse})
        return out

    def contact_bodies(self) -> set[str]:
        return {c["body"] for c in self.contacts()}

    def total_contact_force(self) -> float:
        return float(sum(c["force"] for c in self.contacts()))

    # ── State summary ─────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        """Everything a verifier needs about this object, in one call."""
        return {
            "prim_path": self.prim_path,
            "position": self.position.round(4).tolist(),
            "linear_velocity": self.linear_velocity.round(4).tolist(),
            "angular_velocity": self.angular_velocity.round(4).tolist(),
            "speed": round(self.speed, 4),
            "mass": self.mass,
            "contacts": self.contacts(),
        }
