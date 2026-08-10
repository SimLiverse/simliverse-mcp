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

from ._compat import as_quat, as_vec3, get_stage

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
        self._reject_articulation()
        self._reject_wrapper()
        self._enable_contact_reporting()

    def _reject_wrapper(self) -> None:
        """Refuse to make a rigid body out of a prim that only contains one.

        Also destructive, and quietly so. A referenced asset puts its body on a
        child — `basic_block.usd` on `/Root/Cube` — so the prim a caller holds
        is an Xform wrapping the body, not the body. Wrapping *that* applies
        RigidBodyAPI to a prim with no collider beneath it, and PhysX then
        computes a negative mass:

            The rigid body at /World/Box_0 has a possibly invalid inertia
            tensor of {1.0, 1.0, 1.0} and a negative mass

        The blast radius is what makes this worth a refusal rather than a
        warning. One degenerate body stops dynamics for the *whole scene*:
        measured here, reading three props' positions froze a Franka scene so
        completely that an unrelated control cube dropped from 0.6 m never
        moved, while every read kept returning well-formed numbers. The
        measurement destroyed what it was measuring, and nothing said so.

        `scene.spawn_rigid` is unaffected — it puts body and collider on the one
        prim it creates, which is why cube-stacking always worked and props
        never did.
        """
        from pxr import Usd, UsdPhysics

        prim = self.prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            return

        bodies = [
            str(child.GetPath())
            for child in Usd.PrimRange(prim)
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not bodies:
            return

        # One body, no ambiguity: retarget to it silently. Refusing here was the
        # first fix and it is the worse one — it makes every caller remember
        # which spawn path they used and look up an inner path by hand, to avoid
        # a trap they cannot see. There is nothing to decide when a wrapper
        # contains exactly one body, so the library decides it.
        if len(bodies) == 1:
            logger.debug(
                "%s contains its rigid body at %s; measuring that instead",
                self.prim_path, bodies[0],
            )
            self.prim_path = bodies[0]
            return

        # Several bodies is a real choice, and guessing would put the caller
        # back where they started — holding a number from something they did not
        # pick.
        raise ValueError(
            f"{self.prim_path!r} is not a rigid body — it contains {len(bodies)}: "
            f"{', '.join(bodies[:4])}. Name the one you mean; wrapping the "
            f"container would apply RigidBodyAPI to a prim with no collider, "
            f"which PhysX reads as a negative mass, and one such body stops "
            f"dynamics for the entire scene."
        )

    def _reject_articulation(self) -> None:
        """Refuse to treat a robot as a rigid body.

        This is destructive, not merely wrong. `SingleRigidPrim` applies
        RigidBodyAPI to whatever path it is given, and an articulation root with
        a rigid body on it is not a valid PhysX object: its inertia goes
        invalid, its mass goes negative, and the robot is flung out of the
        world. A measured run passed a robot path in a list of objects and the
        arm ended up at z = -14140 m, with nothing in the traceback pointing
        here — the failure surfaced as "the robot broke when I pressed Play".
        """
        from pxr import UsdPhysics

        if not self.prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return
        raise ValueError(
            f"{self.prim_path!r} is an articulation root — a robot, not a rigid "
            f"body. Wrapping it as a RigidObject would apply RigidBodyAPI to it "
            f"and destroy the articulation, sending the robot out of the world.\n\n"
            f"Use Robot.attach({self.prim_path!r}) for the robot itself, or name a "
            f"single link (e.g. {self.prim_path}/panda_hand) if you really want one "
            f"body's state."
        )

    def __repr__(self) -> str:
        p = self.position
        return f"<RigidObject {self.prim_path} at [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]>"

    # ── Prim access ───────────────────────────────────────────────────────────

    @property
    def prim(self) -> Any:
        return get_stage().GetPrimAtPath(self.prim_path)

    def _rigid_view(self) -> Any:
        """The PhysX view of this body, or None while physics is not running.

        `SingleRigidPrim` reaches into the physics simulation view, which does
        not exist until the timeline plays. Built any earlier it raises

            AttributeError: 'NoneType' object has no attribute 'max_shapes'

        from somewhere inside isaacsim.core, naming neither this prim nor the
        timeline nor anything the caller wrote. That is the whole failure: the
        object is fine, the code is fine, and the only problem is that it was
        asked one line too soon — which is the *natural* order to write, since
        creating an object and then reading it back is how anyone checks their
        own work.

        So this returns None rather than raising, is cached only once it really
        worked, and is retried on the next call. Readers fall back to USD;
        writers say plainly what is wrong.
        """
        if self._view is None:
            try:
                from isaacsim.core.prims import SingleRigidPrim

                view = SingleRigidPrim(prim_path=self.prim_path)
            except Exception:
                logger.debug("No physics view for %s yet", self.prim_path, exc_info=True)
                return None
            try:
                view.initialize()
            except Exception:
                logger.debug("RigidPrim init deferred for %s", self.prim_path, exc_info=True)
            self._view = view
        return self._view

    def _require_view(self, action: str) -> Any:
        view = self._rigid_view()
        if view is None:
            raise RuntimeError(
                f"Cannot {action} on {self.prim_path!r}: PhysX has no view of it. "
                f"Rigid-body state only exists while the simulation is running — "
                f"call scene.play() first."
            )
        return view

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
        try:
            _, quat = self._rigid_view().get_world_pose()
            return np.asarray(quat, dtype=float)
        except Exception:
            from pxr import UsdGeom

            matrix = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(0)
            rotation = matrix.ExtractRotationQuat()
            imaginary = rotation.GetImaginary()
            return np.asarray(
                [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=float
            )

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
        """Mass in kg, as PhysX actually simulates it.

        The authored `UsdPhysics.MassAPI` attribute is only the override. A body
        created without one still has a real mass, derived from its volume and
        density — so reading the attribute alone reported 0.0 for every object
        that did not set it explicitly, which reads as "massless" rather than
        "not overridden". The simulation view is asked first and the attribute
        is the fallback.
        """
        view = self._rigid_view()
        # SingleRigidPrim exposes get_mass(); the batched RigidPrim view
        # exposes get_masses(). Which one is here depends on the backend.
        for method in ("get_mass", "get_masses") if view is not None else ():
            getter = getattr(view, method, None)
            if getter is None:
                continue
            try:
                value = float(np.asarray(getter()).reshape(-1)[0])
            except Exception:
                logger.debug("%s() failed on %s", method, self.prim_path, exc_info=True)
                continue
            if value > 0.0:
                return value

        from pxr import UsdPhysics

        attr = UsdPhysics.MassAPI(self.prim).GetMassAttr()
        return float(attr.Get()) if attr and attr.Get() is not None else 0.0

    def set_pose(self, position: Any = None, orientation: Any = None) -> None:
        """Teleport the body.

        Use this for *initial placement only*. Teleporting an object into a
        gripper is not a grasp — that shortcut is precisely why the previous
        skills never generalised (ADR 012 §1.2).
        """
        view = self._require_view("set a pose")
        pos = as_vec3(position, name="position") if position is not None else None
        view.set_world_pose(
            position=pos,
            orientation=as_quat(orientation) if orientation is not None else None,
        )

    def set_velocity(self, linear: Any = None, angular: Any = None) -> None:
        view = self._require_view("set a velocity")
        if linear is not None:
            view.set_linear_velocity(as_vec3(linear, name="linear"))
        if angular is not None:
            view.set_angular_velocity(as_vec3(angular, name="angular"))

    # ── Contacts ──────────────────────────────────────────────────────────────

    def contacts(self) -> list[dict[str, Any]]:
        """Current contacts on this body: [{"body": path, "force": float}, ...].

        `force` is in newtons, averaged over the last physics step.

        Three things about the PhysX contact report are easy to get wrong, and
        this function got all three wrong until a grasp that visibly worked kept
        reporting itself as a failure:

          * `header.actor0` is an **encoded int**, not a path string. Comparing
            it against `self.prim_path` therefore never matches, and every body
            in the scene reports zero contacts — silently, because an empty list
            is also the correct answer for a body that is genuinely untouched.
            `PhysicsSchemaTools.intToSdfPath` is what turns it back into a path.
          * There is no `total_normal_impulse` on the header. The impulses live
            in the *second* element of the returned tuple, in the slice
            `[contact_data_offset : contact_data_offset + num_contact_data]`.
          * `CONTACT_LOST` headers are published for the step on which a contact
            *ends*, so counting them reports contact with everything the body
            has recently let go of.
        """
        try:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools

            headers, points = get_physx_simulation_interface().get_contact_report()
        except Exception:
            logger.debug("Contact report unavailable", exc_info=True)
            return []

        dt = self._scene.dt if self._scene is not None else 1.0 / 60.0
        out: list[dict[str, Any]] = []
        for header in headers or []:
            if str(getattr(header, "type", "")).endswith("CONTACT_LOST"):
                continue
            try:
                actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            except Exception:
                logger.debug("Could not decode contact actors", exc_info=True)
                continue

            if self.prim_path == actor0:
                other = actor1
            elif self.prim_path == actor1:
                other = actor0
            else:
                continue

            start = int(header.contact_data_offset)
            impulse = 0.0
            for i in range(start, start + int(header.num_contact_data)):
                try:
                    impulse += float(np.linalg.norm(np.asarray(points[i].impulse, dtype=float)))
                except (IndexError, TypeError, ValueError):
                    break
            out.append({"body": other, "force": round(impulse / dt, 4)})
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
