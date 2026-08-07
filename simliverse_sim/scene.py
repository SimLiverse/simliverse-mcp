"""
Scene and physics setup.

Everything here is blocking and steps physics synchronously. Control code reads
top to bottom with no per-tick state machine — which is what makes it writable
by an agent, and reviewable by a human.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import as_vec3, get_physx, get_stage, get_timeline, get_world, update_app

logger = logging.getLogger("simliverse_sim.scene")

DEFAULT_DT = 1.0 / 60.0


@dataclass
class PhysicsConfig:
    gravity: float = -9.81
    dt: float = DEFAULT_DT
    # PhysX defaults are tuned for speed, not for stable finger-on-object
    # contact. Grasping needs more solver iterations than the default 4/1.
    solver_position_iterations: int = 32
    solver_velocity_iterations: int = 4
    gpu_dynamics: bool = False


class Scene:
    """The live stage: physics setup, timeline control, and stepping."""

    def __init__(self, dt: float = DEFAULT_DT) -> None:
        self._dt = dt
        self._world = get_world(physics_dt=dt)
        # PhysX wants a monotonically increasing simulation clock; it is ours
        # to keep now that we step physics directly rather than via World.
        self._sim_time = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, dt: float = DEFAULT_DT) -> "Scene":
        return cls(dt=dt)

    @property
    def world(self) -> Any:
        return self._world

    @property
    def stage(self) -> Any:
        return get_stage()

    def configure_physics(self, config: PhysicsConfig | None = None) -> PhysicsConfig:
        """Apply gravity, timestep and solver settings, and create a ground plane.

        The MCP `set_physics_params` verb silently discards every one of these
        (ADR 012 §1.5) and returns success regardless. This actually applies them.
        """
        from pxr import PhysxSchema, UsdGeom, UsdPhysics

        cfg = config or PhysicsConfig()
        stage = self.stage

        scene_path = "/World/PhysicsScene"
        prim = stage.GetPrimAtPath(scene_path)
        if not prim.IsValid():
            physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        else:
            physics_scene = UsdPhysics.Scene(prim)

        physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(abs(cfg.gravity))

        physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
        physx.CreateSolverTypeAttr().Set("TGS")
        physx.CreateEnableGPUDynamicsAttr().Set(cfg.gpu_dynamics)
        physx.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / cfg.dt)))

        self._dt = cfg.dt
        try:
            self._world.get_physics_context().set_physics_dt(cfg.dt)
        except Exception:
            logger.debug("Could not set physics dt on the World context", exc_info=True)

        self.ensure_ground_plane()
        self._solver_defaults = (cfg.solver_position_iterations, cfg.solver_velocity_iterations)
        return cfg

    def ensure_ground_plane(self, path: str = "/World/GroundPlane", z: float = 0.0) -> str:
        from pxr import UsdGeom, UsdPhysics

        stage = self.stage
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            plane = UsdGeom.Plane.Define(stage, path)
            plane.CreateAxisAttr().Set("Z")
            plane.AddTranslateOp().Set((0.0, 0.0, z))
            prim = plane.GetPrim()
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        return path

    # ── Timeline and stepping ─────────────────────────────────────────────────

    def play(self) -> None:
        timeline = get_timeline()
        if not timeline.is_playing():
            timeline.play()

    def pause(self) -> None:
        get_timeline().pause()

    def stop(self) -> None:
        get_timeline().stop()
        self._sim_time = 0.0

    def is_playing(self) -> bool:
        return bool(get_timeline().is_playing())

    def step(self, count: int = 1, *, render: bool = False) -> None:
        """Advance physics by `count` steps.

        Stepped through `omni.physx` rather than `World.step()`. Two reasons,
        both learned the hard way:

        `World.step()` is documented as "not intended to be used in the Isaac
        Sim Extensions workflow", and we are an extension — Kit owns the render
        loop. Its `render=False` branch dereferences
        `SimulationContext._physics_context`, which is `None` on Isaac Sim 6.0
        even for a freshly constructed `World`, so it raises
        `'NoneType' object has no attribute '_step'`. That single call produced
        eight failures across all three agent tiers in one grasp run, and the
        error names nothing that appears in this file.

        The MCP `step_simulation` verb is no substitute: it calls `update_app()`,
        which advances rendering and moves physics only as a side effect of the
        timeline already running (ADR 012 §1.5). This advances physics itself, by
        a known dt, whether or not anything is rendering.
        """
        physx = get_physx()
        for _ in range(max(0, int(count))):
            physx.update_simulation(self._dt, self._sim_time)
            self._sim_time += self._dt
        # Push the results into USD/Fabric so reads afterwards see the new poses
        # rather than the ones from before the step.
        physx.update_transformations(False, True, True, False)
        if render:
            update_app()

    def settle(self, seconds: float = 0.5, *, render: bool = False) -> None:
        """Step long enough for contacts and drives to reach steady state."""
        self.step(int(round(seconds / self._dt)), render=render)

    @property
    def dt(self) -> float:
        return self._dt

    # ── Authoring ─────────────────────────────────────────────────────────────

    def spawn_rigid(
        self,
        prim_path: str,
        *,
        shape: str = "Sphere",
        position: Any = (0.0, 0.0, 0.5),
        scale: Any = None,
        radius: float | None = None,
        size: float | None = None,
        mass: float = 0.1,
        color: Any = (0.9, 0.2, 0.15),
        friction: float = 0.9,
        restitution: float = 0.05,
    ) -> "RigidObject":
        """Create a dynamic rigid body with real, tunable contact properties.

        `friction` matters: the MCP verb layer hardcodes 0.5/0.5/0.0 and does not
        expose it at all, which makes a friction grasp untunable. A rubber-ish
        0.9 is what actually holds a ball between two finger pads.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        from .objects import RigidObject

        stage = self.stage
        shape_cls = {
            "Sphere": UsdGeom.Sphere,
            "Cube": UsdGeom.Cube,
            "Cylinder": UsdGeom.Cylinder,
            "Capsule": UsdGeom.Capsule,
            "Cone": UsdGeom.Cone,
        }.get(shape.capitalize())
        if shape_cls is None:
            raise ValueError(f"Unsupported shape {shape!r}")

        geom = shape_cls.Define(stage, prim_path)
        prim = geom.GetPrim()

        if radius is not None and hasattr(geom, "CreateRadiusAttr"):
            geom.CreateRadiusAttr().Set(float(radius))
        if size is not None and hasattr(geom, "CreateSizeAttr"):
            geom.CreateSizeAttr().Set(float(size))

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*as_vec3(position, name="position")))
        if scale is not None:
            xform.AddScaleOp().Set(Gf.Vec3f(*as_vec3(scale, name="scale")))

        if color is not None:
            geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*as_vec3(color, name="color"))])

        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(float(mass))

        self.apply_physics_material(
            prim_path, friction=friction, restitution=restitution
        )
        return RigidObject(prim_path, scene=self)

    def apply_physics_material(
        self,
        prim_path: str,
        *,
        friction: float = 0.9,
        restitution: float = 0.05,
        material_path: str | None = None,
    ) -> str:
        """Bind a physics material with real friction to a prim."""
        from pxr import UsdPhysics, UsdShade

        stage = self.stage
        material_path = material_path or f"/World/PhysicsMaterials/mu{int(friction * 100)}"
        mat_prim = stage.GetPrimAtPath(material_path)
        if not mat_prim.IsValid():
            material = UsdPhysics.MaterialAPI.Apply(
                UsdShade.Material.Define(stage, material_path).GetPrim()
            )
            material.CreateStaticFrictionAttr().Set(float(friction))
            material.CreateDynamicFrictionAttr().Set(float(friction))
            material.CreateRestitutionAttr().Set(float(restitution))
            mat_prim = stage.GetPrimAtPath(material_path)

        target = stage.GetPrimAtPath(prim_path)
        if not target.IsValid():
            raise ValueError(f"Cannot bind material: {prim_path} does not exist")
        binding = UsdShade.MaterialBindingAPI.Apply(target)
        binding.Bind(
            UsdShade.Material(mat_prim),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )
        return material_path

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_prims(self, root: str = "/World", *, recursive: bool = True) -> list[dict[str, str]]:
        """List prims under `root`.

        Recursive by default — the MCP `list_prims` verb only walks one level,
        so finding a robot's TCP link needs one call per level of the hierarchy.
        """
        from pxr import Usd

        stage = self.stage
        start = stage.GetPrimAtPath(root)
        if not start.IsValid():
            return []
        iterator = Usd.PrimRange(start) if recursive else start.GetChildren()
        return [
            {"path": str(p.GetPath()), "type": p.GetTypeName()}
            for p in iterator
            if str(p.GetPath()) != root
        ]

    def find(self, pattern: str, *, root: str = "/World") -> list[str]:
        """Case-insensitive substring search over prim paths."""
        needle = pattern.lower()
        return [p["path"] for p in self.list_prims(root) if needle in p["path"].lower()]
