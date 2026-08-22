# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
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

"""Isaac Sim 5.1.0 adapter implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import IsaacAdapterCommon

if TYPE_CHECKING:
    from pxr import Usd


class IsaacAdapterV5(IsaacAdapterCommon):
    """Adapter for Isaac Sim 5.1.0 (isaacsim.* namespace)."""

    # ── Prims ──────────────────────────────────────────────

    def create_prim(self, prim_path: str, prim_type: str = "Xform", **kwargs) -> Usd.Prim:
        from isaacsim.core.utils.prims import create_prim

        return create_prim(prim_path, prim_type, **kwargs)

    def load_environment(self, env_path: str, prim_path: str = "/Environment") -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(env_path, prim_path)

    def add_reference_to_stage(self, usd_path: str, prim_path: str) -> Usd.Prim:
        from isaacsim.core.utils.stage import add_reference_to_stage

        return add_reference_to_stage(usd_path, prim_path)

    # ── Robots ─────────────────────────────────────────────

    def create_xform_prim(self, prim_path: str) -> Any:
        from isaacsim.core.prims import SingleXFormPrim

        return SingleXFormPrim(prim_path=prim_path)

    def create_articulation(self, prim_path: str, name: str) -> Any:
        from isaacsim.core.prims import SingleArticulation

        return SingleArticulation(prim_path=prim_path, name=name)

    def get_robot_joint_info(self, prim_path: str) -> Dict[str, Any]:
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdPhysics

        # Try to get joint info via articulation API (requires running sim)
        joint_names: List[str] = []
        num_dof = 0
        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            joint_names = list(art.dof_names) if art.dof_names else []
            num_dof = art.num_dof if art.num_dof else 0
        except Exception:
            pass

        # Fallback: discover joints by traversing USD stage
        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not joint_names and root_prim.IsValid():
            for desc in Usd.PrimRange(root_prim):
                if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint):
                    joint_names.append(desc.GetName())
            num_dof = len(joint_names)

        joint_limits = []
        for jname in joint_names:
            limit_entry: Dict[str, Any] = {"name": jname}
            for desc in Usd.PrimRange(root_prim):
                if desc.GetName() != jname:
                    continue
                if desc.IsA(UsdPhysics.RevoluteJoint):
                    rev = UsdPhysics.RevoluteJoint(desc)
                    lo = rev.GetLowerLimitAttr().Get()
                    hi = rev.GetUpperLimitAttr().Get()
                    limit_entry["type"] = "revolute"
                    limit_entry["lower"] = float(lo) if lo is not None else None
                    limit_entry["upper"] = float(hi) if hi is not None else None
                    limit_entry["units"] = "degrees"
                    break
                if desc.IsA(UsdPhysics.PrismaticJoint):
                    pris = UsdPhysics.PrismaticJoint(desc)
                    lo = pris.GetLowerLimitAttr().Get()
                    hi = pris.GetUpperLimitAttr().Get()
                    limit_entry["type"] = "prismatic"
                    limit_entry["lower"] = float(lo) if lo is not None else None
                    limit_entry["upper"] = float(hi) if hi is not None else None
                    limit_entry["units"] = "meters"
                    break
            joint_limits.append(limit_entry)

        return {
            "joint_names": joint_names,
            "num_dof": num_dof,
            "joint_limits": joint_limits,
        }

    def set_joint_positions(
        self,
        prim_path: str,
        positions: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            action = ArticulationAction(
                joint_positions=np.array(positions),
                joint_indices=np.array(joint_indices) if joint_indices else None,
            )
            controller = art.get_articulation_controller()
            controller.apply_action(action)
        except Exception:
            # Fallback: set USD drive targets directly (works when sim is stopped)
            self._set_joint_drive_targets(prim_path, positions, joint_indices)

    def _get_joint_names(self, prim_path: str) -> List[str]:
        """Get joint names, trying articulation API first then USD fallback."""
        from isaacsim.core.prims import SingleArticulation

        self._ensure_physics_world()
        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            if art.dof_names:
                return list(art.dof_names)
        except Exception:
            pass

        # Fallback: traverse USD
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return []
        names: List[str] = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint):
                names.append(desc.GetName())
        return names

    def get_joint_positions(self, prim_path: str) -> List[float]:
        from isaacsim.core.prims import SingleArticulation

        # Ensure physics is initialized so SingleArticulation.initialize() works
        self._ensure_physics_world()

        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            positions = art.get_joint_positions()
            if positions is not None:
                return positions.tolist()
        except Exception:
            pass

        # Fallback: read drive target positions from USD
        # WARNING: these are authored targets, not actual physics positions
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return []
        positions_list: List[float] = []
        for desc in Usd.PrimRange(root_prim):
            if not (desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint)):
                continue
            is_revolute = desc.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(desc, drive_type)
            if drive:
                target = drive.GetTargetPositionAttr().Get()
                if target is not None:
                    if is_revolute:
                        positions_list.append(float(np.radians(target)))
                    else:
                        positions_list.append(float(target / 100.0))
                else:
                    positions_list.append(0.0)
            else:
                positions_list.append(0.0)
        return positions_list

    def get_joint_config(self, prim_path: str) -> Dict[str, Any]:
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdPhysics

        self._ensure_physics_world()
        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        # Get current joint positions and names via articulation (requires running sim)
        joint_names = self._get_joint_names(prim_path)
        current_pos_list = self.get_joint_positions(prim_path)

        runtime_targets: List[float] = []
        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            applied_action = art.get_applied_action()
            if applied_action and applied_action.joint_positions is not None:
                runtime_targets = applied_action.joint_positions.tolist()
        except Exception:
            pass  # Fall back to USD values if articulation controller unavailable

        joints_info = []

        # Walk descendants to find joint prims
        for desc in Usd.PrimRange(prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint):
                joint_data: Dict[str, Any] = {"name": desc.GetName()}

                if desc.IsA(UsdPhysics.RevoluteJoint):
                    joint_data["type"] = "revolute"
                    joint_api = UsdPhysics.RevoluteJoint(desc)
                    lower_attr = joint_api.GetLowerLimitAttr()
                    upper_attr = joint_api.GetUpperLimitAttr()
                else:
                    joint_data["type"] = "prismatic"
                    joint_api = UsdPhysics.PrismaticJoint(desc)
                    lower_attr = joint_api.GetLowerLimitAttr()
                    upper_attr = joint_api.GetUpperLimitAttr()

                joint_data["lower_limit"] = lower_attr.Get() if lower_attr else None
                joint_data["upper_limit"] = upper_attr.Get() if upper_attr else None

                # Get drive config
                for drive_type in ["angular", "linear"]:
                    drive_api = UsdPhysics.DriveAPI.Get(desc, drive_type)
                    if drive_api:
                        joint_data["drive_type"] = drive_type
                        stiffness_attr = drive_api.GetStiffnessAttr()
                        damping_attr = drive_api.GetDampingAttr()
                        target_attr = drive_api.GetTargetPositionAttr()
                        joint_data["stiffness"] = stiffness_attr.Get() if stiffness_attr else None
                        joint_data["damping"] = damping_attr.Get() if damping_attr else None
                        # USD default as fallback
                        joint_data["target_position"] = target_attr.Get() if target_attr else None
                        break

                # Match actual position from articulation if possible
                joint_name = desc.GetName()
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    if idx < len(current_pos_list):
                        joint_data["actual_position"] = current_pos_list[idx]

                    # Override target_position with runtime value if available
                    if idx < len(runtime_targets):
                        joint_data["target_position"] = float(runtime_targets[idx])

                    # Calculate position_error using (possibly runtime) target
                    if joint_data.get("target_position") is not None and "actual_position" in joint_data:
                        joint_data["position_error"] = joint_data["target_position"] - joint_data["actual_position"]

                joints_info.append(joint_data)

        # Warn about joints with zero stiffness (broken drive config)
        warnings = []
        for j in joints_info:
            stiff = j.get("stiffness")
            damp = j.get("damping")
            if stiff is not None and stiff == 0 and (damp is None or damp == 0):
                warnings.append(
                    f"Joint '{j['name']}' has stiffness=0 and damping=0 — "
                    f"its drive is effectively disabled and will not respond to position targets."
                )

        result: Dict[str, Any] = {
            "prim_path": prim_path,
            "joint_count": len(joints_info),
            "joints": joints_info,
        }
        if warnings:
            result["warnings"] = warnings
        return result

    # ── Physics ────────────────────────────────────────────

    def create_world(self, **kwargs) -> Any:
        from isaacsim.core.api import World

        return World(**kwargs)

    def create_simulation_context(self, **kwargs) -> Any:
        from isaacsim.core.api import SimulationContext

        return SimulationContext(**kwargs)

    def get_physics_state(self, prim_path: str) -> Dict[str, Any]:
        from pxr import UsdPhysics

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        result: Dict[str, Any] = {"prim_path": prim_path}

        # Check rigid body API
        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        result["has_rigid_body"] = has_rb

        if has_rb:
            rb = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rb.GetKinematicEnabledAttr()
            result["is_kinematic"] = kinematic_attr.Get() if kinematic_attr else False

        # Check mass
        has_mass = prim.HasAPI(UsdPhysics.MassAPI)
        if has_mass:
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            result["mass"] = mass_attr.Get() if mass_attr else None

        # Check collision
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        result["collision_enabled"] = has_collision

        # Get velocities from PhysX runtime API (not USD attributes which may be stale)
        if has_rb:
            try:
                import omni.physx

                physx = omni.physx.get_physx_interface()
                rb_data = physx.get_rigidbody_transformation(prim_path)
                if rb_data and rb_data.get("ret_val", False):
                    vel = rb_data.get("linear_velocity", (0.0, 0.0, 0.0))
                    ang_vel = rb_data.get("angular_velocity", (0.0, 0.0, 0.0))
                    result["linear_velocity"] = [float(vel[0]), float(vel[1]), float(vel[2])]
                    result["angular_velocity"] = [float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])]
                else:
                    result["linear_velocity"] = [0.0, 0.0, 0.0]
                    result["angular_velocity"] = [0.0, 0.0, 0.0]
            except Exception:
                result["linear_velocity"] = [0.0, 0.0, 0.0]
                result["angular_velocity"] = [0.0, 0.0, 0.0]

        # Get contact info if available
        try:
            contacts = []
            result["contacts"] = contacts
        except Exception:
            result["contacts"] = []

        return result

    # ── Sensors ────────────────────────────────────────────

    def create_camera(self, prim_path: str, resolution: Tuple[int, int] = (1280, 720), **kwargs) -> Any:
        from isaacsim.sensors.camera import Camera

        return Camera(prim_path=prim_path, resolution=resolution, **kwargs)

    def capture_camera_image(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.camera import Camera

        cam = Camera(prim_path=prim_path)
        return cam.get_rgba()

    def create_lidar(self, prim_path: str, config: Optional[str] = None, **kwargs) -> Any:
        from isaacsim.sensors.rtx import LidarRtx

        return LidarRtx(prim_path=prim_path, config=config or "Example_Rotary", **kwargs)

    def get_lidar_point_cloud(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.rtx import LidarRtx

        lidar = LidarRtx(prim_path=prim_path)
        return lidar.get_point_cloud()

    # ── Materials ──────────────────────────────────────────

    def apply_material(self, material_path: str, target_prim_path: str) -> None:
        from pxr import UsdShade

        stage = self.get_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        target = stage.GetPrimAtPath(target_prim_path)
        UsdShade.MaterialBindingAPI(target).Bind(material)

    # ── Assets ─────────────────────────────────────────────

    def import_urdf(self, urdf_path: str, prim_path: str = "/World/robot", **kwargs) -> Any:
        import os

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        omni.kit.commands.execute("URDFParseFile", urdf_path=urdf_path, import_config=import_config)
        result = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            import_config=import_config,
            dest_path=prim_path,
        )
        return result

    # ── Simulation ─────────────────────────────────────────

    def play(self) -> None:
        import omni.timeline

        self._ensure_physics_world()
        omni.timeline.get_timeline_interface().play()

    def pause(self) -> None:
        import omni.timeline

        omni.timeline.get_timeline_interface().pause()

    def stop(self) -> None:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()

    def step(
        self, num_steps: int = 1, observe_prims: Optional[List[str]] = None, observe_joints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        import omni.kit.app

        for _ in range(num_steps):
            omni.kit.app.get_app().update()

        result: Dict[str, Any] = {"stepped": num_steps}

        # Observe prim states
        if observe_prims:
            from pxr import UsdPhysics

            prim_states = []
            stage = self.get_stage()
            for path in observe_prims:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    prim_states.append({"prim_path": path, "error": "Prim not found"})
                    continue
                state: Dict[str, Any] = {"prim_path": path}
                # Prefer PhysX runtime position for rigid bodies (always current)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        import omni.physx

                        physx = omni.physx.get_physx_interface()
                        rb_data = physx.get_rigidbody_transformation(path)
                        if rb_data and rb_data.get("ret_val", False):
                            pos = rb_data["position"]
                            state["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]
                        else:
                            transform = self.get_prim_transform(path)
                            state["position"] = transform.get("position", [0, 0, 0])
                    except Exception:
                        transform = self.get_prim_transform(path)
                        state["position"] = transform.get("position", [0, 0, 0])
                else:
                    transform = self.get_prim_transform(path)
                    state["position"] = transform.get("position", [0, 0, 0])
                # Add velocity if rigid body
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        physics_state = self.get_physics_state(path)
                        state["linear_velocity"] = physics_state.get("linear_velocity", [0, 0, 0])
                        state["angular_velocity"] = physics_state.get("angular_velocity", [0, 0, 0])
                    except Exception:
                        pass
                prim_states.append(state)
            result["prim_states"] = prim_states

        # Observe joint states
        if observe_joints:
            joint_states = []
            for path in observe_joints:
                try:
                    positions = self.get_joint_positions(path)
                    names = self._get_joint_names(path)
                    joints_dict = dict(zip(names, positions)) if names else {"positions": positions}
                    joint_states.append({"prim_path": path, "joints": joints_dict})
                except Exception as e:
                    joint_states.append({"prim_path": path, "error": str(e)})
            result["joint_states"] = joint_states

        return result

    def get_simulation_state(self) -> Dict[str, Any]:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        is_playing = timeline.is_playing()
        is_stopped = timeline.is_stopped()

        if is_playing:
            state = "playing"
        elif is_stopped:
            state = "stopped"
        else:
            state = "paused"

        current_time = timeline.get_current_time()
        # Get physics dt from physics scene if available
        from pxr import UsdPhysics

        stage = self.get_stage()
        physics_dt = 1.0 / 60.0  # default
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.Scene):
                time_step_attr = prim.GetAttribute("physxScene:timeStepsPerSecond")
                if time_step_attr and time_step_attr.Get():
                    steps_per_sec = time_step_attr.Get()
                    if steps_per_sec > 0:
                        physics_dt = 1.0 / steps_per_sec
                break

        return {
            "timeline_state": state,
            "current_time": current_time,
            "physics_dt": physics_dt,
        }

    # Track exec() namespaces to clean up subscriptions on reload
    _exec_namespaces: Dict[str, dict] = {}
