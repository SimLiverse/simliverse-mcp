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

"""Isaac Sim 6.0.0 adapter implementation (experimental API based)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import IsaacAdapterCommon

if TYPE_CHECKING:
    from pxr import Usd


class IsaacAdapterV6(IsaacAdapterCommon):
    """Adapter for Isaac Sim 6.0.0 using experimental APIs."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            self._engine = SimulationManager.get_active_physics_engine()
        except ImportError:
            self._engine = "unknown"
            
        try:
            import isaacsim.core.version
            self._isaacsim_version = isaacsim.core.version.get_version()
        except Exception:
            self._isaacsim_version = "6.0.0"

    # ── Prims ──────────────────────────────────────────────

    def create_prim(self, prim_path: str, prim_type: str = "Xform", **kwargs) -> Usd.Prim:
        from isaacsim.core.experimental.utils.stage import define_prim
        return define_prim(prim_path, prim_type, **kwargs)

    def load_environment(self, env_path: str, prim_path: str = "/Environment") -> None:
        from isaacsim.core.experimental.utils.stage import add_reference_to_stage
        add_reference_to_stage(env_path, prim_path)

    def add_reference_to_stage(self, usd_path: str, prim_path: str) -> Usd.Prim:
        from isaacsim.core.experimental.utils.stage import add_reference_to_stage
        return add_reference_to_stage(usd_path, prim_path)

    # ── Robots ─────────────────────────────────────────────

    def create_xform_prim(self, prim_path: str) -> Any:
        from isaacsim.core.experimental.prims import XformPrim
        return XformPrim(prim_paths=[prim_path])

    def create_articulation(self, prim_path: str, name: str) -> Any:
        from isaacsim.core.experimental.prims import Articulation
        return Articulation(prim_paths=[prim_path])

    def get_robot_joint_info(self, prim_path: str) -> Dict[str, Any]:
        from pxr import Usd, UsdPhysics

        joint_names: List[str] = []
        num_dof = 0
        try:
            from isaacsim.core.experimental.prims import Articulation
            art = Articulation(prim_paths=[prim_path])
            if not art.initialized:
                art.initialize()
            names = art.get_dof_names()
            if names and len(names) > 0:
                joint_names = list(names[0])
                num_dof = len(joint_names)
        except BaseException:
            pass

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
        try:
            from isaacsim.core.experimental.prims import Articulation
            import warp as wp
            art = Articulation(prim_paths=[prim_path])
            if not art.initialized:
                art.initialize()
            
            # Articulation is batched (takes lists of prim paths), so we pass [[pos]]
            if joint_indices:
                art.set_joint_positions(
                    positions=wp.array([positions], dtype=wp.float32), 
                    joint_indices=wp.array(joint_indices, dtype=wp.int32)
                )
            else:
                art.set_joint_positions(
                    positions=wp.array([positions], dtype=wp.float32)
                )
        except BaseException:
            self._set_joint_drive_targets(prim_path, positions, joint_indices)

    def _get_joint_names(self, prim_path: str) -> List[str]:
        try:
            from isaacsim.core.experimental.prims import Articulation
            art = Articulation(prim_paths=[prim_path])
            if not art.initialized:
                art.initialize()
            names = art.get_dof_names()
            if names and len(names) > 0:
                return list(names[0])
        except BaseException:
            pass

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
        try:
            from isaacsim.core.experimental.prims import Articulation
            art = Articulation(prim_paths=[prim_path])
            if not art.initialized:
                art.initialize()
            positions = art.get_joint_positions()
            if positions is not None and len(positions) > 0:
                return positions.numpy()[0].tolist()
        except BaseException:
            pass

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
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        joint_names = self._get_joint_names(prim_path)
        current_pos_list = self.get_joint_positions(prim_path)

        runtime_targets: List[float] = []
        try:
            from isaacsim.core.experimental.prims import Articulation
            art = Articulation(prim_paths=[prim_path])
            if not art.initialized:
                art.initialize()
            targets = art.get_joint_position_targets()
            if targets is not None and len(targets) > 0:
                runtime_targets = targets.numpy()[0].tolist()
        except BaseException:
            pass 

        joints_info = []
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

                for drive_type in ["angular", "linear"]:
                    drive_api = UsdPhysics.DriveAPI.Get(desc, drive_type)
                    if drive_api:
                        joint_data["drive_type"] = drive_type
                        stiffness_attr = drive_api.GetStiffnessAttr()
                        damping_attr = drive_api.GetDampingAttr()
                        target_attr = drive_api.GetTargetPositionAttr()
                        joint_data["stiffness"] = stiffness_attr.Get() if stiffness_attr else None
                        joint_data["damping"] = damping_attr.Get() if damping_attr else None
                        joint_data["target_position"] = target_attr.Get() if target_attr else None
                        break

                joint_name = desc.GetName()
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    if idx < len(current_pos_list):
                        joint_data["actual_position"] = current_pos_list[idx]
                    if idx < len(runtime_targets):
                        joint_data["target_position"] = float(runtime_targets[idx])
                    if joint_data.get("target_position") is not None and "actual_position" in joint_data:
                        joint_data["position_error"] = joint_data["target_position"] - joint_data["actual_position"]
                joints_info.append(joint_data)

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
        from isaacsim.core.simulation_manager import SimulationManager
        return SimulationManager()

    def create_simulation_context(self, **kwargs) -> Any:
        from isaacsim.core.simulation_manager import SimulationManager
        return SimulationManager()

    def get_physics_state(self, prim_path: str) -> Dict[str, Any]:
        from pxr import UsdPhysics
        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        result: Dict[str, Any] = {"prim_path": prim_path}

        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        result["has_rigid_body"] = has_rb

        if has_rb:
            rb = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rb.GetKinematicEnabledAttr()
            result["is_kinematic"] = kinematic_attr.Get() if kinematic_attr else False

        has_mass = prim.HasAPI(UsdPhysics.MassAPI)
        if has_mass:
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            result["mass"] = mass_attr.Get() if mass_attr else None

        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        result["collision_enabled"] = has_collision

        if has_rb:
            try:
                from isaacsim.core.simulation_manager import SimulationManager
                sim = SimulationManager()
                view = sim.get_physics_simulation_view()
                from isaacsim.core.experimental.prims import RigidPrim
                rp = RigidPrim(prim_paths=[prim_path])
                if not rp.initialized:
                    rp.initialize()
                lv = rp.get_linear_velocities()
                av = rp.get_angular_velocities()
                if lv is not None and av is not None and len(lv) > 0 and len(av) > 0:
                    l_val = lv.numpy()[0]
                    a_val = av.numpy()[0]
                    result["linear_velocity"] = [float(l_val[0]), float(l_val[1]), float(l_val[2])]
                    result["angular_velocity"] = [float(a_val[0]), float(a_val[1]), float(a_val[2])]
                else:
                    result["linear_velocity"] = [0.0, 0.0, 0.0]
                    result["angular_velocity"] = [0.0, 0.0, 0.0]
            except Exception:
                result["linear_velocity"] = [0.0, 0.0, 0.0]
                result["angular_velocity"] = [0.0, 0.0, 0.0]

        result["contacts"] = []
        return result

    # ── Sensors ────────────────────────────────────────────

    def create_camera(self, prim_path: str, resolution: Tuple[int, int] = (1280, 720), **kwargs) -> Any:
        from isaacsim.sensors.experimental.rtx import RtxCamera
        RtxCamera(prim_paths=[prim_path], resolutions=[resolution])
        return prim_path

    def capture_camera_image(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.experimental.rtx import CameraSensor
        sensor = CameraSensor(prim_path=prim_path)
        sensor.initialize()
        sensor.add_annotator("rgb")
        import omni.kit.app
        omni.kit.app.get_app().update()
        data = sensor.get_data()
        return data["rgb"] if "rgb" in data else np.zeros((0,0,3))

    def create_lidar(self, prim_path: str, config: Optional[str] = None, **kwargs) -> Any:
        from isaacsim.sensors.experimental.rtx import Lidar
        Lidar(prim_paths=[prim_path], configs=[config or "Example_Rotary"])
        return prim_path

    def get_lidar_point_cloud(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.experimental.rtx import LidarSensor
        sensor = LidarSensor(prim_path=prim_path)
        sensor.initialize()
        sensor.add_annotator("point_cloud")
        import omni.kit.app
        omni.kit.app.get_app().update()
        data = sensor.get_data()
        return data["point_cloud"] if "point_cloud" in data else np.zeros((0,3))

    # ── Materials ──────────────────────────────────────────

    def apply_material(self, material_path: str, target_prim_path: str) -> None:
        from pxr import UsdShade

        stage = self.get_stage()
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            raise ValueError(f"Material prim not found: {material_path}. Did you create it first?")
            
        target = stage.GetPrimAtPath(target_prim_path)
        if not target.IsValid():
            raise ValueError(f"Target prim not found: {target_prim_path}. You MUST use list_prims to find exact paths.")
            
        material = UsdShade.Material(material_prim)
        # Modern USD requires .Apply() for MaterialBindingAPI
        binding_api = UsdShade.MaterialBindingAPI.Apply(target)
        binding_api.Bind(material)

    # ── Assets ─────────────────────────────────────────────

    def import_urdf(self, urdf_path: str, prim_path: str = "/World/robot", **kwargs) -> Any:
        import os
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            
        config = URDFImporterConfig(urdf_path=urdf_path, dest_path=prim_path, **kwargs)
        importer = URDFImporter(config)
        importer.import_urdf()
        return {"prim_path": prim_path}

    # ── Simulation ─────────────────────────────────────────
    
    def _ensure_physics_world(self) -> None:
        from isaacsim.core.simulation_manager import SimulationManager
        try:
            sim = SimulationManager()
            if not sim.is_playing():
                sim.play()
        except Exception:
            pass

    def play(self) -> None:
        from isaacsim.core.experimental.utils.app import play
        self._ensure_physics_world()
        play()

    def pause(self) -> None:
        from isaacsim.core.experimental.utils.app import pause
        pause()

    def stop(self) -> None:
        from isaacsim.core.experimental.utils.app import stop
        stop()

    def step(
        self, num_steps: int = 1, observe_prims: Optional[List[str]] = None, observe_joints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        from isaacsim.core.experimental.utils.app import update_app

        update_app(steps=num_steps)

        result: Dict[str, Any] = {"stepped": num_steps}

        if observe_prims:
            prim_states = []
            for path in observe_prims:
                try:
                    physics_state = self.get_physics_state(path)
                    state = {"prim_path": path}
                    
                    transform = self.get_prim_transform(path)
                    state["position"] = transform.get("position", [0, 0, 0])
                    
                    state["linear_velocity"] = physics_state.get("linear_velocity", [0, 0, 0])
                    state["angular_velocity"] = physics_state.get("angular_velocity", [0, 0, 0])
                    prim_states.append(state)
                except Exception as e:
                    prim_states.append({"prim_path": path, "error": str(e)})
            result["prim_states"] = prim_states

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
        from isaacsim.core.simulation_manager import SimulationManager
        
        timeline = omni.timeline.get_timeline_interface()
        sim = SimulationManager()

        if timeline.is_playing():
            state = "playing"
        elif timeline.is_stopped():
            state = "stopped"
        else:
            state = "paused"

        current_time = timeline.get_current_time()
        try:
            current_time = sim.get_simulation_time()
        except Exception:
            pass

        physics_dt = 1.0 / 60.0
        from pxr import UsdPhysics
        stage = self.get_stage()
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
            "engine": self._engine,
            "isaacsim_version": self._isaacsim_version,
        }

    _exec_namespaces: Dict[str, dict] = {}
