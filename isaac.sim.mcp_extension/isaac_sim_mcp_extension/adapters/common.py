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

"""Version-independent adapter behaviour.

`IsaacAdapterBase` states the contract; this layer implements the part of it
that does not vary between Isaac Sim versions, and `v5.py` / `v6.py` are left
holding only what genuinely differs.

Everything here reaches Isaac through APIs that did not move between 5.1 and
6.0 -- `pxr`, `omni.usd`, `omni.kit.commands`, `isaacsim.storage.native` -- or
through no Isaac API at all. That is why these eighteen methods were
byte-identical or comment-only-different across the two adapters: they were
never version-specific, they were copied.

The copy was literal. `rewrite_v6.py` generated v6.py by emitting v5's source as
a string, so v6 carried the same bodies with the comments stripped out. The
bodies below are v5's, comments intact.

Anything a version DOES change -- the `isaacsim.core.*` to
`isaacsim.core.experimental.*` move, the timeline and articulation rewrites --
stays an override in the version module, where the diff between them is now the
whole file rather than something to be found in it.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import IsaacAdapterBase

if TYPE_CHECKING:
    from pxr import Usd


class IsaacAdapterCommon(IsaacAdapterBase):
    """Implementations shared by every Isaac Sim version adapter."""

    # ── Scene ─────────────────────────────────────────────────

    def get_stage(self) -> Usd.Stage:
        import omni.usd

        return omni.usd.get_context().get_stage()

    def get_assets_root_path(self) -> str:
        from isaacsim.storage.native import get_assets_root_path

        return get_assets_root_path()

    def discover_environments(self) -> Dict[str, Dict[str, str]]:
        """Scan the Isaac Sim asset server for available environment USD files."""
        import omni.client
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        discovered: Dict[str, Dict[str, str]] = {}

        search_bases = ["/Isaac/Environments/", "/NVIDIA/Assets/Scenes/Templates/"]
        for base in search_bases:
            result, entries = omni.client.list(root + base)
            if result != omni.client.Result.OK:
                continue
            for entry in entries:
                name = entry.relative_path.rstrip("/")
                dir_path = root + base + name + "/"
                r2, files = omni.client.list(dir_path)
                if r2 != omni.client.Result.OK:
                    continue
                # Find USD files at this level
                for f in files:
                    if f.relative_path.endswith(".usd") or f.relative_path.endswith(".usda"):
                        key = name.lower().replace(" ", "_")
                        if key not in discovered:
                            discovered[key] = {
                                "asset_path": base + name + "/" + f.relative_path,
                                "description": name.replace("_", " "),
                            }
                        break
                # Also check one level deeper for nested envs
                for f in files:
                    subname = f.relative_path.rstrip("/")
                    r3, subfiles = omni.client.list(dir_path + subname + "/")
                    if r3 != omni.client.Result.OK:
                        continue
                    for sf in subfiles:
                        if sf.relative_path.endswith(".usd") or sf.relative_path.endswith(".usda"):
                            key = f"{name}_{subname}".lower().replace(" ", "_")
                            if key not in discovered:
                                discovered[key] = {
                                    "asset_path": base + name + "/" + subname + "/" + sf.relative_path,
                                    "description": f"{name} {subname}".replace("_", " "),
                                }
                            break
        return discovered

    def create_physics_scene(self, gravity: Optional[Sequence[float]] = None, scene_name: str = "PhysicsScene") -> str:
        import omni.kit.commands

        scene_path = f"/World/{scene_name}"
        omni.kit.commands.execute("CreatePrim", prim_path=scene_path, prim_type="PhysicsScene")
        return scene_path

    # ── Prims ─────────────────────────────────────────────────

    def clone_prim(self, source_path: str, target_path: str) -> None:
        import omni.kit.commands

        omni.kit.commands.execute("CopyPrim", path_from=source_path, path_to=target_path)

    def delete_prim(self, prim_path: str) -> bool:
        import omni.kit.commands

        omni.kit.commands.execute("DeletePrims", paths=[prim_path])
        return True

    def set_prim_transform(
        self,
        prim_path: str,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        from pxr import Gf, UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        xformable = UsdGeom.Xformable(prim)

        # Merge with what is already authored. ClearXformOpOrder() drops every
        # op, so re-adding only the supplied ones silently reset the rest to
        # identity: `transform_object(path, scale=[...])` used to wipe position
        # and rotation. A ramp lost its 20-degree tilt the moment an agent tried
        # to widen it, and the damage was invisible until the geometry was
        # measured — the call reports success either way.
        current = {}
        try:
            current = self.get_prim_transform(prim_path) or {}
        except Exception:
            pass

        def _pick(supplied, key, default):
            if supplied is not None:
                return list(supplied)
            value = current.get(key)
            return list(value) if value is not None else list(default)

        pos = _pick(position, "position", (0.0, 0.0, 0.0))
        rot = _pick(rotation, "rotation", (0.0, 0.0, 0.0))
        scl = _pick(scale, "scale", (1.0, 1.0, 1.0))

        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*rot))
        xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scl))

    def get_prim_transform(self, prim_path: str) -> Dict[str, Any]:
        from pxr import UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        return {
            "position": [translation[0], translation[1], translation[2]],
        }

    def get_prim_info(self, prim_path: str) -> Dict[str, Any]:
        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        transform = self.get_prim_transform(prim_path)
        children = [str(c.GetPath()) for c in prim.GetAllChildren()]
        info: Dict[str, Any] = {
            "path": prim_path,
            "type": prim.GetTypeName(),
            "transform": transform,
            "children": children,
        }
        if prim.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Cone", "Capsule"):
            try:
                actual_size, _bbox = self.get_prim_actual_size(prim_path)
                info["actual_size"] = actual_size
            except Exception:
                pass
        return info

    def get_prim_actual_size(self, prim_path: str) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Return actual dimensions and bounding box for a geometric prim."""
        from pxr import UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        prim_type = prim.GetTypeName()

        # Read scale from xform
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        # Extract scale from the matrix diagonal (assuming uniform or axis-aligned scale)
        scale = [
            float(local_transform.GetRow3(0).GetLength()),
            float(local_transform.GetRow3(1).GetLength()),
            float(local_transform.GetRow3(2).GetLength()),
        ]

        if prim_type == "Cube":
            geom = UsdGeom.Cube(prim)
            size_attr = geom.GetSizeAttr()
            size = float(size_attr.Get()) if size_attr and size_attr.Get() is not None else 1.0
            dims = [size * scale[0], size * scale[1], size * scale[2]]
        elif prim_type == "Sphere":
            geom = UsdGeom.Sphere(prim)
            radius_attr = geom.GetRadiusAttr()
            radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 0.5
            diameter = radius * 2.0
            dims = [diameter * scale[0], diameter * scale[1], diameter * scale[2]]
        elif prim_type == "Cylinder":
            geom = UsdGeom.Cylinder(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            axis_attr = geom.GetAxisAttr()
            radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 0.5
            height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 1.0
            axis = axis_attr.Get() if axis_attr and axis_attr.Get() is not None else "Z"
            diameter = radius * 2.0
            if axis == "X":
                dims = [height * scale[0], diameter * scale[1], diameter * scale[2]]
            elif axis == "Y":
                dims = [diameter * scale[0], height * scale[1], diameter * scale[2]]
            else:  # Z (default)
                dims = [diameter * scale[0], diameter * scale[1], height * scale[2]]
        elif prim_type == "Cone":
            geom = UsdGeom.Cone(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            axis_attr = geom.GetAxisAttr()
            radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 0.5
            height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 1.0
            axis = axis_attr.Get() if axis_attr and axis_attr.Get() is not None else "Z"
            diameter = radius * 2.0
            if axis == "X":
                dims = [height * scale[0], diameter * scale[1], diameter * scale[2]]
            elif axis == "Y":
                dims = [diameter * scale[0], height * scale[1], diameter * scale[2]]
            else:  # Z (default)
                dims = [diameter * scale[0], diameter * scale[1], height * scale[2]]
        elif prim_type == "Capsule":
            geom = UsdGeom.Capsule(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 0.5
            height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 1.0
            total_height = height + 2.0 * radius
            diameter = radius * 2.0
            dims = [diameter * scale[0], diameter * scale[1], total_height * scale[2]]
        else:
            raise ValueError(f"Unsupported prim type for size calculation: {prim_type}")

        # Compute world-space position for bounding box
        from pxr import Usd

        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        pos = [float(translation[0]), float(translation[1]), float(translation[2])]
        half = [d / 2.0 for d in dims]
        bbox_min = [pos[0] - half[0], pos[1] - half[1], pos[2] - half[2]]
        bbox_max = [pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]]

        return dims, (bbox_min, bbox_max)

    def list_prims(self, root_path: str = "/", prim_type: Optional[str] = None) -> List[Dict[str, str]]:
        stage = self.get_stage()
        root = stage.GetPrimAtPath(root_path)
        results: List[Dict[str, str]] = []
        for prim in root.GetAllChildren():
            ptype = prim.GetTypeName()
            if prim_type and ptype != prim_type:
                continue
            results.append({"path": str(prim.GetPath()), "type": ptype})
        return results

    # ── Lighting and materials ────────────────────────────────

    def create_light(
        self,
        light_type: str,
        prim_path: str,
        intensity: float = 1000.0,
        color: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> Any:
        from pxr import Gf, UsdLux

        stage = self.get_stage()
        light_classes = {
            "DistantLight": UsdLux.DistantLight,
            "DomeLight": UsdLux.DomeLight,
            "SphereLight": UsdLux.SphereLight,
            "RectLight": UsdLux.RectLight,
            "DiskLight": UsdLux.DiskLight,
            "CylinderLight": UsdLux.CylinderLight,
        }
        cls = light_classes.get(light_type)
        if not cls:
            raise ValueError(f"Unknown light type: {light_type}. Options: {list(light_classes.keys())}")
        light = cls.Define(stage, prim_path)
        light.CreateIntensityAttr(intensity)
        if color:
            light.CreateColorAttr(Gf.Vec3f(*color[:3]))
        position = kwargs.get("position")
        if position:
            self.set_prim_transform(prim_path, position=position)
        rotation = kwargs.get("rotation")
        if rotation:
            self.set_prim_transform(prim_path, rotation=rotation)
        return light

    def modify_light(
        self,
        prim_path: str,
        intensity: Optional[float] = None,
        color: Optional[Sequence[float]] = None,
    ) -> None:
        from pxr import Gf

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Light not found: {prim_path}")
        if intensity is not None:
            prim.GetAttribute("inputs:intensity").Set(intensity)
        if color is not None:
            prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color[:3]))

    def create_pbr_material(
        self,
        prim_path: str,
        color: Optional[Sequence[float]] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> Any:
        from pxr import Gf, Sdf, UsdShade

        stage = self.get_stage()
        material = UsdShade.Material.Define(stage, prim_path)
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        if color:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        return material

    def create_physics_material(
        self,
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
    ) -> Any:
        from pxr import UsdPhysics

        stage = self.get_stage()
        material = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(prim_path))
        material.CreateStaticFrictionAttr(static_friction)
        material.CreateDynamicFrictionAttr(dynamic_friction)
        material.CreateRestitutionAttr(restitution)
        return material

    # ── Robots ────────────────────────────────────────────────

    def _set_joint_drive_targets(
        self,
        prim_path: str,
        positions: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Set joint drive targets via USD API — works regardless of simulation state."""
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        # Collect all joints under the articulation
        joints = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint):
                joints.append(desc)

        if joint_indices is not None:
            targets = list(zip(joint_indices, positions))
        else:
            targets = list(enumerate(positions))

        for idx, value in targets:
            if idx >= len(joints):
                continue
            joint_prim = joints[idx]
            is_revolute = joint_prim.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
            if is_revolute:
                drive.GetTargetPositionAttr().Set(float(np.degrees(value)))
            else:
                # Prismatic joints: positions in meters, USD targets in cm
                drive.GetTargetPositionAttr().Set(float(value * 100.0))

    # ── Scripting ─────────────────────────────────────────────

    def execute_script(self, code: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        import io
        import sys

        import carb
        import omni
        from pxr import Gf, Sdf, Usd, UsdGeom

        # Auto-add cwd to sys.path
        if cwd and cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Alias deprecated omni.isaac -> isaacsim for seamless backwards compatibility
        if "omni.isaac" not in sys.modules:
            try:
                import isaacsim
                import isaacsim.core as isaac_core
                sys.modules["omni.isaac"] = isaacsim
                sys.modules["omni.isaac.core"] = isaac_core
                sys.modules["omni.isaac.core.api"] = isaac_core.api
                sys.modules["omni.isaac.core.prims"] = isaac_core.prims
                sys.modules["omni.isaac.core.utils"] = getattr(isaac_core, "utils", isaac_core)
            except Exception:
                pass

        isaacsim_mod = sys.modules.get("isaacsim")
        usd_utils_mod = None
        prim_utils_mod = None
        try:
            import isaacsim.core.utils.stage as usd_utils_mod
            import isaacsim.core.utils.prims as prim_utils_mod
        except Exception:
            pass

        local_ns = {
            "omni": omni,
            "carb": carb,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "Sdf": Sdf,
            "Gf": Gf,
            "isaacsim": isaacsim_mod,
            "usd_utils": usd_utils_mod,
            "prim_utils": prim_utils_mod,
            "prims": prim_utils_mod,
        }

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_out = io.StringIO()
        sys.stderr = captured_err = io.StringIO()
        try:
            self._ensure_physics_world()
            exec(code, local_ns)
            return {
                "status": "success",
                "message": "Script executed successfully",
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def reload_script(self, file_path: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        import importlib
        import io
        import os
        import sys

        # Auto-add parent directory to sys.path
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Clean up previous exec() namespace for this file (unsubscribe orphaned callbacks)
        abs_path = os.path.abspath(file_path)
        old_ns = self._exec_namespaces.get(abs_path)
        if old_ns:
            for key, val in old_ns.items():
                if hasattr(val, "unsubscribe"):
                    try:
                        val.unsubscribe()
                    except Exception:
                        pass

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_out = io.StringIO()
        sys.stderr = captured_err = io.StringIO()
        try:
            if module_name:
                # Reload existing module or import for first time
                if module_name in sys.modules:
                    _module = importlib.reload(sys.modules[module_name])
                    msg = f"Module '{module_name}' reloaded successfully"
                else:
                    _module = importlib.import_module(module_name)
                    msg = f"Module '{module_name}' imported successfully"
            else:
                # Execute file contents (hot-patch)
                if not os.path.isfile(file_path):
                    return {"status": "error", "message": f"File not found: {file_path}"}
                with open(file_path, "r") as f:
                    code = f.read()
                import carb
                import omni
                from pxr import Gf, Sdf, Usd, UsdGeom

                local_ns = {
                    "omni": omni,
                    "carb": carb,
                    "Usd": Usd,
                    "UsdGeom": UsdGeom,
                    "Sdf": Sdf,
                    "Gf": Gf,
                    "__file__": file_path,
                }
                self._ensure_physics_world()
                exec(code, local_ns)
                # Track namespace so we can clean up subscriptions on next reload
                self._exec_namespaces[abs_path] = local_ns
                msg = f"Script '{os.path.basename(file_path)}' executed successfully"

            return {
                "status": "success",
                "message": msg,
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # Namespaces from `exec()`, kept so a reload can unsubscribe the callbacks
    # its previous run registered. One adapter is live per process, so a single
    # registry on the shared base is the same object the two version adapters
    # each used to declare for themselves.
    _exec_namespaces: Dict[str, dict] = {}
