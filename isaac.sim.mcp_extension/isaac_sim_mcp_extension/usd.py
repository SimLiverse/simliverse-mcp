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

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import carb
import omni
import omni.usd
import requests
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, UsdGeom, UsdShade


class USDLoader:
    """
    A class to load USD models and textures from local directories,
    with methods to bind materials and transform models.
    """

    def __init__(self):
        """
        Initialize the USDLoader class with default values.
        """
        self.usd_prim = None
        self.material = None
        self.working_dir = Path(os.environ.get("USD_WORKING_DIR", "/tmp/usd"))
        self.stage = omni.usd.get_context().get_stage()

    def load_usd_model(self, abs_path=None, task_id=None):
        """
        Load a USD model from either an absolute path or by task_id.

        Args:
            abs_path (str, optional): Absolute path to the USD file
            task_id (str, optional): Task ID to load model from working_dir

        Returns:
            str: Path to the loaded USD prim
        """
        if not (abs_path or task_id):
            raise ValueError("Either abs_path or task_id must be provided")

        if task_id:
            usd_path = self.working_dir / task_id / "output.usd"
        else:
            usd_path = Path(abs_path)

        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found at: {usd_path}")

        # Create a unique prim path name based on task_id or file name
        if task_id:
            prim_id = task_id[-5:]  # Last 5 chars of task ID
        else:
            prim_id = usd_path.stem[:5]  # First 5 chars of filename

        usd_prim_path = f"/World/model_{prim_id}"

        # Add the USD to the stage
        self.usd_prim = add_reference_to_stage(str(usd_path), usd_prim_path)

        print(f"Loaded USD model from {usd_path} at {usd_prim_path}")
        return usd_prim_path

    def load_texture_and_create_material(self, abs_path=None, task_id=None):
        """
        Load a texture from either an absolute path or by task_id and create a material with it.

        Args:
            abs_path (str, optional): Absolute path to the texture file
            task_id (str, optional): Task ID to load texture from working_dir

        Returns:
            tuple: (str, UsdShade.Material) - Path to the loaded texture and the created material
        """
        if not (abs_path or task_id):
            raise ValueError("Either abs_path or task_id must be provided")

        # Load texture
        if task_id:
            texture_path = self.working_dir / task_id / "textures" / "material_0.png"
            material_id = task_id[-5:]  # Last 5 chars of task ID
        else:
            texture_path = Path(abs_path)
            material_id = texture_path.stem[:5]  # First 5 chars of filename

        if not texture_path.exists():
            raise FileNotFoundError(f"Texture file not found at: {texture_path}")

        # Create a unique material name
        material_path = f"/World/SimpleMaterial_{material_id}"

        # Create material
        material = UsdShade.Material.Define(self.stage, material_path)

        # Create shader
        shader = UsdShade.Shader.Define(self.stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))

        # Create texture
        texture = UsdShade.Shader.Define(self.stage, f"{material_path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))

        # Connect texture to shader's diffuse color
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )

        print(f"Created material with texture at {material_path}")
        self.material = material
        return str(texture_path), material

    def bind_texture_to_model(self, prim=None, material=None):
        """
        Bind a texture to a USD model.

        Args:
            prim (UsdGeom.Xformable, optional): The USD prim to bind the material to.
                                               If None, uses self.usd_prim
            material (UsdShade.Material, optional): The material to bind.
                                                  If None, uses self.material

        Returns:
            bool: True if binding succeeded, False otherwise
        """
        if prim is None:
            prim = self.usd_prim

        if material is None:
            material = self.material

        if prim is None or material is None:
            raise ValueError("Both prim and material must be provided or previously set")

        try:
            binding_api = UsdShade.MaterialBindingAPI(prim)
            binding_api.Bind(material)
            print(f"Successfully bound material to {prim.GetPath()}")
            return True
        except Exception as e:
            print(f"Failed to bind material: {str(e)}")
            return False

    def transform(self, prim=None, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        Transform a USD model by applying position and scale.

        Args:
            prim (UsdGeom.Xformable, optional): The USD prim to transform.
                                               If None, uses self.usd_prim
            position (tuple, optional): The position to set (x, y, z)
            scale (tuple, optional): The scale to set (x, y, z)

        Returns:
            UsdGeom.Xformable: The transformed prim
        """
        if prim is None:
            prim = self.usd_prim

        if prim is None:
            raise ValueError("Prim must be provided or previously set")

        # Get the Xformable interface
        xformable = UsdGeom.Xformable(prim)

        # Check if transform operations already exist and use them
        xform_ops = xformable.GetOrderedXformOps()

        # Handle translation
        translate_op = None
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op:
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
        else:
            xformable.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))
        print(f"Model positioned at {position}")

        # Handle scaling
        scale_op = None
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op:
            scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
        else:
            xformable.AddScaleOp().Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
        print(f"Model scaled to {scale}")

        return xformable

    def _load_prim(self, url, path: str = "/World/my_usd", size=1.0):
        """Helper method to load a USD prim from url to specific scene path."""
        try:
            stage = omni.usd.get_context().get_stage()

            # check path exists or not
            prim = stage.GetPrimAtPath(path)
            if prim:
                # Generate unique name based on path
                for p in stage.Traverse():
                    carb.log_info(f"Error in _load_prim: {p.GetPrimPath()}")
                count = len([p for p in stage.Traverse()])
                path = f"{path}_{count + 1}"

            carb.log_info(f"loading from {url} to {path}")
            # Create a new prim
            prim = stage.DefinePrim(path)
            carb.log_info(f"prim loaded: {path, prim}")

            # Add a reference to the external USD file
            prim.GetReferences().AddReference(url)

            return prim
        except Exception as e:
            carb.log_info(f"Error in _load_prim: {str(e)}")
            return {"error": str(e)}

    def _set_transform(self, prim, location=None, rotation=None, scale=None):
        """Set transform operations on a USD prim."""
        if not prim.IsA(UsdGeom.Xformable):
            return

        xformable = UsdGeom.Xformable(prim)

        # Reset transform stack
        xformable.ClearXformOpOrder()

        # Build transform operations in order: scale, rotation, translation
        ops = []

        # Add scale operation if provided
        if scale is not None:
            scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
            ops.append(scale_op)
        # Add rotation operation if provided
        if rotation is not None:
            rot_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.PrecisionDouble)
            rot_op.Set(Gf.Vec3d(rotation[0], rotation[1], rotation[2]))
            ops.append(rot_op)
        # Add translation operation if provided
        if location is not None:
            trans_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
            trans_op.Set(Gf.Vec3d(location[0], location[1], location[2]))
            ops.append(trans_op)
        # Apply transform operations in order
        xformable.SetXformOpOrder(ops)

    def _set_color(self, prim, color):
        """Set a simple color material on a USD prim."""
        stage = omni.usd.get_context().get_stage()

        # Only apply to geometric prims
        if not UsdGeom.Gprim(prim):
            return

        # Create material with unique path
        material_path = str(prim.GetPath()) + "_material"
        material = UsdShade.Material.Define(stage, material_path)

        # Create preview surface shader
        shader_path = material_path + "/PreviewSurface"
        pbrShader = UsdShade.Shader.Define(stage, shader_path)
        pbrShader.CreateIdAttr("UsdPreviewSurface")

        # Set diffuse color
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[0], color[1], color[2]))

        # Connect shader to material surface
        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # Bind material to prim
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def load_usd_from_url(
        self, url_path=None, target_path=None, location=None, rotation=None, scale=None, size=1.0, color=None
    ):
        """
        Load a usd object from url and insert into current scene at target path.

        Args:
            url (str): url of the usd object to be loaded
            target_path (str, optional): target path for the loaded object
            location (list, optional): [x, y, z] position
            rotation (list, optional): [x, y, z] rotation
            scale (list, optional): [x, y, z] scale
            size (float, optional): Size parameter
            color (list, optional): [r, g, b] color

        Returns:
            dict: Information about the loaded object
        """
        try:
            # Create the prim based on type
            # url = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Storage/Drums/Plastic_A/PlasticDrum_A04_PR_V_NVD_01.usd"
            prim = self._load_prim(url_path, path=target_path)
            path = str(prim.GetPath())

            # Apply transform if provided
            if location or rotation or scale:
                self._set_transform(prim, location, rotation, scale)

            # Apply color if provided
            if color:
                self._set_color(prim, color)

            # Return object info
            prim_info = {
                "target_path": path,
            }

            # Add transform information
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                trans = xformable.GetLocalTransformation()
                translation = trans.ExtractTranslation()

                prim_info["transform"] = {
                    "translation": [translation[0], translation[1], translation[2]],
                }

            _details = json.dumps(prim_info, indent=2)

            print(f"Loaded USD model from {url_path} at {path}")
            return path

        except Exception as e:
            carb.log_info(f"Error in load_usd_from_url: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def test_tasks_load():
        """
        Test method to load and process multiple task IDs.
        """
        loader = USDLoader()

        # List of task IDs to test
        task_ids = ["cgt-20250408202506-6tdc4", "cgt-20250408202622-rzcgg", "cgt-20250408202753-44b9f"]

        for task_id in task_ids:
            try:
                # Load model
                _prim_path = loader.load_usd_model(task_id=task_id)

                # # Load texture
                # texture_path = loader.load_texture(task_id=task_id)

                # Create material with texture
                texture_path, material = loader.load_texture_and_create_material(task_id=task_id)

                # Bind texture to model
                loader.bind_texture_to_model()

                # Transform model
                loader.transform(position=(0, 0, 50 + task_ids.index(task_id) * 20))

                print(f"Successfully processed task {task_id}")
            except Exception as e:
                print(f"Error processing task {task_id}: {str(e)}")

    @staticmethod
    def test_absolute_paths():
        """
        Test method for loading from absolute paths.
        """
        loader = USDLoader()

        # Test with absolute paths
        model_path = "/tmp/usd/cgt-20250408202506-6tdc4/output.usd"
        texture_path = "/tmp/usd/cgt-20250408202506-6tdc4/textures/material_0.png"

        try:
            # Load model
            _prim_path = loader.load_usd_model(abs_path=model_path)

            # Load texture
            texture_path, material = loader.load_texture_and_create_material(abs_path=texture_path)

            # Create material with texture
            # material = loader.create_material_with_texture(texture_path)

            # Bind texture to model
            loader.bind_texture_to_model()

            # Transform model
            loader.transform()

            print("Successfully tested absolute paths")
        except Exception as e:
            print(f"Error testing absolute paths: {str(e)}")


import threading
import time


class AssetIndexCache:
    """Singleton background asset indexer.

    Pre-indexes local mounts (/mnt/assets) and remote Omniverse S3/Nucleus repositories
    in a background thread so queries resolve in 0ms!
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._index = []
                cls._instance._is_indexing = False
                cls._instance._last_indexed = 0.0
                cls._instance._start_background_indexing()
            return cls._instance

    def _start_background_indexing(self):
        if self._is_indexing:
            return
        t = threading.Thread(target=self._build_index, daemon=True)
        t.start()

    def _build_index(self):
        self._is_indexing = True
        try:
            new_index = []

            # 1. Scan Local Mounts (/mnt/assets, /workspace/job_repo, /tmp/usd)
            dirs_to_scan = ["/mnt/assets", "/workspace/job_repo", "/tmp/usd"]
            for base_dir in dirs_to_scan:
                p = Path(base_dir)
                if not p.exists():
                    continue
                try:
                    for ext in ["*.usd", "*.usda", "*.usdc", "*.urdf"]:
                        for filepath in p.rglob(ext):
                            new_index.append({
                                "name": filepath.stem,
                                "path": str(filepath.absolute()),
                                "type": filepath.suffix.lstrip("."),
                                "source": f"local:{base_dir}"
                            })
                except Exception as e:
                    carb.log_warn(f"Background index local scan error {base_dir}: {e}")

            # 2. Crawl Omniverse S3 Bucket via omni.client
            try:
                import omni.client
                try:
                    from isaacsim.storage.native import get_assets_root_path
                    assets_root = get_assets_root_path()
                except Exception:
                    try:
                        from omni.isaac.core.utils.nucleus import get_assets_root_path
                        assets_root = get_assets_root_path()
                    except Exception:
                        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"

                categories = ["/Isaac/Robots", "/Isaac/Environments", "/Isaac/Props"]

                def _crawl(url_dir: str, depth: int = 0, max_depth: int = 3):
                    if depth > max_depth:
                        return
                    try:
                        res, entries = omni.client.list(url_dir)
                        if res != omni.client.Result.OK or not entries:
                            return
                        for entry in entries:
                            rel = entry.relative_path
                            child_url = f"{url_dir.rstrip('/')}/{rel}"
                            is_dir = bool(entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
                            if is_dir:
                                _crawl(child_url, depth + 1, max_depth)
                            elif rel.endswith((".usd", ".usda", ".usdc")):
                                new_index.append({
                                    "name": Path(rel).stem,
                                    "path": child_url,
                                    "type": rel.split(".")[-1],
                                    "source": "omni_client_catalog"
                                })
                    except Exception as e:
                        carb.log_warn(f"Background index crawl error {url_dir}: {e}")

                for cat in categories:
                    _crawl(f"{assets_root}{cat}")
            except Exception as e:
                carb.log_warn(f"Background omni.client index error: {e}")

            self._index = new_index
            self._last_indexed = time.time()
            carb.log_info(f"AssetIndexCache build complete. Total indexed assets: {len(self._index)}")
        finally:
            self._is_indexing = False

    def query(self, search_term: str, search_paths: Optional[list] = None) -> list:
        if not self._index and not self._is_indexing:
            self._start_background_indexing()

        extra_matches = []
        if search_paths:
            for base_dir in search_paths:
                p = Path(base_dir)
                if not p.exists():
                    continue
                try:
                    for ext in ["*.usd", "*.usda", "*.usdc", "*.urdf"]:
                        for filepath in p.rglob(ext):
                            extra_matches.append({
                                "name": filepath.stem,
                                "path": str(filepath.absolute()),
                                "type": filepath.suffix.lstrip("."),
                                "source": f"local:{base_dir}"
                            })
                except Exception:
                    pass

        all_items = self._index + extra_matches
        query_lower = search_term.lower()
        tokens = [t for t in query_lower.split() if len(t) > 1]

        matches = []
        for item in all_items:
            name_lower = item["name"].lower()
            path_lower = item["path"].lower()
            if any(tok in name_lower or tok in path_lower for tok in tokens) or not tokens:
                matches.append(item)

        return matches


class USDSearch3d:
    """Dynamic Asset Discovery Service.

    Supports:
    1. Fast background-indexed search (/mnt/assets + Isaac Sim catalog)
    2. NVIDIA USD Search API (if accessible)
    """

    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        self.usd_search_server = "https://ai.api.nvidia.com/v1/omniverse/nvidia/usdsearch"

    def discover_assets(self, query: str, search_paths: Optional[list] = None) -> list:
        """Search local directories (/mnt/assets) and Isaac Sim asset catalogs in 0ms via AssetIndexCache."""
        cache = AssetIndexCache()
        return cache.query(query, search_paths)

    def search(self, text_prompt: str) -> str:
        """Search for a USD model, trying NVIDIA Cloud NIM first, then falling back to local + Isaac discovery."""
        if self.api_key:
            try:
                response = requests.post(
                    url=self.usd_search_server,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(
                        dict(
                            description=text_prompt,
                            file_extension_include="usd*",
                            return_images="true",
                            return_metadata="true",
                            return_vision_generated_metadata="true",
                            cutoff_threshold="1.05",
                            limit="50",
                        )
                    ),
                    timeout=5.0
                )
                if response.status_code == 200:
                    details = response.json()
                    if isinstance(details, list) and len(details) > 0 and "url" in details[0]:
                        url = details[0]["url"]
                        if url.startswith("s3://deepsearch-demo-content"):
                            url = url.replace(
                                "s3://deepsearch-demo-content", "https://omniverse-content-production.s3.us-west-2.amazonaws.com"
                            )
                        return url
            except Exception as e:
                carb.log_info(f"NIM USD search call failed, falling back to dynamic asset finder: {e}")

        # Fallback to Dynamic Asset Finder
        matches = self.discover_assets(text_prompt)
        if matches:
            return matches[0]["path"]

        raise ValueError(f"No asset found for '{text_prompt}' in local mounts (/mnt/assets) or Isaac Sim catalog.")

    @staticmethod
    def test_search_and_load():
        text_prompt = "a rusty desk"
        searcher3d = USDSearch3d()
        url = searcher3d.search(text_prompt)
        target_path = "/World/search_usd"
        loader = USDLoader()
        _prim_path = loader.load_usd_from_url(url, target_path)

    @staticmethod
    def usd_search_3d_from_text(text_prompt: str, target_path: str, position=(0, 0, 50), scale=(10, 10, 10)):
        """
        search a USD assets in USD Search service, load it into the scene and transform it.

        Args:
            text_prompt (str, ): Text prompt for 3D generation
            target_path (str, ): target path in current scene stage
            position (tuple, optional): Position to place the model
            scale (tuple, optional): Scale of the model

        Returns:
            dict: Dictionary with the task_id and prim_path
        """
        try:
            searcher3d = USDSearch3d()
            url = searcher3d.search(text_prompt)
            loader = USDLoader()
            # TODO： need to validate transform location
            prim_path = loader.load_usd_from_url(url, target_path, location=position, scale=scale)
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            loader.transform(prim=prim, position=position, scale=scale)

            return {"url": url, "prim_path": prim_path}
        except Exception as e:
            carb.log_error(f"Error in usd_search_3d_from_text: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # USDLoader.main()
    # USDLoader.test_tasks_load()
    # USDLoader.test_absolute_paths()
    # USDSearch3d.test_search_and_load()

    USDSearch3d.usd_search_3d_from_text(
        text_prompt="a rusty desk", target_path="/World/search_desk", position=(0, 0, 0), scale=(3, 3, 3)
    )
    # USDSearch3d.usd_search_3d_from_text(text_prompt="a rusty apple", target_path="/World/search_apple", position=(0, 0, 0), scale=(3, 3, 3))
    USDSearch3d.usd_search_3d_from_text(
        text_prompt="a rusty chair", target_path="/World/search_chair", position=(10, 0, 0), scale=(3, 3, 3)
    )
    USDSearch3d.usd_search_3d_from_text(
        text_prompt="a rusty sofa", target_path="/World/search_chair", position=(20, 0, 0), scale=(3, 3, 3)
    )
