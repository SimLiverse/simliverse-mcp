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

"""Robot creation and control MCP tools."""

import json
from typing import TYPE_CHECKING, Callable, List, Optional

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:

    @mcp.tool("create_robot")
    def create_robot(
        robot_type: str = "franka",
        position: Optional[List[float]] = None,
        name: Optional[str] = None,
        prim_path: Optional[str] = None,
    ) -> str:
        """Create a robot in the scene from the Isaac Sim asset library.

        Supports fuzzy matching — e.g. "franka", "spot", "g1", "go1".
        Call list_available_robots first to see all available robots.
        Call create_physics_scene before creating robots.

        Returns prim_path, robot_key, joint_names, and num_dof so you can
        immediately use set_joint_positions without a follow-up get_robot_info call.

        CRITICAL: The returned `prim_path` is the exact path where the robot was created.
        You MUST save and use this exact string for any future operations on this robot.
        DO NOT GUESS OR MODIFY IT.

        Args:
            robot_type: Robot name or search term. Fuzzy matched against available robots.
            position: [x, y, z] world position.
            name: Custom name for the robot prim.
            prim_path: Exact USD prim path (e.g. "/World/Franka"). Overrides name-based path.
        """
        try:
            conn = get_connection()
            params = {"robot_type": robot_type}
            if position:
                params["position"] = position
            if name:
                params["name"] = name
            if prim_path:
                params["prim_path"] = prim_path
            result = conn.send_command("robots.create", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("list_available_robots")
    def list_available_robots(
        search: Optional[str] = None,
        manufacturer: Optional[str] = None,
    ) -> str:
        """List robots available on the Isaac Sim asset server.

        Around 200 robots are discovered, so an unfiltered call returns an index
        of keys plus manufacturer counts rather than full descriptions. Pass
        `search` or `manufacturer` to narrow it, and a small result comes back
        in full with asset paths and descriptions.

        Search matches the key, description and manufacturer, so
        `search="franka"`, `search="humanoid"` and `manufacturer="Unitree"` all
        work. Prefer searching to listing everything — the full list is several
        thousand tokens and stays in the conversation afterwards.

        Args:
            search: Substring to match against key, description or manufacturer.
            manufacturer: Restrict to one vendor, e.g. "FrankaRobotics".
        """
        try:
            conn = get_connection()
            params = {}
            if search:
                params["search"] = search
            if manufacturer:
                params["manufacturer"] = manufacturer
            result = conn.send_command("robots.list", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("refresh_robot_library")
    def refresh_robot_library() -> str:
        """Force re-scan the asset server for available robots. Use this if new robot assets were added."""
        try:
            conn = get_connection()
            result = conn.send_command("robots.refresh")
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("inspect_robot")
    def inspect_robot(prim_path: str) -> str:
        """Inspect any robot in the scene (KUKA, Franka, Universal Robots, Boston Dynamics, Unitree, etc.)
        to dynamically discover its kinematics, joint names, degrees of freedom, gripper/finger joints,
        and end-effector links without assuming a specific robot brand.

        Use this tool whenever you interact with a robot to understand how to drive its joints.

        Args:
            prim_path: The prim path of the robot (e.g. '/World/kuka', '/World/Franka', '/World/ur10').
        """
        try:
            conn = get_connection()
            info_res = conn.send_command("robots.get_info", {"prim_path": prim_path})
            
            joint_names = info_res.get("joint_names", [])
            num_dof = info_res.get("num_dof", len(joint_names))
            joint_types = info_res.get("joint_types", {})
            joint_limits = info_res.get("joint_limits", {})

            # Dynamically detect arm vs mobile base vs quadruped vs humanoid
            finger_joints = [
                j for j in joint_names
                if any(k in j.lower() for k in ["finger", "gripper", "knuckle", "thumb", "jaw"])
            ]
            arm_joints = [
                j for j in joint_names
                if j not in finger_joints and any(k in j.lower() for k in ["joint", "shoulder", "elbow", "wrist", "arm", "a1", "a2", "a3", "a4", "a5", "a6", "a7"])
            ]
            wheel_joints = [
                j for j in joint_names
                if any(k in j.lower() for k in ["wheel", "steer", "drive"])
            ]

            category = "manipulator"
            if len(wheel_joints) >= 2 and len(arm_joints) == 0:
                category = "wheeled_vehicle"
            elif num_dof >= 12 and any(k in prim_path.lower() for k in ["dog", "go1", "go2", "spot", "anymal", "quad"]):
                category = "quadruped"
            elif num_dof >= 16 and any(k in prim_path.lower() for k in ["h1", "g1", "humanoid"]):
                category = "humanoid"

            result = {
                "status": "success",
                "prim_path": prim_path,
                "category": category,
                "num_dof": num_dof,
                "arm_joints": arm_joints,
                "finger_joints": finger_joints,
                "has_gripper": len(finger_joints) > 0,
                "joint_types": joint_types,
                "joint_limits": joint_limits,
                "raw_info": info_res,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("get_robot_info")
    def get_robot_info(prim_path: str) -> str:
        """Get robot joint information including names, DOF count, joint types, and limits.

        Call this after create_robot to understand the robot's kinematic structure.
        Returns joint names ordered by DOF index, joint types (revolute/prismatic),
        and joint limits (degrees for revolute, meters for prismatic).

        CRITICAL: `prim_path` must be an exact match to a known path in the scene.
        Do not guess. Use the exact string returned by `create_robot` or `get_scene_info`.

        Args:
            prim_path: The prim path of the robot.
        """
        try:
            conn = get_connection()
            result = conn.send_command("robots.get_info", {"prim_path": prim_path})
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("set_joint_positions")
    def set_joint_positions(
        prim_path: str, joint_positions: List[float], joint_indices: Optional[List[int]] = None
    ) -> str:
        """Set target joint positions on a robot via ArticulationAction.

        Units: radians for revolute joints, meters for prismatic joints (e.g. gripper fingers).
        Use get_robot_info to discover joint names, types, and limits first.
        If you only want to set a subset of joints, provide joint_indices.
        Wait for a few steps after spawning before setting targets.

        CRITICAL: `prim_path` must be an exact match to a known path in the scene.
        Do not guess. Use the exact string returned by `create_robot` or `get_scene_info`.
        CRITICAL JSON FORMAT: `joint_positions` MUST be a real JSON array of numbers (e.g. `[0.1, 0.2]`).
        NEVER pass a string containing Python code like `"[random.uniform(...)]"`. You must
        evaluate any logic yourself and pass only the final raw numerical values.

        Args:
            prim_path: The prim path of the robot.
            joint_positions: List of joint target values. MUST be actual numbers, not Python code.
            joint_indices: Optional subset of joint indices to control.
        """
        try:
            conn = get_connection()
            params = {"prim_path": prim_path, "joint_positions": joint_positions}
            if joint_indices:
                params["joint_indices"] = joint_indices
            result = conn.send_command("robots.set_joints", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("get_joint_positions")
    def get_joint_positions(prim_path: str) -> str:
        """Read current joint positions from a robot.

        Units: radians for revolute joints, meters for prismatic joints.
        Joint order matches the joint_names from get_robot_info.
        For a combined step-and-read, prefer step_simulation with observe_joints.

        Args:
            prim_path: The prim path of the robot.
        """
        try:
            conn = get_connection()
            result = conn.send_command("robots.get_joints", {"prim_path": prim_path})
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
