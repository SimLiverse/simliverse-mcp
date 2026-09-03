# MIT License
#
# Copyright (c) 2026 SimLiverse
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

"""Robot control and perception tools.

These are the two primitives that make manipulation tractable:

  * `run_control` executes Python against the `simliverse_sim` control library
    inside the simulator, with a namespace that persists across calls.
  * `capture_view` returns actual pixels, so a vision model can check its work.

Together they close the write -> run -> look -> fix loop. The 40-odd scene verbs
in the sibling modules remain the right tool for trivial edits; they are not the
right tool for coordinated motion.
"""

import base64
import json
from typing import Callable, List, Optional

from mcp.server.fastmcp import FastMCP, Image

from isaac_mcp.connection import IsaacCommandError, IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:
    @mcp.tool("run_control")
    def run_control(code: str, reset_namespace: bool = False) -> str:
        """Run Python against the simliverse_sim control library inside Isaac Sim.

        This is the primary tool for anything involving robot motion, grasping,
        throwing, or physics verification. Prefer it over set_joint_positions:
        commanding raw joint angles requires solving inverse kinematics, which
        this library does for you.

        The namespace persists between calls, so objects stay alive across steps:

            # call 1
            from simliverse_sim import Scene, Robot
            scene = Scene.get(); scene.configure_physics(); scene.play()
            robot = Robot.spawn("franka")
            ball = scene.spawn_rigid("/World/Ball", shape="Sphere", radius=0.04,
                                     position=[0.45, 0.0, 0.04], mass=0.05)
            print(robot.describe())

            # call 2 — `robot` and `ball` are still bound
            print("grasped:", robot.grasp(ball))

        Anything you print is returned to you, along with the full traceback if
        the code raises. Read the traceback and fix the code — that iteration is
        the intended workflow, not a failure mode.

        Key API (full reference: docs/control_library.md):
          Scene.get() / .configure_physics() / .play() / .step(n) / .settle(s)
          scene.clear_world()                  # stop() does NOT empty the stage
          scene.spawn_rigid(path, shape=, radius=, position=, mass=, friction=)
          scene.list_prims(root, recursive=True) / scene.find("ball")
          Robot.spawn(type, position=) / Robot(prim_path)
          robot.describe() -> joints, gripper, end effector, drive problems
          robot.move_ee_to([x, y, z])          # Cartesian, blocking
          robot.plan_to(pos, quat, robot_name=) + robot.follow(plan)   # cuMotion
          robot.gripper.open() / .close()
          robot.grasp(obj) -> bool             # approach, close, verify
          robot.is_grasping(obj) -> bool       # from contact reports
          robot.throw(obj, direction=, speed=) -> trajectory report
          verify_grasp(robot, obj) / verify_throw(obj, result) -> Report
          list_props(q) / find_prop(q) / spawn_prop(q, position=)
          Conveyor.build(...) / belt.dress("conveyorbelt_a05") / belt.start()
          SafetyFence.build(centre=, size=, gate=, crossings=)
          spawn_pedestal(...) / spawn_operator(...) / vision.look(scale=)
          fence_from_sketch(text) / zones_from_sketch(text)

        IF THE USER DREW A LAYOUT, BUILD WHAT THEY DREW. A message carrying a
        `[LAYOUT SKETCH ...]` block holds plan-view shapes in metres, taken off
        a grid by hand. Those are the requested layout, not an approximation to
        re-derive: pass the block to `fence_from_sketch(text)` and it returns
        the guarding, or to `zones_from_sketch(text)` for the pallet spots and
        travel directions. Isaac is Z-up so the numbers transfer one-to-one; do
        not rescale or re-project them. A rectangle is the cell, an arrow that
        crosses it is a conveyor entering and becomes an opening, a circle is
        where something goes. The result reports `chosen_by` — say so if it
        reads "largest, unlabelled", because then nobody told you which
        rectangle was the cell and it guessed.

        BUILD CELLS OUT OF REAL ASSETS. The library indexes 175 props,
        including 47 conveyor sections and 23 people. A cell authored from
        cubes and cylinders reads as a mock-up however good the physics is.
        Search first — `list_props("conveyor")`, `list_props("worker")` — and
        say so if a search comes up empty rather than quietly building the
        thing out of primitives.

        Placement traps, all measured, none of which raise:
          - Props are placed by their CENTRE and are large. A pallet is 1.21 m
            long, so pallet_y=0.60 puts its near edge at -0.005 and the arm's
            base inside the pallet.
          - A character's origin is NOT at its feet: the bound sits 0.12-0.16 m
            below it. Use guarding.spawn_operator, which measures and drops.
          - A conveyor prop carries at 0.767 m (its rollers). Its bounding box
            says 1.166 because that includes the side frames.
          - `size` means height, and only Cube has a size attribute; for a
            cylinder it maps to height. Getting this wrong is silent.
          - scene.stop() leaves every prim on the stage. Two cells in one
            session share it and the older one is still solid.
          - A halted belt is a sleeping belt: PhysX does not wake a body
            because the surface under it started moving. belt.start() nudges.

        LOOK BEFORE YOU BELIEVE IT. `vision.look()` renders four viewpoints,
        because every visual defect found in this cell was visible from one
        direction and invisible from the others. Pass `scale=` for a cell
        bigger than about a metre.

        Args:
            code: Python source to execute in the simulator process.
            reset_namespace: Discard all previously bound variables first. Use
                this after clearing the stage, since stale handles will point at
                prims that no longer exist.
        """
        try:
            conn = get_connection()
            result = conn.send_command(
                "control.run",
                {"code": code, "reset_namespace": reset_namespace},
            )
            return json.dumps(result, indent=2)
        except IsaacCommandError as e:
            # The traceback is the useful part — return it rather than str(e).
            return json.dumps(e.payload, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("capture_view")
    def capture_view(
        camera_path: Optional[str] = None,
        position: Optional[List[float]] = None,
        look_at: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
    ) -> Image:
        """Render the scene and return the image so you can actually look at it.

        Use this to verify visually what the numbers claim: whether the gripper
        is really around the object, whether the robot is posed sensibly, whether
        anything is intersecting or floating.

        Pair it with state assertions rather than trusting it alone — a render
        can look right while contact forces say the grip is slipping.

        Args:
            camera_path: Existing camera prim to render from. If omitted, a
                temporary camera is placed using position/look_at.
            position: [x, y, z] camera position. Defaults to a three-quarter view.
            look_at: [x, y, z] point to aim at. Defaults to the world origin.
            resolution: [width, height]. Defaults to [1280, 720].
        """
        conn = get_connection()
        params = {}
        if camera_path:
            params["camera_path"] = camera_path
        if position:
            params["position"] = position
        if look_at:
            params["look_at"] = look_at
        if resolution:
            params["resolution"] = resolution

        result = conn.send_command("control.capture_view", params)
        encoded = result.get("image_base64")
        if not encoded:
            raise RuntimeError(
                f"Render returned no image data: {result.get('message', 'unknown error')}"
            )
        return Image(data=base64.b64decode(encoded), format=result.get("format", "png"))

    @mcp.tool("observe")
    def observe(
        prim_paths: Optional[List[str]] = None,
        robot_paths: Optional[List[str]] = None,
        steps: int = 0,
    ) -> str:
        """Step physics and report measured state for the prims you care about.

        Returns positions, velocities, and real contact lists — contacts come
        from the PhysX contact-report API, so an empty list means genuinely no
        contact rather than "not implemented".

        Args:
            prim_paths: Rigid bodies to report on.
            robot_paths: Robots to report joint state and end-effector pose for.
            steps: Physics steps to advance before measuring. 0 measures now.
        """
        try:
            conn = get_connection()
            result = conn.send_command(
                "control.observe",
                {
                    "prim_paths": prim_paths or [],
                    "robot_paths": robot_paths or [],
                    "steps": steps,
                },
            )
            return json.dumps(result, indent=2)
        except IsaacCommandError as e:
            return json.dumps(e.payload, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
