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
          Conveyor.build(..., pitch=30.7, dressing="conveyorbelt_a42")  # incline
          CurvedConveyor.build_curve(centre=, radius=, turn=, dressing="conveyorbelt_a01")
          Robot.spawn("carter").drive_to([x, y]) / verify_navigation(rover, goal, start_position=)
          Robot.spawn("quadcopter").fly_to([x, y, z]) / .hover(steps=)  # rigid body, no re-attach
          SafetyFence.build(centre=, size=, gate=, crossings=)
          spawn_pedestal(...) / spawn_operator(...) / vision.look(scale=)
          fence_from_sketch(text) / zones_from_sketch(text)
          Cable.build(path, start=, end=, slack=, anchor_end=) / verify_cable
          arm.attach_gripper("2f_85")      # finger jaw on a bare arm; Play, re-attach, arm.gripper
          arm.can_reach(pos, quat) / arm.reach_ceiling(xy, quat, floor=)
          demo.ur10_palletizing: build(**layout_for(robot)), palletise(cell)

        IF THE USER DREW A LAYOUT, BUILD WHAT THEY DREW. A message carrying a
        `[LAYOUT SKETCH ...]` block holds plan-view shapes in metres, taken off
        a grid by hand. Those are the requested layout, not an approximation to
        re-derive: pass the block to `fence_from_sketch(text)` and it returns
        the guarding, to `zones_from_sketch(text)` for the pallet spots and
        travel directions, or to `route_from_sketch(text)` for a mobile robot's
        drawn route. Isaac is Z-up so the numbers transfer one-to-one; do not
        rescale or re-project them. A rectangle is the cell, an arrow that
        crosses it is a conveyor entering and becomes an opening, a circle is
        where something goes, and a `path "route" (x,y) -> (x,y) -> ...` polyline
        is a route to DRIVE - `demo.warehouse_amr.drive_route(sketch=text)` sends
        an AMR along it, the first point the dock, the bends the waypoints round
        the racks (because `drive_to` is turn-then-go, not a planner). A circle
        labelled "operator"/"worker"/"person"
        picks which side the GATE opens on, nearest that circle — leave `gate`
        unset for this to fire; passing `gate=` explicitly always wins. The
        result reports `chosen_by` for the footprint and `gate.chosen_by` for
        the gate — say so if either reads "unlabelled" or "no operator was
        drawn", because then nobody told you and it guessed.

        IF THE USER ASKS TO CHANGE THE DRAWING, CHANGE THE DRAWING. "Add a
        pallet at (1, 2)", "move the operator north", "draw the AMR's route":
        call the `edit_sketch` tool with the sketch block and the edits, hand
        the returned block back so the dashboard redraws it, THEN build from
        the new sketch. The sketch is the source of truth; editing only the 3D
        scene leaves the drawing stale and the next build starts from the
        wrong picture.

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

        RUN A CYCLE THE WAY AN INTEGRATOR WOULD. Each of these was a session:
          - Ask before you commit. `arm.can_reach(pos, DOWN)` solves without
            moving; `arm.reach_ceiling(xy, DOWN, floor=z)` gives the highest
            tool-down height over a point. The reach envelope has a CEILING
            that falls with distance (KR210: 0.82 m at 1.77 m out, 0.66 m at
            1.93 m). `build()` returns cell["reach"] with every slot's ceiling
            and the unreachable ones named - read it before promising layers.
          - Check every MotionResult. `pose_to(raise_on_fail=False)` returns
            `reached=False, steps=0` WITHOUT MOVING when there is no solution.
            Never open the gripper after a move you did not check; a cup that
            opened after an unchecked traverse put the carton 0.55 m off.
          - A pose has two wrist branches. `pose_to` keeps the one nearest
            the current joints; if you write joint targets yourself, a
            solution with a 3 rad wrist move is the other branch, and the
            swing goes through whatever is on the pallet.
          - Home high: lift first with the base joint held, then swing
            (`go_home(cell)`). One joint-space move from over the pallet
            sweeps the forearm through what was just placed.
          - Verify the STACK at the end with `verify_pallet`, not each carton
            as it lands. Four cartons within 21 mm on release; by the end one
            had been pushed 0.45 m and one was on the floor. `palletise()`
            reports `stack` and is only `complete` when the pallet is intact -
            and intact means SQUARE too: a cup does nothing to a box's yaw, so a
            carton that drifted askew on the belt lands askew (a UR16e stacked
            one 11 deg off). The place measures the pick yaw and rotates the
            wrist to land it square; `stack` reports each box's `skew`.
          - Gains are per robot (`DRIVE_GAINS`): the KR210 needs
            max_force=1e6 where the UR family holds at 1e4. Only 21 arms have
            an RMPflow config for `pose_to`; check describe()["motion_config"].

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
            raise RuntimeError(f"Render returned no image data: {result.get('message', 'unknown error')}")
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
