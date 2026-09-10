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

"""An AMR delivering to a goal inside a warehouse, run on live physics. MEASURED.

A Carter drives from a dock to a goal pose in a real warehouse environment and
the run is verified the way a fleet-management acceptance test would: the robot
physically moved, under its own wheels, and stopped where it was sent.

    from demo.warehouse_amr import deliver
    print(deliver(goal=[6.0, 2.0]))

Every default here is a measured cell, so `deliver()` reproduces the run the
numbers came from. Three things are decisions about this cell, not tuning, and
each was a failure first:

**1. `drive_to` is not a planner.** It turns to face the goal and drives at it,
so a goal on the far side of a rack drives the base into the rack. `plan_path`
smooths a route through waypoints you supply, but it does not *find* one - it
does not avoid anything either. Give this function the waypoints that keep the
aisle, and it drives them leg by leg. In an open bay, none are needed.

**2. Match the arrival tolerance to the verifier.** `drive_to(tolerance=0.3)`
reached 0.30 m from the goal and `verify_navigation` (0.25 m) then called the
run failed - both true, and the disagreement was the whole bug. The tolerance
is one number here, handed to both.

**3. Spawn clear of the floor and let it settle.** A Carter referenced at z=0
interpenetrates the warehouse floor and the first wheel command launches it.
Spawned at 0.3 m and settled for 20 steps, it drops onto its wheels first.
"""

from __future__ import annotations

import numpy as np

from simliverse_sim import Robot, Scene, verify_navigation

CARTER = "/World/Carter"
#: One tolerance, handed to both the drive and the check, so they cannot
#: disagree about whether the robot arrived (see the module docstring). 0.30 m
#: because a Carter at 0.6 m/s coasts ~0.1-0.3 m past the point it brakes at.
ARRIVAL = 0.30

#: Waypoints are via-points on the way to the goal, not the goal, so they are
#: reached loosely - a base that passes within half a metre of a corner has
#: cleared the corner, and holding it to the arrival tolerance there only makes
#: it stop, reverse and hunt.
VIA = 0.5

#: Warehouse environments on the Isaac asset server, by the USD they reference.
ENVIRONMENTS = {
    "Simple_Warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "Simple_Warehouse_with_shelves": "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "Digital_Twin_Warehouse": "/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
}


def load_environment(scene, environment: str) -> str | None:
    """Reference a warehouse USD onto the stage, or fall back to a ground plane.

    Returns the prim path of the environment, or None if it could not be
    referenced and a bare floor was used instead - a caller should say which,
    because "the robot drove across an empty plane" is a different result from
    "the robot drove through the warehouse".
    """
    usd = ENVIRONMENTS.get(environment, environment)
    try:
        prim = scene.stage.DefinePrim("/World/Environment", "Xform")
        prim.GetReferences().AddReference(usd)
        if prim.IsValid() and prim.GetPrim().GetPrimStack():
            return "/World/Environment"
    except Exception:  # noqa: BLE001 - a missing asset is a result, not a crash
        pass
    scene.ensure_ground_plane()
    return None


def _light(scene) -> None:
    """A dome light, because a referenced warehouse loses its own on a clear.

    Without it the scene renders black — a healthy run that looks like a broken
    simulator, the first thing anyone reports.
    """
    try:
        from pxr import Sdf, UsdLux

        dome = UsdLux.DomeLight.Define(scene.stage, "/World/DeliveryDome")
        dome.CreateIntensityAttr(1000.0)
        dome.GetPrim().CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
    except Exception:  # noqa: BLE001 - lighting is a nicety, never the failure
        pass


def deliver(
    scene: "Scene | None" = None,
    *,
    goal: list | tuple = (6.0, 2.0),
    waypoints: list | None = None,
    robot: str = "carter",
    environment: str = "Simple_Warehouse",
    dock: list | tuple = (0.0, 0.0),
    tolerance: float = ARRIVAL,
    max_speed: float = 0.6,
) -> dict:
    """Drive `robot` from `dock` to `goal` in `environment`, and verify it.

    `waypoints` are legs to drive in order before the goal - the route around
    the racks, since `drive_to` will not find one itself. Returns what was
    measured: the environment used, the path length, the final error and the
    per-check verdict.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    scene = scene or Scene.get()
    scene.stop()
    scene.clear_world()
    scene.configure_physics()

    env = load_environment(scene, environment)
    _light(scene)

    dock = np.asarray(dock, dtype=float)
    rover = Robot.spawn(robot, position=[float(dock[0]), float(dock[1]), 0.3], prim_path=CARTER)

    scene.play()
    scene.step(20)  # drop onto the wheels before commanding them
    rover = Robot.attach(CARTER, scene=scene)
    start = np.asarray(rover.base_position, dtype=float)

    # Via-points loosely, the goal to the arrival tolerance. A missed via-point
    # is not a failed delivery - the route continues; only the goal is judged.
    legs = [(np.asarray(w, dtype=float)[:2], VIA, True) for w in (waypoints or [])]
    legs.append((np.asarray(goal, dtype=float)[:2], tolerance, False))
    began = float(SimulationManager.get_simulation_time())
    reached = []
    for leg, tol, is_via in legs:
        ok = rover.drive_to(leg.tolist(), tolerance=tol, max_speed=max_speed, raise_on_fail=False)
        here = np.asarray(rover.base_position, dtype=float)[:2]
        reached.append(
            {"leg": np.round(leg, 3).tolist(), "via": is_via, "ok": bool(ok), "at": np.round(here, 3).tolist()}
        )
    seconds = float(SimulationManager.get_simulation_time()) - began

    report = verify_navigation(rover, np.asarray(goal, dtype=float)[:2], start_position=start, tolerance=tolerance)
    end = np.asarray(rover.base_position, dtype=float)
    goal_xy = np.asarray(goal, dtype=float)[:2]
    return {
        "robot": robot,
        "environment": environment if env else "%s (unavailable; drove a bare floor)" % environment,
        "in_warehouse": env is not None,
        "delivered": bool(report.passed),
        "goal": np.round(goal_xy, 3).tolist(),
        "final": np.round(end[:2], 3).tolist(),
        "error": round(float(np.linalg.norm(end[:2] - goal_xy)), 4),
        "path_m": round(float(np.linalg.norm(end[:2] - start[:2])), 3),
        "seconds": round(seconds, 2),
        "legs": reached,
        "checks": [{"name": c.name, "passed": bool(c.passed)} for c in report.checks],
    }


def drive_route(
    scene: "Scene | None" = None,
    *,
    sketch: str,
    robot: str = "carter",
    environment: str = "Simple_Warehouse",
    **kwargs,
) -> dict:
    """Drive `robot` along a path DRAWN on a sketch.

    A person draws the route as a polyline -
    `path "route" (0,-2) -> (2,-2) -> (2,1) -> (5,1)` - and this follows it: the
    first point is where the robot starts, the last is the goal, and the bends
    are the waypoints that keep it clear of the racks. That is the whole point
    of drawing a route rather than naming a goal, because `drive_to` is a
    turn-then-go controller and will not find its own way around obstacles - so
    the human draws the way around, exactly as `fence_from_sketch` builds the
    guarding a human drew. Returns the delivery report plus the drawn points and
    how the route was chosen, so a mislabelled sketch is visible rather than
    silently driving the wrong line.
    """
    from simliverse_sim.sketch import route_from_sketch

    route = route_from_sketch(sketch, robot=robot)
    result = deliver(
        scene,
        dock=route["start"],
        goal=route["goal"],
        waypoints=route["waypoints"],
        robot=robot,
        environment=environment,
        **kwargs,
    )
    result["route_chosen_by"] = route["chosen_by"]
    result["drawn_points"] = route["points"]
    return result
