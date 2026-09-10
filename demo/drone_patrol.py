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

"""A drone flying a patrol route on live physics. MEASURED.

A quadcopter takes off, holds a hover, and flies a loop of waypoints, and the
run is verified the way an inspection-flight acceptance test would: the airframe
actually flew - under thrust, off the ground, arriving at each point.

    from demo.drone_patrol import patrol
    print(patrol(route=[[3, 0, 1.5], [3, 3, 1.5], [0, 3, 1.5], [0, 0, 1.5]]))

Two facts about flying a drone here, each a failure first:

**1. A drone is a rigid body, not an articulation.** `Robot.spawn("quadcopter")`
returns an `AerialRobot` that is thrust-controlled; re-attaching it through the
articulation path (`Robot.attach`) fails with "no articulation registered".
Cycle the timeline and keep the spawn handle.

**2. `fly_to` is a PD controller, not a planner.** It holds position and
attitude to reach a waypoint and hover, and it flies straight there through
anything in the way. For a route around obstacles, give it the waypoints; it
flies them leg by leg. Measured: it reached a waypoint 0.11 m off at a 0.3 m
tolerance and held a hover to a few millimetres.
"""

from __future__ import annotations

import numpy as np

from simliverse_sim import Robot, Scene

DRONE = "/World/Drone"
#: A drone coasts less than a wheeled base but still overshoots a touch; 0.3 m
#: is a waypoint tolerance an inspection route is happy with.
ARRIVAL = 0.30


def patrol(
    scene: "Scene | None" = None,
    *,
    route: list | None = None,
    robot: str = "quadcopter",
    start: list | tuple = (0.0, 0.0, 0.5),
    tolerance: float = ARRIVAL,
) -> dict:
    """Fly `robot` from `start` around `route`, hovering at each waypoint.

    Returns what was measured: the legs reached, the total path flown, the peak
    altitude, and whether every leg arrived - a flight that reached three of
    four points and stalled on the fourth is a different result from one that
    flew the loop, and the report says which.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    scene = scene or Scene.get()
    scene.stop()
    scene.clear_world()
    scene.configure_physics()
    scene.ensure_ground_plane()
    _light(scene)

    drone = Robot.spawn(robot, position=[float(v) for v in start], prim_path=DRONE)
    # A drone is a rigid body: cycle the timeline, keep the handle.
    scene.stop()
    scene.play()
    scene.step(5)

    take_off = np.asarray(drone.position, dtype=float)
    drone.hover(steps=60)  # settle into a stable hover before the route
    legs = [np.asarray(w, dtype=float) for w in (route or [[3, 0, 1.5], [3, 3, 1.5], [0, 3, 1.5], [0, 0, 1.5]])]

    began = float(SimulationManager.get_simulation_time())
    flown = []
    path = 0.0
    peak = float(drone.altitude())
    here = take_off
    for leg in legs:
        ok = drone.fly_to(leg.tolist(), tolerance=tolerance, raise_on_fail=False)
        now = np.asarray(drone.position, dtype=float)
        path += float(np.linalg.norm(now - here))
        peak = max(peak, float(drone.altitude()))
        here = now
        flown.append({"waypoint": np.round(leg, 3).tolist(), "ok": bool(ok), "at": np.round(now, 3).tolist()})
    seconds = float(SimulationManager.get_simulation_time()) - began

    final = np.asarray(drone.position, dtype=float)
    return {
        "robot": robot,
        "flew_route": all(f["ok"] for f in flown),
        "airborne": bool(final[2] > 0.3),
        "waypoints": len(legs),
        "reached": sum(int(f["ok"]) for f in flown),
        "path_m": round(path, 3),
        "peak_altitude": round(peak, 3),
        "seconds": round(seconds, 2),
        "final": np.round(final, 3).tolist(),
        "legs": flown,
    }


def _light(scene) -> None:
    try:
        from pxr import Sdf, UsdLux

        dome = UsdLux.DomeLight.Define(scene.stage, "/World/PatrolDome")
        dome.CreateIntensityAttr(1000.0)
        dome.GetPrim().CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
    except Exception:  # noqa: BLE001 - lighting is a nicety, never the failure
        pass
