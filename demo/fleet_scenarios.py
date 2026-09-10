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

"""One harness across the whole fleet, not just the palletiser.

The palletising sweep (`scenario_sweep`) crosses arms, grippers and cartons.
This crosses the *other* morphologies the agent has to handle - mobile bases
across warehouse floors, and drones on routes - so a run answers "does the
library still work on a different body, in a different environment" and not
just "does the one cell we tuned still pass".

Each scenario is a small dict pointing at the verified demo that runs it, so
the harness stays a catalogue, not a second copy of the control code.

    from demo.fleet_scenarios import fleet, run_fleet, report
    print(report(run_fleet(fleet(seed=1))))
"""

from __future__ import annotations

import time
from typing import Any

#: The mobile bases that drive with `WheeledRobot.drive_to`, measured to work.
MOBILE_ROBOTS = ["carter", "turtlebot3", "novacarter"]

#: Warehouse and factory floors an AMR can be dropped into (see warehouse_amr).
ENVIRONMENTS = ["Simple_Warehouse", "Digital_Twin_Warehouse"]

#: Airframes flown with `AerialRobot.fly_to`.
DRONES = ["quadcopter"]


def mobile_scenarios(
    robots: list[str] | None = None, environments: list[str] | None = None, *, seed: int = 0
) -> list[dict[str, Any]]:
    """Every (mobile robot x environment) pair, each with a goal and a via-point."""
    import random as _random

    rng = _random.Random(seed)
    out = []
    for robot in robots or MOBILE_ROBOTS:
        for env in environments or ENVIRONMENTS:
            goal = [round(rng.uniform(3.0, 6.0), 1), round(rng.uniform(-2.0, 3.0), 1)]
            out.append(
                {
                    "kind": "mobile",
                    "name": "%s @ %s -> %s" % (robot, env, goal),
                    "robot": robot,
                    "environment": env,
                    "goal": goal,
                    "waypoints": [[round(goal[0] / 2, 1), 0.0]],
                }
            )
    return out


def drone_scenarios(robots: list[str] | None = None, *, seed: int = 0) -> list[dict[str, Any]]:
    """Drones flying a square patrol at a sampled altitude."""
    import random as _random

    rng = _random.Random(seed)
    out = []
    for robot in robots or DRONES:
        side = round(rng.uniform(2.0, 4.0), 1)
        alt = round(rng.uniform(1.2, 2.0), 1)
        route = [[side, 0, alt], [side, side, alt], [0, side, alt], [0, 0, alt]]
        out.append(
            {"kind": "drone", "name": "%s square %gm @ %gm" % (robot, side, alt), "robot": robot, "route": route}
        )
    return out


def fleet(*, seed: int = 0) -> list[dict[str, Any]]:
    """The whole cross-morphology catalogue: every mobile x floor, plus drones."""
    return mobile_scenarios(seed=seed) + drone_scenarios(seed=seed)


def run_fleet(scenarios: list[dict[str, Any]] | None = None, *, scene=None) -> list[dict[str, Any]]:
    """Run each scenario through its verified demo, one row of result per cell."""
    from demo.drone_patrol import patrol
    from demo.warehouse_amr import deliver
    from simliverse_sim import Scene

    scene = scene or Scene.get()
    rows = []
    for s in scenarios if scenarios is not None else fleet():
        started = time.time()
        row = {"name": s["name"], "kind": s["kind"]}
        try:
            if s["kind"] == "mobile":
                r = deliver(
                    scene, robot=s["robot"], environment=s["environment"], goal=s["goal"], waypoints=s.get("waypoints")
                )
                row.update(ok=bool(r["delivered"]), detail="in_warehouse=%s err=%.2f" % (r["in_warehouse"], r["error"]))
            elif s["kind"] == "drone":
                r = patrol(scene, robot=s["robot"], route=s["route"])
                row.update(
                    ok=bool(r["flew_route"] and r["airborne"]),
                    detail="reached %d/%d peak %.1f" % (r["reached"], r["waypoints"], r["peak_altitude"]),
                )
            else:
                row.update(ok=False, detail="unknown kind")
        except Exception as exc:  # noqa: BLE001 - a broken scenario is a result
            row.update(ok=False, detail="%s: %s" % (type(exc).__name__, str(exc)[:120]))
        row["wall_s"] = round(time.time() - started, 1)
        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]]) -> str:
    lines = ["%-34s %-7s %-8s %s" % ("scenario", "kind", "ok", "detail")]
    lines.append("-" * 74)
    for r in rows:
        lines.append(
            "%-34s %-7s %-8s %s" % (r["name"][:34], r["kind"], "OK" if r.get("ok") else "FAIL", r.get("detail", ""))
        )
    passed = sum(int(r.get("ok", False)) for r in rows)
    lines.append("-" * 74)
    lines.append("%d/%d passed" % (passed, len(rows)))
    return "\n".join(lines)
