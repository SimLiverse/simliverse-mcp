"""The drone patrol report and its two documented facts.

Measured live: a quadcopter took off, held a hover to a few millimetres, and
flew a four-waypoint loop, reaching each within 0.3 m. It is a rigid body, so
it is flown from the spawn handle after a timeline cycle, not re-attached
through the articulation path; and fly_to is a PD controller, not a planner.
"""

from __future__ import annotations

import inspect

import demo.drone_patrol as dp


def test_patrol_is_generic_over_route_robot_and_start():
    params = inspect.signature(dp.patrol).parameters
    for name in ("route", "robot", "start", "tolerance"):
        assert name in params, "patrol is not generic over %s" % name


def test_a_drone_is_flown_from_the_spawn_handle_not_reattached():
    """Re-attaching a rigid-body drone through the articulation path fails; the
    demo keeps the spawn handle after a timeline cycle."""
    src = inspect.getsource(dp.patrol)
    assert "Robot.spawn(robot" in src
    assert "Robot.attach" not in src
    assert "rigid body" in dp.patrol.__doc__ or "rigid body" in dp.__doc__


def test_fly_to_is_documented_as_not_a_planner():
    assert "not a planner" in dp.__doc__


def test_the_report_says_whether_it_flew_the_whole_route():
    src = inspect.getsource(dp.patrol)
    for key in ("flew_route", "airborne", "reached", "peak_altitude", "path_m"):
        assert '"%s"' % key in src, "the report omits %s" % key
