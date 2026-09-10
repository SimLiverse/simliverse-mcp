"""The AMR delivery report, and the one number that must be shared.

Measured live: a Carter drove ~4 m to a goal in the Simple_Warehouse and
verify_navigation passed physics and motion. It once reported the run failed
while the drive reported success — because `drive_to(0.3)` and the verifier
(0.25) held different tolerances. `deliver` hands one tolerance to both.
"""

from __future__ import annotations

import inspect

import demo.warehouse_amr as amr


def test_one_tolerance_reaches_the_drive_and_the_check():
    """The bug that made a good run look failed: two tolerances. There must be
    a single knob, defaulting to ARRIVAL, passed to drive_to and verify."""
    src = inspect.getsource(amr.deliver)
    assert "tolerance: float = ARRIVAL" in src
    # It reaches both the leg drive and the verifier.
    assert "drive_to(leg.tolist(), tolerance=tol" in src
    assert "verify_navigation(rover" in src and "tolerance=tolerance" in src


def test_the_environments_are_named_by_a_real_usd_path():
    for name, usd in amr.ENVIRONMENTS.items():
        assert usd.startswith("/Isaac/Environments/")
        assert usd.endswith(".usd")
    assert "Simple_Warehouse" in amr.ENVIRONMENTS


def test_deliver_is_generic_over_goal_robot_and_environment():
    params = inspect.signature(amr.deliver).parameters
    for name in ("goal", "waypoints", "robot", "environment", "dock", "tolerance"):
        assert name in params, "deliver is not generic over %s" % name


def test_waypoints_are_driven_before_the_goal():
    """drive_to is not a planner, so the route is the caller's waypoints then
    the goal — the docstring and code must both say so."""
    src = inspect.getsource(amr.deliver)
    assert "waypoints or []" in src
    assert "not a planner" in amr.__doc__
