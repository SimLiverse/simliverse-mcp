"""The cross-morphology catalogue: bodies and floors, not just the palletiser.

It points each scenario at a verified demo (deliver, patrol) rather than
copying the control code, so the harness stays a catalogue.
"""

from __future__ import annotations

import demo.fleet_scenarios as fs


def test_the_fleet_crosses_every_mobile_and_every_floor():
    m = fs.mobile_scenarios()
    assert len(m) == len(fs.MOBILE_ROBOTS) * len(fs.ENVIRONMENTS)
    pairs = {(s["robot"], s["environment"]) for s in m}
    assert pairs == {(r, e) for r in fs.MOBILE_ROBOTS for e in fs.ENVIRONMENTS}
    for s in m:
        assert s["kind"] == "mobile"
        assert len(s["goal"]) == 2
        assert s["waypoints"]  # a via-point, since drive_to is not a planner


def test_the_fleet_includes_a_drone_route():
    d = fs.drone_scenarios()
    assert d and all(s["kind"] == "drone" for s in d)
    assert len(d[0]["route"]) == 4  # a square patrol
    assert all(len(w) == 3 for w in d[0]["route"])  # x, y, z


def test_the_fleet_is_deterministic_under_a_seed():
    assert [s["name"] for s in fs.fleet(seed=5)] == [s["name"] for s in fs.fleet(seed=5)]
    assert fs.fleet(seed=1) != fs.fleet(seed=2)


def test_the_environments_are_real_warehouse_names():
    from demo.warehouse_amr import ENVIRONMENTS as KNOWN

    for env in fs.ENVIRONMENTS:
        assert env in KNOWN, "%s is not a known warehouse environment" % env


def test_the_report_totals_the_passes():
    rows = [
        {"name": "a", "kind": "mobile", "ok": True, "detail": ""},
        {"name": "b", "kind": "drone", "ok": False, "detail": "x"},
    ]
    text = fs.report(rows)
    assert "1/2 passed" in text
    assert "OK" in text and "FAIL" in text
