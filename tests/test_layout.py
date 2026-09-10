"""A cell laid out to the arm, not the arm squeezed into the UR10's cell.

The measured cell picks 0.85 m out and stacks on a pallet whose near edge is
0.35 m from the base. Scaled to a UR5e that pick is outside the arm, and a
full pallet cannot get close enough to be inside it: the load has to change,
not just the distances.
"""

from __future__ import annotations

import pytest

import demo.ur10_palletizing as demo
from demo import scenario_sweep
from simliverse_sim.robots.library import _infer_motion_config


def test_the_ur10_layout_is_the_measured_cell():
    layout = demo.layout_for("ur10")
    assert layout["stop_x"] == pytest.approx(demo.STOP_X, abs=0.01)
    assert layout["offset_y"] == pytest.approx(demo.OFFSET_Y, abs=0.01)
    assert layout["pallet"] == "pallet"
    assert 0.65 <= layout["pallet_y"] <= 0.95


def test_a_small_arm_gets_a_tote_it_can_span():
    layout = demo.layout_for("ur5e")
    assert layout["pallet"] == "tote"
    assert layout["stop_x"] < demo.STOP_X
    assert 0.25 <= layout["deck"] < demo.DECK
    assert demo.layout_for("ur10e")["deck"] == demo.DECK
    # Near edge of the deck clear of the base.
    assert layout["pallet_y"] - demo.DECKS["tote"][1] / 2.0 >= 0.25


def test_a_big_arm_gets_a_pedestal_and_a_far_pallet():
    layout = demo.layout_for("kuka_kr210")
    assert layout["pedestal"] == 0.35
    assert layout["pallet"] == "pallet"
    assert layout["pallet_y"] > 1.2


def test_the_fanuc_carries_its_config_name():
    assert demo.layout_for("crx10ia_l")["rmp_config"] == "Fanuc_CRX10IAL"
    assert "rmp_config" not in demo.layout_for("ur10e")


def test_every_layout_is_something_build_accepts():
    import inspect

    accepted = set(inspect.signature(demo.build).parameters)
    for robot in scenario_sweep.ROBOTS:
        extra = set(demo.layout_for(robot)) - accepted
        assert not extra, "%s: build() has no %s" % (robot, sorted(extra))


def test_an_unknown_arm_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        demo.layout_for("abb_irb6700")


@pytest.mark.parametrize("robot,is_ur", [("ur10", True), ("ur5e", True), ("UR16e", True), ("kuka_kr210", False)])
def test_the_ur_family_shares_a_home(robot, is_ur):
    assert demo._is_ur(robot) is is_ur


def test_robot_scenarios_name_each_arm():
    names = [name for name, _ in scenario_sweep.robot_scenarios(["ur10", "ur5e"])]
    assert names == ["ur10", "ur5e"]


SUPPORTED = ["Fanuc_CRX10IAL", "UR3", "UR30", "UR10e", "Kuka_KR210", "Franka", "FrankaFR3"]


@pytest.mark.parametrize(
    "vendor,model,expected",
    [
        ("Fanuc", "crx10ia_l", "Fanuc_CRX10IAL"),
        ("Fanuc", "lrmate200id", None),
        ("UniversalRobots", "ur3", "UR3"),
        ("UniversalRobots", "ur30", "UR30"),
        ("UniversalRobots", "ur10e", "UR10e"),
        ("Kuka", "KR210", "Kuka_KR210"),
    ],
)
def test_the_config_matcher_finds_names_that_tokenise_differently(vendor, model, expected):
    """Fanuc_CRX10IAL is {crx, 10, ial}; the asset is {crx, 10, ia, l}.
    Whole-string equality catches it; ur3 still does not claim ur30."""
    assert _infer_motion_config(vendor, model, SUPPORTED) == expected
