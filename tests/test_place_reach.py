"""A place that cannot get there must not let go.

Measured on a KR210: the traverse pose over the far pallet column had no IK
solution, `pose_to(raise_on_fail=False)` returned without moving, the descent
went part-way, and the cup opened 0.55 m from the slot. The run reported a
placement error. It was a move that never happened, and nothing had checked.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import demo.ur10_palletizing as demo
from simliverse_sim.robots.manipulator import MotionResult

DOWN = demo.DOWN


class FakeScene:
    def settle(self, seconds):
        pass


class FakeCup:
    tip_offset = 0.04

    def __init__(self):
        self.holding = True
        self.gripped_objects = ["/World/Box0"]
        self.opened = 0

    def open(self, settle_steps=0):
        self.opened += 1
        self.holding = False
        self.gripped_objects = []


class FakeArm:
    """Reaches anything below `ceiling`; above it the solver says no."""

    def __init__(self, ceiling=1.0):
        self.ceiling = ceiling
        self.scene = FakeScene()
        self.moves = []
        self.ee_position = np.zeros(3)

    def can_reach(self, position, orientation=None):
        return float(position[2]) <= self.ceiling

    def pose_to(self, position, orientation=None, **kwargs):
        target = [float(v) for v in position]
        self.moves.append(target)
        if not self.can_reach(target, orientation):
            return MotionResult(False, 0, float("inf"), target, 180.0)
        self.ee_position = np.asarray(target)
        return MotionResult(True, 100, 0.001, target, 0.5)


def _reach_ceiling(self, xy, orientation, *, floor, limit=2.0, resolution=0.005):
    from simliverse_sim.robots.manipulator import Manipulator

    return Manipulator.reach_ceiling(self, xy, orientation, floor=floor, limit=limit, resolution=resolution)


FakeArm.reach_ceiling = _reach_ceiling


class FakeBox:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)
        self.prim_path = "/World/Box0"
        self.speed = 0.0


def _slot(index=0, x=-1.88, y=0.42, z=0.2175):
    return {
        "index": index,
        "place": [x, y, z],
        "approach": [x, y, z + 0.30],
        "rest": [x, y, z],
    }


def _cell(arm, cup, box=0.15):
    return {
        "arm": arm,
        "cup": cup,
        "belt": SimpleNamespace(boxes=[FakeBox([0, 0, 0])]),
        "slots": [_slot(0), _slot(1, x=-1.72)],
        "box_size": box,
        "spec": {"robot": "kuka_kr210"},
    }


def test_the_cup_stays_shut_when_the_traverse_finds_no_solution():
    arm, cup = FakeArm(ceiling=0.0), FakeCup()
    result = demo.place_on_slot(_cell(arm, cup), _slot(), box=FakeBox([0, 0, 1]))
    assert result["placed"] is False
    assert "out of reach" in result["reason"]
    assert cup.opened == 0
    assert arm.moves == [], "an unreachable slot is known before the arm moves"


def test_the_traverse_drops_under_the_ceiling_instead_of_failing():
    """0.764 m wanted, 0.664 m ceiling: the KR210 numbers."""
    arm, cup = FakeArm(ceiling=0.664), FakeCup()
    box = FakeBox([-1.88, 0.42, 0.2175])
    result = demo.place_on_slot(_cell(arm, cup), _slot(), box=box)
    assert result["placed"] is True, result
    traverse = arm.moves[0]
    assert traverse[2] <= 0.664 - demo.CEILING_MARGIN + 1e-6
    assert traverse[2] >= 0.60
    assert cup.opened == 1


def test_a_reachable_wanted_height_is_used_as_is():
    arm, cup = FakeArm(ceiling=1.5), FakeCup()
    box = FakeBox([-1.88, 0.42, 0.2175])
    demo.place_on_slot(_cell(arm, cup), _slot(), box=box)
    wanted = max(0.2175 + 0.30, 0.55) + cup.tip_offset + 0.15 / 2.0
    assert arm.moves[0][2] == pytest.approx(wanted)


def test_a_move_that_stops_short_does_not_release():
    class ShortArm(FakeArm):
        def pose_to(self, position, orientation=None, **kwargs):
            target = [float(v) for v in position]
            self.moves.append(target)
            if len(self.moves) == 2:  # the descent
                return MotionResult(False, 800, 0.12, target, 3.0)
            return MotionResult(True, 100, 0.001, target, 0.5)

    arm, cup = ShortArm(ceiling=1.5), FakeCup()
    result = demo.place_on_slot(_cell(arm, cup), _slot(), box=FakeBox([0, 0, 1]))
    assert result["placed"] is False
    assert "descent not reached" in result["reason"]
    assert "0.120 m short" in result["reason"]
    assert cup.opened == 0
    assert len(arm.moves) == 3, "it goes back up, still holding"


def test_a_millimetre_miss_still_counts_as_arrived():
    assert demo._arrived(MotionResult(False, 900, 0.006, [0, 0, 0], 0.4))
    assert not demo._arrived(MotionResult(False, 900, 0.05, [0, 0, 0], 0.4))
    assert demo._describe(MotionResult(False, 0, float("inf"), [0, 0, 0], 180.0)).startswith("no IK solution")


def test_slot_reach_reports_the_ceiling_per_slot_before_any_carton():
    arm, cup = FakeArm(ceiling=0.664), FakeCup()
    cell = _cell(arm, cup)
    cell["slots"].append(_slot(2, z=0.7))  # a third layer, above the ceiling
    reach = demo.slot_reach(cell)
    assert reach["reachable"] == [0, 1]
    assert reach["unreachable"] == [2]
    assert reach["ceilings"][0] == pytest.approx(0.664, abs=0.006)
    assert reach["ceilings"][2] is None


def test_homing_lifts_before_it_swings():
    """From over the pallet (base at -3.1 rad) the first stage keeps the base
    where it is and only straightens the arm; the swing comes second."""

    class ParkedArm(FakeArm):
        dof = 6

        def __init__(self):
            super().__init__()
            self.joint_positions = np.array([-3.1, 1.01, 0.04, 0.04, 2.09, 0.02])
            self.commands = []

        def set_joint_positions(self, positions, settle_steps=0, **kwargs):
            self.commands.append(list(positions))
            self.joint_positions = np.asarray(positions, dtype=float)

    arm = ParkedArm()
    demo.go_home(_cell(arm, FakeCup()))
    assert len(arm.commands) == 2
    lift, swing = arm.commands
    assert lift[0] == pytest.approx(-3.1)
    assert lift[1:] == [0.0] * 5
    assert swing == [0.0] * 6


@pytest.mark.parametrize(
    "robot,max_force",
    [("ur10", 1.0e4), ("ur5e", 1.0e4), ("ur16e", 1.0e4), ("kuka_kr210", 1.0e6), ("crx10ia_l", 1.0e6)],
)
def test_drive_gains_are_per_robot(robot, max_force):
    assert demo.drive_gains(robot)["max_force"] == max_force
