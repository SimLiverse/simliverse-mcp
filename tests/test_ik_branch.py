"""The same tool pose has two wrist solutions; take the one the arm is near.

Measured on a KR210 carrying a carton to the same slot four times: three
cycles solved the wrist as [0.05, 2.17, 0.03], the fourth as [3.08, -2.18,
3.10], and the swing between them put the tool through two placed cartons.
"""

from __future__ import annotations

import numpy as np

from simliverse_sim.robots.manipulator import Manipulator

PICK = np.array([-1.327, -0.026, 0.716, 1.744, 1.391, 0.883])
CLEAN = np.array([-3.1, 0.87, 0.1, 0.05, 2.17, 0.03])
FLIPPED = np.array([-3.1, 0.87, 0.1, 3.19, -2.17, 3.17])
CLEAN_FAR_BASE = np.array([3.18, 0.87, 0.1, 0.05, 2.17, 0.03])


def _lula_like(seed):
    """Converge to whichever branch the seed's wrist is nearest, and keep the
    base on the seed's side of +/-pi - which is what Lula does."""
    wrist_flipped = abs(seed[3]) > np.pi / 2
    out = (FLIPPED if wrist_flipped else CLEAN).copy()
    out[0] = 3.18 if seed[0] > 0 else -3.1
    return out, True


def test_the_flipped_branch_is_traded_for_the_near_one():
    chosen = Manipulator._nearest_branch(_lula_like, FLIPPED.copy(), PICK.copy())
    np.testing.assert_allclose(chosen, CLEAN)


def test_a_base_wrapped_the_long_way_round_is_traded_too():
    chosen = Manipulator._nearest_branch(_lula_like, CLEAN_FAR_BASE.copy(), PICK.copy())
    assert chosen[0] == -3.1


def test_the_near_branch_is_kept_when_it_is_already_the_answer():
    chosen = Manipulator._nearest_branch(_lula_like, CLEAN.copy(), PICK.copy())
    np.testing.assert_allclose(chosen, CLEAN)


def test_a_solver_that_admits_nothing_else_changes_nothing():
    def only_this(seed):
        return np.zeros(6), False

    chosen = Manipulator._nearest_branch(only_this, FLIPPED.copy(), PICK.copy())
    np.testing.assert_allclose(chosen, FLIPPED)


def test_a_solver_that_raises_on_a_seed_is_survived():
    def touchy(seed):
        if seed[0] > 0:
            raise RuntimeError("seed outside joint limits")
        return _lula_like(seed)

    chosen = Manipulator._nearest_branch(touchy, FLIPPED.copy(), PICK.copy())
    np.testing.assert_allclose(chosen, CLEAN)


def test_arms_with_fewer_than_six_joints_are_left_alone():
    sol = np.array([0.1, 0.2, 0.3])
    assert Manipulator._nearest_branch(_lula_like, sol, np.zeros(3)) is sol
