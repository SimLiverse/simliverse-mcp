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

"""`Manipulator.throw` -- the release decision, and the result it reports.

Physics is faked here, because what broke was never physics. A live Franka run
asked for `speed=2.8` and released the ball at 0.145 m/s. `speed` only ever set
the *distance* of the swing (`speed * 0.35 + windup`), and for any interesting
value that distance puts the swing target outside the arm's reach -- 1.23 m for
an arm that reaches 0.85 m. RMPflow damps hard against the workspace boundary,
so the arm crawled to its geometric release point and let go at a standstill.
The ball left at 0.099 m/s and rolled; nine of the twelve reported trajectory
samples were the ball sliding along the floor at z=0.03, and the function called
that a 0.5565 m throw.

The agent under test measured the ball speed, correctly concluded the throw had
not happened, and spent the rest of its turn budget hand-rolling a swing out of
`_policy` internals. It got there, at 2x the cost of the task before it.

A second call returned `released: True` beside `still_held: True` and
`apex_height: -inf`, which is not a state the world can be in: the gripper had
been commanded open but never given a physics step to open in.

The fake arm below is a speed profile. The release decision is allowed to depend
on hand speed and geometry, and on nothing else, so a profile is enough to test
it.
"""

import numpy as np
import pytest

from simliverse_sim.robots.manipulator import Manipulator, MotionError

DT = 1.0 / 60.0
GROUND = 0.03


class FakeGripper:
    """A gripper that opens when told, unless it is built stuck."""

    open_width = 0.04

    def __init__(self, arm, *, opens=True):
        self.arm = arm
        self.opens = opens
        self.set_position_calls = []

    def set_position(self, width, settle_steps=0):
        self.set_position_calls.append(width)
        if self.opens:
            self.arm.holding = False


class FakeObject:
    """Ballistic once released, and it rolls where it lands."""

    def __init__(self, arm):
        self.arm = arm
        self._position = np.array([0.45, 0.0, 0.5])
        self._velocity = np.zeros(3)

    @property
    def position(self):
        if self.arm.holding:
            return self.arm.ee_position
        return self._position.copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self._velocity))

    def tick(self):
        if self.arm.holding:
            self._position = self.arm.ee_position
            self._velocity = self.arm.velocity.copy()
            return
        self._velocity[2] -= 9.81 * DT
        self._position = self._position + self._velocity * DT
        if self._position[2] <= GROUND:
            self._position[2] = GROUND
            self._velocity[2] = 0.0
            self._velocity[:2] *= 0.97  # it rolls, and slows down doing it


class FakeScene:
    dt = DT

    def __init__(self, arm):
        self.arm = arm
        self.steps = 0

    def play(self):
        pass

    def step(self, count=1):
        for _ in range(count):
            self.steps += 1
            self.arm.tick()


class FakeArm:
    """Enough of a Manipulator for `throw` to run against.

    `profile` maps swing step to hand speed in m/s.
    """

    def __init__(self, profile, *, gripper_opens=True, ceiling=1e9):
        self.profile = profile
        self.ceiling = ceiling  # the fastest this arm's hand can ever go
        self.base_dt = DT
        self.dt_scale = 1.0
        self.dt_history = []
        self.holding = True
        self.swing_step = -1
        self.unit = np.array([1.0, 0.0, 0.0])
        self.velocity = np.zeros(3)
        self._ee = np.array([0.45, 0.0, 0.5])
        self.scene = FakeScene(self)
        self.gripper = FakeGripper(self, opens=gripper_opens)
        self.obj = FakeObject(self)
        self.released_by_fallback = False

        # `throw` reaches for these three through self.
        self._rmpflow = self
        self._policy = self

    @property
    def ee_position(self):
        return self._ee.copy()

    def is_grasping(self, obj):
        return self.holding

    def move_ee_to(self, position, orientation=None, **kwargs):
        self._ee = np.asarray(position, dtype=float).copy()
        self.obj.tick()
        return None

    def _ensure_motion_policy(self):
        pass

    def get_default_physics_dt(self):
        return self.base_dt

    def set_default_physics_dt(self, value):
        self.dt_history.append(value)
        self.dt_scale = value / DT

    def set_end_effector_target(self, target_position=None):
        direction = np.asarray(target_position, dtype=float) - self._ee
        norm = float(np.linalg.norm(direction))
        self.unit = direction / norm if norm else np.array([1.0, 0.0, 0.0])
        self.swing_step = -1

    def update_world(self):
        pass

    def get_next_articulation_action(self):
        return None

    def _controller(self):
        return self

    def apply_action(self, action):
        self.swing_step += 1

    def release(self, settle_steps=20):
        self.released_by_fallback = True
        self.holding = False

    def tick(self):
        if self.swing_step >= 0:
            driven = float(self.profile(self.swing_step)) * self.dt_scale
            self.velocity = self.unit * min(self.ceiling, driven)
            self._ee = self._ee + self.velocity * DT
        self.obj.tick()


def run_throw(profile, *, gripper_opens=True, ceiling=1e9, **kwargs):
    arm = FakeArm(profile, gripper_opens=gripper_opens, ceiling=ceiling)
    return arm, Manipulator.throw(arm, arm.obj, **kwargs)


def reaches(target):
    """A swing that accelerates to `target` m/s and holds it."""
    return lambda step: min(target, 0.15 * (step + 1))


# An arm driven as hard as the policy will drive it plateaus at its ceiling; it
# does not slow down. `ceiling` on FakeArm is what makes a swing run out of arm.
TOPS_OUT = 0.6


def test_releases_at_the_ceiling_when_the_arm_cannot_go_any_faster():
    _, result = run_throw(reaches(9.9), ceiling=TOPS_OUT, speed=2.8, observe_steps=90)

    assert "as fast as it swings" in result["release_reason"]
    # The number that mattered: the old code carried on to the geometric release
    # point and let go at whatever crawl was left, which measured 0.145 m/s.
    assert result["release_hand_speed"] == pytest.approx(TOPS_OUT, abs=1e-3)
    assert result["peak_hand_speed"] == pytest.approx(TOPS_OUT, abs=1e-3)


def test_a_slow_swing_is_driven_harder_rather_than_reported_as_a_throw():
    arm, result = run_throw(reaches(9.9), ceiling=TOPS_OUT, speed=2.8, observe_steps=10)

    # RMPflow plans to arrive, so it plans to stop, and no release rule recovers
    # a velocity it never produced. The only lever that does is telling it more
    # time passed than really did.
    assert arm.dt_history, "the swing never tried to drive the policy harder"
    assert max(arm.dt_history) > DT
    assert result["speed_shortfall"] > 0


def test_the_policy_timestep_is_put_back_when_the_swing_ends():
    arm, _ = run_throw(reaches(9.9), ceiling=TOPS_OUT, speed=2.8, observe_steps=5)

    # Leaving it scaled turns the next ordinary move_ee_to into another swing.
    assert arm.dt_history[-1] == pytest.approx(DT)
    assert arm.dt_scale == pytest.approx(1.0)


def test_reports_the_shortfall_rather_than_implying_the_speed_was_met():
    _, result = run_throw(reaches(9.9), ceiling=TOPS_OUT, speed=2.8, observe_steps=30)

    assert result["requested_speed"] == 2.8
    assert result["release_hand_speed"] < 2.8
    assert result["speed_shortfall"] == pytest.approx(2.8 - result["release_hand_speed"], abs=1e-3)


def test_releases_on_speed_when_the_arm_can_actually_deliver_it():
    _, result = run_throw(reaches(3.0), speed=1.5, observe_steps=60)

    assert result["release_reason"] == "reached the requested hand speed"
    assert result["release_hand_speed"] >= 1.5
    assert result["speed_shortfall"] == 0.0


def test_released_is_never_true_while_the_object_is_still_held():
    _, result = run_throw(reaches(3.0), gripper_opens=False, speed=1.5, observe_steps=20)

    assert result["still_held"] is True
    assert result["released"] is False


def test_apex_height_is_a_height_even_with_no_observation_steps():
    _, result = run_throw(reaches(3.0), speed=1.5, observe_steps=0)

    assert np.isfinite(result["apex_height"])
    assert result["apex_height"] > 0.0


def test_the_gripper_is_given_physics_steps_to_actually_open():
    arm, result = run_throw(reaches(3.0), speed=1.5, observe_steps=0)

    assert arm.gripper.set_position_calls == [FakeGripper.open_width]
    # With observe_steps=0 the old code stepped zero times after commanding the
    # open, so the fingers were still closed when `still_held` was read.
    assert result["still_held"] is False
    assert result["released"] is True


def test_flight_and_roll_are_reported_separately():
    _, result = run_throw(reaches(3.0), speed=1.5, observe_steps=200)

    assert result["flight_distance"] is not None
    assert result["rolled_distance"] >= 0.0
    assert result["landing_position"][2] == pytest.approx(GROUND, abs=1e-3)
    assert result["flight_distance"] < result["horizontal_distance"]


def test_throwing_something_that_is_not_held_says_so():
    arm = FakeArm(reaches(3.0))
    arm.holding = False

    with pytest.raises(MotionError, match="not currently grasped"):
        Manipulator.throw(arm, arm.obj)


def test_a_negative_observation_window_is_rejected():
    arm = FakeArm(reaches(3.0))

    with pytest.raises(ValueError, match="observe_steps"):
        Manipulator.throw(arm, arm.obj, observe_steps=-1)


def test_a_zero_direction_is_rejected_before_anything_moves():
    arm = FakeArm(reaches(3.0))

    with pytest.raises(ValueError, match="non-zero"):
        Manipulator.throw(arm, arm.obj, direction=[0.0, 0.0, 0.0])
    assert arm.holding is True
