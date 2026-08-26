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

"""Recording a run as a trajectory a real controller could take.

The point of these is the export contract. A trajectory that leaves here is
meant to be handed to someone whose arm will execute it, so the tests pin the
things that would make that dangerous or useless: joint names present and in
order, time monotonic from zero, limit violations reported rather than
swallowed, and the caveats travelling inside the payload instead of alongside it.
"""

import json

import numpy as np
import pytest

from simliverse_sim.recording import CAVEATS, SCHEMA, JointRecorder, replay

NAMES = ["joint_1", "joint_2", "joint_3"]
LIMITS = [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]
DT = 1.0 / 60.0


class FakeScene:
    dt = DT

    def __init__(self):
        self.listeners = []
        self.time = 0.0
        self.played = False

    def add_step_listener(self, fn):
        if fn not in self.listeners:
            self.listeners.append(fn)

    def remove_step_listener(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def play(self):
        self.played = True

    def step(self, count=1, **_kw):
        for _ in range(count):
            self.time += DT
            for fn in list(self.listeners):
                fn(self.time)


class FakeArticulation:
    def __init__(self, max_velocity):
        self.dof_properties = np.array(
            [(v,) for v in max_velocity],
            dtype=[("maxVelocity", "f4")],
        )


class FakeRobot:
    prim_path = "/World/Arm"

    def __init__(self, *, max_velocity=(3.0, 3.0, 3.0)):
        self.joint_names = list(NAMES)
        self.joint_limits = list(LIMITS)
        self.joint_positions = np.zeros(3)
        self.joint_velocities = np.zeros(3)
        self.scene = FakeScene()
        self._articulation = FakeArticulation(max_velocity)
        self.commands = []

    def set_joint_positions(self, targets, *, indices=None, settle_steps=0):
        self.commands.append((list(indices) if indices else None, list(targets)))
        for slot, value in zip(indices or range(3), targets):
            self.joint_positions[slot] = value
        if settle_steps:
            self.scene.step(settle_steps)


def drive(robot, steps=10, *, per_step=0.05, velocity=0.5):
    for _ in range(steps):
        robot.joint_positions = robot.joint_positions + per_step
        robot.joint_velocities = np.full(3, velocity)
        robot.scene.step(1)


# ── capturing the run ────────────────────────────────────────────────────────


def test_it_records_one_point_per_physics_step():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=10)

    assert len(rec.times) == 10
    assert len(rec.positions) == 10


def test_it_detaches_on_exit_so_a_recording_cannot_keep_growing():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=5)
    drive(robot, steps=50)

    assert len(rec.times) == 5
    assert robot.scene.listeners == []


def test_time_starts_at_zero_and_only_increases():
    """A controller reads `time_from_start`; a run that began at sim t=12 is
    still a trajectory that starts now."""
    robot = FakeRobot()
    robot.scene.time = 12.0
    with JointRecorder(robot) as rec:
        drive(robot, steps=8)

    times = [p["time_from_start"] for p in rec.trajectory()["points"]]
    assert times[0] == 0.0
    assert all(b > a for a, b in zip(times, times[1:]))


def test_decimation_keeps_every_nth_step():
    robot = FakeRobot()
    with JointRecorder(robot, every=5) as rec:
        drive(robot, steps=20)

    assert len(rec.times) == 4


# The guard for a listener that raises lives in the real `Scene._notify_step`,
# which cannot be constructed without Isaac Sim. Faking a scene to check it
# would only test the fake, so it is verified live rather than pretended here.


# ── the export contract ──────────────────────────────────────────────────────


def test_the_payload_is_shaped_like_a_ros_joint_trajectory():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=6)
    payload = rec.trajectory()

    assert payload["schema"] == SCHEMA
    assert payload["joint_names"] == NAMES
    for point in payload["points"]:
        assert set(point) == {"positions", "velocities", "time_from_start"}
        assert len(point["positions"]) == len(NAMES)
        assert len(point["velocities"]) == len(NAMES)


def test_the_caveats_travel_inside_the_payload():
    """Not in a README beside it. A trajectory that arrives without the
    conditions it was recorded under is the one that gets someone hurt."""
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=3)

    assert rec.trajectory()["caveats"] == CAVEATS
    assert "hardware" in CAVEATS


def test_it_saves_json_that_round_trips(tmp_path):
    robot = FakeRobot()
    with JointRecorder(robot, label="demo") as rec:
        drive(robot, steps=4)
    path = rec.save(str(tmp_path / "run.json"))

    loaded = json.loads(open(path, encoding="utf-8").read())
    assert loaded["label"] == "demo"
    assert len(loaded["points"]) == 4


# ── what would be dangerous on hardware ──────────────────────────────────────


def test_a_clean_run_reports_no_violations():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=10, per_step=0.05, velocity=0.5)

    assert rec.violations() == []


def test_a_joint_driven_past_its_limit_is_reported():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=40, per_step=0.05)  # joint_3 has a 1.0 rad limit

    found = rec.violations()
    assert any("joint_3" in v and "above its upper limit" in v for v in found)


def test_a_joint_driven_faster_than_declared_is_reported():
    robot = FakeRobot(max_velocity=(3.0, 0.2, 3.0))
    with JointRecorder(robot) as rec:
        drive(robot, steps=5, per_step=0.01, velocity=1.5)

    found = rec.violations()
    assert any("joint_2" in v and "above its declared maximum" in v for v in found)


def test_violations_ride_along_in_the_exported_payload():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=40, per_step=0.05)

    assert rec.trajectory()["violations"], "an unsafe trajectory exported as clean"


def test_no_declared_maxima_is_not_an_error():
    """Plenty of assets ship without them; that is missing data, not a fault."""
    robot = FakeRobot(max_velocity=(0.0, 0.0, 0.0))
    with JointRecorder(robot) as rec:
        drive(robot, steps=5, velocity=99.0)

    assert all("declared maximum" not in v for v in rec.violations())


def test_an_empty_recording_reports_nothing_rather_than_guessing():
    robot = FakeRobot()
    rec = JointRecorder(robot)

    assert rec.violations() == []
    assert rec.trajectory()["points"] == []
    assert rec.duration == 0.0


# ── playing it back ──────────────────────────────────────────────────────────


def test_replay_commands_the_recorded_joints_in_order():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=4, per_step=0.1)
    payload = rec.trajectory()

    target = FakeRobot()
    replay(target, payload)

    assert target.scene.played
    assert len(target.commands) == 4
    assert target.commands[0][0] == [0, 1, 2]
    assert target.commands[-1][1] == pytest.approx(payload["points"][-1]["positions"])


def test_replay_maps_joints_by_name_not_by_position():
    """A controller that trusted index order would drive the wrong joints on a
    robot whose DOFs are enumerated differently."""
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=2, per_step=0.1)
    payload = rec.trajectory()
    payload["joint_names"] = ["joint_3", "joint_1", "joint_2"]
    for point in payload["points"]:
        point["positions"] = list(reversed(point["positions"]))

    target = FakeRobot()
    replay(target, payload)

    assert target.commands[0][0] == [2, 0, 1]


def test_replaying_nothing_does_nothing():
    target = FakeRobot()
    replay(target, {"joint_names": NAMES, "points": []})

    assert target.commands == []


# ── which joints belong to which controller ──────────────────────────────────


class FakeGripper:
    joint_names = ["joint_3"]


def test_gripper_joints_are_labelled_not_silently_dropped():
    """On real hardware the arm and the gripper are different controllers.

    A bridge that fed a finger joint to the arm controller would be asking it to
    move an axis it does not have — so the split is declared. Declared, not
    applied: dropping data is worse than labelling it.
    """
    robot = FakeRobot()
    robot.gripper = FakeGripper()
    with JointRecorder(robot) as rec:
        drive(robot, steps=3)
    payload = rec.trajectory()

    assert payload["groups"] == {"arm": ["joint_1", "joint_2"], "gripper": ["joint_3"]}
    # every joint is still in the points
    assert len(payload["points"][0]["positions"]) == 3


def test_a_robot_with_no_gripper_reports_every_joint_as_arm():
    robot = FakeRobot()
    with JointRecorder(robot) as rec:
        drive(robot, steps=2)

    assert rec.trajectory()["groups"] == {"arm": NAMES, "gripper": []}


# ── surviving the agent's namespace resets ───────────────────────────────────


def test_a_started_recording_can_be_found_again_by_label():
    """The harness starts a recording, hands the scene to an agent, and collects
    it afterwards — across tool calls that may reset the namespace in between."""
    from simliverse_sim.recording import active, start_recording, stop_recording

    robot = FakeRobot()
    start_recording(robot, label="t2")
    drive(robot, steps=4)

    assert active("t2") is not None
    assert len(active("t2").times) == 4
    stop_recording("t2")
    assert active("t2") is None


def test_starting_the_same_label_twice_replaces_rather_than_doubles():
    from simliverse_sim.recording import start_recording, stop_all

    robot = FakeRobot()
    start_recording(robot, label="t2")
    drive(robot, steps=3)
    start_recording(robot, label="t2")
    drive(robot, steps=2)

    assert len(active_len(robot)) == 1, "the first recorder was left attached"
    stop_all()


def active_len(robot):
    return robot.scene.listeners


def test_stop_all_detaches_everything():
    from simliverse_sim.recording import start_recording, stop_all

    robot = FakeRobot()
    start_recording(robot, label="a")
    start_recording(robot, label="b")
    stop_all()

    assert robot.scene.listeners == []


# ── frame numbers come from time, not from list position ─────────────────────


class FakePrimStage:
    """Enough USD for `bake` to be exercised without Isaac."""


def test_pose_samples_record_their_own_timestamps():
    """`bake` needs times, because a sample's position in the list says nothing
    about when it happened once `every > 1`."""
    from simliverse_sim.recording import PoseRecorder

    robot = FakeRobot()
    rec = PoseRecorder([], scene=robot.scene, every=3)
    rec._sample = lambda: {}
    with rec:
        drive(robot, steps=12)

    assert len(rec.frames) == 4
    assert len(rec.times) == 4
    assert rec.times[0] == 0.0
    # Every third step of a 1/60 s tick: 3/60 apart.
    assert rec.times[1] == pytest.approx(3 * DT, abs=1e-9)
    assert rec.times[-1] == pytest.approx(9 * DT, abs=1e-9)


def test_decimated_recording_keeps_real_time_spacing():
    """A recording sampled every 5th step still covers the same wall clock.

    This is what the frame-number fix protects: indexing would have compressed
    12 steps of motion into 3 frames' worth of time.
    """
    from simliverse_sim.recording import PoseRecorder

    robot = FakeRobot()
    rec = PoseRecorder([], scene=robot.scene, every=5)
    rec._sample = lambda: {}
    with rec:
        drive(robot, steps=15)

    assert len(rec.frames) == 3
    assert rec.times[-1] == pytest.approx(10 * DT, abs=1e-9)
    # At 24 fps that last sample belongs at frame 4, not frame 2.
    assert round(rec.times[-1] * 24) == 4
