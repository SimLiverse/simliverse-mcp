"""Stack three cubes largest-to-smallest, on Play.

Delivered and measured. The numbers below are readings, not guesses.

Two things this encodes that cost a run each to find.

**Home the arm before every pick.** The Franka's wrist winds up over a sequence
of solves, and once joint 6 sits near its 3.752 rad limit the DOWN orientation
can only be satisfied by driving into the stop -- so RMPflow trades position
away for it and every target comes back 9 to 22 cm short. Measured directly:
with the wrist wound, `[0.45, 0.2, 0.135]` was unreachable at 0.095 m error;
after homing, the same target solved to 0.009 m at 0.0 degrees of orientation
error. An earlier run read that as "a real physical/kinematic limit" and gave up
on the smallest cube. It is not a limit, it is a starting pose.

**Release 3 mm above the target, not 12.** Letting go higher drops the cube hard
enough to knock a two-high tower apart.

`servo_to`, never `move_ee_to`: same motion, one tick, no stepping. A controller
that steps physics from inside `compute` deadlocks the graph.
"""

import carb
import numpy as np

ARM = "/World/Arm"
BASE = "/World/CubeLarge"  # stays put; the tower is built on it
JOBS = [("/World/CubeMedium", 0.070), ("/World/CubeSmall", 0.105)]

DOWN = [0.0, 1.0, 0.0, 0.0]
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

STACK_X, STACK_Y = 0.45, -0.20
CARRY_Z = 0.36  # clear of the finished tower
APPROACH = 0.13
GRASP_LIFT = 0.004
COARSE = 0.06  # first stage of the descent onto the tower
FINAL = 0.003  # set down, do not drop

HOME_FRAMES = 90
GRIP_FRAMES = 45
RELEASE_FRAMES = 45
SETTLE_FRAMES = 40
WARMUP_FRAMES = 30
LIMIT = 1400  # per state; a servo move is ~100-300 ticks

(
    WARMUP,
    INIT,
    HOME_ARM,
    OVER_PICK,
    DOWN_PICK,
    GRIP,
    LIFT,
    TRAVERSE,
    OVER_STACK,
    PLACE,
    RELEASE,
    RETREAT,
    NEXT,
    CHECK,
    DONE,
    FAILED,
) = range(16)

NAMES = [
    "WARMUP",
    "INIT",
    "HOME_ARM",
    "OVER_PICK",
    "DOWN_PICK",
    "GRIP",
    "LIFT",
    "TRAVERSE",
    "OVER_STACK",
    "PLACE",
    "RELEASE",
    "RETREAT",
    "NEXT",
    "CHECK",
    "DONE",
    "FAILED",
]

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_cubes = None
_base = None
_pick = None
_why = ""


def _go(state):
    global _state, _frame
    carb.log_warn("[stack3] %s -> %s after %d frames" % (NAMES[_state], NAMES[state], _frame))
    _state, _frame = state, 0


def _fail(reason):
    global _why
    _why = reason
    carb.log_warn("[stack3] FAILED: %s" % reason)
    _go(FAILED)


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _cubes, _base, _pick, _why
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _cubes, _base, _pick, _why = None, None, None, None, ""
        carb.log_warn("[stack3] reset -- next Play rebuilds the tower from scratch")


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    global _state, _frame, _job, _arm, _cubes, _base, _pick
    _frame += 1

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import RigidObject, Robot, Scene

        scene = Scene.get()
        _arm = Robot.attach(ARM, scene=scene)
        _cubes = [RigidObject(path, scene=scene) for path, _ in JOBS]
        _base = RigidObject(BASE, scene=scene)
        _arm.gripper.open()
        _go(HOME_ARM)
        return True

    # No state waits forever. A servo that cannot converge would otherwise
    # neither finish nor fail -- the graph runs, the timeline plays, and nothing
    # happens, which is the hardest failure to see from outside.
    if _frame > LIMIT:
        _fail("%s ran %d frames without progressing (job %d)" % (NAMES[_state], _frame, _job))
        return True

    if _state == HOME_ARM:
        # Re-homed before every pick, not once at the start: the wind-up
        # accumulates across picks, and pick 2 is where it first bites.
        if _frame == 1:
            _arm.set_joint_positions(HOME)
        if _frame >= HOME_FRAMES:
            _go(OVER_PICK)
        return True

    # Past the last job there is no current cube, and CHECK is the only state
    # that still runs. Indexing unconditionally here threw IndexError on every
    # frame of CHECK, so the tower was never measured; the state ran out its
    # frame limit and the controller ended in FAILED with a correct tower
    # standing in front of it. `deliver()` still reported reproduced=True,
    # because it measures the cubes and not the controller -- which is exactly
    # how a verification step manages to never run and never be missed.
    cube = _cubes[_job] if _job < len(_cubes) else None
    target_z = JOBS[_job][1] if _job < len(JOBS) else None

    if _state == OVER_PICK:
        # Read the cube where it actually is. On a replay it is back at its
        # authored pose, which is the whole point of reading the scene rather
        # than trusting remembered coordinates.
        if _pick is None:
            _pick = [float(v) for v in cube.position]
        if _arm.servo_to([_pick[0], _pick[1], _pick[2] + APPROACH], DOWN, tolerance=0.012):
            _go(DOWN_PICK)

    elif _state == DOWN_PICK:
        if _arm.servo_to([_pick[0], _pick[1], _pick[2] + GRASP_LIFT], DOWN, tolerance=0.010):
            _arm.gripper.close()
            _go(GRIP)

    elif _state == GRIP:
        if _frame >= GRIP_FRAMES:
            if not _arm.is_grasping(cube):
                _fail("closed on %s and it is not held" % JOBS[_job][0])
            else:
                _go(LIFT)

    elif _state == LIFT:
        if _arm.servo_to([_pick[0], _pick[1], CARRY_Z], DOWN, tolerance=0.015):
            _go(TRAVERSE)

    elif _state == TRAVERSE:
        if _arm.servo_to([STACK_X, STACK_Y, CARRY_Z], DOWN, tolerance=0.015):
            _go(OVER_STACK)

    elif _state == OVER_STACK:
        if _arm.servo_to([STACK_X, STACK_Y, target_z + COARSE], DOWN, tolerance=0.012):
            _go(PLACE)

    elif _state == PLACE:
        # Two stages. One move overshoots on arrival and nudges the tower.
        if _arm.servo_to([STACK_X, STACK_Y, target_z + FINAL], DOWN, tolerance=0.008):
            _arm.gripper.open()
            _go(RELEASE)

    elif _state == RELEASE:
        if _frame >= RELEASE_FRAMES:
            _go(RETREAT)

    elif _state == RETREAT:
        if _arm.servo_to([STACK_X, STACK_Y, CARRY_Z], DOWN, tolerance=0.015):
            _go(NEXT)

    elif _state == NEXT:
        if _frame < SETTLE_FRAMES:
            return True
        _pick = None
        _job += 1
        _go(HOME_ARM if _job < len(JOBS) else CHECK)

    elif _state == CHECK:
        # Measure the outcome. Arriving at the last state is not the same as
        # having done the task: a controller reported DONE having stacked two of
        # three cubes and left the third on the floor.
        if _frame < SETTLE_FRAMES:
            return True
        heights = [float(c.position[2]) for c in _cubes]
        spread = [float(np.hypot(c.position[0] - STACK_X, c.position[1] - STACK_Y)) for c in _cubes]
        wanted = [z for _, z in JOBS]
        carb.log_warn(
            "[stack3] heights=%s wanted=%s spread=%s"
            % (np.round(heights, 4).tolist(), wanted, np.round(spread, 4).tolist())
        )

        for i, (path, want) in enumerate(JOBS):
            if abs(heights[i] - want) > 0.015:
                _fail("%s at z=%.3f, wanted %.3f -- not stacked" % (path, heights[i], want))
                return True
            if spread[i] > 0.02:
                _fail("%s is %.3f m off the tower axis" % (path, spread[i]))
                return True
        if float(np.hypot(_base.position[0] - STACK_X, _base.position[1] - STACK_Y)) > 0.02:
            _fail("the base cube was pushed off its mark")
            return True
        _go(DONE)

    return True
