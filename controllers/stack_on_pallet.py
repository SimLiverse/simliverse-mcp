"""A Franka stacking three cubes on a pallet, on Play. NOT YET REPLAYED.

Unlike its neighbours in this directory, this one has not been through
`controller.deliver` — no replay has run, so nothing here is measured *as a
controller*. The constants below come from the live run described next, which
is a different thing and a weaker one: they are the numbers that worked when a
human-driven sequence did the task, not numbers a tick-driven state machine has
reproduced. Treat it as a draft until a delivery report says otherwise.

The live version of this worked first — arm drives, tower goes up, numbers look
right — and it was worth nothing. PhysX returns every dynamic body to its
authored pose when the timeline stops, so a task performed through live calls
replays as its own starting state. What the user keeps is the scene, and a
scene that merely ended up in the right arrangement is indistinguishable from
one where the cubes were teleported.

So the deliverable is this: a ScriptNode on `OnPlaybackTick` that rebuilds the
tower from physics every time Play is pressed.

Three things here are shaped by measurement rather than taste:

* Release **3 mm** above the target, not 12 mm. The first attempt let go higher
  and the third cube dropped hard enough onto a two-high tower to knock it
  apart — final heights came out [0.168, 0.168, 0.218] instead of a stack.

* The descent onto the tower is **two stages**, a coarse move to 6 cm above and
  then a slow one to 3 mm. A single move overshoots on arrival and nudges what
  is already stacked.

* `servo_to`, never `move_ee_to`. Same motion, one tick, no stepping — a
  controller that steps physics from inside `compute` deadlocks the graph.
"""

(WARMUP, INIT, OVER_CUBE, DOWN_TO_CUBE, GRIP, LIFT, TRAVERSE, OVER_STACK,
 PLACE, RELEASE, RETREAT, NEXT, DONE, FAILED) = range(14)

NAMES = ["WARMUP", "INIT", "OVER_CUBE", "DOWN_TO_CUBE", "GRIP", "LIFT", "TRAVERSE",
         "OVER_STACK", "PLACE", "RELEASE", "RETREAT", "NEXT", "DONE", "FAILED"]

ARM = "/World/Arm"
CUBES = ["/World/Cube0", "/World/Cube1", "/World/Cube2"]
TRACE = "/tmp/stack_on_pallet.log"

# Franka flange pointing at the floor.
DOWN = [0.0, 1.0, 0.0, 0.0]

PALLET_TOP = 0.143
HALF = 0.025
STACK_X, STACK_Y = 0.56, 0.0
CARRY_Z = PALLET_TOP + 0.32        # travel height, clear of the tower
APPROACH = 0.13                    # above a cube before descending onto it
GRASP_LIFT = 0.005                 # tool height above the cube when closing
COARSE = 0.06                      # first stage of the descent onto the tower
FINAL = 0.003                      # release height: set down, do not drop

GRIP_FRAMES = 40
RELEASE_FRAMES = 45
SETTLE_FRAMES = 25
WARMUP_FRAMES = 30
LIMIT = 1600                       # per state; a servo move is ~100-300 ticks

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_cubes = None
_pick = None


def _trace(line):
    with open(TRACE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _go(state):
    global _state, _frame
    _trace("job=%d %-12s -> %-12s after %4d frames" % (_job, NAMES[_state], NAMES[state], _frame))
    _state, _frame = state, 0


def _target_z(index):
    """Where the centre of cube `index` belongs in the finished tower."""
    return PALLET_TOP + HALF + index * (2 * HALF)


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _cubes, _pick
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _cubes, _pick = None, None, None
        open(TRACE, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    global _state, _frame, _job, _arm, _cubes, _pick
    _frame += 1

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:
        # Physics needs a few frames before an articulation can be read.
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import RigidObject, Robot, Scene

        scene = Scene.get()
        _arm = Robot.attach(ARM, scene=scene)
        _cubes = [RigidObject(path, scene=scene) for path in CUBES]
        _arm.gripper.open()
        _go(OVER_CUBE)
        return True

    if _frame > LIMIT:
        _trace("  TIMEOUT in %s" % NAMES[_state])
        _go(FAILED)
        return True

    cube = _cubes[_job]

    if _state == OVER_CUBE:
        # Read the cube where it actually is. On a replay it is back at its
        # authored pose, which is the whole point of doing this from the scene
        # rather than from remembered coordinates.
        if _pick is None:
            _pick = [float(v) for v in cube.position]
        if _arm.servo_to([_pick[0], _pick[1], _pick[2] + APPROACH], DOWN, tolerance=0.012):
            _go(DOWN_TO_CUBE)

    elif _state == DOWN_TO_CUBE:
        if _arm.servo_to([_pick[0], _pick[1], _pick[2] + GRASP_LIFT], DOWN, tolerance=0.010):
            _arm.gripper.close()
            _go(GRIP)

    elif _state == GRIP:
        if _frame >= GRIP_FRAMES:
            _go(LIFT)

    elif _state == LIFT:
        if _arm.servo_to([_pick[0], _pick[1], CARRY_Z], DOWN, tolerance=0.015):
            _go(TRAVERSE)

    elif _state == TRAVERSE:
        if _arm.servo_to([STACK_X, STACK_Y, CARRY_Z], DOWN, tolerance=0.015):
            _go(OVER_STACK)

    elif _state == OVER_STACK:
        if _arm.servo_to([STACK_X, STACK_Y, _target_z(_job) + COARSE], DOWN, tolerance=0.012):
            _go(PLACE)

    elif _state == PLACE:
        if _arm.servo_to([STACK_X, STACK_Y, _target_z(_job) + FINAL], DOWN, tolerance=0.008):
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
        if _job >= len(CUBES):
            _trace("ALL DONE")
            _go(DONE)
        else:
            _go(OVER_CUBE)

    return True
