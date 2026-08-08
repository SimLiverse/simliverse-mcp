"""Stack three 5 cm cubes into one tower.

Reactive control throughout, and no obstacles registered. That is the right
choice here rather than an omission: a planner earns its cost by routing around
something the arm must not touch, and this workspace contains nothing of the
kind — every object present is either the cube being carried or the cube being
placed onto. Registering the tower would be actively wrong, since the tool has
to approach it.

The one real hazard, knocking the tower over in passing, is handled by lifting
clear and traversing above it, which costs nothing.
"""

(WARMUP, INIT, OPEN, HOVER, DESCEND, CLOSE, LIFT, OVER,
 LOWER, RELEASE, RETREAT, NEXT, DONE, FAILED) = range(14)

NAMES = ["WARMUP", "INIT", "OPEN", "HOVER", "DESCEND", "CLOSE", "LIFT", "OVER",
         "LOWER", "RELEASE", "RETREAT", "NEXT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/ctl_trace.log"

DOWN = [0.0, 1.0, 0.0, 0.0]     # tool z pointing at the table
CUBE = 0.05                      # edge length
TRANSIT = 0.35                   # above a finished three-cube tower (top 0.15)
HOVER_Z = 0.18
GRASP_Z = 0.035                  # cube centred at 0.025, tool a touch above centre
WARMUP_FRAMES = 30
GRIP_FRAMES = 100                # closing takes real time; the state waits, never steps
LIMIT = 250                      # a state that cannot converge should say so in ~4 s
COARSE = 0.012
FINE = 0.008

BASE = "/World/Cube1"
JOBS = [
    {"cube": "/World/Cube2", "onto": "/World/Cube1", "level": 1},
    {"cube": "/World/Cube3", "onto": "/World/Cube2", "level": 2},
]

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_cubes = None
_pick = None


def _trace(line):
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _timeout():
    if _frame > LIMIT:
        _trace("  TIMEOUT %s err=%s" % (NAMES[_state], getattr(_arm, "_servo_error", None)))
        return True
    return False


def _go(state):
    global _state, _frame
    try:
        ee = [round(float(v), 3) for v in _arm.ee_position]
    except Exception:
        ee = None
    _trace("job=%d %-8s -> %-8s frames=%4d ee=%s" % (_job, NAMES[_state], NAMES[state], _frame, ee))
    _state, _frame = state, 0


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _cubes, _pick
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _cubes, _pick = None, None, None
        open(TRACE_PATH, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    """One frame, one transition. Never loops, never steps physics."""
    global _state
    try:
        return _compute(db)
    except Exception:
        import traceback

        _trace("RAISED in " + NAMES[_state])
        _trace(traceback.format_exc())
        _state = FAILED
        return True


def _compute(db=None):
    global _state, _frame, _job, _arm, _cubes, _pick
    _frame += 1

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import RigidObject, Scene
        from simliverse_sim.robots.manipulator import Manipulator

        scene = Scene.get()
        _arm = Manipulator("/World/Franka", scene=scene)
        paths = {BASE} | {j["cube"] for j in JOBS} | {j["onto"] for j in JOBS}
        _cubes = {p: RigidObject(p, scene=scene) for p in paths}
        _trace("init: obstacles=%s (none by design)" % _arm.obstacles())
        _go(OPEN)
        return True

    if _state in (DONE, FAILED):
        return True

    job = JOBS[_job]
    cube = _cubes[job["cube"]]
    onto = _cubes[job["onto"]]
    base = _cubes[BASE]

    if _state == OPEN:
        if _frame == 1:
            _arm.gripper.open()
        if _frame >= GRIP_FRAMES:
            # Read the pose once, before touching it: after the grasp the cube
            # travels with the gripper and a live read would chase itself.
            _pick = [float(v) for v in cube.position]
            _go(HOVER)
        return True

    if _state == HOVER:
        if _arm.servo_to([_pick[0], _pick[1], HOVER_Z], DOWN, tolerance=COARSE) or _timeout():
            _go(DESCEND)
        return True

    if _state == DESCEND:
        if _arm.servo_to([_pick[0], _pick[1], GRASP_Z], DOWN, tolerance=FINE) or _timeout():
            _go(CLOSE)
        return True

    if _state == CLOSE:
        if _frame == 1:
            _arm.gripper.close()
        if _frame >= GRIP_FRAMES:
            _trace("  grasped: %s" % _arm.is_grasping(cube))
            _go(LIFT)
        return True

    if _state == LIFT:
        if _arm.servo_to([_pick[0], _pick[1], TRANSIT], DOWN, tolerance=COARSE) or _timeout():
            _go(OVER)
        return True

    if _state == OVER:
        # Aim at the base's live x,y. The tower is vertical, so the column is
        # wherever the base actually is rather than where it was authored — and
        # converging here, high above everything, is what makes the descent land
        # on the stack instead of beside it.
        b = base.position
        if _arm.servo_to([float(b[0]), float(b[1]), TRANSIT], DOWN, tolerance=FINE) or _timeout():
            _go(LOWER)
        return True

    if _state == LOWER:
        # Stop on contact with the cube below, not on reaching a height. The
        # cube resting on the stack is the actual end condition and it is
        # measurable, so a descent that converges early or late still releases
        # at the right moment.
        b = base.position
        target = 0.05 * job["level"] + GRASP_Z
        if onto.prim_path in cube.contact_bodies():
            _trace("  touched down on %s at %s"
                   % (onto.prim_path, [round(float(v), 3) for v in cube.position]))
            _go(RELEASE)
            return True
        if _arm.servo_to([float(b[0]), float(b[1]), target], DOWN, tolerance=FINE) or _timeout():
            _trace("  lowered without contact at %s"
                   % [round(float(v), 3) for v in cube.position])
            _go(RELEASE)
        return True

    if _state == RELEASE:
        if _frame == 1:
            _arm.gripper.open()
        if _frame >= GRIP_FRAMES:
            _go(RETREAT)
        return True

    if _state == RETREAT:
        b = base.position
        if _arm.servo_to([float(b[0]), float(b[1]), TRANSIT], DOWN, tolerance=COARSE) or _timeout():
            _go(NEXT)
        return True

    if _state == NEXT:
        _job += 1
        _go(OPEN if _job < len(JOBS) else DONE)
        return True

    return True
