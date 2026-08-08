"""Worked example: two pick-and-places past an obstacle, planned.

Delivered and measured — both cubes land on the platform and the post is
untouched, identically across independent replays. Reproduced verbatim from
the run that proved it, so the numbers in the comments are measurements.

Move both cubes onto the platform without disturbing the post.

Gross motion is planned, the last centimetres are servoed. The reactive policy
alone could not do this: crossing over the post wedged it 18 cm short of the
target, and the post's repulsive field pulled the descent far enough off-centre
to push the cube away instead of gripping it. Both obstacles are therefore
registered for *planning only* — the tool never goes near them under servo
control, so giving them to the policy costs accuracy and buys nothing.
"""

(WARMUP, INIT, OPEN, PLAN_PICK, GOTO_PICK, DESCEND, CLOSE, LIFT,
 PLAN_PLACE, GOTO_PLACE, LOWER, RELEASE, PLAN_CLEAR, GOTO_CLEAR,
 NEXT, DONE, FAILED) = range(17)

NAMES = ["WARMUP", "INIT", "OPEN", "PLAN_PICK", "GOTO_PICK", "DESCEND", "CLOSE",
         "LIFT", "PLAN_PLACE", "GOTO_PLACE", "LOWER", "RELEASE", "PLAN_CLEAR",
         "GOTO_CLEAR", "NEXT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/planned_transfer.log"

DOWN = [0.0, 1.0, 0.0, 0.0]      # tool z pointing at the table
HOVER_Z = 0.16                   # above the cube, where the plan hands over
GRASP_Z = 0.030                  # fingertips around a 4 cm cube
PLACE_HOVER = 0.32               # well clear of the platform, so the plan may end there
# Lift the object clear before planning the transfer. Right for any
# pick-and-place — you do not drag a grasped object across a surface — and
# it also plans far better: from down at the table the same transfer took
# 2.8 s to find and sometimes failed outright, against 0.9 s from up here.
TRANSIT_Z = 0.40
PLACE_Z = 0.145                  # platform top 0.10 + half a cube + clearance
WARMUP_FRAMES = 30
GRIP_FRAMES = 45
LIMIT = 600
FINE = 0.007

OBSTACLES = ["/World/Post", "/World/Platform"]
JOBS = [
    {"cube": "/World/CubeA", "place": [0.42, 0.34]},
    {"cube": "/World/CubeB", "place": [0.50, 0.34]},
]

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_cubes = None
_pick = None
_plan = None


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
    _trace("job=%d %-11s -> %-11s frames=%4d ee=%s" % (_job, NAMES[_state], NAMES[state], _frame, ee))
    _state, _frame = state, 0


def _plan_to(target, next_state):
    """Plan once and advance, or fail loudly. Never silently fall back.

    A planner that found no route has said something true about the scene, and
    servoing at the same target would drive into whatever it just refused to
    route around.
    """
    global _plan
    try:
        _plan = _arm.plan_to(target, DOWN)
        _trace("  planned %s -> %s" % ([round(v, 3) for v in target], _plan))
        _go(next_state)
    except Exception as exc:
        _trace("  PLAN FAILED for %s: %s" % ([round(v, 3) for v in target], exc))
        _go(FAILED)


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _cubes, _pick, _plan
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _cubes, _pick, _plan = None, None, None, None
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
    global _state, _frame, _job, _arm, _cubes, _pick, _plan
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
        _cubes = {j["cube"]: RigidObject(j["cube"], scene=scene) for j in JOBS}
        _arm.clear_obstacles()
        for path in OBSTACLES:
            _arm.add_obstacle(path, reactive=False)
        # Build the planner here rather than on first use: loading the robot and
        # compiling the GPU kernels takes a couple of seconds, and that belongs
        # in a state that is expecting to wait rather than mid-transfer.
        _arm.planner()
        _trace("init: obstacles=%s servo-blind=%s" % (_arm.obstacles(), _arm.unavoidable_by_servo()))
        _go(OPEN)
        return True

    if _state in (DONE, FAILED):
        return True

    job = JOBS[_job]
    cube = _cubes[job["cube"]]
    px, py = job["place"]

    if _state == OPEN:
        if _frame == 1:
            _arm.gripper.open()   # non-blocking: never step inside compute()
        if _frame >= GRIP_FRAMES:
            # Read the pose once, before touching it: after the grasp the cube
            # travels with the gripper and a live read would chase itself.
            _pick = [float(v) for v in cube.position]
            _go(PLAN_PICK)
        return True

    if _state == PLAN_PICK:
        _plan_to([_pick[0], _pick[1], HOVER_Z], GOTO_PICK)
        return True

    if _state == GOTO_PICK:
        if _arm.follow(_plan) or _timeout():
            _go(DESCEND)
        return True

    if _state == DESCEND:
        if _arm.servo_to([_pick[0], _pick[1], GRASP_Z], DOWN, tolerance=FINE) or _timeout():
            _go(CLOSE)
        return True

    if _state == CLOSE:
        if _frame == 1:
            _arm.gripper.close()  # non-blocking: never step inside compute()
        if _frame >= GRIP_FRAMES:
            _go(LIFT)
        return True

    if _state == LIFT:
        # Straight up, no obstacle anywhere near: the reactive policy is the
        # right tool and planning a vertical retreat would be ceremony.
        if _arm.servo_to([_pick[0], _pick[1], TRANSIT_Z], DOWN, tolerance=0.012) or _timeout():
            _go(PLAN_PLACE)
        return True

    if _state == PLAN_PLACE:
        # One plan for the whole transfer: up, across, and over the post. The
        # route is the planner's problem, which is the entire point.
        _plan_to([px, py, PLACE_HOVER], GOTO_PLACE)
        return True

    if _state == GOTO_PLACE:
        if _arm.follow(_plan) or _timeout():
            _go(LOWER)
        return True

    if _state == LOWER:
        if _arm.servo_to([px, py, PLACE_Z], DOWN, tolerance=FINE) or _timeout():
            _go(RELEASE)
        return True

    if _state == RELEASE:
        if _frame == 1:
            _arm.gripper.open()   # non-blocking: never step inside compute()
        if _frame >= GRIP_FRAMES:
            _go(PLAN_CLEAR)
        return True

    if _state == PLAN_CLEAR:
        _plan_to([px, py, PLACE_HOVER], GOTO_CLEAR)
        return True

    if _state == GOTO_CLEAR:
        if _arm.follow(_plan) or _timeout():
            _go(NEXT)
        return True

    if _state == NEXT:
        _job += 1
        _go(OPEN if _job < len(JOBS) else DONE)
        return True

    return True
