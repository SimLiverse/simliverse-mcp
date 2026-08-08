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

(WARMUP, INIT, OPEN, PLAN_PICK, GOTO_PICK, HOVER, DESCEND, CLOSE, LIFT, GOTO_LIFT,
 PLAN_PLACE, GOTO_PLACE, LOWER, RELEASE, PLAN_CLEAR, GOTO_CLEAR,
 NEXT, DONE, FAILED) = range(19)

NAMES = ["WARMUP", "INIT", "OPEN", "PLAN_PICK", "GOTO_PICK", "HOVER", "DESCEND", "CLOSE", "LIFT", "GOTO_LIFT", "PLAN_PLACE", "GOTO_PLACE", "LOWER", "RELEASE", "PLAN_CLEAR",
         "GOTO_CLEAR", "NEXT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/ctl_trace.log"

DOWN = [0.0, 1.0, 0.0, 0.0]      # tool z pointing at the table
HOVER_Z = 0.16                   # above the cube, where the plan hands over
GRASP_Z = 0.030                  # fingertips around a 4 cm cube
# Just outside the planner's margin around the platform (top 0.10 + 0.06),
# and no further. Everything between here and the surface is servoed, and
# the reactive policy is fighting the post's field the whole way — a 17 cm
# descent from 0.32 stalled with 27 cm of error and drifted off carrying the
# cube. Keep the reactive segment as short as the margin allows.
PLACE_HOVER = 0.19
# Lift the object clear before planning the transfer. Right for any
# pick-and-place — you do not drag a grasped object across a surface — and
# it also plans far better: from down at the table the same transfer took
# 2.8 s to find and sometimes failed outright, against 0.9 s from up here.
TRANSIT_Z = 0.40
# Touch-down height, not hover height: platform top 0.10 + half a 4 cm cube
# is 0.12, so this leaves 5 mm. The old 0.145 released with the underside
# 2.5 cm clear and the cube was dropped rather than set down.
PLACE_Z = 0.125
WARMUP_FRAMES = 30
GRIP_FRAMES = 100   # closing takes real time; the state waits instead of stepping
LIMIT = 900
FINE = 0.007

POST = "/World/Post"
PLATFORM = "/World/Platform"
OBSTACLES = [POST, PLATFORM]
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
_hit = None      # the state that first touched something it must not


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
        globals()['_hit'] = None
        open(TRACE_PATH, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def _watch_post():
    """Name the state that first touches the post. One line of evidence."""
    global _hit
    if _hit is not None or _arm is None:
        return
    from simliverse_sim import RigidObject, Scene

    bodies = RigidObject(POST, scene=Scene.get()).contact_bodies()
    hits = sorted(b for b in bodies if b.startswith("/World/Franka"))
    if hits:
        _hit = NAMES[_state]
        _trace("  FIRST POST CONTACT during %s (job=%d frame=%d) by %s"
               % (_hit, _job, _frame, ", ".join(hits)))


def compute(db=None):
    """One frame, one transition. Never loops, never steps physics."""
    global _state
    try:
        _watch_post()
        return _compute(db)
    except Exception:
        import traceback
        _trace("RAISED in " + NAMES[_state])
        _trace(traceback.format_exc())
        _state = FAILED
        return True


def _compute(db=None):
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
        # Visible to BOTH backends by default. Planner-only was wrong: it left
        # servo_to blind, and the servo that lowers onto the platform swings the
        # wrist and forearm back across the post — measured, as link5/6/7 in the
        # contact report. The planner reasons about the whole arm; the reactive
        # policy only avoids what it has been given.
        for path in OBSTACLES:
            _arm.add_obstacle(path)
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
            _arm.gripper.open()
            # Hide the post from the reactive policy for the pick only. Its
            # repulsion reaches 23 cm and pulls the descent 1.4 cm off-centre,
            # which lands the fingers beside a 4 cm cube. The planner still
            # routes around it; the arm never goes near it while picking.
            _arm.remove_obstacle(POST)
            _arm.add_obstacle(POST, reactive=False)   # non-blocking: never step inside compute()
        if _frame >= GRIP_FRAMES:
            # Read the pose once, before touching it: after the grasp the cube
            # travels with the gripper and a live read would chase itself.
            _pick = [float(v) for v in cube.position]
            _go(PLAN_PICK)
        return True

    if _state == PLAN_PICK:
        # Cross at transit height, then descend. Asking the planner for a
        # target below the obstacle's own height on the far side of it makes
        # the route thread down past the post; the outbound leg already lifts
        # clear before transiting and the return has to be symmetric.
        _plan_to([_pick[0], _pick[1], TRANSIT_Z], GOTO_PICK)
        return True

    if _state == GOTO_PICK:
        if _arm.follow(_plan) or _timeout():
            _go(HOVER)
        return True

    if _state == HOVER:
        # Straight down, directly above the cube, nothing near: servo.
        if _arm.servo_to([_pick[0], _pick[1], HOVER_Z], DOWN, tolerance=0.012) or _timeout():
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
            # Give the post back to the reactive policy BEFORE the lift, not
            # after. Measured: link5 first contacts the post 26 frames into
            # LIFT. The tool rises 23 cm away from it and the elbow does not —
            # servo_to steers the end effector and lets the arm behind it go
            # where it likes, so an obstacle nowhere near the tool is still very
            # much near the robot. The field bias that made hiding it necessary
            # only ever mattered for the fine descent onto the cube.
            _arm.remove_obstacle(POST)
            _arm.add_obstacle(POST)
            _go(LIFT)
        return True

    if _state == LIFT:
        # Plan the lift; do not servo it.
        #
        # Raising the tool 37 cm is gross motion through the workspace, not a
        # final approach, and it was misfiled as the latter. Both ways of
        # servoing it fail, which is what makes the classification the bug: with
        # the post hidden from the reactive policy, link5 sweeps through it as
        # the elbow comes up; with the post visible, the policy wedges against
        # its own repulsion and drives the arm back to x=0.02 instead of up.
        # The planner reasons about the whole arm and about a route, and has
        # neither failure.
        _plan_to([_pick[0], _pick[1], TRANSIT_Z], GOTO_LIFT)
        return True

    if _state == GOTO_LIFT:
        if _arm.follow(_plan) or _timeout():
            _go(PLAN_PLACE)
        return True

    if _state == PLAN_PLACE:
        # One plan for the whole transfer: up, across, and over the post. The
        # route is the planner's problem, which is the entire point.
        _plan_to([px, py, PLACE_HOVER], GOTO_PLACE)
        return True

    if _state == GOTO_PLACE:
        if _arm.follow(_plan) or _timeout():
            # Hide the platform from the reactive policy for the descent onto
            # it. A registered obstacle is somewhere RMPflow will not take the
            # tool, and the place target is 3.5 cm above the platform's top —
            # so the policy fights the last move of every place, stalls with
            # 30 cm of error and drifts away carrying the cube. Same rule as the
            # post during the pick: whatever the tool must approach cannot be in
            # the reactive set while it approaches it. The planner still has it.
            # The platform only: the tool has to approach it, so it cannot be
            # in the reactive set. The post stays, because the arm still swings
            # near it while reaching across and nothing else protects the links.
            _arm.remove_obstacle(PLATFORM)
            _arm.add_obstacle(PLATFORM, reactive=False)
            _trace("  arrived over place; still holding: %s" % _arm.is_grasping(cube))
            _go(LOWER)
        return True

    if _state == LOWER:
        # Stop on contact, not on arriving at a number.
        #
        # The cube resting on the platform is the actual end condition, and it
        # is measurable — so a descent that converges early, late, or not at all
        # still releases at the right moment. Releasing on a position target
        # meant a servo that timed out let go from wherever it had got to, which
        # is how a cube ends up on the floor beside the platform rather than on
        # it.
        if PLATFORM in cube.contact_bodies():
            _trace("  touched down: cube on platform at %s"
                   % [round(float(v), 3) for v in cube.position])
            _go(RELEASE)
            return True
        if _arm.servo_to([px, py, PLACE_Z], DOWN, tolerance=FINE) or _timeout():
            _trace("  lowered without contact; releasing anyway at %s"
                   % [round(float(v), 3) for v in cube.position])
            _go(RELEASE)
        return True

    if _state == RELEASE:
        if _frame == 1:
            _arm.gripper.open()   # non-blocking: never step inside compute()
        if _frame >= GRIP_FRAMES:
            _go(PLAN_CLEAR)
        return True

    if _state == PLAN_CLEAR:
        # Servo out, do not plan out.
        #
        # After releasing, the tool sits ~4 cm above the platform, which is
        # inside the planner's safety margin around it — so there is no
        # collision-free route out, and asking for one fails with "walled off by
        # an obstacle" while the arm is in perfectly good shape. The margin that
        # keeps a plan clear of a surface is exactly what makes a pose next to
        # that surface unplannable. Retreat with the reactive policy until the
        # arm is somewhere the planner considers free, then plan.
        if _arm.servo_to([px, py, PLACE_HOVER], DOWN, tolerance=0.015) or _timeout():
            # Clear of the platform; hand it back before anything transits.
            _arm.remove_obstacle(PLATFORM)
            _arm.add_obstacle(PLATFORM)
            _go(NEXT)
        return True

    if _state == NEXT:
        _job += 1
        _go(OPEN if _job < len(JOBS) else DONE)
        return True

    return True
