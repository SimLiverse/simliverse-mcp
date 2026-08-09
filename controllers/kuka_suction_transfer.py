"""Worked example: a Kuka KR210 on suction, moving boxes between two tables.

Move two boxes from the pick table to the place table, ninety degrees round,
without disturbing the third.

Three things here are not obvious, and each one was a failure first:

* The tool is held **square to the box** by `pose_to`, not by `move_ee_to`.
  RMPflow treats orientation as a soft objective and settles 11 degrees off
  vertical however long it runs, and a suction cup 11 degrees off its surface
  does not seal. `pose_to` solves the pose with IK and closes the loop in joint
  space; measured residuals here are under 1 mm and 0.1 degrees.

* Every move is **ramped through Cartesian waypoints**, one per tick group.
  Commanded straight to a pose, this arm swings: a descent to a point 25 cm
  clear of a box took the tool through the table on the way, and threw a 5 kg
  box 28 m. The waypoints are warm-started from each other, because solved
  independently they come back in alternating IK branches.

* The turn from tool-horizontal to tool-down happens **once, at HOME, high and
  away from the tables**. That rotation swings the arm through half a metre
  whatever speed it is run at, so it is done where there is nothing to hit.
"""

(WARMUP, INIT, HOME, TRANSIT, DESCEND, GRIP, LIFT, SWING, LOWER, RELEASE,
 RETREAT, NEXT, DONE, FAILED) = range(14)

NAMES = ["WARMUP", "INIT", "HOME", "TRANSIT", "DESCEND", "GRIP", "LIFT", "SWING",
         "LOWER", "RELEASE", "RETREAT", "NEXT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/kuka_suction_trace.log"

ARM = "/World/Arm"
JOBS = [
    {"box": "/World/Box1", "place": [0.0, 1.90]},
    {"box": "/World/Box0", "place": [0.0, 1.45]},
]
UNDISTURBED = "/World/Box2"

# Flange +X onto world -Z: a +90 degree turn about world Y. This arm's tool
# points along its own +X, which `attach_suction_gripper` works out by measuring
# rather than being told.
DOWN = [0.70710678, 0.0, 0.70710678, 0.0]

BOX_TOP = 1.30        # table top 1.00 + a 30 cm box
TIP = 0.04            # the cup tip stands this far past the flange
HOME_POSE = [1.2, 0.0, 2.30]
CLEAR = 0.45          # transit height above the box tops
GRASP = 0.02          # cup tip above the box face when suction is applied
SET_DOWN = 0.05       # release height: the box is lowered, not dropped

# Long moves get more waypoints and more time per waypoint. Short vertical ones
# do not need it - a pure descent tracks its command exactly - and the whole
# replay is otherwise dominated by them.
FAR = (16, 24)
NEAR = (8, 12)
SETTLE = 90
GRIP_FRAMES = 60
RELEASE_FRAMES = 45
WARMUP_FRAMES = 30
LIMIT = 1400          # generous: a FAR move is legitimately ~700 frames

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_cup = None
_move_phase = 0
_move_ticks = 0
_hit = None


def _trace(line):
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _go(state):
    global _state, _frame, _move_phase, _move_ticks
    try:
        ee = [round(float(v), 3) for v in _arm.ee_position]
    except Exception:
        ee = None
    _trace("job=%d %-8s -> %-8s frames=%4d ee=%s held=%s"
           % (_job, NAMES[_state], NAMES[state], _frame, ee,
              _cup.gripped_objects if _cup else None))
    _state, _frame = state, 0
    _move_phase, _move_ticks = 0, 0


def _move(target, ramp, waypoint_frames):
    """Advance one pose move by a single tick. True once it has arrived.

    One transition per call: issue the command, release one waypoint every
    `waypoint_frames`, settle, apply the single joint-space correction, settle
    again. Physics is never stepped from here - the graph ticks it.
    """
    global _move_phase, _move_ticks
    if _move_phase == 0:
        _arm.command_pose(target, DOWN, ramp=ramp, raise_on_fail=False)
        _move_phase, _move_ticks = 1, 0
        return False

    _move_ticks += 1
    if _move_phase == 1:
        if _move_ticks % waypoint_frames == 0 and _arm.advance_pose():
            _move_phase, _move_ticks = 2, 0
        return False
    if _move_phase == 2:
        if _move_ticks >= SETTLE:
            _arm.refine_pose()
            _move_phase, _move_ticks = 3, 0
        return False
    if _move_ticks >= SETTLE:
        _move_phase = 0
        return True
    return False


def _watch_untouched():
    """Name the state that first disturbs the box that must not move."""
    global _hit
    if _hit is not None:
        return
    from simliverse_sim import RigidObject, Scene

    bodies = RigidObject(UNDISTURBED, scene=Scene.get()).contact_bodies()
    if any("Arm" in b or "Suction" in b for b in bodies):
        _hit = NAMES[_state]
        _trace("  TOUCHED %s during %s" % (UNDISTURBED, _hit))


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _cup, _move_phase, _move_ticks, _hit
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _cup = None, None
        _move_phase, _move_ticks, _hit = 0, 0, None
        open(TRACE_PATH, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    global _state, _frame, _job, _arm, _cup
    _frame += 1

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:
        # Physics needs a few frames before an articulation can be read.
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import Robot, Scene
        from simliverse_sim.robots.manipulator import SuctionGripper

        scene = Scene.get()
        _arm = Robot.attach(ARM, scene=scene)
        _cup = SuctionGripper(
            "/World/World_Arm_tool0_SuctionCup/SurfaceGripper",
            scene=scene, max_grip_distance=0.05,
        )
        _go(HOME)
        return True

    _watch_untouched()
    if _frame > LIMIT:
        _trace("  TIMEOUT in %s" % NAMES[_state])
        _go(FAILED)
        return True

    job = JOBS[_job]
    box_x, box_y = 1.90, [-0.45, 0.0, 0.45][int(job["box"][-1])]
    place_x, place_y = job["place"]

    if _state == HOME:
        if _move(HOME_POSE, *FAR):
            _go(TRANSIT)
    elif _state == TRANSIT:
        if _move([box_x, box_y, BOX_TOP + CLEAR + TIP], *FAR):
            _go(DESCEND)
    elif _state == DESCEND:
        if _move([box_x, box_y, BOX_TOP + GRASP + TIP], *NEAR):
            _cup.close(settle_steps=0)
            _go(GRIP)
    elif _state == GRIP:
        if _frame >= GRIP_FRAMES:
            if not _cup.is_holding(job["box"]):
                _trace("  NO LATCH on %s (status=%s)" % (job["box"], _cup.status))
                _go(FAILED)
            else:
                _go(LIFT)
    elif _state == LIFT:
        if _move([box_x, box_y, BOX_TOP + CLEAR + TIP], *NEAR):
            _go(SWING)
    elif _state == SWING:
        if _move([place_x, place_y, BOX_TOP + CLEAR + TIP], *FAR):
            _go(LOWER)
    elif _state == LOWER:
        if _move([place_x, place_y, BOX_TOP + SET_DOWN + TIP], *NEAR):
            _cup.open(settle_steps=0)
            _go(RELEASE)
    elif _state == RELEASE:
        if _frame >= RELEASE_FRAMES:
            _go(RETREAT)
    elif _state == RETREAT:
        if _move([place_x, place_y, BOX_TOP + CLEAR + TIP], *NEAR):
            _go(NEXT)
    elif _state == NEXT:
        _job += 1
        if _job >= len(JOBS):
            _trace("ALL DONE (touched=%s)" % _hit)
            _go(DONE)
        else:
            _go(TRANSIT)
    return True
