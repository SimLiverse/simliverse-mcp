"""Slalom a rover past three staggered walls to a goal 5.2 m out.

Closed-loop point-to-point, via `WheelBasePoseController`. The route is authored
because nothing in the wheeled stack finds one: the trajectory tools smooth and
track a path, they do not decide it.

Measured, not estimated: the rover is 0.673 x 0.626 m, so half a length is
0.337 and half a width 0.313. Every number below comes from those.

    WallA  x 1.15..1.25  blocks y <= 0.3   -> cross at y = 0.8  (edge 0.487)
    WallB  x 2.75..2.85  blocks y >= 0.3   -> cross at y = -0.2 (edge 0.113)
    WallC  x 4.35..4.45  blocks y <= 0.3   -> cross at y = 0.8

`PoseDriver` pivots in place at every waypoint, so each waypoint is also a place
the rover has to be able to turn: all of them sit at least half a length clear
of the nearest barrier, and every crossing is a straight run between two of
them. The spacing matters as much as the control — at 1.0 m between walls the
window for a 1.0 m lateral swing is 0.19 m wide and no controller can drive it.
"""

WARMUP, INIT, DRIVE, HALT, DONE, FAILED = range(6)
NAMES = ["WARMUP", "INIT", "DRIVE", "HALT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/ctl_trace.log"

ROUTE = [
    [0.70, 0.80],   # line up short of A
    [1.70, 0.80],   # straight past A
    [2.30, -0.20],  # swing down, clear of B
    [3.30, -0.20],  # straight past B
    [3.90, 0.80],   # swing up, clear of C
    [4.90, 0.80],   # straight past C
    [5.20, 0.00],   # in to the goal
]
WALLS = ("/World/WallA", "/World/WallB", "/World/WallC")
WARMUP_FRAMES = 30
HALT_FRAMES = 60
LIMIT = 2000

_state = WARMUP
_frame = 0
_leg = 0
_rover = None
_driver = None
_bumped = False


def _trace(line):
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _go(state):
    global _state, _frame
    try:
        p = [round(float(v), 3) for v in _rover.base_position[:2]]
    except Exception:
        p = None
    _trace("leg=%d %-6s -> %-6s frames=%5d at=%s" % (_leg, NAMES[_state], NAMES[state], _frame, p))
    _state, _frame = state, 0


def _watch_walls():
    """The walls are the constraint; note the first touch and where it happened."""
    global _bumped
    if _bumped:
        return
    for wall in WALLS:
        if _rover.touching(wall):
            _bumped = True
            _trace("  BUMPED %s on leg %d at %s"
                   % (wall, _leg, [round(float(v), 3) for v in _rover.base_position[:2]]))
            return


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _leg, _rover, _driver, _bumped
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _leg = WARMUP, 0, 0
        _rover, _driver, _bumped = None, None, False
        open(TRACE_PATH, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
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
    global _state, _frame, _leg, _rover, _driver

    _frame += 1

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import Robot, Scene

        _rover = Robot.attach("/World/Rover", scene=Scene.get())
        _driver = _rover.pose_driver(max_linear=0.4, max_angular=1.0, position_tol=0.10)
        _trace("init: %s, %d waypoints" % (type(_rover).__name__, len(ROUTE)))
        _go(DRIVE)
        return True

    if _state in (DONE, FAILED):
        return True

    _watch_walls()

    if _state == DRIVE:
        if _driver.step(ROUTE[_leg]) or _frame > LIMIT:
            _trace("  leg %d done at %s"
                   % (_leg, [round(float(v), 3) for v in _rover.base_position[:2]]))
            if _leg + 1 < len(ROUTE):
                _leg += 1
                _go(DRIVE)
            else:
                _go(HALT)
        return True

    if _state == HALT:
        # drive(0, 0), not stop(): stop() settles, and settling steps physics,
        # which a controller must never do from inside compute().
        _rover.drive(0.0, 0.0)
        if _frame >= HALT_FRAMES:
            _trace("  finished; touched a wall: %s" % _bumped)
            _go(DONE)
        return True

    return True
