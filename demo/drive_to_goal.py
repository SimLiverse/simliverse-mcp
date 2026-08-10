"""Drive the rover to a goal and stop there.

There is no `servo_to` for a wheeled base — `drive_to` blocks, which the brief
says cannot go inside compute(), so the control loop is written here from
`drive(linear, angular)`, which commands one tick and returns.

`stop()` settles by default, and settling steps physics, so the halt is
`drive(0, 0)` instead.
"""

import math

WARMUP, INIT, TURN, DRIVE, HALT, DONE, FAILED = range(7)
NAMES = ["WARMUP", "INIT", "TURN", "DRIVE", "HALT", "DONE", "FAILED"]
TRACE_PATH = "/tmp/ctl_trace.log"

GOAL = [2.0, 0.0]
ARRIVED = 0.08          # metres
AIMED = 0.05            # radians
MAX_LINEAR = 0.5
MAX_ANGULAR = 1.0
WARMUP_FRAMES = 30
HALT_FRAMES = 60
LIMIT = 2000            # driving 2 m takes longer than an arm reach

_state = WARMUP
_frame = 0
_rover = None


def _trace(line):
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _go(state):
    global _state, _frame
    try:
        p = [round(float(v), 3) for v in _rover.base_position]
    except Exception:
        p = None
    _trace("%-7s -> %-7s frames=%5d at=%s" % (NAMES[_state], NAMES[state], _frame, p))
    _state, _frame = state, 0


def _yaw():
    """Heading from the base quaternion (w, x, y, z)."""
    w, x, y, z = [float(v) for v in _rover.base_orientation]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _to_goal():
    p = _rover.base_position
    dx, dy = GOAL[0] - float(p[0]), GOAL[1] - float(p[1])
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    error = math.atan2(math.sin(bearing - _yaw()), math.cos(bearing - _yaw()))
    return distance, error


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _rover
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _rover = WARMUP, 0, None
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
    global _state, _frame, _rover
    _frame += 1

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import Robot, Scene

        _rover = Robot.attach("/World/Rover", scene=Scene.get())
        _trace("init: %s dof=%d wheels=%s" % (type(_rover).__name__, _rover.dof,
                                              _rover.wheel_indices))
        _go(TURN)
        return True

    if _state in (DONE, FAILED):
        return True

    distance, error = _to_goal()

    if _state == TURN:
        # Point at the goal before moving, so the drive leg is a straight line.
        if abs(error) < AIMED or _frame > LIMIT:
            _rover.drive(0.0, 0.0)
            _go(DRIVE)
            return True
        _rover.drive(0.0, max(-MAX_ANGULAR, min(MAX_ANGULAR, 2.0 * error)))
        return True

    if _state == DRIVE:
        if distance < ARRIVED or _frame > LIMIT:
            _trace("  arrived: distance=%.3f heading_err=%.3f" % (distance, error))
            _go(HALT)
            return True
        # Ease off as the goal approaches, and keep steering the whole way.
        speed = max(0.08, min(MAX_LINEAR, 0.8 * distance))
        _rover.drive(speed, max(-MAX_ANGULAR, min(MAX_ANGULAR, 1.5 * error)))
        return True

    if _state == HALT:
        # drive(0, 0) rather than stop(): stop() settles, and settling steps
        # physics, which a controller must never do from inside compute().
        _rover.drive(0.0, 0.0)
        if _frame >= HALT_FRAMES:
            d, _ = _to_goal()
            _trace("  halted at distance=%.3f from goal" % d)
            _go(DONE)
        return True

    return True
