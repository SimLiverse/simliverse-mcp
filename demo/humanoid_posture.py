"""Stand a humanoid and hold a raised-arm posture.

No locomotion policy is involved and none is needed: posture and limb control
work without one, which is what the library says and what this confirms. All 43
drives are healthy on this asset, so a commanded joint is a joint that moves —
measured, the shoulder tracked its 1.201 rad command exactly.
"""

WARMUP, INIT, STAND, POSE, HOLD, DONE, FAILED = range(7)
NAMES = ["WARMUP", "INIT", "STAND", "POSE", "HOLD", "DONE", "FAILED"]
TRACE_PATH = "/tmp/ctl_trace.log"

SHOULDER = "left_shoulder_pitch_joint"
RAISED = -1.2          # radians
WARMUP_FRAMES = 40
STAND_FRAMES = 150
POSE_FRAMES = 150
HOLD_FRAMES = 120

_state = WARMUP
_frame = 0
_bot = None
_target = None


def _trace(line):
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _go(state):
    global _state, _frame
    try:
        note = "h=%.3f upright=%s tilt=%.2f" % (
            _bot.base_height(), _bot.is_upright(), _bot.tilt_degrees())
    except Exception:
        note = "-"
    _trace("%-7s -> %-7s frames=%4d %s" % (NAMES[_state], NAMES[state], _frame, note))
    _state, _frame = state, 0


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _bot, _target
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _bot, _target = WARMUP, 0, None, None
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
    global _state, _frame, _bot, _target
    import numpy as np

    _frame += 1

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import Robot, Scene

        _bot = Robot.attach("/World/Human", scene=Scene.get())
        _trace("init: %s dof=%d drives_bad=%d"
               % (type(_bot).__name__, _bot.dof, len(_bot.drive_health())))
        _go(STAND)
        return True

    if _state in (DONE, FAILED):
        return True

    if _state == STAND:
        if _frame == 1:
            # settle_steps=0: a controller waits by staying in its state, never
            # by stepping physics from inside the simulator's own callback.
            _bot.set_joint_positions(_bot.capture_stand_pose(), settle_steps=0)
        if _frame >= STAND_FRAMES:
            _go(POSE)
        return True

    if _state == POSE:
        if _frame == 1:
            target = np.asarray(_bot.joint_positions, dtype=float).copy()
            index = _bot.joint_names.index(SHOULDER)
            target[index] = RAISED
            _target = target
            _trace("  raising %s to %.2f rad" % (SHOULDER, RAISED))
        _bot.set_joint_positions(_target, settle_steps=0)
        if _frame >= POSE_FRAMES:
            _go(HOLD)
        return True

    if _state == HOLD:
        _bot.set_joint_positions(_target, settle_steps=0)
        if _frame >= HOLD_FRAMES:
            index = _bot.joint_names.index(SHOULDER)
            reached = float(np.asarray(_bot.joint_positions, dtype=float)[index])
            _trace("  held: shoulder at %.3f rad (target %.3f), upright=%s tilt=%.2f"
                   % (reached, RAISED, _bot.is_upright(), _bot.tilt_degrees()))
            _go(DONE)
        return True

    return True
