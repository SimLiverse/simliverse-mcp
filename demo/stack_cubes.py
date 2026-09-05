"""Stack three cubes — ScriptNode action script.

Wired to OnPlaybackTick, so pressing Play *performs* the task. The tower is
re-derived by physics every run rather than being a set of poses somebody wrote
into the stage: stop, play, and it builds itself again from the flat row.

That distinction is the point. A scene whose cubes merely sit in a stack proves
nothing — it looks identical whether a robot placed them or a script teleported
them there. This is the reproducible form.

Structure follows the ScriptNode rules: setup() and compute() are both defined,
persistent state lives in module globals, and a timeline STOP subscription
resets it so the next Play reruns from the beginning. Motion goes through
`Manipulator.servo_to`, which advances one control tick without stepping
physics -- the timeline owns stepping here, and calling the blocking
`move_ee_to` from inside compute() would double-advance the world.
"""

import carb
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
ROBOT_PATH = "/World/Franka"
CUBE_PATHS = ["/World/CubeA", "/World/CubeB", "/World/CubeC"]
BASE_INDEX = 1  # CubeB stays put and becomes the bottom of the tower
CUBE_SIZE = 0.05
APPROACH_HEIGHT = 0.15
DROP_CLEARANCE = 0.004
RETREAT_HEIGHT = 0.18
WARMUP_FRAMES = 30
GRIPPER_FRAMES = 45
SETTLE_FRAMES = 30

# ── States ───────────────────────────────────────────────────────────────────
WARMUP, INIT, NEXT_CUBE = 0, 1, 2
OPEN_GRIPPER, APPROACH_PICK, DESCEND_PICK, CLOSE_GRIPPER = 3, 4, 5, 6
LIFT, APPROACH_PLACE, DESCEND_PLACE, RELEASE, RETREAT = 7, 8, 9, 10, 11
SETTLE, DONE = 12, 13
_NAMES = [
    "WARMUP",
    "INIT",
    "NEXT_CUBE",
    "OPEN_GRIPPER",
    "APPROACH_PICK",
    "DESCEND_PICK",
    "CLOSE_GRIPPER",
    "LIFT",
    "APPROACH_PLACE",
    "DESCEND_PLACE",
    "RELEASE",
    "RETREAT",
    "SETTLE",
    "DONE",
]

# ── Persistent state ─────────────────────────────────────────────────────────
_state = WARMUP
_frame = 0
_arm = None
_cubes = None
_order = []
_layer = 0
_pick = None
_place = None
_timeline_sub = None


def _log(message):
    carb.log_warn(f"[StackCubes] {message}")


def _go(state):
    global _state, _frame
    _log(f"{_NAMES[_state]} -> {_NAMES[state]}")
    _state = state
    _frame = 0


def _reset():
    global _state, _frame, _arm, _cubes, _order, _layer, _pick, _place
    _state, _frame = WARMUP, 0
    _arm, _cubes, _order, _layer = None, None, [], 0
    _pick, _place = None, None


def _on_timeline(event):
    import omni.timeline

    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _reset()
        _log("Reset — next Play reruns the stack from scratch.")


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)
        _log("setup()")


def compute(db=None):
    global _state, _frame, _arm, _cubes, _order, _layer, _pick, _place

    _frame += 1

    # Physics needs a few frames before articulation views can be built.
    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        try:
            from simliverse_sim import RigidObject, Scene
            from simliverse_sim.robots.manipulator import Manipulator

            scene = Scene.get()
            _arm = Manipulator(ROBOT_PATH, scene=scene)
            _cubes = {path: RigidObject(path, scene=scene) for path in CUBE_PATHS}
            _order = [p for i, p in enumerate(CUBE_PATHS) if i != BASE_INDEX]
            _layer = 0
            _go(NEXT_CUBE)
        except Exception as exc:  # retry on the next tick rather than wedging
            _log(f"init failed ({exc}); retrying")
            _state, _frame = WARMUP, 0
        return True

    if _state == DONE:
        return True

    base = _cubes[CUBE_PATHS[BASE_INDEX]]

    if _state == NEXT_CUBE:
        if not _order:
            _go(SETTLE)
            return True
        _layer += 1
        target_cube = _cubes[_order[0]]
        _pick = np.asarray(target_cube.position, dtype=float)
        origin = np.asarray(base.position, dtype=float)
        _place = np.array([origin[0], origin[1], origin[2] + _layer * CUBE_SIZE])
        _log(f"cube {_order[0]} -> layer {_layer} at {_place.round(3).tolist()}")
        _go(OPEN_GRIPPER)
        return True

    if _state == OPEN_GRIPPER:
        _arm.gripper.set_position(_arm.gripper.open_width, settle_steps=0)
        if _frame >= GRIPPER_FRAMES:
            _go(APPROACH_PICK)
        return True

    if _state == APPROACH_PICK:
        if _arm.servo_to(_pick + np.array([0.0, 0.0, APPROACH_HEIGHT])):
            _go(DESCEND_PICK)
        return True

    if _state == DESCEND_PICK:
        # Re-read the cube: it may have been nudged while the arm approached.
        _pick = np.asarray(_cubes[_order[0]].position, dtype=float)
        if _arm.servo_to(_pick):
            _go(CLOSE_GRIPPER)
        return True

    if _state == CLOSE_GRIPPER:
        _arm.gripper.set_position(0.0, settle_steps=0)
        _arm.servo_to(_pick)  # hold station while the fingers close
        if _frame >= GRIPPER_FRAMES:
            _go(LIFT)
        return True

    if _state == LIFT:
        if _arm.servo_to(_pick + np.array([0.0, 0.0, APPROACH_HEIGHT])):
            _go(APPROACH_PLACE)
        return True

    if _state == APPROACH_PLACE:
        if _arm.servo_to(_place + np.array([0.0, 0.0, APPROACH_HEIGHT])):
            _go(DESCEND_PLACE)
        return True

    if _state == DESCEND_PLACE:
        if _arm.servo_to(_place + np.array([0.0, 0.0, DROP_CLEARANCE])):
            _go(RELEASE)
        return True

    if _state == RELEASE:
        _arm.gripper.set_position(_arm.gripper.open_width, settle_steps=0)
        _arm.servo_to(_place + np.array([0.0, 0.0, DROP_CLEARANCE]))
        if _frame >= GRIPPER_FRAMES:
            _order.pop(0)
            _go(RETREAT)
        return True

    if _state == RETREAT:
        if _arm.servo_to(_place + np.array([0.0, 0.0, RETREAT_HEIGHT])):
            _go(NEXT_CUBE)
        return True

    if _state == SETTLE:
        # Park clear of the tower so the arm is not holding it up.
        _arm.servo_to(np.array([0.35, -0.35, 0.45]))
        if _frame >= SETTLE_FRAMES * 4:
            heights = {p: round(float(c.position[2]), 4) for p, c in _cubes.items()}
            _log(f"done — heights {heights}")
            _go(DONE)
        return True

    return True
