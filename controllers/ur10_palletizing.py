"""UR10 palletising, driven from the physics tick so it runs on Play.

Everything else in this branch is driven from outside: a script calls
`pick_waiting_box`, blocks while physics steps, and reads the result. Press Play
in the viewport and nothing happens, because nothing is wired to the tick.

This is that wiring. A ScriptNode on `OnPlaybackTick` runs `compute()` once per
frame, and one frame does one thing. The whole cell then belongs to the scene
rather than to whoever is holding the MCP connection.

## What changes when the driver is a tick rather than a script

**No blocking calls.** `pose_to`, `move_ee_to` and `settle` all step physics
internally, and stepping physics from inside a physics callback is re-entrant -
`_refuse_reentrant_step` throws on the first grasp. Motion here is `servo_to`,
which commands one tick and returns whether it has arrived. Every wait is a
frame count.

**Handles are rebuilt in INIT, never captured at authoring time.** Articulations
and gripper views are invalid across a stop, and this file is loaded once and
run on every Play. `INIT` re-attaches the arm, rebinds the cup and rebuilds the
belt handle from a description literal.

**The belt is not driven from here.** It is a PhysX surface velocity applied at
authoring time, so it conveys on its own from the moment Play starts. The
controller only waits for what it delivers.

## The measured numbers this depends on

From the same cell run outside: 58.61 s per carton, four of four placed, 3.7 to
7.5 mm from target. The pick sequence is the one that produced them - approach
high, descend until the cup seals, and require a *named* gripped object rather
than trusting the status token, which reaches Closed while holding nothing.
"""

import carb
import numpy as np

WARMUP_FRAMES = 30

#: Frames a servo state may spend before it is called stuck. At 60 Hz this is
#: ten seconds - ample for a 30 cm lift or a descent.
STATE_LIMIT = 600

#: The traverse gets its own, much larger budget. It carries a carton right
#: across the cell - about 1.07 m from the belt to the far pallet slot - and
#: `servo_to` is a reactive policy taking one step per tick, not a plan. On the
#: shared budget it timed out mid-swing and reported "could not traverse to slot
#: 0", which reads as unreachable and was only slow.
TRAVERSE_LIMIT = 2400

#: Frames to hold still after arriving, before the cup is asked to seal. The
#: descent overshoots and settles; sealing mid-oscillation is what made the
#: standoff sweep look like a gripper fault.
SETTLE_FRAMES = 36

#: Frames given to a single seal attempt before dropping 2 mm and retrying.
SEAL_FRAMES = 30

#: How far down each retry steps, and how many are allowed.
SEAL_STEP = 0.002
SEAL_TRIES = 10

BOX = 0.15
STANDOFF = 0.006
DOWN = [0.0, 1.0, 0.0, 0.0]
HOME = [0.0, -1.8, 1.5, -1.3, -1.57, 0.0]
ARM_PATH = "/World/UR"

#: Pasted, not re-derived. Re-deriving the belt inside the controller is what
#: put its idea of the belt a metre from the real one when the cell moved.
#: Pasted from `belt.describe()` verbatim, not written by hand.
#:
#: The first version of this dict was hand-typed from what the belt "obviously"
#: needs, and INIT died on `KeyError: 'belt_path'` - a key the description
#: carries and my transcription did not. Re-deriving it inside the controller is
#: worse still: that is what put a controller's idea of the belt a metre from
#: the real one when the cell moved.
BELT = {
    "belt_path": "/World/Belt",
    "body_path": "/World/Belt",
    "box_size": [0.15, 0.15, 0.15],
    "boxes": ["/World/Box0", "/World/Box1", "/World/Box2", "/World/Box3"],
    "centre": [-0.05, -0.4, 0.45],
    "direction": [1.0, 0.0, 0.0],
    "gate_path": "/World/BeltGate",
    "length": 1.6,
    "mechanism": "PhysxSurfaceVelocityAPI",
    "running": True,
    "speed": 0.2,
    "top_z": 0.45,
    "width": 0.4,
}

SLOTS = [
    {"index": 0, "place": [-0.08, 0.67, 0.220], "rest": [-0.08, 0.67, 0.2175]},
    {"index": 1, "place": [0.08, 0.67, 0.220], "rest": [0.08, 0.67, 0.2175]},
    {"index": 2, "place": [0.08, 0.83, 0.220], "rest": [0.08, 0.83, 0.2175]},
    {"index": 3, "place": [-0.08, 0.83, 0.220], "rest": [-0.08, 0.83, 0.2175]},
]

(WARMUP, INIT, WAIT_BOX, OVER_BOX, DESCEND, SEAL, LIFT,
 TRAVERSE, OVER_SLOT, RELEASE, RETREAT, NEXT, DONE, FAILED) = range(14)

_state = WARMUP
_frame = 0
_arm = None
_cup = None
_belt = None
_scene = None
_slot = 0
_tries = 0
_carton = None
_pick = None
_lift_z = 0.0
_why = ""
_placed = 0


#: A ScriptNode cannot report. It runs in its own namespace inside the graph, so
#: nothing outside can import it and read its state, and `carb.log_warn` did not
#: surface anywhere reachable either - the first delivered run came back
#: "reproduced: True, moved: []" with no way at all to ask what it had done.
#: One line per transition, in a file, is enough to answer that.
STATUS_PATH = "/tmp/ur10_palletizing.status"

_NAMES = {}


def _note(tag):
    """Record where the machine is, so a stalled run can be asked about."""
    try:
        if not _NAMES:
            for name, value in list(globals().items()):
                if name.isupper() and isinstance(value, int) and 0 <= value < 14:
                    _NAMES.setdefault(value, name)
        with open(STATUS_PATH, "a", encoding="utf-8") as handle:
            line = "%-9s state=%-10s frame=%-5d slot=%d tries=%d why=%s" % (
                tag, _NAMES.get(_state, _state), _frame, _slot, _tries, _why)
            handle.write(line + chr(10))
    except Exception:  # noqa: BLE001 - reporting must never break the run
        pass


def _go(state):
    global _state, _frame
    _state, _frame = state, 0
    _note("enter")


def _fail(reason):
    global _why
    _why = reason
    carb.log_warn("ur10_palletizing FAILED: %s" % reason)
    _go(FAILED)


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _arm, _cup, _belt, _scene
    global _slot, _tries, _carton, _pick, _lift_z, _why, _placed
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame = WARMUP, 0
        _arm = _cup = _belt = _scene = _carton = _pick = None
        _slot = _tries = _placed = 0
        _lift_z = 0.0
        _why = ""
        try:
            open(STATUS_PATH, "w", encoding="utf-8").close()
        except Exception:  # noqa: BLE001
            pass


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def _top_of(carton):
    return float(np.asarray(carton.position, dtype=float)[2]) + BOX / 2.0


def compute(db=None):
    global _arm, _cup, _belt, _scene, _slot, _tries, _carton, _pick
    global _lift_z, _placed
    _frame_tick()

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        try:
            from simliverse_sim import Conveyor, RigidObject, Robot, Scene

            _scene = Scene.get()
            _arm = Robot.attach(ARM_PATH, scene=_scene)
            _cup = _arm.rebind_suction()
            _belt = Conveyor.from_description(BELT, scene=_scene)
            _belt.track([RigidObject(p, scene=_scene) for p in BELT["boxes"]])
            # Idempotent, and required: a stop between authoring and Play drops
            # the surface velocity, and INIT is the first thing that runs after
            # Play by construction.
            _belt.start()
            _arm.set_joint_positions(HOME, settle_steps=0)
        except Exception as exc:  # noqa: BLE001 - INIT must report, not raise
            _fail("INIT: %s: %s" % (type(exc).__name__, exc))
            return True
        _go(WAIT_BOX)
        return True

    if _state == WAIT_BOX:
        if _slot >= len(SLOTS):
            _go(DONE)
            return True
        _carton = _belt.box_at_gate()
        if _carton is not None:
            _belt.halt()
            _tries = 0
            _go(OVER_BOX)
        elif _frame > STATE_LIMIT * 2:
            _fail("no carton settled at the stop for slot %d" % _slot)
        return True

    if _state == OVER_BOX:
        here = np.asarray(_carton.position, dtype=float)
        target = [float(here[0]), float(here[1]),
                  _top_of(_carton) + _cup.tip_offset + 0.18]
        if _arm.servo_to(target, DOWN, tolerance=0.02):
            _pick = [float(here[0]), float(here[1])]
            _go(DESCEND)
        elif _frame > STATE_LIMIT:
            _fail("could not reach the approach above %s" % _carton.prim_path)
        return True

    if _state == DESCEND:
        target = [_pick[0], _pick[1],
                  _top_of(_carton) + _cup.tip_offset + STANDOFF - _tries * SEAL_STEP]
        if _arm.servo_to(target, DOWN, tolerance=0.006) or _frame > STATE_LIMIT // 2:
            _go(SEAL)
        return True

    if _state == SEAL:
        if _frame == SETTLE_FRAMES:
            _cup.close(settle_steps=0)
        if _frame > SETTLE_FRAMES:
            # A named object, not the status token: the token reaches Closed
            # while holding nothing, and believing it once cost a whole session.
            if _cup.holding and _cup.gripped_objects:
                _go(LIFT)
                return True
            if _frame > SETTLE_FRAMES + SEAL_FRAMES:
                _tries += 1
                if _tries >= SEAL_TRIES:
                    _fail("cup did not seal on %s after %d descents"
                          % (_carton.prim_path, SEAL_TRIES))
                    return True
                _cup.open(settle_steps=0)
                _go(DESCEND)
        return True

    if _state == LIFT:
        # Fix the height once, on the frame the lift begins.
        #
        # Computing it from the carton every frame makes the goal rise with the
        # thing being raised: the arm lifts the carton, `_top_of` returns a
        # larger number, the target moves up, and the arm chases it until the
        # frame limit and reports "could not lift clear". It was lifting fine.
        if _frame == 1:
            _lift_z = _top_of(_carton) + _cup.tip_offset + 0.30
        if _arm.servo_to([_pick[0], _pick[1], _lift_z], DOWN, tolerance=0.03):
            _go(TRAVERSE)
        elif _frame > STATE_LIMIT:
            _fail("could not lift %s clear" % _carton.prim_path)
        return True

    if _state == TRAVERSE:
        slot = SLOTS[_slot]
        target = [float(slot["place"][0]), float(slot["place"][1]),
                  0.55 + _cup.tip_offset + BOX / 2.0]
        if not (_cup.holding and _cup.gripped_objects):
            _fail("dropped %s during the traverse" % _carton.prim_path)
            return True
        if _arm.servo_to(target, DOWN, tolerance=0.03):
            _go(OVER_SLOT)
        elif _frame > TRAVERSE_LIMIT:
            _fail("could not traverse to slot %d" % _slot)
        return True

    if _state == OVER_SLOT:
        slot = SLOTS[_slot]
        target = [float(slot["place"][0]), float(slot["place"][1]),
                  float(slot["place"][2]) + BOX / 2.0 + _cup.tip_offset]
        if _arm.servo_to(target, DOWN, tolerance=0.01) or _frame > STATE_LIMIT:
            _go(RELEASE)
        return True

    if _state == RELEASE:
        if _frame == 1:
            _cup.open(settle_steps=0)
        if _frame > 24 and not _cup.holding:
            _go(RETREAT)
        elif _frame > STATE_LIMIT // 4:
            _go(RETREAT)          # released or not, do not sit here
        return True

    if _state == RETREAT:
        slot = SLOTS[_slot]
        target = [float(slot["place"][0]), float(slot["place"][1]),
                  float(slot["place"][2]) + BOX / 2.0 + _cup.tip_offset + 0.25]
        if _arm.servo_to(target, DOWN, tolerance=0.04) or _frame > STATE_LIMIT:
            _go(NEXT)
        return True

    if _state == NEXT:
        rest = np.asarray(SLOTS[_slot]["rest"], dtype=float)
        where = np.asarray(_carton.position, dtype=float)
        if float(np.linalg.norm(where - rest)) <= 0.05:
            _placed += 1
        _slot += 1
        _belt.start()
        _go(WAIT_BOX)
        return True

    return True


def _frame_tick():
    global _frame
    _frame += 1
