"""A KR210 palletising boxes off a driven conveyor. NOT YET REPLAYED.

Like `stack_on_pallet.py` was before it, this has not been through
`controller.deliver` — no replay has run against a live Isaac session, so
nothing here is measured *as a controller*. The structure is proven (it is the
state machine from `two_cube_transfer.py`, which was delivered and does replay)
and the geometry comes from assets measured offline with usd-core. What has not
happened is the part that counts. Treat every tolerance below as a starting
value, run `controller.deliver`, and fix what the report names.

The task: boxes ride a belt driven by PhysX surface velocity, come to rest
against a stop, and get lifted onto a pallet in a 2 x 2 x 2 pattern. Eight
picks from one fixed pose to eight different ones.

Three things here are shaped by the conveyor rather than by the arm:

* **The pick pose is read from the box, not assumed.** The stop makes every box
  arrive at the same place, which is what makes the cycle repeatable — but
  "the same place" is a fact to measure once per box, not a constant to hard
  code, because a box that arrives skewed or riding up on its neighbour is
  exactly the case a hard-coded pose gets wrong silently.

* **WAIT_FOR_BOX is a real state.** `belt.box_at_gate()` returns None until a
  box has both reached the stop and stopped moving. Picking on contact closes
  the cup on a box that is still being pushed, and the grasp fails in a way
  that reads as a gripper fault. The belt keeps running underneath the queue
  the whole time; that is what a real infeed does.

* **The belt is halted while the arm is over it.** Not for physics reasons —
  for the next box. With the belt running, the moment box N is lifted, box N+1
  starts moving into the stop, and it can arrive while the arm is still
  retreating through the same space.
"""

(WARMUP, INIT, WAIT_FOR_BOX, OVER_BOX, DOWN_TO_BOX, GRIP, LIFT, TRAVERSE,
 OVER_SLOT, PLACE, RELEASE, RETREAT, NEXT, DONE, FAILED) = range(15)

NAMES = ["WARMUP", "INIT", "WAIT_FOR_BOX", "OVER_BOX", "DOWN_TO_BOX", "GRIP",
         "LIFT", "TRAVERSE", "OVER_SLOT", "PLACE", "RELEASE", "RETREAT",
         "NEXT", "DONE", "FAILED"]

ARM = "/World/Arm"
BELT = "/World/Conveyor"
TRACE = "/tmp/kuka_palletizing.log"

# KR210 flange pointing at the floor.
DOWN = [0.0, 1.0, 0.0, 0.0]

BOX = 0.30                      # full box size, metres
PALLET_Y = 1.90
PALLET_DECK_Z = 0.1425
ROWS, COLS, LAYERS = 2, 2, 2

CARRY_Z = 1.65                  # travel height: clear of a two-layer stack
APPROACH = 0.35                 # above a box before descending onto it
GRIP_LIFT = 0.010               # cup height above the box face when it seals
COARSE = 0.12                   # first stage of the descent onto the pallet
FINAL = 0.003                   # release height: set down, do not drop

GRIP_FRAMES = 45
RELEASE_FRAMES = 45
SETTLE_FRAMES = 30
WARMUP_FRAMES = 30
WAIT_LIMIT = 3000               # a box should reach the stop well inside this
LIMIT = 2000                    # per motion state

# The KR210's ready pose. Re-homed before every pick: the wrist winds up over a
# sequence of solves, and once a joint is against its travel limit a demanded
# orientation can only be met by driving into the stop, so the solver gives up
# position instead and every target lands short. It bites on pick two, not one.
HOME = [0.0, -1.396, 1.396, 0.0, 1.571, 0.0]

_state = WARMUP
_frame = 0
_job = 0
_arm = None
_belt = None
_slots = None
_pick = None
_held = None


def _trace(line):
    with open(TRACE, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _go(state):
    global _state, _frame
    _trace("job=%d %-13s -> %-13s after %4d frames" % (_job, NAMES[_state], NAMES[state], _frame))
    _state, _frame = state, 0


def _on_timeline(event):
    import omni.timeline

    global _state, _frame, _job, _arm, _belt, _slots, _pick, _held
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _job = WARMUP, 0, 0
        _arm, _belt, _slots, _pick, _held = None, None, None, None, None
        open(TRACE, "w", encoding="utf-8").close()


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    global _state, _frame, _job, _arm, _belt, _slots, _pick, _held
    _frame += 1

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:
        # Physics needs a few frames before an articulation can be read.
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import Conveyor, RigidObject, Robot, Scene, pallet_slots

        scene = Scene.get()
        _arm = Robot.attach(ARM, scene=scene)
        # The cup was authored while the scene was built, because a surface
        # gripper created after the timeline starts is never registered. This
        # handle is new on every Play, so bind it to the cup already there —
        # calling attach_suction_gripper() here would author a second one.
        _arm.rebind_suction()
        # Attach to the belt already on the stage rather than rebuilding it —
        # `build()` from inside compute() would author a second belt over the
        # first one on every Play. The belt's geometry and drive come from the
        # stamp its builder wrote on the prim, so nothing here is pasted in:
        # an earlier version carried the builder's numbers as constants, and
        # they went stale the moment the cell layout moved.
        _belt = Conveyor.attach(BELT, scene=scene)
        paths = sorted(scene.find("Box"))
        _belt.track([RigidObject(path, scene=scene) for path in paths])
        import numpy as _np

        # The stamp predates load(), so it carries no box size; the queue
        # reader needs it to tell "at the stop" from "half a box short".
        _belt.box_size = _np.array([BOX, BOX, BOX])
        _trace("bound %d boxes: %s" % (len(paths), paths))

        # After Play, always. A stop between the belt being switched on and the
        # timeline starting drops the drive, and authoring the suction cup stops
        # the timeline. Cheap and idempotent, so it is done unconditionally.
        _belt.start()

        _slots = pallet_slots(
            origin=[0.0, PALLET_Y, PALLET_DECK_Z], box=(BOX, BOX, BOX),
            rows=ROWS, cols=COLS, layers=LAYERS, gap=0.01,
        )
        _arm.set_joint_positions(HOME, settle_steps=0)
        _go(WAIT_FOR_BOX)
        return True

    if _state == WAIT_FOR_BOX:
        if _frame > WAIT_LIMIT:
            _trace("  no box reached the stop in %d frames" % WAIT_LIMIT)
            _go(FAILED)
            return True
        ready = _belt.box_at_gate()
        if ready is not None:
            _held = ready
            _pick = [float(v) for v in ready.position]
            _belt.halt()
            _go(OVER_BOX)
        return True

    if _frame > LIMIT:
        _trace("  TIMEOUT in %s" % NAMES[_state])
        _go(FAILED)
        return True

    slot = _slots[_job]

    if _state == OVER_BOX:
        if _arm.servo_to([_pick[0], _pick[1], _pick[2] + APPROACH], DOWN, tolerance=0.020):
            _go(DOWN_TO_BOX)

    elif _state == DOWN_TO_BOX:
        # Onto the box's top face, not its centre.
        target_z = _pick[2] + BOX / 2.0 + GRIP_LIFT
        if _arm.servo_to([_pick[0], _pick[1], target_z], DOWN, tolerance=0.012):
            _arm.suction.close(settle_steps=0)
            _go(GRIP)

    elif _state == GRIP:
        if _frame >= GRIP_FRAMES:
            if not _arm.suction.holding:
                _trace("  cup did not seal on %s" % _held.prim_path)
                _go(FAILED)
                return True
            _go(LIFT)

    elif _state == LIFT:
        if _arm.servo_to([_pick[0], _pick[1], CARRY_Z], DOWN, tolerance=0.025):
            _go(TRAVERSE)

    elif _state == TRAVERSE:
        if _arm.servo_to([slot["rest"][0], slot["rest"][1], CARRY_Z], DOWN, tolerance=0.025):
            _go(OVER_SLOT)

    elif _state == OVER_SLOT:
        if _arm.servo_to([slot["rest"][0], slot["rest"][1],
                          slot["rest"][2] + BOX / 2.0 + COARSE], DOWN, tolerance=0.018):
            _go(PLACE)

    elif _state == PLACE:
        # Two-stage descent. A single move overshoots on arrival and nudges what
        # is already stacked; measured on a three-cube tower that came apart.
        if _arm.servo_to([slot["rest"][0], slot["rest"][1],
                          slot["rest"][2] + BOX / 2.0 + FINAL], DOWN, tolerance=0.008):
            _arm.suction.open(settle_steps=0)
            _go(RELEASE)

    elif _state == RELEASE:
        if _frame >= RELEASE_FRAMES:
            _go(RETREAT)

    elif _state == RETREAT:
        if _arm.servo_to([slot["rest"][0], slot["rest"][1], CARRY_Z], DOWN, tolerance=0.025):
            _go(NEXT)

    elif _state == NEXT:
        if _frame < SETTLE_FRAMES:
            return True
        _job += 1
        _pick, _held = None, None
        if _job >= len(_slots):
            _trace("ALL DONE")
            _go(DONE)
        else:
            # Re-home between picks, and start the belt again so the next box
            # runs down to the stop while the arm is on its way back.
            _arm.set_joint_positions(HOME, settle_steps=0)
            _belt.start()
            _go(WAIT_FOR_BOX)

    return True
