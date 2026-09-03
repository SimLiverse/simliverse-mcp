# MIT License
#
# Copyright (c) 2026 SimLiverse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The full infeed cell: belt, discharge, dead plate, stop, palletise.

`ur10_palletizing.py` bolts the stop to the belt, which makes the pick pose
fixed and the cell reproducible. This one is built the way a real cell is: the
belt *discharges*, the carton drops onto a fixed plate, slides on its own
momentum, and comes to rest against a mechanical stop. The arm picks from the
plate, not from the belt.

    belt deck 0.45 ─────────────┐
                                │ 120 mm drop
                                ▼
                  ┌──────────────────────────┐ ◀ stop
                  │   dead plate  0.33       │
                  └──────────────────────────┘

Measured on a live worker, cartons discharging at 0.30 m/s:

    t=2s   box1 mid-drop, z 0.424, v 0.795
    t=4s   box0 at the stop, x 0.945, v 0.003

x = stop − half a carton, which is the pose the pick expects.

## Everything here that is not obvious

**The plate must not overlap the belt, and its guides must sit below belt
level.** First attempt overlapped by 100 mm with guides topping out at 0.49
against a 0.45 belt deck: the queue ran into the guides and jammed. The 120 mm
drop exists to put the guide tops at 0.43, clear of the belt.

**Clear the previous cell before building this one.** `scene.stop()` rewinds
physics, it does not remove prims. Built over the older cell, this one inherited
that cell's gate still standing at x=0.75, and the cartons queued against a wall
from a scene that no longer existed. Every reading said "stuck at 0.675" and
none of them said why.

## OPEN: the traverse scatters the queue

One carton goes all the way through - belt, drop, plate, stop, pick, pallet,
placed 6.8 mm from its slot in 85.3 s. The second cycle then finds nothing at
the stop, because the first has thrown the rest of the queue across the cell:

    box0  [-0.085,  0.674, 0.218]   on the pallet, correct
    box1  [-0.033,  3.457, 0.075]   3.5 m away, on the floor
    box2  [ 0.634,  0.049, 0.075]   on the floor
    box3  [ 0.589, -0.627, 0.505]   off the side of the plate

y = 3.457 is not a nudge; something struck it hard. The belt cell had the same
class of problem and it was the home pose sweeping the queue at belt height —
fixed there by parking the tool 203 mm above the cartons. This cell has a
second deck 120 mm lower and a longer traverse to the pallet, and the same home
pose no longer clears everything it crosses.

The traverse is the suspect: the path from the plate at (0.805, -0.40) to the
pallet at (0.0, 0.75) passes over the plate and its queue, and nothing
currently constrains its height. The fix is likely a via-point at a height
measured from the plate rather than a single home pose, which is what the belt
cell got away with because it had only one deck.

Not yet attempted. The single-carton path is measured and works.

**The stop sits at 0.88, not further out.** The pick pose is
`stop − half a carton`, and at the first geometry that put it 1.03 m from the
base — inside a UR10's 1.3 m reach on paper, and close enough to the edge that
the approach above it is not. Reach is measured to the *tool*, and the tool is
161 mm past the flange.
"""

import numpy as np

from simliverse_sim import (
    Conveyor,
    DeadPlate,
    Escapement,
    RigidObject,
    Robot,
    Scene,
    pallet_slots,
    spawn_prop,
)

from .ur10_palletizing import DOWN, STANDOFF, light_the_cell, place_on_slot

ARM = "/World/UR"
BELT = "/World/Belt"
PLATE = "/World/Plate"
PALLET = "/World/Pallet"
BLADE = "/World/Escapement"

BOX = 0.15
BOX_MASS = 1.0
BELT_DECK = 0.45
PLATE_DECK = 0.33              #: 120 mm below the belt. See the module docstring.
BELT_END = 0.50                #: where the belt stops and the carton leaves it
PLATE_STOP = 0.78              #: inner face of the mechanical stop
#: A singulated carton has to reach the stop on its own momentum.
#:
#: With the whole queue on the plate the cartons behind push the leader onto the
#: stop, and none of this matters. Once the escapement releases them one at a
#: time that help disappears: the first singulated carton stopped 91 mm short,
#: at x=0.714 against a stop at 0.88, and the cycle reported "no carton reached
#: the stop" while a carton sat plainly on the plate.
#:
#: So the plate is shorter and slipperier than it was. Both, rather than one:
#: friction alone would need a value low enough to make a carton skate off the
#: guides, and distance alone pushes the stop back toward the belt until there
#: is no plate left to land on.
PLATE_FRICTION = 0.08
OFFSET_Y = -0.40
BELT_LENGTH, BELT_WIDTH = 1.5, 0.40
SPEED = 0.30
PALLET_Y = 0.75

#: Parked clear of both decks, and back from the plate.
HOME = [0.0, -1.8, 1.5, -1.3, -1.57, 0.0]

#: Where a settled carton's centre ends up on the plate.
PICK_X = PLATE_STOP - BOX / 2.0
PICK_Z = PLATE_DECK + BOX / 2.0

_STALE = (
    "Belt", "BeltGate", "Plate", "Plate_Stop", "Plate_GuideL", "Plate_GuideR",
    "UR", "Pallet", "Escapement",
    "Box0", "Box1", "Box2", "Box3", "Box4", "Box5",
)


def clear_cell(scene) -> None:
    """Remove a previous cell's prims. `scene.stop()` does not do this."""
    for name in _STALE:
        path = f"/World/{name}"
        if scene.stage.GetPrimAtPath(path):
            scene.stage.RemovePrim(path)


def build(scene: Scene | None = None, *, boxes: int = 4) -> dict:
    """Author the cell and leave it playing, with the belt running."""
    scene = scene or Scene.get()
    scene.stop()
    clear_cell(scene)
    scene.configure_physics()
    scene.ensure_ground_plane()
    light_the_cell(scene)

    arm = Robot.spawn("ur10", position=[0.0, 0.0, 0.0], prim_path=ARM)
    gains = arm.tune_drives(stiffness=1.0e5, damping=1.0e4, max_force=1.0e4)

    belt = Conveyor.build(
        BELT, length=BELT_LENGTH, width=BELT_WIDTH,
        position=[BELT_END - BELT_LENGTH / 2.0, OFFSET_Y, BELT_DECK],
        direction=(1, 0, 0), speed=SPEED,
        gate=False,                      # discharge; the stop is on the plate
        scene=scene,
    )
    cartons = belt.load(boxes, box=(BOX, BOX, BOX), mass=BOX_MASS,
                        spacing=0.30, start_offset=0.20)

    # Upstream of the discharge by a carton and a half, so a released carton
    # is clear of the blade before it comes back up behind the next one.
    blade = Escapement.build(
        BLADE, at_x=BELT_END - BOX * 1.5, centre_y=OFFSET_Y,
        deck_z=BELT_DECK, width=BELT_WIDTH + 0.04, scene=scene,
    )

    plate = DeadPlate.build(
        PLATE, deck_z=PLATE_DECK, stop_x=PLATE_STOP,
        # Guides close to the carton, not to the belt. At 0.42 wide there was
        # 135 mm of slack either side of a 150 mm carton, and on a plate slick
        # enough to reach the stop the cup's own descent skated it 103 mm
        # sideways - the seal then failed and the failure named the gripper.
        # Low friction and loose guides are the same mistake twice.
        length=PLATE_STOP - BELT_END, width=BOX + 0.05,
        centre_y=OFFSET_Y, guide_height=0.10,
        friction=PLATE_FRICTION, scene=scene,
    )
    plate.set_box_size((BOX, BOX, BOX)).track(cartons)

    spawn_prop("pallet", prim_path=PALLET,
               position=[0.0, PALLET_Y, 0.0], scene=scene)
    slots = pallet_slots(origin=[0.0, PALLET_Y, 0.1425], box=(BOX, BOX, BOX),
                         rows=2, cols=2, layers=1, gap=0.01)

    arm.attach_suction_gripper(
        approach_axis="Z", max_grip_distance=0.10,
        cup_radius=0.045, cup_length=0.04,
        coaxial_force_limit=1.0e6, shear_force_limit=1.0e6, retry_interval=0.1,
    )
    described_belt = belt.describe()
    described_plate = plate.describe()
    described_blade = blade.describe()

    scene.play()
    scene.step(10)
    belt.start()
    scene.settle(6.0)

    arm = Robot.attach(ARM, scene=scene)
    cup = arm.rebind_suction()
    arm.set_joint_positions(HOME, settle_steps=120)

    belt = Conveyor.from_description(described_belt, scene=scene)
    belt.track([RigidObject(p, scene=scene) for p in described_belt["boxes"]])
    plate = DeadPlate.from_description(described_plate, scene=scene)
    plate.track([RigidObject(p, scene=scene) for p in described_plate["boxes"]])
    blade = Escapement.from_description(described_blade, scene=scene)

    # Let the first carton through immediately; the rest wait for their cycle.
    blade.release()
    scene.settle(3.0)
    blade.hold()

    return {
        "arm": arm, "cup": cup, "belt": belt, "plate": plate,
        "blade": blade, "slots": slots,
        "gains": gains, "box_size": BOX,
        "described": {"belt": described_belt, "plate": described_plate,
                      "blade": described_blade},
    }


def wait_at_stop(plate, *, seconds: float = 15.0, step: float = 0.5):
    """Run until a carton is settled against the plate's stop, or give up.

    Unlike the belt cell, the belt is *not* halted while picking. Cartons keep
    arriving and queueing behind the one being picked, which is what a real
    infeed does — and it is only safe because the stop, not the belt, is what
    holds the pick pose.
    """
    elapsed = 0.0
    while elapsed < seconds:
        carton = plate.box_at_stop()
        if carton is not None:
            return carton
        plate.scene.settle(step)
        elapsed += step
    return None


def pick_from_plate(cell: dict) -> dict:
    """Seal on the carton at the plate's stop and lift it clear."""
    arm, cup, plate = cell["arm"], cell["cup"], cell["plate"]
    carton = plate.box_at_stop() or wait_at_stop(plate)
    if carton is None:
        return {"picked": False, "reason": "no carton reached the stop"}

    start = np.asarray(carton.position, dtype=float).copy()
    arm.set_joint_positions(HOME, settle_steps=90)
    arm.scene.settle(0.5)

    here = np.asarray(carton.position, dtype=float)
    top = float(here[2]) + BOX / 2.0
    # Coarse with RMPflow, then precise with IK - and `move_ee_to` has to come
    # first for a second reason.
    #
    # Reaching out to the plate's stop at x=0.88 the reactive policy plateaus
    # 41.9 mm short of a 15 mm tolerance, so the tolerance here is 60 mm: this
    # move only has to get the arm into the neighbourhood.
    #
    # Replacing it with `pose_to` outright - the obvious fix - fails with "no
    # inverse-kinematics solver. The end-effector frame did not resolve." The
    # solver is initialised lazily by the first motion-generation call, and in
    # the belt cell that is always `move_ee_to`. Removing it removed the thing
    # that resolves the frame `pose_to` then needs.
    approach = [float(here[0]), float(here[1]), top + cup.tip_offset + 0.18]
    arm.move_ee_to(approach, DOWN, tolerance=0.06)
    arm.pose_to(approach, DOWN, corrections=8, raise_on_fail=False)
    arm.scene.settle(0.5)

    here = np.asarray(carton.position, dtype=float)
    top = float(here[2]) + BOX / 2.0

    sealed = False
    for attempt in range(10):
        arm.pose_to([float(here[0]), float(here[1]),
                     top + cup.tip_offset + STANDOFF - attempt * 0.002],
                    DOWN, corrections=8, raise_on_fail=False)
        arm.scene.settle(0.6)
        cup.close(settle_steps=0)
        for _ in range(8):
            arm.scene.settle(0.25)
            if cup.holding and cup.gripped_objects:
                break
        if cup.holding and cup.gripped_objects:
            sealed = True
            break
        cup.open(settle_steps=0)
        arm.scene.settle(0.2)

    if not sealed:
        return {"picked": False,
                "reason": f"cup did not seal after 10 descents (status {cup.status})"}

    arm.pose_to([float(here[0]), float(here[1]), top + cup.tip_offset + 0.30],
                DOWN, corrections=8, raise_on_fail=False)
    arm.scene.settle(1.2)

    end = np.asarray(carton.position, dtype=float)
    return {
        "picked": bool(cup.holding and cup.gripped_objects),
        "box": carton.prim_path,
        "rise": round(float(end[2] - start[2]), 4),
        "from": start.round(4).tolist(),
        "to": end.round(4).tolist(),
        "gripped": cup.gripped_objects,
    }


def palletise(cell: dict, *, count: int | None = None) -> dict:
    """Pick from the plate and stack, timing each cycle on the sim clock."""
    from isaacsim.core.simulation_manager import SimulationManager

    plate, slots = cell["plate"], cell["slots"]
    count = len(slots) if count is None else min(int(count), len(slots))

    blade = cell.get("blade")
    cycles, placed = [], 0
    for index in range(count):
        started = float(SimulationManager.get_simulation_time())
        # Release exactly one carton for this cycle, then close behind it. The
        # blade is what makes the arm's rate and the belt's rate independent.
        if blade is not None and index > 0:
            blade.release()
            plate.scene.settle(2.5)
            blade.hold()
        carton = wait_at_stop(plate)
        if carton is None:
            cycles.append({"slot": index, "ok": False,
                           "reason": "no carton reached the stop"})
            break

        picked = pick_from_plate(cell)
        if not picked.get("picked"):
            cycles.append({"slot": index, "ok": False,
                           "reason": picked.get("reason", "pick failed")})
            break

        result = place_on_slot(cell, slots[index], box=carton)
        finished = float(SimulationManager.get_simulation_time())
        ok = bool(result.get("placed"))
        placed += int(ok)
        cycles.append({
            "slot": index, "ok": ok,
            "seconds": round(finished - started, 2),
            "error": result.get("error"),
            "reason": result.get("reason"),
        })
        if not ok:
            break

    times = [c["seconds"] for c in cycles if c.get("ok")]
    mean = (sum(times) / len(times)) if times else None
    return {
        "placed": placed,
        "of": count,
        "complete": placed == count,
        "seconds_per_carton": round(mean, 2) if mean else None,
        "cartons_per_hour": round(3600.0 / mean, 1) if mean else None,
        "cycles": cycles,
    }


if __name__ == "__main__":
    cell = build()
    print(palletise(cell))
