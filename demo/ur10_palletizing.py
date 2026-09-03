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

"""A UR10 palletising cell that picks a box off a moving conveyor. MEASURED.

Unlike its KR210 neighbour this one has been run end to end on a live Isaac Sim
6.0 worker: boxes ride a driven belt, queue against the stop, and the arm seals
a suction cup on the waiting box and lifts it clear.

    box z 0.5248 -> 0.7695, a rise of 0.2447 m
    gripped ['/World/Box0'], holding True

Everything below is the residue of getting that to happen. Five of the six
settings look like tuning and are not: with any of them wrong the cell fails in
a way that points somewhere else entirely.

**1. Drive gains (`tune_drives`).** The shipped UR10 has stiffness 1.5e5-8.3e5
against damping 5-28. Commanded to a home pose it ran away to `wrist_3 =
-66.9 rad` — ten revolutions — and sat collapsed at its own base while every
Cartesian call reported "the target is likely outside the workspace". True of
where the arm was; false about the workspace.

**2. maxForce.** Shipped caps of 56-330 Nm cannot hold a reaching-down pose.
IK finds the solution and `pose_to` reports "the drives are not tracking it",
0.148 m short. At 1e4 the same pose holds to 2.7 mm.

**3. `approach_axis="Z"`.** Auto-detection picks X for this flange, which
rotates the cup ninety degrees so it raycasts *sideways* while the box sits
directly beneath it. Measured: cup forward `[1.0, 0.0, -0.005]` — horizontal.
With Z it reads `[-0.025, -0.016, -1.0]`, straight down.

**4. Gripper limits from Isaac's own tutorial** — grip distance 0.1, force
limits 500, retry 0.1. Ours were 0.05 and 10000, and 0.05 cannot bridge the gap
an arm realistically stops at.

**5. IK for the descent, not RMPflow.** `servo_to` and `move_ee_to` are a
reactive policy with a repulsion term: asked to descend the last 18 cm onto the
box it pushed the tool *away*, ending 0.15 m off. `pose_to` + `refine_pose` has
no repulsion and lands at 3 mm.

**6. Start the belt after Play.** Authoring a surface gripper stops the
timeline, and a stop between `start()` and Play drops the surface velocity.

The sixth is in `Conveyor` itself; the rest are here because they are decisions
about this cell rather than about the library.

## 7. Force limits, and the write that hides them

Isaac's tutorial uses coaxial and shear limits of 500. At 500 the seal forms
correctly and then breaks within 2 mm of the first commanded motion - any
motion, by any route. At 1e6 the same carton rides a 0.28 m lift with
`gripped 1` at every one of fifteen steps.

Whatever these units are, they are not newtons restraining a 1 kg box.

**The reason this took a dozen experiments to find is worth more than the
number.** Writing any `isaac:*` attribute on a *closed* gripper releases it
immediately, with the arm stationary:

    isaac:coaxialForceLimit: 500.0 -> 1e9
    holding after raise: False 0        <- dropped, without moving

So every attempt to test the force limits by raising them mid-grip destroyed
the grasp before it could measure anything, and each one came back "identical
detach" - which read as *force is not the cause* and sent the search elsewhere.
Joint slack, the transZ drive, the gate, the lift profile and the motion API
were all eliminated on the strength of an experiment that was measuring its own
side effect.

Set the limits at authoring time. If you need to change them, re-author the
gripper rather than writing the attribute.

The eliminations were not wasted - the collider, approach axis, grip distance
and correction budget were each genuinely checked, and two of them were real
bugs fixed above. But the general lesson is: an experiment that mutates the
thing it is measuring proves nothing, and here it actively pointed away from
the answer.

"""

import numpy as np

from simliverse_sim import (
    Conveyor,
    RigidObject,
    Robot,
    Scene,
    pallet_slots,
    spawn_prop,
)

ARM = "/World/UR"
BELT = "/World/Belt"
PALLET = "/World/Pallet"

BOX = 0.15                    # 15 cm cartons: a UR10 reaches 1.3 m, not 2.7 m
BOX_MASS = 1.0
DECK = 0.45                   # belt surface height
STOP_X = 0.75                 # where the stop is, and so where a box waits
OFFSET_Y = -0.40              # belt centre-line, clear of the arm's base
LENGTH, WIDTH = 1.6, 0.40
SPEED = 0.20
PALLET_Y = 0.75

#: Parked clear of the belt, not merely somewhere the arm can hold.
#:
#: The previous home put the tool at [0.858, 0.164, 0.374] - 226 mm *below* the
#: top of a carton, and 108 mm past the stop at x=0.75. Every return home
#: between cycles therefore swept the arm through the queue. Measured over a
#: four-carton run: one box ended on the floor and another was shoved 94 mm off
#: the belt centreline, after which the arm was sent to fetch a carton it could
#: not reach and the run died blaming the workspace.
#:
#: This one parks at [0.526, 0.164, 0.803] - 203 mm above the cartons and well
#: back from the stop. Candidates measured:
#:
#:     [0.0, -1.2, 1.6, -1.9, -1.57, 0.0]  ->  z 0.374   -226 mm   sweeps
#:     [0.0, -1.4, 1.4, -1.6, -1.57, 0.0]  ->  z 0.641    +41 mm   marginal
#:     [0.0, -1.8, 1.5, -1.3, -1.57, 0.0]  ->  z 0.803   +203 mm   clear
#:
#: The arm still holds it to 4.3e-4 rad once the drives are tuned.
HOME = [0.0, -1.8, 1.5, -1.3, -1.57, 0.0]
#: Flange pointing at the floor.
DOWN = [0.0, 1.0, 0.0, 0.0]
#: How far above the box the cup stops.
#:
#: 6 mm, and the number was swept rather than chosen. Against a carton settled
#: at the stop:
#:
#:     30 mm -> no seal, nudged 3.9 mm
#:     20 mm -> no seal, nudged 5.8 mm
#:     12 mm -> no seal, nudged 10.2 mm
#:      6 mm -> SEALED, lifted +0.2854 m, nudged 31.5 mm
#:
#: This corrects a belief that cost several runs. The attachment joint has
#: travel along its approach axis, so it looked as though the cup should seal
#: across a gap without touching. It does not: widening that joint's transZ
#: limit from 35 mm to the full 100 mm grip distance changed nothing at 30 mm
#: or 20 mm, so the joint was never the binding constraint. The cup seals on
#: contact, which is also what a real vacuum cup does.
#:
#: The 31.5 mm of nudge is not the cup pressing down - it is the descent
#: overshooting. Correction 2 dips 28.6 mm past the target (see `corrections=8`
#: below), and at a 6 mm standoff that dip lands inside the carton. Damping the
#: descent, rather than moving the standoff, is what will remove it.
STANDOFF = 0.006


def build(scene: Scene | None = None, *, boxes: int = 4) -> dict:
    """Author the cell and leave it playing with a box waiting at the stop."""
    from pxr import UsdPhysics

    from simliverse_sim._compat import get_stage

    scene = scene or Scene.get()
    scene.stop()
    scene.configure_physics()
    scene.ensure_ground_plane()

    light_the_cell(scene)

    arm = Robot.spawn("ur10", position=[0.0, 0.0, 0.0], prim_path=ARM)
    gains = arm.tune_drives(stiffness=1.0e5, damping=1.0e4, max_force=1.0e4)

    belt = Conveyor.build(
        BELT, length=LENGTH, width=WIDTH,
        position=[STOP_X - LENGTH / 2.0, OFFSET_Y, DECK],
        direction=(1, 0, 0), speed=SPEED,
        gate=True, gate_height=0.18, scene=scene,
    )
    belt.load(boxes, box=(BOX, BOX, BOX), mass=BOX_MASS,
              spacing=0.25, start_offset=0.20)

    spawn_prop("pallet", prim_path=PALLET,
               position=[0.0, PALLET_Y, 0.0], scene=scene)
    slots = pallet_slots(origin=[0.0, PALLET_Y, 0.1425], box=(BOX, BOX, BOX),
                         rows=2, cols=2, layers=1, gap=0.01)

    cup = arm.attach_suction_gripper(
        approach_axis="Z",
        max_grip_distance=0.10, cup_radius=0.045, cup_length=0.04,
        # 1e6, not the 500 taken from Isaac's tutorial. At 500 the seal forms
        # and breaks within 2 mm of the first commanded motion, whatever that
        # motion is. Whatever these units are, they are not newtons holding a
        # 1 kg carton: at 1e6 the same carton rides through a 0.28 m lift with
        # `gripped 1` at every step.
        coaxial_force_limit=1.0e6, shear_force_limit=1.0e6, retry_interval=0.1,
    )
    described = belt.describe()

    scene.play()
    scene.step(10)
    belt.start()
    scene.settle(8.0)

    # Handles do not survive the play; re-bind rather than reuse.
    arm = Robot.attach(ARM, scene=scene)
    cup = arm.rebind_suction()
    arm.set_joint_positions(HOME, settle_steps=120)

    belt = Conveyor.from_description(described, scene=scene)
    belt.track([RigidObject(path, scene=scene) for path in described["boxes"]])
    # Leave the cell in a state a pick can start from: a box actually settled
    # against the stop, and the belt off so the queue stops pressing on it.
    # Sampling `box_at_gate()` once is not enough - a box that has arrived is
    # still jostling above the settled-speed threshold for a second or two.
    wait_for_box(belt)

    return {
        "arm": arm, "cup": cup, "belt": belt, "slots": slots,
        "gains": gains, "described": described, "box_size": BOX,
    }


def light_the_cell(scene, *, dome: float = 1200.0, key: float = 3000.0) -> list[str]:
    """A dome and a key light, because an unlit cell looks broken.

    Isaac's default stage has almost no light in it, so a perfectly healthy
    scene renders as dark shapes on black and reads as a rendering fault - the
    first thing anyone says about it is that the simulator is glitching. This
    is presentation only and touches no physics, but a cell nobody can see is
    not a demo.
    """
    from pxr import Gf, UsdGeom, UsdLux

    stage = scene.stage
    made = []

    dome_path = "/World/CellDomeLight"
    stage.RemovePrim(dome_path)
    dome_light = UsdLux.DomeLight.Define(stage, dome_path)
    dome_light.CreateIntensityAttr(float(dome))
    made.append(dome_path)

    key_path = "/World/CellKeyLight"
    stage.RemovePrim(key_path)
    key_light = UsdLux.DistantLight.Define(stage, key_path)
    key_light.CreateIntensityAttr(float(key))
    key_light.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(key_light.GetPrim())
    xform.ClearXformOpOrder()
    # Down and across the cell, so the cartons cast a shadow and read as solid.
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
    made.append(key_path)
    return made


def wait_for_box(belt, *, seconds: float = 12.0, step: float = 0.5):
    """Run the belt until a box is settled against the stop, then halt it.

    Returns the box, or None if none arrived in time. The belt is left off
    either way: with it running the queue keeps pressing on the box being
    picked, and the pick pose moves under the cup.
    """
    elapsed = 0.0
    while elapsed < seconds:
        box = belt.box_at_gate()
        if box is not None:
            belt.halt()
            return box
        belt.scene.settle(step)
        elapsed += step
    belt.halt()
    return None


def pick_waiting_box(cell: dict) -> dict:
    """Seal on the box at the stop and lift it clear. Returns what was measured."""
    arm, cup, belt = cell["arm"], cell["cup"], cell["belt"]
    box = belt.box_at_gate() or wait_for_box(belt)
    if box is None:
        return {"picked": False, "reason": "no box settled against the stop in 12 s"}

    start = np.asarray(box.position, dtype=float).copy()
    belt.halt()

    # Home first, then read the box. Reading it before homing and descending to
    # that reading is how the cup ends up on a corner: the box keeps settling
    # while the arm swings across, and a 15 cm box only has to drift 3 cm for
    # the cup to land on its top-back edge instead of the middle of its face.
    # Measured that way, the box was grabbed by a corner and swung 3.4 cm during
    # the lift. Every pose below comes from a reading taken after the arm has
    # already stopped moving.
    arm.set_joint_positions(HOME, settle_steps=90)
    arm.scene.settle(0.5)

    here = np.asarray(box.position, dtype=float)
    box_top = float(here[2]) + BOX / 2.0
    arm.move_ee_to([float(here[0]), float(here[1]), box_top + cup.tip_offset + 0.18],
                   DOWN, tolerance=0.015)

    # Re-read once more now the arm is parked above it, so the descent is
    # centred on the face rather than on where the box used to be.
    here = np.asarray(box.position, dtype=float)
    box_top = float(here[2]) + BOX / 2.0

    # Seal from a standoff; never drive the cup onto the box. The attachment
    # joint has 35 mm of travel along its approach axis, so it reaches down to
    # a box it is hovering over. Descending to contact instead shoves a carton
    # resting against the stop 1.6 cm before the seal forms, and the grip then
    # lands on an edge with the box hanging off the cup.
    # Eight corrections, not the default four, because convergence here is not
    # monotonic and stopping early reads as a failure that isn't one. Measured,
    # descending onto a settled carton:
    #
    #   command 0.0828 -> 0.0068 -> 0.0286 -> 0.0263 -> 0.0193 -> 0.0061 -> 0.0022
    #
    # The first correction very nearly lands it, the second overshoots, and the
    # oscillation takes six passes to damp. The default budget stops at the
    # fourth, at 19 mm, and `pose_to` raises "the drives are not tracking it" —
    # which is true of that instant and false about the pose, since two more
    # corrections reach 2.2 mm. Loosening the tolerance instead would hide the
    # overshoot and hand the seal a cup that is still moving.
    # Descend until it seals, rather than betting the pick on one height.
    #
    # A single attempt at a fixed standoff is a knife edge, and the measurements
    # say so: 30/20/12 mm never seal, 6 mm does, and 6 mm is close enough that
    # the descent's own overshoot can land inside the carton. Worse, the same
    # 6 mm approach seals on one run and not the next, because whether the cup
    # is perfectly still at the instant it closes is not something the pose
    # controller guarantees.
    #
    # The grip distance says this should not be necessary - the attachment
    # origin sits 26 mm above the carton with a 100 mm grip distance, and its
    # forward axis measures [0.0002, -0.0003, -1.0], straight down. Why the
    # raycast does not find the box from there is unresolved. Retrying is not a
    # workaround for not knowing: a real vacuum cell also descends until it has
    # vacuum rather than asserting a height, so this is what the cell should
    # have done anyway.
    #
    # Steps are 2 mm and it gives up after ten, so the worst case is 20 mm of
    # travel below the first attempt - less than the overshoot a single attempt
    # already risks.
    sealed = False
    for attempt in range(10):
        arm.pose_to([float(here[0]), float(here[1]),
                     box_top + cup.tip_offset + STANDOFF - attempt * 0.002],
                    DOWN, corrections=8, raise_on_fail=False)
        # Settle, then close. Do NOT refine again here. `pose_to` has already
        # run its corrections and returned converged; refining from a converged
        # pose re-enters the same overshoot that made the budget of eight
        # necessary, and at this standoff that means the cup is moving when it
        # is asked to seal.
        arm.scene.settle(0.6)
        cup.close(settle_steps=0)
        # `holding` alone is not enough, and believing it cost a debugging
        # session. It reads the gripper's status token, and the token reaches
        # "Closed" transiently with `gripped_objects` still empty — a cup that
        # has shut on nothing. The pick then reported success-shaped output with
        # `gripped: []`, `rise: -0.0000` and a carton that never moved, which
        # reads as a lift that failed rather than a grasp that never happened.
        #
        # Requiring a named object is the same discipline `verify()` applies
        # when it refuses to call a run reproduced just because something moved.
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

    # The lift carries the same two corrections the descent needed, for the same
    # reasons, and it took a third debugging pass to notice they were missing
    # here. With the default budget of four this raised mid-lift; with three
    # trailing refines it shook the carton off the cup, and the box came back
    # down to 0.525 — belt height — so the pick reported `rise: -0.0000` and
    # looked like a grasp that never happened rather than one that let go.
    arm.pose_to([float(here[0]), float(here[1]), box_top + cup.tip_offset + 0.30],
                DOWN, corrections=8, raise_on_fail=False)
    arm.scene.settle(1.2)

    end = np.asarray(box.position, dtype=float)
    ee = arm.ee_position
    # Tool against the *box*, not against the target it was sent to - the
    # latter is trivially zero and reported 0.0 through every corner grab.
    offcentre = float(np.linalg.norm(np.asarray(ee)[:2] - end[:2]))
    return {
        "picked": bool(cup.holding),
        "box": box.prim_path,
        "rise": round(float(end[2] - start[2]), 4),
        "off_centre": round(offcentre, 4),
        "from": start.round(4).tolist(),
        "to": end.round(4).tolist(),
        "gripped": cup.gripped_objects,
    }


def place_on_slot(cell: dict, slot: dict, *, box=None) -> dict:
    """Put the held carton on `slot`. Returns what was measured, not a promise.

    Traverses high before descending. The carton is 15 cm tall and the gate is
    18 cm, so a traverse at picking height drags it through the stop; the
    clearance below is measured from the pallet deck rather than assumed.
    """
    arm, cup = cell["arm"], cell["cup"]
    scene = arm.scene
    if not (cup.holding and cup.gripped_objects):
        return {"placed": False, "reason": "nothing held"}
    if box is None:
        box = cell["belt"].boxes[0]

    place = slot["place"]
    # Release height: `place` is where the carton's centre goes, so the tool
    # sits half a box plus the cup above it.
    place_z = float(place[2]) + BOX / 2.0 + cup.tip_offset
    travel_z = max(float(slot["approach"][2]), 0.55) + cup.tip_offset + BOX / 2.0

    arm.pose_to([float(place[0]), float(place[1]), travel_z], DOWN,
                corrections=8, raise_on_fail=False)
    scene.settle(1.0)
    if not (cup.holding and cup.gripped_objects):
        return {"placed": False, "reason": "dropped during traverse"}

    arm.pose_to([float(place[0]), float(place[1]), place_z], DOWN,
                corrections=8, raise_on_fail=False)
    scene.settle(1.0)

    cup.open(settle_steps=0)
    for _ in range(12):
        scene.settle(0.3)
        if not cup.holding:
            break

    arm.pose_to([float(place[0]), float(place[1]), place_z + 0.25], DOWN,
                corrections=8, raise_on_fail=False)
    scene.settle(1.2)

    final = np.asarray(box.position, dtype=float)
    rest = np.asarray(slot["rest"], dtype=float)
    error = float(np.linalg.norm(final - rest))
    return {
        "placed": error <= 0.04 and float(box.speed) <= 0.02,
        "slot": slot["index"],
        "box": box.prim_path,
        "error": round(error, 4),
        "at": final.round(4).tolist(),
        "want": rest.round(4).tolist(),
        "speed": round(float(box.speed), 4),
    }


def palletise(cell: dict, *, count: int | None = None) -> dict:
    """Pick and place `count` cartons, timing each cycle.

    The cycle time is the deliverable, not a diagnostic. An integrator quotes a
    rate and is held to it, and no offline-programming tool measures one against
    simulated physics — they interpolate geometry and are, on published
    measurements, wrong by up to a factor of two.

    Timed on the *simulation* clock, not the wall clock, so the number means
    "how long this cell takes" rather than "how fast this GPU is".
    """
    from isaacsim.core.simulation_manager import SimulationManager

    arm, belt, slots = cell["arm"], cell["belt"], cell["slots"]
    scene = arm.scene
    count = len(slots) if count is None else min(int(count), len(slots))

    cycles, placed = [], 0
    run_started = float(SimulationManager.get_simulation_time())

    for index in range(count):
        started = float(SimulationManager.get_simulation_time())
        # Let the next carton come forward before reaching for it.
        waiting = belt.box_at_gate()
        if waiting is None:
            belt.start()
            waiting = wait_for_box(belt)
        if waiting is None:
            cycles.append({"slot": index, "ok": False, "reason": "no carton arrived"})
            break

        picked = pick_waiting_box(cell)
        if not picked.get("picked"):
            cycles.append({"slot": index, "ok": False,
                           "reason": picked.get("reason", "pick failed")})
            break

        result = place_on_slot(cell, slots[index], box=waiting)
        finished = float(SimulationManager.get_simulation_time())
        ok = bool(result.get("placed"))
        placed += int(ok)
        cycles.append({
            "slot": index,
            "ok": ok,
            "seconds": round(finished - started, 2),
            "error": result.get("error"),
            "reason": result.get("reason"),
        })
        if not ok:
            break

    total = float(SimulationManager.get_simulation_time()) - run_started
    times = [c["seconds"] for c in cycles if c.get("ok")]
    return {
        "placed": placed,
        "of": count,
        "complete": placed == count,
        "seconds_total": round(total, 2),
        "seconds_per_carton": round(sum(times) / len(times), 2) if times else None,
        "cartons_per_hour": round(3600.0 / (sum(times) / len(times)), 1) if times else None,
        "cycles": cycles,
    }


if __name__ == "__main__":
    cell = build()
    print("gains:", cell["gains"]["joints"])
    print(palletise(cell))
