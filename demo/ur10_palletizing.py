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

"""A UR10 palletising cell, run end to end on live hardware. MEASURED.

Four cartons ride a driven belt, queue against a stop, and are picked one at a
time and stacked into a 2x2 layer on a pallet. Measured on an Isaac Sim 6.0
worker, four cycles, no failures:

    slot 0   rise 0.2835   placed 7.5 mm from target   59.30 s
    slot 1   rise 0.2773   placed 6.3 mm from target   58.08 s
    slot 2   rise 0.2818   placed 4.5 mm from target   58.42 s
    slot 3   rise 0.3662   placed 3.7 mm from target   58.65 s

    58.61 s per carton  ->  61.4 cartons/hour

The cycle time is on the *simulation* clock, so it describes the cell rather
than the GPU. It is also the number an integrator quotes and is held to, and
the reason this file exists: offline-programming tools interpolate geometry and
publish no accuracy figure, while a measured cycle here comes out of contact,
friction and payload dynamics.

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

**3. `approach_axis` is measured, not "Z".** The flange's own tool axis is X on
a UR, and `down_at_yaw` sends *that* axis at the floor, so the cup faces down
AND the wrist stands vertical over the carton, the way a real UR palletiser
holds it. "Z" was an old workaround from before `down_at_yaw` was axis-aware: a
fixed down quat aimed tool Z at the floor, which sealed the carton but laid the
flange on its side (`wrist_3` Z reads `[1,0,0]`, horizontal). Measured with the
axis, `wrist_3` Z is `[0,0,-1]` and the cup faces `[0,0,-1]` - both right.

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
    verify_pallet,
)

ARM = "/World/UR"
BELT = "/World/Belt"
PALLET = "/World/Pallet"

BOX = 0.15  # 15 cm cartons: a UR10 reaches 1.3 m, not 2.7 m
BOX_MASS = 1.0
DECK = 0.45  # belt surface height
STOP_X = 0.75  # where the stop is, and so where a box waits
OFFSET_Y = -0.40  # belt centre-line, clear of the arm's base
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


def cell_geometry(box: float) -> dict:
    """The dimensions that follow from the carton, in one place.

    Kept as a function rather than four expressions inside `build` so a test
    can ask what a 22 cm carton implies without a GPU and a running stage.
    Inlined, these are exactly the kind of number that stays at its measured
    value while everything around it changes.
    """
    if box <= 0:
        raise ValueError("box=%r: a carton has a positive size." % (box,))
    return {
        # A gate is a stop, so it must be taller than what it stops.
        "gate_height": box + 0.03,
        # Room for a carton to sit askew without fouling the side rails.
        "width": max(WIDTH, box + 0.10),
        # Cartons spaced tighter than they are wide arrive as one block.
        "spacing": max(0.25, box + 0.10),
        # A cup wider than the top face grips the corner it hangs over.
        "cup_radius": min(0.045, box * 0.30),
        # How far the cup may reach for something to seal against. A queue
        # accumulates against the stop, so the next carton is *touching* the
        # one being picked - and a reach comparable to the carton itself then
        # latches onto the neighbour. At 0.10 m, which is what this was for
        # every carton size, a 10 cm box was picked by grabbing the box behind
        # it and knocking the intended one off the belt. The cup only ever has
        # to cross the standoff, so this wants to be small.
        "max_grip_distance": max(0.02, min(0.06, box * 0.40)),
    }


#: Drive gains by robot. One set does not fit every arm: at the UR10's 1e4
#: force cap the KR210's first three joints barely moved, a home reset landed
#: 0.1 m from home, and every cycle after the first raised MotionError. At 1e6
#: the joints converge in about 60 steps and a cycle takes 36 s, not 91. The UR
#: family and the Fanuc CRX hold to under 3 mm at the UR10 values.
DRIVE_GAINS = {
    "ur": {"stiffness": 1.0e5, "damping": 1.0e4, "max_force": 1.0e4},
    "default": {"stiffness": 1.0e5, "damping": 1.0e4, "max_force": 1.0e6},
}


def drive_gains(robot: str) -> dict:
    """The UR family holds at 1e4. Nothing heavier has: a KR210's first three
    joints barely moved, and a Fanuc CRX's shoulder sat sagged at -0.82 rad
    while joints 3-6 tracked their solution to the milliradian - the IK was
    right and the tool ended 1.24 m from its target."""
    return dict(DRIVE_GAINS["ur" if _is_ur(robot) else "default"])


#: Nominal reach in metres, from the vendors' data sheets. The library records
#: none of this, and the cell's layout scales with it: a pick point 0.85 m out
#: is comfortable for a UR10 and outside a UR5e altogether.
REACH = {
    "ur3": 0.50,
    "ur3e": 0.50,
    "ur5": 0.85,
    "ur5e": 0.85,
    "ur10": 1.30,
    "ur10e": 1.30,
    "ur16e": 0.90,
    "ur20": 1.75,
    "ur30": 1.30,
    "crx5ia": 0.99,
    "crx10ia": 1.25,
    "crx10ia_l": 1.42,
    "crx20ia_l": 1.42,
    "kuka_kr210": 2.70,
    "tm12": 1.30,
    "rs007l": 0.93,
    "rs013n": 1.46,
    "rs025n": 1.73,
    "rs080n": 2.10,
    "cobottapro900": 0.90,
    "cobottapro1300": 1.30,
    "franka": 0.855,
    "fr3": 0.855,
}

#: RMPflow configurations the library cannot match by name on its own.
RMP_CONFIG = {"crx10ia_l": "Fanuc_CRX10IAL"}

#: Arms that build and drive but do not yet complete a pick, with why. A
#: sweep should say "known, and here is the reason" rather than print a bare
#: failure that reads as a regression.
KNOWN_GAPS = {
    "crx10ia_l": (
        "clears its shoulder (FOOTPRINT moves the belt to -0.57) and drives right "
        "(1e6), but the suction cup does not seal: the flange-axis heuristic measures "
        "the structural offset (-Y) and mounts the cup on the SIDE of the wrist. The "
        "tool point is tool0, offset along the flange, and its frame differs from the "
        "mount link by a rotation that neither the mount nor down_at_yaw reconciles - "
        "so the cup faces ~horizontal at the tool-down pose. An empirical measured-down "
        "(search the orientation that points the cup at the floor, then aim the cup - "
        "not the flange - at the carton) placed one to 0.2 mm in a probe; the real fix "
        "is a per-asset tool-frame registration, and every attempt so far regressed the "
        "UR or KR210, so it needs doing carefully in the library, not the cell"
    ),
}

#: Decks a stack can be built on: the real pallet prop, or a static slab for
#: loads a small arm can actually span. (length, width, height) in metres.
DECKS = {
    "pallet": None,
    "half": (0.80, 0.60, 0.1425),
    "tote": (0.60, 0.40, 0.1425),
}


def layout_for(robot: str, *, box: float = BOX) -> dict:
    """Build arguments that put this arm's cell inside its reach.

    The measured UR10 cell picks at 0.85 m (65 % of reach) and stacks at
    0.75-0.95 m. Scaled by reach, that ratio is where every arm here was
    happy; the pallet is the constraint. A full pallet is 0.80 m wide and is
    placed by its centre, so it needs pallet_y >= 0.65 m to clear the base -
    beyond a UR5e's stack radius - and gets a tote instead.
    """
    reach = REACH.get(robot)
    if reach is None:
        raise ValueError("%s: no reach on record; add it to REACH" % robot)
    pick_r = 0.65 * reach
    stop_x = round(pick_r * (STOP_X / 0.85), 3)
    offset_y = round(-pick_r * (-OFFSET_Y / 0.85), 3)
    stack_r = 0.60 * reach
    deck = "pallet" if stack_r >= 0.65 + 0.10 else "half" if stack_r >= 0.55 else "tote"
    half_width = 0.40 if deck == "pallet" else DECKS[deck][1] / 2.0
    pallet_y = round(max(stack_r, half_width + 0.30), 3)
    out = {
        "robot": robot,
        "stop_x": stop_x,
        "offset_y": offset_y,
        "pallet_y": pallet_y,
        "pallet": deck,
        "box": box,
        # The belt too. At the UR10's 0.45 m deck a UR5e's pick hover sat
        # 0.82 m up at 0.49 m out - the edge of the arm - and RMPflow got
        # within 65 mm and no closer. A small arm gets a low belt.
        "deck": round(max(0.25, DECK * reach / REACH["ur10"]), 3) if reach < REACH["ur10"] else DECK,
    }
    if robot in RMP_CONFIG:
        out["rmp_config"] = RMP_CONFIG[robot]
    if reach >= 2.0:
        # A big arm cannot fold in to a deck at its own feet; raise the base.
        out["pedestal"] = 0.35
    return out


def max_pattern(robot: str, box: float, pallet_y: float, *, gap: float = 0.01) -> tuple[int, int]:
    """The largest (rows, cols) whose far slot stays inside this arm's reach.

    A UR10e carrying a 22 cm carton to the far slot of a 2x3 pattern was 0.31 m
    short - a real reach limit the cell only found mid-place. The pallet grid
    spreads (rows-1)*pitch along X and (cols-1)*pitch along Y about pallet_y,
    so the far corner sits at that planar distance from the base; keep it under
    the tool-down stack radius (0.60 x reach) and the cell is placeable, not a
    late failure.
    """
    reach = REACH.get(robot, 1.3)
    limit = 0.60 * reach
    pitch = float(box) + gap
    best_rows, best_cols = 1, 1
    for rows in (1, 2, 3):
        for cols in (1, 2, 3):
            x = (rows - 1) / 2.0 * pitch
            y = float(pallet_y) + (cols - 1) / 2.0 * pitch
            if float(np.hypot(x, y)) <= limit and rows * cols >= best_rows * best_cols:
                best_rows, best_cols = rows, cols
    return best_rows, best_cols


#: Air between the arm's body and anything built beside it.
CLEARANCE = 0.05


#: A measured floor on the belt-height footprint, for arms whose shoulder sits
#: at belt height in their spawn pose and so is not caught by the base+shoulder
#: links a fresh handle can see. Measured live at the home pose (deck 0.45,
#: base on the floor): a Fanuc CRX's J2 shoulder reaches 0.31 m across the belt
#: band, where the UR family folds below the deck (0.0) and the KR210's own
#: layout already clears its reach. Only arms that need more than the cheap
#: lower bound appear here.
FOOTPRINT = {"crx10ia_l": 0.32}


def arm_footprint(arm, *, links: int = 2) -> float:
    """A cheap lower bound on how far the arm's body reaches from its base.

    The horizontal extent of the first `links` chain links - the base and the
    shoulder housing, which do not fold away. Measuring every link took the
    whole arm on a UR10, whose asset spawns lying flat (1.3 m), so the belt and
    pallet were pushed out past it and every slot came back unreachable. This
    stays a lower bound on purpose; an arm whose shoulder sits at belt height
    (a Fanuc CRX) carries a measured floor in FOOTPRINT, because measuring it
    at the true home pose needs a play the build cannot spare mid-authoring.
    """
    from simliverse_sim.conveyor import _world_bounds

    base = np.asarray(arm.base_position, dtype=float)[:2]
    chain = arm.chain_links() if hasattr(arm, "chain_links") else list(arm.links())
    radius = 0.0
    for link in chain[: max(1, int(links))]:
        bounds = _world_bounds(link)
        if bounds is None:
            continue
        low, high = (np.asarray(b, dtype=float) for b in bounds)
        for x in (low[0], high[0]):
            for y in (low[1], high[1]):
                radius = max(radius, float(np.hypot(x - base[0], y - base[1])))
    return radius


def _deck_half_width(pallet: str) -> float:
    return 0.40 if DECKS.get(pallet) is None else DECKS[pallet][1] / 2.0


def clear_offsets(
    offset_y: float, pallet_y: float, *, width: float, pallet_half_width: float, footprint: float
) -> tuple[float, float, dict]:
    """Push the belt and the deck out until their near edges clear `footprint`.

    Returns the (possibly moved) offsets and a record of what moved, so a
    cell that had to be re-laid says so instead of quietly being a different
    cell from the one asked for.
    """
    need = footprint + CLEARANCE
    record = {"footprint": round(float(footprint), 3), "moved": {}}
    belt_edge = abs(float(offset_y)) - float(width) / 2.0
    if belt_edge < need:
        sign = -1.0 if float(offset_y) < 0 else 1.0
        moved = sign * (need + float(width) / 2.0)
        record["moved"]["offset_y"] = {"from": round(float(offset_y), 3), "to": round(moved, 3)}
        offset_y = moved
    deck_edge = float(pallet_y) - float(pallet_half_width)
    if deck_edge < need:
        moved = need + float(pallet_half_width)
        record["moved"]["pallet_y"] = {"from": round(float(pallet_y), 3), "to": round(moved, 3)}
        pallet_y = moved
    return round(float(offset_y), 3), round(float(pallet_y), 3), record


def build(
    scene: Scene | None = None,
    *,
    boxes: int = 4,
    box: float = BOX,
    box_mass: float = BOX_MASS,
    deck: float = DECK,
    stop_x: float = STOP_X,
    offset_y: float = OFFSET_Y,
    speed: float = SPEED,
    pallet_y: float = PALLET_Y,
    rows: int = 2,
    cols: int = 2,
    layers: int = 1,
    robot: str = "ur10",
    guides: bool = False,
    grip_distance: float | None = None,
    pedestal: float = 0.0,
    dressing: str | None = None,
    pallet: str = "pallet",
    rmp_config: str | None = None,
) -> dict:
    """Author the cell and leave it playing with a box waiting at the stop.

    Every default here is the measured cell, so calling `build(scene)` still
    reproduces the run these numbers came from. They are arguments rather than
    constants because a cell that only works at one carton size, one belt
    height and one pallet distance has been fitted to its own demo - and the
    only way to find out which of these numbers is load-bearing is to be able
    to change them.

    Dimensions that follow from the carton follow from it here too: a gate
    sized for a 15 cm box stops nothing when the box is 22 cm, and a 45 mm cup
    overhangs a 10 cm carton. Deriving them is what keeps a scenario sweep
    honest rather than a list of ways to mis-author a cell.
    """

    scene = scene or Scene.get()
    scene.stop()
    # Not housekeeping. A previous cell's prims survive `stop()`, and the
    # infeed experiment's escapement blade once held this cell's carton queue
    # while every belt observable insisted the conveyor was running.
    scene.clear_world()
    scene.configure_physics()
    scene.ensure_ground_plane()

    light_the_cell(scene)

    # A pedestal is structure, not scenery: its top *is* the robot's base
    # height. Drawing a plinth under an arm that is still on the floor gives a
    # render of a robot growing out of a crate, so the base moves with it.
    # Zero by default, because every number in this cell was measured with the
    # base on the floor and a pedestal moves the whole working envelope up.
    base_z = 0.0
    plinth = None
    if pedestal > 0.0:
        from simliverse_sim import spawn_pedestal

        plinth = spawn_pedestal(
            "/World/Pedestal", position=[0.0, 0.0, 0.0], height=float(pedestal), size=(0.45, 0.45), scene=scene
        )
        base_z = plinth["top"]

    spawn_kwargs = {"rmp_config": rmp_config} if rmp_config else {}
    arm = Robot.spawn(robot, position=[0.0, 0.0, base_z], prim_path=ARM, **spawn_kwargs)
    gains = arm.tune_drives(**drive_gains(robot))

    shape = cell_geometry(box)
    gate_height, width, spacing = (shape["gate_height"], shape["width"], shape["spacing"])

    # The belt and the deck must clear the arm's own body, not just its base
    # point. A Fanuc CRX's shoulder reaches 0.31 m out where the belt runs; with
    # the belt edge 0.23 m from the base the base joint stopped at -43 degrees
    # on every move toward it, the IK right, the drives right, and the pick
    # reported "hover not reached: 1.08 m short". The base+shoulder links give a
    # cheap lower bound; arms whose shoulder sits at belt height carry a
    # measured floor in FOOTPRINT (a live play-home-measure was tried and
    # de-initialised the articulation mid-build - it is not worth a stop cycle).
    footprint = max(arm_footprint(arm), FOOTPRINT.get(robot, 0.0))
    offset_y, pallet_y, clearance = clear_offsets(
        offset_y, pallet_y, width=width, pallet_half_width=_deck_half_width(pallet), footprint=footprint
    )

    belt = Conveyor.build(
        BELT,
        length=LENGTH,
        width=width,
        position=[stop_x - LENGTH / 2.0, offset_y, deck],
        direction=(1, 0, 0),
        speed=speed,
        gate=True,
        gate_height=gate_height,
        # Off by default, and that is a known limitation rather than a
        # preference. Rails stop a queue squirting cartons off a 400 mm belt,
        # which is what fails every carton at 1.5 kg and above. They also cost
        # placement accuracy - 208 mm out, deterministically - because the pick
        # has never been centred and nobody could tell: a carton sealed
        # off-centre slides into line under the cup while it is lifted, and the
        # rails stop it sliding. The 3-5 mm this cell reports is partly the
        # carton correcting the approach on its way up.
        #
        # Turning them on without fixing the approach trades a failure at 1.5 kg
        # for a failure at every mass, so the default stays off until the pick
        # lands centred on its own.
        guides=guides,
        guide_height=max(0.10, box * 0.8),
        dressing=dressing,
        scene=scene,
    )
    belt.load(boxes, box=(box, box, box), mass=box_mass, spacing=spacing, start_offset=0.20)

    if pallet not in DECKS:
        raise ValueError("pallet=%r: one of %s" % (pallet, sorted(DECKS)))
    if DECKS[pallet] is None:
        spawn_prop("pallet", prim_path=PALLET, position=[0.0, pallet_y, 0.0], scene=scene)
        deck_top = 0.1425
        # A pallet is 1.21 m long and is placed by its centre, so a pallet_y
        # that sounds merely close - 0.60 - puts its near edge at -0.005 and
        # the arm's base inside it. `spawn_prop` warns; a warning in a sweep
        # of twelve cells scrolls past. Recording it lets a caller tell "this
        # cell is impossible" apart from "this code is wrong".
        from simliverse_sim.props import _overlapping_robots

        try:
            fouled = _overlapping_robots(PALLET)
        except Exception:  # noqa: BLE001 - a diagnostic must not break the build
            fouled = []
    else:
        # A static slab, sized for what a small arm can span: a full pallet is
        # 0.80 m wide and 0.65 m out at the nearest, past a UR5e's stack.
        length, width_, height = DECKS[pallet]
        scene.spawn_rigid(
            PALLET,
            shape="Cube",
            size=1.0,
            scale=(length, width_, height),
            position=[0.0, pallet_y, height / 2.0],
            color=(0.55, 0.40, 0.25),
            static=True,
        )
        deck_top = height
        fouled = [{"robot": ARM}] if pallet_y - width_ / 2.0 < 0.25 else []
    slots = pallet_slots(
        origin=[0.0, pallet_y, deck_top], box=(box, box, box), rows=rows, cols=cols, layers=layers, gap=0.01
    )

    cup_radius = shape["cup_radius"]

    # Measure the flange's own tool axis for every arm - a UR answers X, a
    # KR210 X, a Fanuc CRX -Y. "Z" was once hard-coded for the UR family as a
    # workaround from before `down_at_yaw` was axis-aware: a fixed down quat
    # sent tool Z at the floor, so an X-mounted cup raycast sideways and "Z"
    # was the only thing that pointed it down. Now down_at_yaw sends the
    # MEASURED axis down, so "auto" gives both a down-facing cup and a flange
    # that stands vertical over the carton - measured, wrist_3 Z is [0,0,-1]
    # with the axis (correct look), and [1,0,0] with "Z" (flange on its side).
    cup = arm.attach_suction_gripper(
        approach_axis="auto",
        max_grip_distance=(shape["max_grip_distance"] if grip_distance is None else float(grip_distance)),
        cup_radius=cup_radius,
        cup_length=0.04,
        # 1e6, not the 500 taken from Isaac's tutorial. At 500 the seal forms
        # and breaks within 2 mm of the first commanded motion, whatever that
        # motion is. Whatever these units are, they are not newtons holding a
        # 1 kg carton: at 1e6 the same carton rides through a 0.28 m lift with
        # `gripped 1` at every step.
        coaxial_force_limit=1.0e6,
        shear_force_limit=1.0e6,
        retry_interval=0.1,
    )
    described = belt.describe()

    scene.play()
    scene.step(10)
    belt.start()
    scene.settle(8.0)

    # Handles do not survive the play; re-bind rather than reuse.
    # The configuration name given at spawn does not survive a re-attach on
    # its own: the Fanuc came back "No RMPflow configuration matches" with
    # Fanuc_CRX10IAL in the list it printed.
    arm = Robot.attach(ARM, scene=scene, **spawn_kwargs)
    cup = arm.rebind_suction()
    if _is_ur(robot):
        arm.set_joint_positions(HOME, settle_steps=120)
    else:
        # HOME was measured on a UR10's six joints. Handing it to another arm
        # is either a shape error or, worse, a silent pose on a different
        # kinematic chain.
        arm.set_joint_positions([0.0] * arm.dof, settle_steps=120)

    belt = Conveyor.from_description(described, scene=scene)
    belt.track([RigidObject(path, scene=scene) for path in described["boxes"]])
    # Leave the cell in a state a pick can start from: a box actually settled
    # against the stop, and the belt off so the queue stops pressing on it.
    # Sampling `box_at_gate()` once is not enough - a box that has arrived is
    # still jostling above the settled-speed threshold for a second or two.
    wait_for_box(belt)

    cell = {
        "arm": arm,
        "cup": cup,
        "belt": belt,
        "slots": slots,
        "gains": gains,
        "described": described,
        "box_size": box,
        "fouled": fouled,
        "pedestal": plinth,
        "base_z": base_z,
        # The tool-down orientation for this flange: down_at_yaw sends the
        # measured approach axis to world -Z. A UR answers Z, a KR210 X. It is
        # right for both; a Fanuc CRX is the one arm it is not (KNOWN_GAPS).
        "down": list(arm.down_at_yaw(0.0)),
        "clearance": clearance,
    }
    # Say now which slots the arm cannot serve with the tool down, rather than
    # one carton at a time. On a KR210 with the pallet 1.9 m out the far
    # column's ceiling was 0.664 m; a third layer there is not a cycle that
    # will fail, it is a cell that was laid out wrong.
    cell["reach"] = slot_reach(cell)
    cell["spec"] = {
        "boxes": boxes,
        "box": box,
        "box_mass": box_mass,
        "deck": deck,
        "stop_x": stop_x,
        "offset_y": offset_y,
        "speed": speed,
        "pallet_y": pallet_y,
        "rows": rows,
        "cols": cols,
        "layers": layers,
        "robot": robot,
        "guides": guides,
        "grip_distance": grip_distance,
        "pedestal": pedestal,
        "dressing": dressing,
        "pallet": pallet,
        "rmp_config": rmp_config,
    }
    return cell


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


def _down_of(cell: dict) -> list:
    """The tool-down quaternion for this cell's flange; DOWN when unrecorded."""
    return list(cell.get("down") or DOWN)


def _box_of(cell: dict) -> float:
    """The carton this cell was actually built with, not the demo's default.

    Reading the module constant here is the whole overfitting failure in one
    line: the cell builds a 22 cm carton, the pick reaches for the top of a
    15 cm one, and the cup descends 35 mm into the box it meant to seal.
    """
    return float(cell.get("box_size", BOX))


# A move counts if the tool got within this of where it was sent. Tighter than
# the 4 cm a placement is judged by, looser than pose_to's 5 mm hold criterion,
# which a settled arm can miss by a millimetre and still have arrived.
ARRIVAL = 0.03


def _arrived(moved) -> bool:
    return bool(moved.reached) or float(moved.final_error) <= ARRIVAL


def _describe(moved) -> str:
    if moved.steps == 0:
        return "no IK solution, arm did not move"
    return "%.3f m short after %d steps" % (float(moved.final_error), int(moved.steps))


#: What an arm in this cell can be stopped by. Named, because "1.08 m short"
#: says nothing and "1.08 m short, touching /World/Belt" names the layout.
OBSTACLES = (BELT, "%sGate" % BELT, PALLET, "/World/Pedestal")


def _blocked_by(arm) -> str:
    touching = []
    for path in OBSTACLES:
        try:
            if arm.touching(path):
                touching.append(path)
        except Exception:  # noqa: BLE001 - a diagnostic, never the failure
            continue
    return (", touching " + " and ".join(touching)) if touching else ""


#: How far under the reach ceiling a traverse runs. The ceiling is where the
#: solver stops finding solutions; right at it the drives track badly.
CEILING_MARGIN = 0.03


def travel_height(arm, place, place_z: float, wanted: float, down=DOWN) -> float | None:
    """The traverse height for a slot: `wanted` if the arm can hold the tool
    down there, otherwise the highest it can, or None if even the release
    height is out of reach.

    Measured on a KR210 on a 0.35 m pedestal with the pallet 1.9 m out: the
    tool-down ceiling over the far column is 0.664 m and over the near column
    0.818 m. A traverse fixed at 0.764 m reached the near slots and none of the
    far ones.
    """
    top = arm.reach_ceiling(place[:2], down, floor=place_z, limit=wanted)
    if top is None:
        return None
    if top >= wanted:
        return wanted
    return max(place_z, top - CEILING_MARGIN)


def slot_reach(cell: dict) -> dict:
    """Which slots this cell's arm can serve, before a carton is committed.

    Returns `{"reachable": [...], "unreachable": [...], "ceilings": {...}}`
    so an unreachable pallet position is reported when the cell is built,
    not discovered one carton at a time.
    """
    arm, cup = cell["arm"], cell["cup"]
    down = _down_of(cell)
    size = _box_of(cell)
    reachable, unreachable, ceilings = [], [], {}
    for slot in cell["slots"]:
        place = slot["place"]
        place_z = float(place[2]) + size / 2.0 + cup.tip_offset
        top = arm.reach_ceiling(place[:2], down, floor=place_z, limit=place_z + 1.0)
        ceilings[slot["index"]] = None if top is None else round(float(top), 3)
        (unreachable if top is None else reachable).append(slot["index"])
    return {"reachable": reachable, "unreachable": unreachable, "ceilings": ceilings}


def _home_of(cell: dict) -> list[float]:
    """A parked pose with the right number of joints for this cell's arm."""
    spec = cell.get("spec") or {}
    if _is_ur(spec.get("robot", "ur10")):
        return list(HOME)
    return [0.0] * int(cell["arm"].dof)


def _is_ur(robot: str) -> bool:
    """The whole UR family shares the UR10's kinematic chain, so HOME fits."""
    return str(robot).lower().startswith("ur")


def go_home(cell: dict, *, settle_steps: int = 90) -> None:
    """Park the arm: lift first, then swing.

    A single joint-space move from over the pallet to home rotates the base
    while the shoulder and elbow are still folded down at release height, so
    the forearm sweeps through whatever was just placed. Measured on a KR210,
    four cartons placed to within 21 mm: at the end of the run one had been
    pushed 0.45 m along the deck and another was on the floor, both shoved
    away from the base. Holding the base joint for the first stage sends the
    arm straight up; the swing happens with the tool high.
    """
    arm = cell["arm"]
    home = list(_home_of(cell))
    lift = list(home)
    lift[0] = float(np.asarray(arm.joint_positions, dtype=float)[0])
    arm.set_joint_positions(lift, settle_steps=settle_steps)
    arm.set_joint_positions(home, settle_steps=settle_steps)


def pick_waiting_box(cell: dict) -> dict:
    """Seal on the box at the stop and lift it clear. Returns what was measured."""
    arm, cup, belt = cell["arm"], cell["cup"], cell["belt"]
    down = _down_of(cell)
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
    size = _box_of(cell)
    go_home(cell)
    arm.scene.settle(0.5)

    here = np.asarray(box.position, dtype=float)
    box_top = float(here[2]) + size / 2.0
    # Route the hover, do not servo it. RMPflow never moved a Fanuc CRX
    # toward its hover point at all (closest approach 0.67 m, i.e. where it
    # started) while Lula IK drove the same arm to 0.3 mm; and on a UR5e the
    # servo stalled 65 mm out at the edge of reach. `pose_to` either arrives
    # or says so.
    hover = arm.pose_to(
        [float(here[0]), float(here[1]), box_top + cup.tip_offset + 0.18], down, corrections=6, raise_on_fail=False
    )
    if not _arrived(hover):
        return {
            "picked": False,
            "reason": "hover over the carton not reached: %s%s" % (_describe(hover), _blocked_by(arm)),
        }

    # Re-read once more now the arm is parked above it, so the descent is
    # centred on the face rather than on where the box used to be.
    here = np.asarray(box.position, dtype=float)
    box_top = float(here[2]) + size / 2.0

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
        # Re-read every attempt. `here` was measured once, before the first
        # descent, and a carton that moves between attempts then gets the cup
        # driven at where it used to be - which shoves it further, so the next
        # attempt is aimed further out still. At 1.0 kg the carton does not
        # move and this never showed. At 1.5 kg ten descents walked it off the
        # side of the belt and onto the floor, and the pick reported "cup did
        # not seal", which is true and says nothing about why.
        here = np.asarray(box.position, dtype=float)
        box_top = float(here[2]) + size / 2.0
        arm.pose_to(
            [float(here[0]), float(here[1]), box_top + cup.tip_offset + STANDOFF - attempt * 0.002],
            down,
            corrections=8,
            raise_on_fail=False,
        )
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
        return {"picked": False, "reason": f"cup did not seal after 10 descents (status {cup.status})"}

    # The lift carries the same two corrections the descent needed, for the same
    # reasons, and it took a third debugging pass to notice they were missing
    # here. With the default budget of four this raised mid-lift; with three
    # trailing refines it shook the carton off the cup, and the box came back
    # down to 0.525 — belt height — so the pick reported `rise: -0.0000` and
    # looked like a grasp that never happened rather than one that let go.
    arm.pose_to(
        [float(here[0]), float(here[1]), box_top + cup.tip_offset + 0.30], down, corrections=8, raise_on_fail=False
    )
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
    down = _down_of(cell)
    scene = arm.scene
    if not (cup.holding and cup.gripped_objects):
        return {"placed": False, "reason": "nothing held"}
    if box is None:
        box = cell["belt"].boxes[0]

    place = slot["place"]
    # Release height: `place` is where the carton's centre goes, so the tool
    # sits half a box plus the cup above it.
    size = _box_of(cell)
    place_z = float(place[2]) + size / 2.0 + cup.tip_offset
    wanted = max(float(slot["approach"][2]), 0.55) + cup.tip_offset + size / 2.0
    travel_z = travel_height(arm, place, place_z, wanted, down=down)
    if travel_z is None:
        return {
            "placed": False,
            "slot": slot["index"],
            "reason": "slot out of reach with the tool down at %s; move the pallet closer or lower the stack"
            % np.round([place[0], place[1], place_z], 3).tolist(),
        }

    # Every move is checked before the cup opens. `pose_to` returns without
    # moving when the solver finds no solution, and on the KR210 that happened
    # on the traverse from over the belt: the arm stayed put, the descent then
    # went partway, and the cup opened wherever it was - 0.55 m off the slot,
    # reported as a placement error rather than the move failure it was.
    moved = arm.pose_to([float(place[0]), float(place[1]), travel_z], down, corrections=8, raise_on_fail=False)
    scene.settle(1.0)
    if not (cup.holding and cup.gripped_objects):
        return {"placed": False, "reason": "dropped during traverse"}
    if not _arrived(moved):
        return {"placed": False, "reason": "traverse not reached: %s" % _describe(moved)}

    moved = arm.pose_to([float(place[0]), float(place[1]), place_z], down, corrections=8, raise_on_fail=False)
    scene.settle(1.0)
    if not _arrived(moved):
        # Still holding: go back up rather than let go over the wrong spot.
        arm.pose_to([float(place[0]), float(place[1]), travel_z], down, corrections=4, raise_on_fail=False)
        return {"placed": False, "reason": "descent not reached: %s" % _describe(moved)}

    cup.open(settle_steps=0)
    for _ in range(12):
        scene.settle(0.3)
        if not cup.holding:
            break

    # Retreat to a height the solver has already said yes to. `place_z + 0.25`
    # sits above the far column's 0.661 m ceiling, so the retreat did nothing
    # there and the joint-space home that followed started with the cup 3 mm
    # over the carton — and swept it 0.34 m along the deck, turned 43 degrees.
    retreat_z = min(place_z + 0.25, travel_z)
    lifted = arm.pose_to([float(place[0]), float(place[1]), retreat_z], down, corrections=8, raise_on_fail=False)
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
        "retreated": _arrived(lifted),
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

    belt, slots = cell["belt"], cell["slots"]
    count = len(slots) if count is None else min(int(count), len(slots))

    cycles, placed, stacked = [], 0, []
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
            cycles.append({"slot": index, "ok": False, "reason": picked.get("reason", "pick failed")})
            break

        result = place_on_slot(cell, slots[index], box=waiting)
        finished = float(SimulationManager.get_simulation_time())
        ok = bool(result.get("placed"))
        placed += int(ok)
        cycles.append(
            {
                "slot": index,
                "ok": ok,
                "seconds": round(finished - started, 2),
                "error": result.get("error"),
                "reason": result.get("reason"),
            }
        )
        if not ok:
            break
        stacked.append((slots[index], waiting))

    total = float(SimulationManager.get_simulation_time()) - run_started
    times = [c["seconds"] for c in cycles if c.get("ok")]
    # The stack at the end, not each carton as it landed. Four cartons each
    # measured within 21 mm on release; by the end of the run one had been
    # pushed 0.45 m and another was on the floor, and the per-cycle numbers
    # said nothing. A run is complete when the pallet is, not when the last
    # cycle was.
    stack = verify_pallet([box for _, box in stacked], [slot for slot, _ in stacked]) if stacked else None
    intact = stack["complete"] if stack else False
    return {
        "placed": placed,
        "of": count,
        "complete": placed == count and intact,
        "stack": stack,
        "seconds_total": round(total, 2),
        "seconds_per_carton": round(sum(times) / len(times), 2) if times else None,
        "cartons_per_hour": round(3600.0 / (sum(times) / len(times)), 1) if times else None,
        "cycles": cycles,
    }


if __name__ == "__main__":
    cell = build()
    print("gains:", cell["gains"]["joints"])
    print(palletise(cell))
