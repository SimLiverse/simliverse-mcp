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

"""A palletising cell: boxes down a belt, a stop, and a KR210 stacking a pallet.

The scene the whole conveyor work exists to make possible. Boxes ride a driven
belt, come to rest against a stop at the end of it, and a Kuka KR210 with a
suction cup lifts them one at a time onto a pallet in a 2 x 2 x 2 pattern.

Sized to the robot, not to taste. Every number here is either measured from the
shipped assets or forced by the arm:

* **30 cm boxes.** The KR210 is a 2.7 m, 150 kg-payload palletising arm and its
  wrist is physically wider than a 10 cm box, so small cartons cannot be
  approached without the wrist fouling the ones beside them. 30 cm boxes at
  ~1.9 m are the job this machine is built for.
* **Belt deck at 0.90 m.** `Conveyor.build()` rather than the shipped asset,
  because `ConveyorBelt_A09` decks at 1.781 m — fine for the arm, but it puts
  the pick 1.8 m up while the pallet decks at 0.14 m, and a 1.6 m vertical
  travel per box makes a slow cycle look like a broken one. Pass
  `asset=True` to build it from the real prop instead; the geometry is
  re-measured from the belt either way.
* **Suction, not fingers.** A KR210 ships with a bare flange. Real palletisers
  use vacuum, a 30 cm box has a flat top, and a parallel gripper would have to
  span the whole carton. `attach_suction_gripper` is the honest fitting, and it
  is authored before physics starts because a surface gripper created while the
  timeline runs never registers.

    from demo.kuka_palletizing import build
    info = build()

`build()` returns the paths and the pattern the controller expects. It does not
press Play — `controllers/kuka_palletizing.py` is what performs the task, and it
performs it from the scene rather than from remembered coordinates.
"""

from simliverse_sim import Conveyor, Robot, Scene, pallet_slots, spawn_prop

# ── The cell ─────────────────────────────────────────────────────────────────

ARM = "/World/Arm"
BELT = "/World/Conveyor"
PALLET = "/World/Pallet"

BOX = (0.30, 0.30, 0.30)      # full size, metres. Sized to the KR210's wrist.
BOX_MASS = 5.0

BELT_DECK = 0.90              # height of the belt surface
BELT_LENGTH = 3.2
BELT_WIDTH = 0.70
BELT_SPEED = 0.30             # m/s. A real palletising infeed runs 0.2-0.5.

# The cell layout, and the number that matters is BELT_OFFSET_Y.
#
# The obvious arrangement - belt along +X with its stop at the arm's reach -
# puts the *middle* of a 3.2 m belt directly on top of a robot standing at the
# origin, because a belt is placed by its centre. Measured on the live sim: the
# KR210 ended up inside the conveyor, its joints stopped responding to position
# commands entirely, and every target came back "outside the workspace" with
# the end effector pinned at the belt's deck height. Nothing in the error said
# "your robot is inside a conveyor".
#
# So the belt runs *past* the arm, offset across its travel, which is also how
# a real infeed is laid out: the arm stands beside the line, not on it.
BELT_STOP_X = 1.30            # where the stop is, and so where a box waits
BELT_OFFSET_Y = -1.05         # belt centre-line, clear of the arm's base
PALLET_REACH = 1.70           # pallet centre, on +Y

PALLET_DECK_Z = 0.1425        # measured from Isaac's pallet.usd
PALLET_DECK = (1.2132, 0.8023)

ROWS, COLS, LAYERS = 2, 2, 2  # eight boxes


def build(scene: Scene | None = None, *, asset: bool = False, boxes: int = 4) -> dict:
    """Author the cell. Returns what the controller needs to drive it."""
    scene = scene or Scene.get()
    scene.stop()
    scene.configure_physics()
    scene.ensure_ground_plane()

    arm = Robot.spawn("kuka_kr210", position=[0.0, 0.0, 0.0], prim_path=ARM)

    # The belt runs along +X and stops short of the arm, so the last box sits at
    # BELT_REACH with its far face against the stop. Centre it so that the stop
    # lands there rather than working backwards from the belt's midpoint later.
    belt_centre_x = BELT_STOP_X - BELT_LENGTH / 2.0
    if asset:
        belt = Conveyor.from_prop(
            "conveyorbelt_a09",
            prim_path=BELT,
            position=[belt_centre_x, BELT_OFFSET_Y, 0.0],
            direction=(1, 0, 0),
            speed=BELT_SPEED,
            scene=scene,
        )
    else:
        belt = Conveyor.build(
            BELT,
            length=BELT_LENGTH,
            width=BELT_WIDTH,
            position=[belt_centre_x, BELT_OFFSET_Y, BELT_DECK],
            direction=(1, 0, 0),
            speed=BELT_SPEED,
            gate=True,
            gate_height=0.35,
            scene=scene,
        )

    queued = belt.load(
        boxes,
        box=BOX,
        mass=BOX_MASS,
        spacing=BOX[0] * 1.5,
        start_offset=0.30,
    )

    # The pallet is a static collider and stays where it is put.
    spawn_prop("pallet", prim_path=PALLET,
               position=[0.0, PALLET_REACH, 0.0], scene=scene)

    slots = pallet_slots(
        origin=[0.0, PALLET_REACH, PALLET_DECK_Z],
        box=BOX,
        rows=ROWS, cols=COLS, layers=LAYERS,
        gap=0.01,
        deck=PALLET_DECK,
    )

    # Authored before anything starts physics. A surface gripper created while
    # the timeline runs is never registered: the plugin logs "Gripper not found"
    # every frame while the Python side keeps reporting a healthy Open.
    cup = arm.attach_suction_gripper(
        max_grip_distance=0.06, cup_radius=0.10, cup_length=0.05
    )

    return {
        "arm": ARM,
        "gripper": cup.prim_path,
        "tip_offset": cup.tip_offset,
        "belt": belt.describe(),
        "boxes": [b.prim_path for b in queued],
        "box_size": list(BOX),
        "pallet": PALLET,
        "slots": slots,
        "pick_height": belt.top_z + BOX[2] / 2.0,
        "carry_z": max(belt.top_z, PALLET_DECK_Z + LAYERS * BOX[2]) + 0.45,
    }


if __name__ == "__main__":
    scene = Scene.get()
    info = build(scene)
    scene.play()
    # The cup's authoring stopped the timeline after the belt was switched on,
    # which drops the drive. Re-assert it now that physics is running.
    Conveyor.from_description(info["belt"], scene=scene).start()
    scene.settle(8.0)

    print("cell built")
    for key in ("arm", "gripper", "pallet", "pick_height", "carry_z"):
        print("  %-13s %s" % (key, info[key]))
    print("  %-13s %s" % ("belt", info["belt"]["mechanism"]))
    print("  %-13s %d queued, %d slots" % (
        "load", len(info["boxes"]), len(info["slots"])))

    print("  first slot     %s" % info["slots"][0]["rest"])
