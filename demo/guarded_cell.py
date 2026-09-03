"""The palletising cell with the guarding an integrator would actually quote.

The working cell - arm, belt, pallet - is the part that answers "does it
palletise". This module adds the part that answers "is it a cell": a fenced
perimeter with a gate, a slot where the conveyor runs through the line, a
control cabinet, a stack light, and a marked standing area outside the gate.

That furniture is not decoration. The fence line is what fixes where the belt
enters, which side the operator works from, and how much room the arm has, and
those are the decisions a customer argues about. A layout drawing that leaves
them out has not answered the question it was asked.

    from demo.guarded_cell import build
    cell = build()
    print(cell["fence"].describe())
"""

from __future__ import annotations

from typing import Any

from simliverse_sim import (
    SafetyFence,
    Scene,
    spawn_beacon,
    spawn_cabinet,
    spawn_operator,
    spawn_operator_platform,
)

from . import ur10_palletizing as base

#: A UR10 reaches about 1.3 m. Guarding goes outside the envelope, not on it:
#: a fence set exactly at the reach is a fence the arm polishes.
REACH = 1.3
STANDOFF = 0.55


def guard(cell: dict, *, scene: Any = None, gate: str = "south",
          margin: float = STANDOFF) -> dict:
    """Fence an existing cell, leaving the conveyor a slot to run through.

    The footprint is measured from what the cell actually contains rather than
    typed in: the arm at the origin with its reach, the pallet, and the belt
    running out to its far end. Typing it in is how a fence ends up crossing
    the belt, and a panel across the belt is invisible from the front and
    stops every carton.
    """
    scene = scene or Scene.get()
    described = cell["described"]
    spec = cell.get("spec") or {}

    belt_y = float(described["centre"][1])
    belt_width = float(described["width"])
    # The belt runs along +x and has to leave the cell somewhere.
    belt_far_x = float(described["centre"][0]) + float(described["length"]) / 2.0

    pallet_y = float(spec.get("pallet_y", base.PALLET_Y))
    # A pallet is 1.21 m long and placed by its centre.
    pallet_far_y = pallet_y + 0.605

    # Every line has to clear both the equipment on that side *and* the arm's
    # reach envelope. Sizing a side off the equipment alone is how the east
    # line ended up 0.95 m from a 1.3 m arm: the belt ends at 0.75, the fence
    # went just past it, and the guarding was inside the working envelope on
    # the one side nothing else stuck out of.
    envelope = REACH + margin
    west = -envelope
    # NOT the belt's far end. The belt is the one thing here that is *meant* to
    # cross the line - an infeed conveyor runs in from outside - so sizing this
    # side to contain it guarantees it never reaches the fence and the crossing
    # is never cut. Sizing it off the equipment that must be enclosed leaves
    # the belt free to pass through.
    east = envelope
    south = min(belt_y - belt_width / 2.0 - margin, -envelope)
    north = max(pallet_far_y + margin, envelope)

    centre = ((west + east) / 2.0, (south + north) / 2.0)
    size = (east - west, north - south)

    # Only cut a slot for the belt if the belt actually reaches the line. Once
    # the east side was widened to clear the reach envelope it moved out to
    # 1.85 m while the belt still ended at 0.75, and the fence was built with a
    # doorway onto nothing - a gap in the guarding that no conveyor uses. The
    # first version of this tested that the slot lined up with the belt's
    # centre-line, which it did, and never that the belt got there.
    reaches = belt_far_x >= east - 1e-6
    crossings = ([{"side": "east", "centre": belt_y,
                   "width": belt_width + 0.30}] if reaches else [])

    fence = SafetyFence.build(
        "/World/Fence", centre=centre, size=size, gate=gate, gate_width=1.0,
        # The slot the conveyor runs through, with room either side: cut to the
        # exact belt width it clips the guide rails when anyone fits them.
        crossings=crossings,
        scene=scene,
    )
    if not reaches:
        # Worth saying. A belt that terminates inside the guarding is a cell
        # with no way in for cartons, which is a layout question, not a bug.
        print("  note: the belt ends %.2f m short of the east guarding, so no "
              "crossing was cut" % (east - belt_far_x))

    fit = fence.fits((0.0, 0.0), reach=REACH)
    if fit["touches_fence"]:
        # Reported, not raised: guarding at the envelope is a real choice.
        # Silently shipping it is not.
        print("  note: the arm reaches within %.2f m of the guarding"
              % fit["clearance"])

    gate_line = fence.centre[1] - fence.size[1] / 2.0
    spawn_cabinet("/World/Cabinet",
                  position=(float(fence.centre[0] - fence.size[0] / 2.0 - 0.5),
                            float(fence.centre[1]), 0.0),
                  scene=scene)
    spawn_beacon("/World/Beacon",
                 position=(float(fence.centre[0] + fence.size[0] / 2.0 - 0.25),
                           float(fence.centre[1] + fence.size[1] / 2.0 - 0.25),
                           0.0),
                 height=1.2, scene=scene)
    platform_y = float(gate_line - 1.0)
    spawn_operator_platform("/World/OperatorPlatform",
                            position=(float(fence.centre[0]), platform_y, 0.0),
                            size=(1.6, 1.6), scene=scene)
    # On the platform, outside the line, facing the gate. Passing the fence
    # means the placement is checked rather than asserted - a figure inside
    # the guarding is a cell nobody may run with the robot live.
    person = spawn_operator("/World/Operator",
                            position=(float(fence.centre[0] + 0.45),
                                      platform_y, 0.05),
                            facing=90.0, fence=fence, scene=scene)

    cell["fence"] = fence
    cell["guarding"] = {
        "fence": fence.describe(),
        "fit": fit,
        "cabinet": "/World/Cabinet",
        "beacon": "/World/Beacon",
        "platform": "/World/OperatorPlatform",
        "operator": person,
    }
    return cell


def build(scene: Any = None, *, boxes: int = 4, pedestal: float = 0.35,
          **spec) -> dict:
    """The measured palletising cell, fenced.

    Guarding is authored after the working cell rather than before it because
    the fence is sized from what the cell contains. Building it first means
    guessing, and the fence is exactly the thing that must not be guessed.
    """
    scene = scene or Scene.get()
    cell = base.build(scene, boxes=boxes, pedestal=pedestal, **spec)
    return guard(cell, scene=scene)


if __name__ == "__main__":
    made = build()
    print(made["guarding"]["fence"])
    print(made["guarding"]["fit"])
