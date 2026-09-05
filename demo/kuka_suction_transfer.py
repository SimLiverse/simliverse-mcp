"""Scene for the Kuka suction transfer: two tables ninety degrees apart.

A KR210 stands between them. Three boxes start on the pick table; the controller
moves two of them to the place table and must leave the third where it is.

Sized to the robot on purpose. The KR210 is a 2.7 m, 150 kg-payload palletising
arm and its wrist is wider than a 10 cm box, so 8 cm cubes could not be
approached without the wrist fouling them. 30 cm boxes at 1.9 m are the job this
machine is actually built for.
"""

from simliverse_sim import Robot, Scene

TABLE_H = 1.0  # table top height
REACH = 1.9  # table centres, on +X and on +Y
BOX = 0.15  # half-extent multiplier: 30 cm boxes
BOX_MASS = 5.0

PICK_TABLE = "/World/PickTable"
PLACE_TABLE = "/World/PlaceTable"
ARM = "/World/Arm"


def build(scene: Scene | None = None) -> dict:
    """Author the scene. Returns the prim paths the controller expects."""
    scene = scene or Scene.get()
    scene.stop()

    arm = Robot.spawn("kuka_kr210", position=[0, 0, 0], prim_path=ARM)

    for path, centre in ((PICK_TABLE, [REACH, 0.0]), (PLACE_TABLE, [0.0, REACH])):
        scene.spawn_rigid(
            path,
            shape="cube",
            scale=[0.70, 0.70, TABLE_H / 2],
            position=[centre[0], centre[1], TABLE_H / 2],
            mass=0.0,
            friction=0.9,
            static=True,
        )

    boxes = []
    for index, offset in enumerate((-0.45, 0.0, 0.45)):
        boxes.append(
            scene.spawn_rigid(
                "/World/Box%d" % index,
                shape="cube",
                scale=[BOX, BOX, BOX],
                position=[REACH, offset, TABLE_H + BOX],
                mass=BOX_MASS,
                friction=0.9,
            ).prim_path
        )

    # Author the cup before anything starts physics. A surface gripper created
    # while the timeline runs is never registered - the plugin then logs
    # "Gripper not found" every frame while the Python side keeps reporting a
    # healthy Open. `create` stops the timeline itself now, but building it here
    # keeps the ordering obvious.
    cup = arm.attach_suction_gripper(max_grip_distance=0.05, cup_radius=0.08, cup_length=0.04)

    return {
        "arm": ARM,
        "gripper": cup.prim_path,
        "tip_offset": cup.tip_offset,
        "boxes": boxes,
        "box_top": TABLE_H + 2 * BOX,
        "place": REACH,
    }


if __name__ == "__main__":
    scene = Scene.get()
    info = build(scene)
    scene.play()
    scene.settle(1.5)
    for key, value in info.items():
        print("%-12s %s" % (key, value))
