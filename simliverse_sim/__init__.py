"""
simliverse_sim — a robot control library for Isaac Sim.

Runs *inside* the Isaac Sim process. Generated control code imports this rather
than driving raw joint angles, because Cartesian motion, locomotion, grasping and
physics verification are things a language model cannot derive in-context but can
call correctly.

`Robot.spawn()` returns a handle whose control surface matches the body:

    Manipulator        move_ee_to, gripper, grasp, throw
    DexterousHand      per-finger control, multi-contact grasp
    WheeledRobot       drive, drive_to
    MobileManipulator  both of the above
    LeggedRobot        stand, limb control, policy-driven walk
    Humanoid           legs, arms, torso
    AerialRobot        thrust control, fly_to, hover

Manipulation example:

    from simliverse_sim import Scene, Robot, verify_grasp

    scene = Scene.get()
    scene.configure_physics()
    scene.play()

    robot = Robot.spawn("franka")
    ball = scene.spawn_rigid("/World/Ball", shape="Sphere", radius=0.04,
                             position=[0.45, 0.0, 0.04], mass=0.05)

    before = ball.position.copy()
    robot.grasp(ball)
    print(verify_grasp(robot, ball, previous_position=before))
    print(robot.throw(ball, direction=[1, 0, 0.8], speed=2.5))

Conveyor palletising example:

    from simliverse_sim import Conveyor, Robot, pallet_slots

    belt = Conveyor.build(length=3.0, width=0.8, position=[0, 0, 0.9], speed=0.3)
    boxes = belt.load(4, box=(0.18, 0.13, 0.11))
    arm = Robot.spawn("kuka_kr210")
    arm.attach_suction_gripper()

    slots = pallet_slots(origin=[0.0, 1.2, 0.145], box=(0.18, 0.13, 0.11),
                         rows=2, cols=2)
    ready = belt.box_at_gate()          # settled against the stop

Navigation example:

    rover = Robot.spawn("carter", position=[0, 0, 0.1])
    rover.drive_to([3.0, 1.0])

See `simliverse-core/docs/adr/012-agent-hierarchy-rewrite.md` for the design.
"""

from ._compat import IsaacSimUnavailable, isaac_version
from .assertions import (
    Check,
    Report,
    airborne,
    grasped,
    moved_under_own_power,
    not_teleported,
    physics_running,
    reached_height,
    reached_position,
    travelled,
    upright,
    verify_grasp,
    verify_navigation,
    verify_throw,
)
from . import controller
from .controller import ControllerError
from .conveyor import Conveyor, ConveyorError, drive_surface
from .deadplate import DeadPlate, DeadPlateError, Escapement
from .sketch import (
    SketchError,
    fence_from_sketch,
    parse_sketch,
    zones_from_sketch,
)
from .guarding import (
    GuardingError,
    SafetyFence,
    spawn_beacon,
    spawn_cabinet,
    spawn_operator,
    spawn_operator_platform,
    spawn_pedestal,
)
from .objects import RigidObject
from .robots import (
    AerialRobot,
    DexterousHand,
    FlightError,
    Gripper,
    Humanoid,
    LeggedRobot,
    LocomotionError,
    Manipulator,
    MobileManipulator,
    Morphology,
    MotionError,
    MotionResult,
    NavigationError,
    Robot,
    StaleArticulation,
    SuctionGripper,
    WheeledRobot,
    list_robots,
    spawn_robot,
)
from .palletizing import PalletError, pallet_slots, verify_pallet
from .vision import STANDARD_VIEWS, VisionUnavailable, capture, look, png, views
from .props import (
    PropNotFound,
    find_prop,
    list_props,
    spawn_prop,
    verify_index,
)
from .scene import PhysicsConfig, Scene

__all__ = [
    "AerialRobot",
    "Check",
    "ControllerError",
    "Conveyor",
    "DeadPlate",
    "GuardingError",
    "SketchError",
    "fence_from_sketch",
    "parse_sketch",
    "zones_from_sketch",
    "SafetyFence",
    "spawn_beacon",
    "spawn_cabinet",
    "spawn_operator",
    "spawn_operator_platform",
    "spawn_pedestal",
    "Escapement",
    "DeadPlateError",
    "STANDARD_VIEWS",
    "VisionUnavailable",
    "views",
    "capture",
    "look",
    "png",
    "ConveyorError",
    "controller",
    "DexterousHand",
    "FlightError",
    "Gripper",
    "SuctionGripper",
    "Humanoid",
    "IsaacSimUnavailable",
    "LeggedRobot",
    "LocomotionError",
    "Manipulator",
    "MobileManipulator",
    "Morphology",
    "MotionError",
    "MotionResult",
    "NavigationError",
    "PalletError",
    "PhysicsConfig",
    "Report",
    "RigidObject",
    "Robot",
    "Scene",
    "StaleArticulation",
    "WheeledRobot",
    "airborne",
    "drive_surface",
    "grasped",
    "isaac_version",
    "list_robots",
    "list_props",
    "find_prop",
    "spawn_prop",
    "PropNotFound",
    "spawn_robot",
    "moved_under_own_power",
    "not_teleported",
    "pallet_slots",
    "physics_running",
    "reached_height",
    "reached_position",
    "travelled",
    "upright",
    "verify_grasp",
    "verify_pallet",
    "verify_navigation",
    "verify_throw",
]

__version__ = "0.2.0"
