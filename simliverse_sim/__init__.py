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
    SuctionGripper,
    WheeledRobot,
    list_robots,
    spawn_robot,
)
from .scene import PhysicsConfig, Scene

__all__ = [
    "AerialRobot",
    "Check",
    "ControllerError",
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
    "PhysicsConfig",
    "Report",
    "RigidObject",
    "Robot",
    "Scene",
    "WheeledRobot",
    "airborne",
    "grasped",
    "isaac_version",
    "list_robots",
    "spawn_robot",
    "not_teleported",
    "physics_running",
    "reached_height",
    "reached_position",
    "travelled",
    "upright",
    "verify_grasp",
    "verify_navigation",
    "verify_throw",
]

__version__ = "0.2.0"
