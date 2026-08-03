"""
Robot handles, one control surface per morphology.

`Robot.spawn(...)` and `Robot.attach(...)` return the right class for the body:

    Manipulator        arms — Cartesian move_ee_to, gripper, grasp, throw
    DexterousHand      standalone multi-finger hands — per-finger control
    WheeledRobot       differential/skid bases — drive, drive_to
    MobileManipulator  wheeled base carrying an arm — both surfaces
    LeggedRobot        quadrupeds — stand, limb control, policy-driven walk
    Humanoid           legs + arms + torso
    AerialRobot        multirotors — thrust control, fly_to, hover

Morphology is inferred from the articulation's joint structure, not from the prim
path, so a robot spawned under any name still gets the correct controller.
"""

from .aerial import AerialRobot, FlightError
from .base import JointGroups, Morphology, Robot, classify_morphology
from .legged import Humanoid, LeggedRobot, LocomotionError
from .library import CATALOGUE, RobotAsset, list_robots, resolve, spawn_robot
from .manipulator import DexterousHand, Gripper, Manipulator, MotionError, MotionResult
from .mobile import MobileManipulator, NavigationError, WheeledRobot

__all__ = [
    "CATALOGUE",
    "AerialRobot",
    "DexterousHand",
    "FlightError",
    "Gripper",
    "Humanoid",
    "JointGroups",
    "LeggedRobot",
    "LocomotionError",
    "Manipulator",
    "MobileManipulator",
    "Morphology",
    "MotionError",
    "MotionResult",
    "NavigationError",
    "Robot",
    "RobotAsset",
    "WheeledRobot",
    "classify_morphology",
    "list_robots",
    "resolve",
    "spawn_robot",
]
