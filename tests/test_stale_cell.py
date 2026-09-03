"""One stage, two cells: the older one is still solid.

`Scene.stop()` ends the simulation and leaves every prim exactly where it
was. Build a second cell on the same stage and the first is still there,
still collidable, still holding things. This is the regression for a real
run: an escapement blade left over from an infeed experiment clamped the
palletising cell's carton queue with 26 N, while the belt reported a
kinematic body, `surfaceVelocityEnabled True`, surface velocity (0.2, 0, 0)
and friction 0.9 on both sides. Every conveyor observable was correct.
"""
from __future__ import annotations

import pytest

from simliverse_sim.scene import Scene


class _FakePrim:
    def __init__(self, name, stage):
        self._name = name
        self._stage = stage

    def GetName(self):
        return self._name

    def GetPath(self):
        return _FakePath("/World/" + self._name)

    def GetChildren(self):
        return [_FakePrim(n, self._stage) for n in self._stage.children]

    def IsValid(self):
        return True


class _FakePath:
    def __init__(self, text):
        self.pathString = text


class _FakeStage:
    def __init__(self, children):
        self.children = list(children)
        self.removed = []

    def GetPrimAtPath(self, path):
        if path == "/World":
            return _FakePrim("World", self)
        return None

    def RemovePrim(self, path):
        self.removed.append(path)
        name = path.rsplit("/", 1)[-1]
        if name in self.children:
            self.children.remove(name)


def _scene(children):
    """A Scene whose stage is fake, without touching Scene itself.

    Patching `Scene.stage` in place would leak the fake into every later
    test in the run, which is its own version of the bug under test.
    """
    stage = _FakeStage(children)

    class _Staged(Scene):
        stage = property(lambda self: stage)

    return _Staged.__new__(_Staged), stage


def test_a_previous_cells_prims_are_removed() -> None:
    scene, stage = _scene([
        "PhysicsScene", "GroundPlane", "PhysicsMaterials",
        "Escapement", "Plate", "Plate_Stop", "Belt", "Box0", "UR",
    ])

    removed = scene.clear_world()

    assert "/World/Escapement" in removed, (
        "the blade that held the queue must not survive a rebuild")
    for path in ("/World/Plate", "/World/Plate_Stop", "/World/Belt",
                 "/World/Box0", "/World/UR"):
        assert path in removed


def test_physics_configuration_and_the_floor_survive() -> None:
    """Re-authoring these mid-session invalidates handles that are still live."""
    scene, stage = _scene([
        "PhysicsScene", "GroundPlane", "PhysicsMaterials", "Escapement",
    ])

    removed = scene.clear_world()

    assert removed == ["/World/Escapement"]
    assert set(stage.children) == {
        "PhysicsScene", "GroundPlane", "PhysicsMaterials"}


def test_the_sweep_is_by_what_is_there_not_by_a_kept_list() -> None:
    """A name nobody wrote down is exactly the one that outlives its cell."""
    scene, stage = _scene(
        ["PhysicsScene", "SomethingNobodyListed", "FutureFixture"])

    removed = scene.clear_world()

    assert "/World/SomethingNobodyListed" in removed
    assert "/World/FutureFixture" in removed


def test_a_caller_can_widen_what_survives() -> None:
    scene, stage = _scene(["PhysicsScene", "CellDomeLight", "Escapement"])

    removed = scene.clear_world(keep=("PhysicsScene", "CellDomeLight"))

    assert removed == ["/World/Escapement"]


def test_clearing_an_empty_world_is_not_an_error() -> None:
    scene, stage = _scene([])
    assert scene.clear_world() == []


def test_a_stage_without_a_world_is_not_an_error() -> None:
    class _Empty:
        def GetPrimAtPath(self, path):
            return None

    class _Staged(Scene):
        stage = property(lambda self: _Empty())

    assert _Staged.__new__(_Staged).clear_world() == []
