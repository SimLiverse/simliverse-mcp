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

"""Binding to a suction cup that is already on the stage.

The controller case. A cup must be authored while the scene is built — one
created after the timeline starts is never registered by the plugin — but a
controller builds fresh handles inside `compute()` on every Play, so it has to
find the cup rather than make one. Getting this wrong authors a second cup on
top of the first, or drives the wrong arm's.

USD is faked here; what is under test is the selection and the error messages,
which are the parts that decide whether a wrong bind is loud or silent.
"""

import pytest

from simliverse_sim.robots import manipulator as M
from simliverse_sim.robots.manipulator import Manipulator, MotionError


class _Attr:
    def __init__(self, value):
        self._value = value

    def __bool__(self):
        return True

    def Get(self):
        return self._value


class _Prim:
    def __init__(self, path, type_name="", attrs=None):
        self._path = path
        self._type = type_name
        self._attrs = attrs or {}

    def GetPath(self):
        return self._path

    def GetTypeName(self):
        return self._type

    def IsValid(self):
        return True

    def GetAttribute(self, name):
        return self._attrs.get(name)


class _Stage:
    def __init__(self, prims):
        self._prims = prims

    def GetPseudoRoot(self):
        return "root"

    def GetPrimAtPath(self, path):
        for prim in self._prims:
            if prim.GetPath() == path:
                return prim
        return _Prim(path, attrs={})


def _arm(prim_path="/World/Arm"):
    """A Manipulator without running its __init__, which needs a live stage."""
    arm = object.__new__(Manipulator)
    arm.prim_path = prim_path
    arm.scene = object()
    arm.suction = None
    return arm


@pytest.fixture
def fake_usd(monkeypatch):
    """Patch the stage, and the one `pxr` the code under test imports.

    One coherent fake module rather than several: `rebind_suction` imports
    `Usd`/`UsdPhysics` to find the gripper and `UsdGeom` to measure the cup, so
    a fake carrying only some of those falls through to the real pxr and fails
    with a Boost argument error that says nothing about the test.
    """

    def install(prims, cup_height=None):
        stage = _Stage(prims)
        monkeypatch.setattr(M, "get_stage", lambda: stage)

        class _Usd:
            @staticmethod
            def PrimRange(_root):
                return list(prims)

        class _UsdPhysics:
            RigidBodyAPI = object()

        class _Cylinder:
            def __init__(self, _prim):
                pass

            def GetHeightAttr(self):
                return _Attr(cup_height)

        fake = type(
            "pxr",
            (),
            {
                "Usd": _Usd,
                "UsdPhysics": _UsdPhysics,
                "UsdGeom": type("UsdGeom", (), {"Cylinder": _Cylinder}),
            },
        )
        monkeypatch.setitem(__import__("sys").modules, "pxr", fake)
        return stage

    return install


def test_the_only_surface_gripper_on_the_stage_is_the_one_bound(fake_usd) -> None:
    fake_usd(
        [
            _Prim("/World/Arm_link6_SuctionCup/SurfaceGripper", "SurfaceGripper"),
            _Prim("/World/Box0", "Cube"),
        ]
    )
    assert _arm()._find_surface_gripper() == "/World/Arm_link6_SuctionCup/SurfaceGripper"


def test_no_gripper_says_it_must_be_authored_before_physics(fake_usd) -> None:
    """The actual constraint, not just 'not found'."""
    fake_usd([_Prim("/World/Box0", "Cube")])
    with pytest.raises(MotionError, match="before physics starts"):
        _arm()._find_surface_gripper()


def test_two_arms_resolve_to_the_one_naming_this_robot(fake_usd) -> None:
    fake_usd(
        [
            _Prim("/World/World_Arm_link6_SuctionCup/SurfaceGripper", "SurfaceGripper"),
            _Prim("/World/World_Other_link6_SuctionCup/SurfaceGripper", "SurfaceGripper"),
        ]
    )
    found = _arm("/World/Arm")._find_surface_gripper()
    assert found == "/World/World_Arm_link6_SuctionCup/SurfaceGripper"


def test_an_ambiguous_stage_raises_rather_than_driving_the_wrong_cup(fake_usd) -> None:
    """Guessing here silently operates another robot's gripper."""
    fake_usd(
        [
            _Prim("/World/CupA/SurfaceGripper", "SurfaceGripper"),
            _Prim("/World/CupB/SurfaceGripper", "SurfaceGripper"),
        ]
    )
    with pytest.raises(MotionError, match="ambiguous"):
        _arm("/World/Arm")._find_surface_gripper()


def test_rebind_reads_the_settings_written_on_the_prim(fake_usd, monkeypatch) -> None:
    """`create` writes these so they survive stop/play; a rebind must use them.

    Otherwise the rebound gripper runs on default limits and the grasp behaves
    differently on the second Play than it did on the first.
    """
    path = "/World/Arm_link6_SuctionCup/SurfaceGripper"
    fake_usd(
        [
            _Prim(
                path,
                "SurfaceGripper",
                {
                    "isaac:maxGripDistance": _Attr(0.06),
                    "isaac:coaxialForceLimit": _Attr(1234.0),
                    "isaac:shearForceLimit": _Attr(555.0),
                    "isaac:retryInterval": _Attr(2.0),
                },
            ),
        ]
    )
    captured = {}

    class _FakeGripper:
        def __init__(self, prim_path, *, scene=None, **settings):
            captured["path"] = prim_path
            captured["settings"] = settings

    monkeypatch.setattr(M, "SuctionGripper", _FakeGripper)

    arm = _arm()
    bound = arm.rebind_suction()
    assert captured["path"] == path
    assert captured["settings"]["max_grip_distance"] == pytest.approx(0.06)
    assert captured["settings"]["coaxial_force_limit"] == pytest.approx(1234.0)
    assert arm.suction is bound


def test_rebind_accepts_an_explicit_path_without_searching(fake_usd, monkeypatch) -> None:
    fake_usd([])  # nothing to find; the explicit path must be used regardless

    class _FakeGripper:
        def __init__(self, prim_path, *, scene=None, **settings):
            self.prim_path = prim_path

    monkeypatch.setattr(M, "SuctionGripper", _FakeGripper)
    arm = _arm()
    assert arm.rebind_suction("/World/Explicit").prim_path == "/World/Explicit"


def test_rebind_measures_the_cup_so_pick_heights_are_right(fake_usd, monkeypatch) -> None:
    """A rebound cup reporting tip_offset 0.0 buries itself in the object.

    Measured on the worker: the flange was sent to `box top + 0.0 + clearance`,
    which put the cup 45 mm inside a 30 cm carton and shoved it off the belt
    sideways rather than sealing on it. The number is the cup cylinder's own
    height and it is on the stage, so nothing has to be remembered.
    """
    path = "/World/Arm_tool0_SuctionCup/SurfaceGripper"
    fake_usd([_Prim(path, "SurfaceGripper", {})], cup_height=0.05)

    class _FakeGripper:
        def __init__(self, prim_path, *, scene=None, **settings):
            self.prim_path = prim_path
            self.tip_offset = 0.0
            self.cup_path = None

    monkeypatch.setattr(M, "SuctionGripper", _FakeGripper)
    bound = _arm().rebind_suction()
    assert bound.tip_offset == pytest.approx(0.05)
    assert bound.cup_path == "/World/Arm_tool0_SuctionCup"


def test_an_unmeasurable_cup_leaves_tip_offset_zero_and_says_so(fake_usd, monkeypatch, caplog) -> None:
    path = "/World/Arm_tool0_SuctionCup/SurfaceGripper"
    fake_usd([_Prim(path, "SurfaceGripper", {})], cup_height=None)

    class _FakeGripper:
        def __init__(self, prim_path, *, scene=None, **settings):
            self.tip_offset = 0.0
            self.cup_path = None

    monkeypatch.setattr(M, "SuctionGripper", _FakeGripper)
    with caplog.at_level("WARNING"):
        bound = _arm().rebind_suction()
    assert bound.tip_offset == 0.0
    assert "bury the cup" in caplog.text
