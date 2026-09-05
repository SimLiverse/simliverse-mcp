"""`route_clearance` must find a link sweeping through a box, and nothing else.

Lula's routes are not collision-checked, and a KR210 on a 286-degree base
turn took its forearm through the north fence panel with a carton in the
cup. This is the check that was missing, driven here over a fake solver so
it runs without Isaac: a two-frame "arm" whose elbow moves along a straight
line, and a box placed either on that line or beside it.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# Loaded by path: importing the package pulls in the simulator.
_root = Path(__file__).resolve().parents[1] / "simliverse_sim" / "robots" / "manipulator.py"


def _load_route_clearance():
    src = _root.read_text()
    start = src.index("    def route_clearance(")
    end = src.index("    def _cspace_generator(")
    body = src[start:end]
    from collections.abc import Sequence
    from typing import Any

    ns = {"np": np, "Sequence": Sequence, "Any": Any}
    # Annotations stay strings, as they do in the real module.
    exec("from __future__ import annotations\nclass _M:\n" + body, ns)  # noqa: S102
    return ns["_M"]


class _Route:
    duration = 1.0

    def sample(self, t):
        return np.array([t, 0.0]), np.zeros(2)


class _Solver:
    """Frame 'elbow' travels from x=0 to x=2 as q[0] goes 0..1."""

    def get_all_frame_names(self):
        return ["base", "elbow"]

    def compute_forward_kinematics(self, frame, q):
        if frame == "base":
            return np.zeros(3), np.eye(3)
        return np.array([2.0 * float(q[0]), 0.0, 1.0]), np.eye(3)


def _arm():
    M = _load_route_clearance()
    arm = M()
    arm._ik = types.SimpleNamespace(get_kinematics_solver=lambda: _Solver())
    arm._ensure_motion_policy = lambda: None
    arm._sync_base_pose = lambda: None
    return arm


def test_a_box_on_the_elbows_path_is_reported():
    hit = _arm().route_clearance(_Route(), [([0.9, -0.1, 0.5], [1.1, 0.1, 1.5], "panel")], margin=0.0)
    assert hit is not None
    assert hit["frame"] == "elbow" and hit["obstacle"] == "panel"
    assert 0.4 < hit["t"] < 0.6, "the hit should be reported where the elbow crosses the box"


def test_a_box_beside_the_path_is_not():
    assert _arm().route_clearance(_Route(), [([0.9, 0.5, 0.5], [1.1, 0.7, 1.5], "panel")], margin=0.0) is None


def test_margin_is_link_thickness():
    """A box 0.2 m to the side is clear for a point but not for a forearm 0.25 m across."""
    box = ([0.9, 0.2, 0.5], [1.1, 0.4, 1.5], "panel")
    assert _arm().route_clearance(_Route(), [box], margin=0.0) is None
    assert _arm().route_clearance(_Route(), [box], margin=0.25) is not None


def test_the_carried_carton_counts():
    """The last frame is extended downward by `carry`: a carton hanging under
    the cup sweeps a box the flange itself would clear."""
    box = ([0.9, -0.1, -0.2], [1.1, 0.1, 0.3], "belt frame")   # below the elbow at z=1
    assert _arm().route_clearance(_Route(), [box], margin=0.0) is None
    assert _arm().route_clearance(_Route(), [box], margin=0.0, carry=0.8) is not None


def test_no_obstacles_means_no_work():
    assert _arm().route_clearance(_Route(), []) is None
