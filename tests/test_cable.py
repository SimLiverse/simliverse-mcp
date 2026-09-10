"""The chain's geometry, before physics has a say.

Laid along the straight chord, links longer than their share overlap and
PhysX throws them apart; laid along a parabola whose arc length matches the
chain, the cable starts near where gravity is about to hang it.
"""

from __future__ import annotations

import numpy as np
import pytest

from simliverse_sim.cable import LINK_LENGTH, Cable, align_z, chain_layout, verify_cable


def _arc_length(centres, link):
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return float(steps.sum() + link)


def test_a_taut_cable_lies_on_its_chord():
    import math

    from simliverse_sim.cable import LINK_LENGTH

    plan = chain_layout([0, 0, 1.5], [2, 0, 1.5], slack=0.0)
    assert plan["depth"] == 0.0
    assert np.allclose(plan["centres"][:, 2], 1.5)
    # The link count follows LINK_LENGTH (shorter links -> a smoother rope), so
    # derive it rather than pin a number that a look-and-feel change would break.
    expected = math.ceil(2.0 / LINK_LENGTH)
    assert plan["links"] == expected
    assert plan["link"] == pytest.approx(2.0 / expected)


def test_slack_hangs_below_the_chord_with_the_right_length():
    plan = chain_layout([0, 0, 1.5], [2, 0, 1.5], slack=0.10)
    assert plan["length"] == pytest.approx(2.2)
    assert plan["depth"] > 0.0
    lowest = float(plan["centres"][:, 2].min())
    assert lowest < 1.5 - 0.2, "10 percent slack on 2 m hangs well below the chord"
    assert _arc_length(plan["centres"], plan["link"]) == pytest.approx(2.2, rel=0.05)


def test_the_ends_sit_half_a_link_inside_the_anchors():
    plan = chain_layout([0, 0, 1.5], [2, 0, 1.5], slack=0.10)
    first, last = plan["centres"][0], plan["centres"][-1]
    assert np.linalg.norm(first - [0, 0, 1.5]) == pytest.approx(plan["link"] / 2.0, abs=0.03)
    assert np.linalg.norm(last - [2, 0, 1.5]) == pytest.approx(plan["link"] / 2.0, abs=0.03)


def test_a_vertical_drop_bulges_sideways_rather_than_nowhere():
    plan = chain_layout([0, 0, 2.0], [0, 0, 0.5], slack=0.2)
    assert plan["centres"][:, 0].max() > 0.05


def test_link_count_follows_length_unless_given():
    assert chain_layout([0, 0, 0], [1, 0, 0], slack=0.0)["links"] == int(np.ceil(1.0 / LINK_LENGTH))
    assert chain_layout([0, 0, 0], [1, 0, 0], slack=0.0, links=5)["links"] == 5


@pytest.mark.parametrize(
    "bad", [dict(start=[0, 0, 0], end=[0, 0, 0], slack=0.1), dict(start=[0, 0, 0], end=[1, 0, 0], slack=-0.1)]
)
def test_impossible_cables_are_refused(bad):
    with pytest.raises(ValueError):
        chain_layout(**bad)


def _rotate(q, v):
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ np.asarray(v, dtype=float)


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 0.3], [-0.2, 0.4, -0.9]])
def test_align_z_turns_the_capsule_axis_onto_the_tangent(direction):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    q = align_z(d)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(_rotate(q, [0, 0, 1]), d, atol=1e-9)


class _Link:
    def __init__(self, position, speed=0.0):
        self.position = np.asarray(position, dtype=float)
        self.speed = speed
        self.prim_path = "/World/Cable/link"


def _hanging(sag=0.08, speed=0.0, torn=False):
    plan = chain_layout([0, 0, 1.5], [2, 0, 1.5], slack=0.1)
    links = []
    for i, c in enumerate(plan["centres"]):
        t = (i + 0.5) / plan["links"]
        p = np.array([c[0], c[1], 1.5 - 4 * sag * t * (1 - t)])
        links.append(_Link(p, speed))
    if torn:
        links[-1].position = np.array([2.0, 0.0, 0.3])
    return Cable("/World/Cable", links, layout=plan, start=[0, 0, 1.5], end=[2, 0, 1.5], scene=None)


def test_verify_reports_a_hanging_settled_cable_as_ok():
    result = verify_cable(_hanging())
    assert result["ok"], result["problems"]
    assert result["sag"] == pytest.approx(0.08, abs=0.01)


def test_verify_names_a_torn_anchor_and_a_moving_cable():
    torn = verify_cable(_hanging(torn=True))
    assert not torn["ok"]
    assert any("far end" in p for p in torn["problems"])
    moving = verify_cable(_hanging(speed=0.5))
    assert any("still moving" in p for p in moving["problems"])
