"""Which RMPflow config a discovered asset gets.

The pairs below are not hypothetical. `UR30 -> UR3` and `FrankaFR3 -> Franka`
were both produced by a substring rule that looked reasonable, and both would
have handed an arm the kinematics of a different arm — a failure that shows up
as an arm which misses everything it reaches for, with no error anywhere.
"""

import re

import pytest

from simliverse_sim.robots.library import _infer_motion_config, _tokens

# A representative slice of what Lula actually ships, including the pairs that
# differ only by a trailing digit or letter.
SUPPORTED = [
    "Franka", "FR3", "UR3", "UR3e", "UR5", "UR5e", "UR10", "UR10e", "UR16e",
    "Kuka_KR210", "Kinova_Gen3", "RS007N", "RS007L", "Techman_TM12",
    "Cobotta_Pro_900", "Cobotta_Pro_1300", "Rizon4",
]


@pytest.mark.parametrize(
    ("vendor", "model", "expected"),
    [
        # The asset that started this: the seeded entry claimed `Kuka_iiwa7`,
        # which Isaac does not ship, for a robot that is a KR210.
        ("Kuka", "KR210_L150", "Kuka_KR210"),
        # Trailing-digit collisions. A UR30 is not a UR3.
        ("UniversalRobots", "ur3", "UR3"),
        ("UniversalRobots", "ur30", None),
        ("UniversalRobots", "ur10", "UR10"),
        ("UniversalRobots", "ur10e", "UR10e"),
        ("UniversalRobots", "ur16e", "UR16e"),
        # Brand token vs model token: both match, the model token is specific.
        ("FrankaRobotics", "FrankaPanda", "Franka"),
        ("FrankaRobotics", "FrankaFR3", "FR3"),
        # Vendor supplies half the config name.
        ("Kinova", "Gen3", "Kinova_Gen3"),
        ("Kawasaki", "RS007N", "RS007N"),
        ("Techman", "TM12", "Techman_TM12"),
        ("Denso", "cobotta_pro_900", "Cobotta_Pro_900"),
        # Nothing plausible: no config is better than a wrong one.
        ("BostonDynamics", "spot", None),
        ("Unitree", "H1", None),
    ],
)
def test_inference(vendor: str, model: str, expected: str | None) -> None:
    assert _infer_motion_config(vendor, model, SUPPORTED) == expected


def test_ambiguity_yields_nothing() -> None:
    """Two configs that fit equally well must produce neither.

    Both names are model-specific, one token each, the same length — nothing
    left to prefer one by. Guessing here is how an arm silently gets the wrong
    kinematics, so the answer is no answer.
    """
    assert _infer_motion_config("Acme", "AlphaGamma", ["Alpha", "Gamma"]) is None
    # But a brand token losing to a model token is not a tie.
    assert _infer_motion_config("Acme", "AcmeBot", ["Acme", "Bot"]) == "Bot"


def test_tokens_split_on_letter_digit_boundaries() -> None:
    assert _tokens("KR210_L150") == ["kr", "210", "l", "150"]
    assert _tokens("FrankaFR3") == ["franka", "fr", "3"]
    assert _tokens("ur30") != _tokens("ur3")


def test_no_config_is_invented() -> None:
    """Every returned name must be one Lula offered."""
    for model in ("KR210_L150", "ur10e", "FrankaFR3", "Gen3", "spot"):
        result = _infer_motion_config("Vendor", model, SUPPORTED)
        assert result is None or result in SUPPORTED


def test_camel_regex_compiles() -> None:
    assert re.compile(_tokens.__globals__["_CAMEL"].pattern)
