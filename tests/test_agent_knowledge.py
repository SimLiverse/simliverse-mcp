"""What the agent is told, and whether it is still true.

The agent's whole picture of this library is the `run_control` docstring plus
`docs/control_library.md`. That is not documentation in the usual sense - it
is the only thing standing between the agent and authoring a factory out of
grey cubes, which is what it did until someone looked at a render.

The docstring already pointed at `docs/control_library.md`, and that file did
not exist. Nothing caught it, because nothing was checking.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "control_library.md"


def _control_source() -> str:
    return (ROOT / "isaac_mcp" / "tools" / "control.py").read_text(encoding="utf-8")


def test_the_reference_the_agent_is_sent_to_exists() -> None:
    """A docstring that points at a missing file teaches nothing."""
    source = _control_source()
    assert "docs/control_library.md" in source
    assert GUIDE.exists(), "run_control tells the agent to read docs/control_library.md"


def test_the_guide_names_the_assets_a_cell_is_built_from() -> None:
    """The agent authored a conveyor out of a cuboid with 47 in the library."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("conveyorbelt_a05", "/Isaac/People/Characters", "list_props", "find_prop"):
        assert token in text, "the guide never mentions %s" % token


@pytest.mark.parametrize(
    "number,what",
    [
        ("0.767", "a real conveyor's carrying height"),
        ("1.21", "how long a pallet is"),
        ("0.1425", "the pallet deck"),
    ],
)
def test_the_guide_carries_the_measured_numbers(number, what) -> None:
    """These cost a session each. None are guessable from the API."""
    assert number in GUIDE.read_text(encoding="utf-8"), "the guide does not record %s (%s)" % (number, what)


def test_the_docstring_tells_the_agent_to_search_before_authoring() -> None:
    """The failure this exists to prevent, in the agent's own instructions."""
    source = _control_source()
    assert "list_props" in source
    assert "REAL ASSETS" in source.upper()


def test_the_docstring_carries_the_placement_traps() -> None:
    """Silent failures need saying out loud; none of these raise."""
    source = _control_source()
    for trap in ("CENTRE", "feet", "0.767", "clear_world", "sleeping"):
        assert trap in source, "run_control never warns the agent about %r" % trap


def test_the_docstring_covers_the_sketch_gate_derivation() -> None:
    """The gate used to open south no matter what was drawn."""
    source = _control_source()
    assert "operator" in source.lower()
    assert "gate" in source.lower()


def test_the_guide_covers_sketch_building_and_dressing_orientation() -> None:
    """Two gaps closed the same day: fence_from_sketch had no section at
    all, and dressing silently faced the wrong way and never tiled."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("fence_from_sketch", "zones_from_sketch", "rotates", "tiles"):
        assert token in text, "the guide never mentions %r" % token


def test_the_agent_is_told_to_look_at_more_than_one_view() -> None:
    source = _control_source()
    assert "vision.look" in source
    assert "four" in source.lower()


def test_every_api_the_docstring_advertises_actually_exists() -> None:
    """A docstring is a promise. This is the part that goes stale first."""
    import simliverse_sim as sim

    for name in (
        "Scene",
        "Robot",
        "Conveyor",
        "SafetyFence",
        "list_props",
        "find_prop",
        "spawn_prop",
        "spawn_pedestal",
        "spawn_operator",
        "vision",
    ):
        assert hasattr(sim, name), "run_control advertises %s and the package has no such name" % name


def test_the_advertised_methods_exist_on_the_objects() -> None:
    from simliverse_sim import Conveyor, SafetyFence
    from simliverse_sim.scene import Scene

    assert hasattr(Scene, "clear_world")
    assert hasattr(Conveyor, "dress")
    assert hasattr(Conveyor, "wake_load")
    assert hasattr(SafetyFence, "build")
    assert hasattr(SafetyFence, "fits")


def test_vision_look_takes_the_scale_the_docstring_promises() -> None:
    from simliverse_sim import vision

    assert "scale" in inspect.signature(vision.look).parameters


def test_the_guide_states_the_working_envelope_rather_than_implying_it() -> None:
    """A demo that only says what works has not said what does not."""
    text = GUIDE.read_text(encoding="utf-8")
    assert "envelope" in text.lower()
    assert "58.69" in text, "the guide should carry the measured baseline"
