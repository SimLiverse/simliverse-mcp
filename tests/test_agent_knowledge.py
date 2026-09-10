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


def test_the_guide_covers_the_incline_belt() -> None:
    """A pitched belt drives its surface up-slope in local space; the friction
    rule and the local-space caveat are both things the agent cannot guess."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("pitch=", "tan(pitch)", "local space", "ride UP"):
        assert token in text, "the guide never mentions %r" % token


def test_the_guide_covers_the_curved_belt() -> None:
    """A curve is a fan of tangent chord slabs; the agent needs to know it is
    generic over any bend and dresses with the real A01 prop."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("build_curve", "chord slab", "spacing_deg", "conveyorbelt_a01"):
        assert token in text, "the guide never mentions %r" % token


def test_the_curve_api_the_guide_advertises_exists() -> None:
    import simliverse_sim as sim
    from simliverse_sim.conveyor import CurvedConveyor

    assert hasattr(sim, "CurvedConveyor")
    for name in ("build_curve", "box_at_gate", "load", "dress", "start", "halt"):
        assert hasattr(CurvedConveyor, name), "CurvedConveyor has no %s" % name


def test_the_guide_covers_the_mobile_robot() -> None:
    """drive_to is not a planner and the arrival tolerance must be shared with
    the verifier - two things the agent cannot guess and both cost a run."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("drive_to", "verify_navigation", "not a planner", "Carter", "One tolerance"):
        assert token in text, "the guide never mentions %r" % token


def test_the_mobile_robot_api_the_guide_advertises_exists() -> None:
    from simliverse_sim import verify_navigation  # noqa: F401
    from simliverse_sim.robots.mobile import WheeledRobot

    for name in ("drive_to", "plan_path", "drive", "stop"):
        assert hasattr(WheeledRobot, name), "WheeledRobot has no %s" % name


def test_the_guide_covers_the_other_morphologies() -> None:
    """Drones fly; quadrupeds and humanoids need a policy to walk - both are
    things the agent must know before it spawns one and asks it to move."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("quadcopter", "fly_to", "trained policy", "rigid body, not an articulation"):
        assert token in text, "the guide never mentions %r" % token


def test_the_aerial_api_the_guide_advertises_exists() -> None:
    from simliverse_sim.robots.aerial import AerialRobot

    for name in ("fly_to", "hover", "apply_thrust", "altitude"):
        assert hasattr(AerialRobot, name), "AerialRobot has no %s" % name


def test_the_guide_points_at_both_scenario_harnesses() -> None:
    """The agent must know both harnesses exist: one crosses palletising cells,
    the other crosses the whole fleet across floors."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("generated_scenarios", "fleet_scenarios", "run_fleet"):
        assert token in text, "the guide never mentions %r" % token


def test_the_fleet_harness_exists_and_dispatches_to_the_verified_demos() -> None:
    import demo.fleet_scenarios as fs

    for name in ("mobile_scenarios", "drone_scenarios", "fleet", "run_fleet", "report"):
        assert hasattr(fs, name), "fleet_scenarios has no %s" % name


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


def test_the_docstring_tells_the_agent_to_ask_before_committing() -> None:
    """A move with no solution returns without moving; the cup opened anyway."""
    source = _control_source()
    for token in ("can_reach", "MotionResult", "verify_pallet", "go_home", "CEILING", "DRIVE_GAINS", "wrist branch"):
        assert token in source, "run_control never mentions %r" % token


def test_the_guide_carries_the_reach_ceiling_and_the_arm_table() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("reach_ceiling", "0.66", "0.82", "crx10ia_l", "1e6", "motion_config", "arm_footprint", "touching"):
        assert token in text, "the guide never mentions %r" % token


def test_the_reach_api_the_agent_is_told_about_exists() -> None:
    import demo.ur10_palletizing as demo
    import simliverse_sim as sim
    from simliverse_sim.robots.manipulator import Manipulator

    assert hasattr(Manipulator, "can_reach")
    assert hasattr(Manipulator, "reach_ceiling")
    for name in ("go_home", "slot_reach", "drive_gains", "DRIVE_GAINS", "layout_for"):
        assert hasattr(demo, name), "the docstring advertises %s" % name
    for name in ("Cable", "verify_cable"):
        assert hasattr(sim, name), "the docstring advertises %s" % name


def test_the_agent_is_told_there_is_no_cable_asset() -> None:
    """`list_props("cable")` is empty; the guide must say what to do instead."""
    text = GUIDE.read_text(encoding="utf-8")
    for token in ("Cable.build", "slack", "deformabletube_tube", "layout_for"):
        assert token in text, "the guide never mentions %r" % token
    assert "Cable.build" in _control_source()


def test_the_agent_is_told_how_to_bolt_a_finger_gripper_on() -> None:
    """A bare arm has no jaw; the guide must say which grippers ship and
    that a Hand-E ships without drives."""
    from simliverse_sim.robots.manipulator import Manipulator

    text = GUIDE.read_text(encoding="utf-8")
    for token in ("attach_gripper", "2f_85", "egk_25", "hand_e", "repair_drives", "simliverse:gripper"):
        assert token in text, "the guide never mentions %r" % token
    assert "attach_gripper" in _control_source()
    assert hasattr(Manipulator, "attach_gripper")
    assert hasattr(Manipulator, "rebind_gripper")


def test_the_guide_covers_palletising_with_a_finger_jaw() -> None:
    """A jaw grips sides not faces, refuses a box wider than it opens, and must
    centre its pads on the box - none guessable, and the honest grasp gap is
    said out loud rather than implied by a demo that only shows what works."""
    import demo.ur10_palletizing as demo

    text = GUIDE.read_text(encoding="utf-8")
    for token in ('build(gripper="2f_85")', "JAW_GEOMETRY", "pad_center", "KNOWN GAP", "box_h"):
        assert token in text, "the guide never mentions %r" % token
    # The working native-parallel-jaw path, and why it works.
    for token in ('build(robot="franka", gripper="native")', "_firm_grip", "reach_ceiling"):
        assert token in text, "the guide never mentions %r" % token
    # The API the note advertises exists.
    for name in ("_JawEE", "_SuctionEE", "_box_w", "JAW_GEOMETRY", "_firm_grip"):
        assert hasattr(demo, name), "the guide advertises %s" % name
    assert "gripper" in inspect.signature(demo.build).parameters
    assert "box_h" in inspect.signature(demo.build).parameters
    # A native gripper's IK frame is the grasp point, so its drop is 0.
    assert demo.JAW_GEOMETRY["native"]["drop"] == 0.0


def test_the_agent_is_told_to_edit_the_drawing_not_just_the_scene() -> None:
    """When the user asks in words to change what they drew, the sketch must
    change - it is the source of truth. The tool exists, is registered, and the
    docstring says to use it and hand the block back."""
    from isaac_mcp.tools import sketch as tool_module
    from simliverse_sim import sketch as S

    source = _control_source()
    assert "edit_sketch" in source
    assert "CHANGE THE DRAWING" in source
    assert hasattr(tool_module, "register_tools")
    for name in ("edit_sketch", "render_sketch", "route_from_sketch"):
        assert hasattr(S, name), "sketch has no %s" % name
    # register_all_tools wires the sketch module in.
    init_src = (ROOT / "isaac_mcp" / "tools" / "__init__.py").read_text(encoding="utf-8")
    assert "sketch" in init_src


def test_the_guide_states_the_working_envelope_rather_than_implying_it() -> None:
    """A demo that only says what works has not said what does not."""
    text = GUIDE.read_text(encoding="utf-8")
    assert "envelope" in text.lower()
    assert "58.69" in text, "the guide should carry the measured baseline"
