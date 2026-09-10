"""Dressing knows which conveyor props are straight, and how long they are.

`dress()` tiled everything as 2.0 m sections carrying at 0.767 m - A05's
numbers - whatever prop it was handed. Handed the 4 m A07 it placed two
sections through each other; handed the 90 degree A01 it laid a curve out
in a line.
"""

from __future__ import annotations

import pytest

from simliverse_sim.conveyor import (
    DRESSING_DECK,
    DRESSING_SECTION_LENGTH,
    NOT_STRAIGHT,
    SECTIONS,
    ConveyorError,
    section_spec,
)


def test_a05_is_the_measured_default():
    spec = section_spec("conveyorbelt_a05")
    assert spec == {"length": DRESSING_SECTION_LENGTH, "deck": DRESSING_DECK}


@pytest.mark.parametrize(
    "prop,length,deck",
    [("conveyorbelt_a07", 4.0, 0.769), ("conveyorbelt_a08", 2.72, 0.769), ("CONVEYORBELT_A09", 4.0, 1.781)],
)
def test_other_straights_carry_their_own_length_and_tier(prop, length, deck):
    spec = section_spec(prop)
    assert spec["length"] == length
    assert spec["deck"] == deck


@pytest.mark.parametrize("prop", ["conveyorbelt_a01", "conveyorbelt_a37", "conveyorbelt_a22"])
def test_curves_ramps_and_branches_are_refused_as_straight_dressing(prop):
    with pytest.raises(ConveyorError) as err:
        section_spec(prop)
    assert NOT_STRAIGHT[prop] in str(err.value)
    assert "conveyorbelt_a05" in str(err.value)


def test_an_unknown_key_falls_back_to_a05_rather_than_failing():
    assert section_spec("conveyorbelt_a99") == section_spec("conveyorbelt_a05")


def test_the_tables_do_not_overlap():
    assert not set(SECTIONS) & set(NOT_STRAIGHT)
