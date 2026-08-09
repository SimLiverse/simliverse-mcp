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

"""The prop catalogue.

Pure lookups over the committed index, so this runs without Isaac Sim. The
index itself is generated from a walk of the asset server and every path and
physics classification below was measured there, not asserted from a datasheet.

The question these exist to answer is the one that started it: "put a box on a
conveyor belt". Before this, an agent had no way to find either and would build
both out of stretched cubes.
"""

import json

import pytest

from simliverse_sim import props


def test_the_things_someone_would_actually_ask_for_are_findable() -> None:
    for query in ("conveyor belt", "pallet", "forklift", "bin", "mug", "cabinet"):
        assert props.list_props(query), f"nothing matches {query!r}"


def test_a_box_finds_a_real_box_not_a_primitive() -> None:
    """`small_KLT.usd` is the standard logistics bin. It shares no word with 'box'."""
    keys = [p["key"] for p in props.list_props("box")]
    assert "small_klt" in keys


@pytest.mark.parametrize(
    "key, path",
    [
        ("conveyorbelt_a01", "/Isaac/Props/Conveyors/ConveyorBelt_A01.usd"),
        ("small_klt", "/Isaac/Props/KLT_Bin/small_KLT.usd"),
        ("pallet", "/Isaac/Props/Pallet/pallet.usd"),
        ("forklift", "/Isaac/Props/Forklift/forklift.usd"),
        ("packing_table", "/Isaac/Props/PackingTable/packing_table.usd"),
    ],
)
def test_keys_resolve_to_the_asset_they_name(key: str, path: str) -> None:
    assert props.find_prop(key)["path"] == path


def test_the_plain_key_is_the_variant_that_works() -> None:
    """YCB ships every object twice; only one of them can be picked up.

    `Axis_Aligned` is render-only and `Axis_Aligned_Physics` has the rigid body.
    An agent asking for a cracker box means the one a gripper can hold.
    """
    cracker = props.find_prop("003_cracker_box")
    assert cracker["physics"] == "dynamic"
    assert "Axis_Aligned_Physics" in cracker["path"]

    # The visual twin stays reachable, just not under the plain name.
    assert props.find_prop("axis_aligned_003_cracker_box")["physics"] == "visual"


def test_physics_filter_excludes_what_cannot_be_manipulated() -> None:
    """The majority of this library is visual-only; a grasp task needs the rest."""
    dynamic = props.list_props(physics="dynamic")
    assert dynamic and all(p["physics"] == "dynamic" for p in dynamic)
    assert len(dynamic) < props._index()["count"], "if everything passed, the filter is a no-op"


def test_a_static_prop_is_labelled_not_silently_unusable() -> None:
    """A pallet collides but has no rigid body: you can stack on it, not lift it."""
    assert props.find_prop("pallet")["physics"] == "static"


def test_capable_variants_sort_first() -> None:
    ranked = [p["physics"] for p in props.list_props("box")]
    assert ranked == sorted(ranked, key=lambda k: -props.PHYSICS_KINDS.index(k))


def test_a_missing_prop_raises_rather_than_substituting() -> None:
    """Handing back a near-miss is how the wrong scene gets built quietly."""
    with pytest.raises(props.PropNotFound) as excinfo:
        props.find_prop("teleporter")
    assert "teleporter" in str(excinfo.value)


def test_every_entry_is_complete_and_unique() -> None:
    index = props._index()
    entries = list(index["props"].values())
    assert len(entries) == index["count"]
    assert len({e["path"] for e in entries}) == len(entries), "two keys share an asset"
    for entry in entries:
        assert entry["path"].startswith("/Isaac/Props/")
        assert entry["physics"] in props.PHYSICS_KINDS


def test_no_support_files_are_offered_as_props() -> None:
    """`instanceable_meshes.usd` exists to back another asset, not to be spawned."""
    paths = [e["path"] for e in props._index()["props"].values()]
    assert not any(p.endswith("instanceable_meshes.usd") for p in paths)
    assert not any(p.endswith("physics_material.usd") for p in paths)


def test_extents_are_recorded_so_reach_and_grip_can_be_checked() -> None:
    """A cracker box is 16 cm; whether a gripper closes on it is arithmetic."""
    cracker = props.find_prop("003_cracker_box")
    assert len(cracker["extent"]) == 3
    assert all(v > 0 for v in cracker["extent"])


def test_the_index_declares_what_it_was_generated_against() -> None:
    """Without this, a release bump moves every path and nothing notices."""
    assert props._index()["assets_root"].endswith("/Isaac/6.0")


def test_index_is_valid_json_on_disk() -> None:
    with props._INDEX_PATH.open(encoding="utf-8") as handle:
        assert json.load(handle)["count"] > 100


def test_an_unknown_adjective_does_not_hide_the_asset() -> None:
    """"cardboard box" used to match nothing at all.

    `list_props` required *every* word, and "cardboard" appears in no keyword
    list, so the query returned empty — from which an agent concludes no box
    exists and builds one out of primitives. Scoring instead of requiring means
    one unrecognised adjective cannot hide what the rest of the query names.
    """
    keys = [p["key"] for p in props.list_props("cardboard box")]
    assert keys, "an unknown adjective emptied the result"
    assert "003_cracker_box" in keys[:3]


def test_a_query_matching_nothing_still_returns_nothing() -> None:
    """Scoring must not degrade into matching everything."""
    assert props.list_props("teleporter") == []


def test_more_query_words_matched_ranks_higher() -> None:
    hits = props.list_props("tomato soup")
    assert hits[0]["key"] == "005_tomato_soup_can"


def test_a_partial_match_refuses_rather_than_substituting_quietly() -> None:
    """Scoring alone traded a silent absence for a silent approximation.

    There is no wooden crate in this library. There is a plastic KLT bin, and
    it scores well on "crate" — so it came back first, unlabelled, and would be
    spawned and then reported as the wooden crate that was asked for. A
    substitution nobody decided on, in a scene the user believes contains
    something else.
    """
    with pytest.raises(props.PropNotFound) as excinfo:
        props.find_prop("wooden crate")
    message = str(excinfo.value)
    assert "small_klt" in message, "the near miss should still be named"
    assert "'wooden'" in message, "and what it failed to account for"


def test_a_partial_match_is_available_once_the_choice_is_made() -> None:
    entry = props.find_prop("wooden crate", allow_partial=True)
    assert entry["key"] == "small_klt"
    assert entry["match"] == "partial"
    assert entry["unmatched"] == ["wooden"]


def test_every_query_word_matching_is_exact() -> None:
    for query in ("conveyor belt", "cardboard box", "banana"):
        entry = props.find_prop(query)
        assert entry["match"] == "exact", query
        assert entry["unmatched"] == []


def test_listing_labels_how_well_each_result_matched() -> None:
    """The caller cannot weigh a result it cannot tell apart from an exact one."""
    hits = props.list_props("glass jar")
    assert hits and all("match" in h and "unmatched" in h for h in hits)
    assert hits[0]["match"] == "partial"
