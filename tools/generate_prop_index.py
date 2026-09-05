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

"""Generate the prop index from a walk of the Isaac asset server.

Run inside Isaac Sim (via `run_control`), then commit the JSON it prints.

The index is static on purpose — an agent should not pay a network walk per
call, and curated keywords match "cardboard box" to `003_cracker_box.usd` in a
way a directory listing never will. But it is *generated*, never hand-written,
and that distinction is the whole reason this file exists rather than a literal
dict in the library.

The one hand-written catalogue entry we ever had said `kuka_iiwa` and pointed at
a KR210 asset with a motion config that does not exist. Spawning it silently
returned a different robot than the one asked for. A generated index cannot
invent an asset that is not there; a typed one can, and did.

Usage:
    python tools/generate_prop_index.py --emit   # prints the code to run in-sim
"""

from __future__ import annotations

import json
import sys

# Files that exist to support another asset rather than to be spawned. Matched
# on the whole name, so a real prop that merely contains the word survives.
_SUPPORT_FILES = {
    "instanceable_meshes.usd",
    "physics_material.usd",
}
_SUPPORT_SUFFIXES = (
    "_visual.usd",  # render-only twin of a physics asset
    "_collisions.usd",  # collision-only twin
    "_visual_collision.usd",
)

# Words the file and directory names never contain but people ask for. This is
# the only hand-maintained part, and it adds *search terms* to generated
# entries — it can never introduce an asset that the walk did not find.
_KEYWORDS = {
    "conveyors": ["conveyor", "belt", "conveyor belt", "transport", "line"],
    "klt_bin": ["bin", "box", "tote", "crate", "container", "klt", "logistics", "cardboard", "plastic"],
    "pallet": ["pallet", "skid", "warehouse"],
    "forklift": ["forklift", "truck", "warehouse", "vehicle"],
    "packingtable": ["table", "workbench", "bench", "packing", "surface"],
    "mounts": ["table", "stand", "mount", "bench", "surface"],
    "blocks": ["block", "cube", "brick", "box"],
    "shapes": ["shape", "primitive", "cube", "sphere", "cone", "cylinder"],
    "mugs": ["mug", "cup", "drink"],
    "food": ["food", "package", "box", "grocery"],
    "ycb": ["ycb", "benchmark", "grocery", "household", "manipulation", "cardboard", "carton"],
    "sektion_cabinet": ["cabinet", "drawer", "cupboard", "furniture"],
    "beaker": ["beaker", "glass", "lab", "laboratory"],
    "factory": ["bolt", "nut", "fastener", "assembly", "peg"],
    "rubiks_cube": ["cube", "rubik", "puzzle", "toy"],
    "dolly": ["dolly", "cart", "trolley", "wheeled"],
    "flip_stack": ["assembly", "part", "mechanical"],
}


WALK_SNIPPET = r"""
import json
import omni.client
from isaacsim.storage.native import get_assets_root_path

root = get_assets_root_path()
SKIP = {".thumbs", "Materials", "materials", "textures", "Textures", "Material Library"}

def ls(path):
    result, entries = omni.client.list(path)
    return [e.relative_path for e in entries] if str(result) == "Result.OK" else []

found = {}
def walk(rel, depth=0):
    if depth > 2:
        return
    for name in ls(root + rel):
        if name.startswith(".") or name in SKIP:
            continue
        if name.lower().endswith((".usd", ".usda")):
            found.setdefault(rel, []).append(name)
        elif "." not in name:
            walk(rel + "/" + name, depth + 1)

walk("/Isaac/Props")
print(json.dumps({"assets_root": root, "tree": found}, sort_keys=True))
"""


def _key_for(directory: str, filename: str) -> str:
    """A stable, typeable key. Directory-qualified only where it must be."""
    stem = filename.rsplit(".", 1)[0].lower().replace(" ", "_").replace("-", "_")
    leaf = directory.rstrip("/").rsplit("/", 1)[-1].lower().replace(" ", "_").replace("-", "_")

    # Do not repeat what the filename already says. `Pallet/pallet.usd` is
    # `pallet`, not `pallet_pallet`; `Conveyors/ConveyorBelt_A01.usd` is
    # `conveyorbelt_a01`; `KLT_Bin/small_KLT.usd` is `small_klt`;
    # `PackingTable/packing_table.usd` is `packing_table`. Compared with
    # separators removed, because the directory and the file rarely agree on
    # where the underscores go.
    flat_stem, flat_leaf = stem.replace("_", ""), leaf.replace("_", "")
    if flat_stem == flat_leaf or flat_leaf.rstrip("s") in flat_stem:
        return stem
    if any(len(tok) > 2 and tok in flat_stem for tok in leaf.split("_")):
        return stem
    # YCB is organised by variant, not by object; the object name is the key and
    # the variant is recorded as a property.
    if leaf in ("axis_aligned", "axis_aligned_physics"):
        return stem
    return f"{leaf}_{stem}"


# Most capable first. Used to settle key collisions.
_RANK = {"dynamic": 3, "static": 2, "visual": 1, "unknown": 0}


def _classify(scan: dict) -> str:
    """What an agent can actually do with this asset.

    Over half of Isaac's prop library turns out to be visual-only -- 72 of 169
    assets declare no rigid body and 90 declare no collider. Spawned into a
    manipulation task, those fall through the floor or ignore the gripper, and
    nothing anywhere says so. Recording it is the difference between choosing a
    prop and discovering the problem after the grasp fails.
    """
    if scan.get("error"):
        return "unknown"
    if scan.get("rigid", 0) > 0:
        return "dynamic"  # has mass, falls, can be picked up
    if scan.get("collision", 0) > 0:
        return "static"  # collides, cannot be moved by contact
    return "visual"  # geometry only: no collider, no rigid body


def _keywords_for(directory: str, key: str) -> list[str]:
    words = set()
    for part in directory.lower().replace("/isaac/props/", "").split("/"):
        for token in part.replace("-", "_").split("_"):
            if len(token) > 2 and not token.isdigit():
                words.add(token)
                words.update(_KEYWORDS.get(token, []))
        words.update(_KEYWORDS.get(part.replace(" ", "_"), []))
    for token in key.split("_"):
        if len(token) > 2 and not token.isdigit():
            words.add(token)
    return sorted(words)


def build_index(walked: dict, scanned: dict | None = None) -> dict:
    root = walked["assets_root"]
    tree = walked["tree"]
    scanned = scanned or {}

    props: dict[str, dict] = {}
    skipped = 0
    for directory in sorted(tree):
        for filename in sorted(tree[directory]):
            if filename in _SUPPORT_FILES or filename.endswith(_SUPPORT_SUFFIXES):
                skipped += 1
                continue
            path = f"{directory}/{filename}"
            key = _key_for(directory, filename)
            scan = scanned.get(path, {})
            physics = _classify(scan)

            entry = {
                "key": key,
                "path": path,
                "category": directory.replace("/Isaac/Props/", "").split("/")[0].lower(),
                "keywords": _keywords_for(directory, key),
                "physics": physics,
            }
            if scan.get("extent"):
                entry["extent"] = scan["extent"]

            # A silent overwrite here would shadow one asset with another, which
            # is the failure this generated index exists to prevent. Both keep a
            # key — and the *dynamic* one keeps the plain name, because YCB ships
            # every object twice and the visual-only copy cannot be picked up.
            # `spawn_prop("003_cracker_box")` should hand back the one that works.
            if key in props:
                incumbent = props[key]
                takes_it = _RANK[physics] > _RANK[incumbent["physics"]]
                winner, loser = (entry, incumbent) if takes_it else (incumbent, entry)
                variant = loser["path"].rstrip("/").rsplit("/", 2)[-2].lower()
                loser = dict(loser, key=f"{variant}_{key}")
                props[key] = dict(winner, key=key)
                props[loser["key"]] = loser
                continue

            props[key] = entry

    by_physics: dict[str, int] = {}
    for entry in props.values():
        by_physics[entry["physics"]] = by_physics.get(entry["physics"], 0) + 1

    return {
        # Stamped so drift is detectable rather than silent: a mismatch against
        # the live server means the index describes a different Isaac release.
        "assets_root": root,
        "generated_from": "/Isaac/Props",
        "count": len(props),
        "support_files_skipped": skipped,
        "by_physics": by_physics,
        "props": props,
    }


def main() -> int:
    if "--emit" in sys.argv:
        print(WALK_SNIPPET)
        return 0
    raw = sys.stdin.read()
    walked = json.loads(raw[raw.index("{") :])
    scanned = None
    for arg in sys.argv[1:]:
        if arg.startswith("--scan="):
            with open(arg.split("=", 1)[1], encoding="utf-8") as handle:
                scanned = json.load(handle)
    print(json.dumps(build_index(walked, scanned), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
