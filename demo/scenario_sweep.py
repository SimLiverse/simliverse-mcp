"""Run the palletising cell against geometries it was never measured on.

Every number in `demo.ur10_palletizing` came from one cell: a 15 cm carton,
a 0.45 m deck, a pallet 0.75 m away, a belt at 0.2 m/s. A demo that only
works there is fitted to itself, and the only way to tell the difference is
to build it somewhere else and watch.

This is deliberately not a test. It needs a GPU, a running Kit session and
several minutes per scenario, and its output is a table to read rather than
a pass/fail - "the far pallet places 2/2 but takes 40% longer" is the kind
of result that matters here and that an assertion would throw away. The
arithmetic these scenarios feed is unit-tested in `tests/test_cell_scenarios`.

Run it inside a live session:

    from demo.scenario_sweep import sweep
    print(report(sweep()))
"""

from __future__ import annotations

import time
import traceback
from typing import Any

#: One knob each, so a failure names its own cause. A scenario that changes
#: three things at once tells you only that something broke.
SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("small carton", {"box": 0.10}),
    ("large carton", {"box": 0.22}),
    # Fails without side guides, and guides currently cost placement accuracy.
    # Kept in the sweep as a standing red result rather than dropped: the cell's
    # working envelope is about 1 kg, and a sweep that only lists what passes
    # is a sweep that stops being worth running.
    ("heavy carton", {"box_mass": 3.0}),
    ("guided belt", {"guides": True}),
    ("light carton", {"box_mass": 0.25}),
    ("slow belt", {"speed": 0.10}),
    ("fast belt", {"speed": 0.40}),
    # Not 0.60: a pallet is 1.21 m long and placed by its centre, so 0.60 puts
    # its near edge at -0.005 and the arm's base inside it. That is an
    # impossible cell, not a failing one, and a sweep that cannot tell those
    # apart reports a geometry mistake as a code defect.
    ("near pallet", {"pallet_y": 0.68}),
    ("far pallet", {"pallet_y": 0.95}),
    ("low deck", {"deck": 0.35}),
    ("high deck", {"deck": 0.55}),
    ("2x3 pattern", {"rows": 2, "cols": 3}),
]

#: The same cell on other arms, each laid out to its own reach by
#: `layout_for`. Built lazily so importing this module needs no reach table.
ROBOTS = ["ur10", "ur10e", "ur5e", "ur16e", "crx10ia_l", "kuka_kr210"]


def robot_scenarios(robots: list[str] | None = None, *, box: float | None = None) -> list[tuple[str, dict[str, Any]]]:
    from demo import ur10_palletizing as cell_mod

    out = []
    for robot in robots or ROBOTS:
        spec = cell_mod.layout_for(robot, box=box or cell_mod.BOX)
        out.append((robot, spec))
    return out


#: The axes a generated cell samples. Each is a real capability the harness is
#: meant to handle; crossing them makes combinations no single scenario tested,
#: which is the point - a cell that works only on the grid it was tuned for is
#: not general. `crx10ia_l` is left out by default (its suction cup is a known
#: gap, see demo.ur10_palletizing.KNOWN_GAPS); pass it in `robots` to include it.
_AXES = {
    "robot": [r for r in ROBOTS if r != "crx10ia_l"],
    "box": [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22],
    "box_mass": [0.3, 0.5, 1.0, 1.5],
    "rows": [1, 2, 3],
    "cols": [1, 2, 3],
    "layers": [1, 2],
    "speed": [0.10, 0.15, 0.20, 0.30, 0.40],
    "deck": [0.35, 0.45, 0.55],
    "dressing": [None, "conveyorbelt_a05"],
    "guides": [False, True],
}


def generated_scenarios(
    count: int = 20, *, seed: int = 0, robots: list[str] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Invent `count` distinct cells by crossing the capability axes, per arm.

    Deterministic under `seed`, so a failure is reproducible and a fix can be
    checked against the same set. Each cell is a robot's reach-aware layout with
    a sampled carton, pattern, stack height, belt speed, deck height, dressing
    and guarding folded in - combinations the fixed scenarios never enumerate,
    which is what tests generalisation rather than memorisation. A cell whose
    numbers are impossible (a small arm reaching a tall stack) is still emitted;
    the sweep reports it as an impossible cell, not a defect.
    """
    import random as _random

    from demo import ur10_palletizing as cell_mod

    rng = _random.Random(seed)
    pool = robots or _AXES["robot"]
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple] = set()
    tries = 0
    while len(out) < count and tries < count * 40:
        tries += 1
        robot = rng.choice(pool)
        box = rng.choice(_AXES["box"])
        pick = {
            k: rng.choice(_AXES[k])
            for k in ("box_mass", "rows", "cols", "layers", "speed", "deck", "dressing", "guides")
        }
        key = (
            robot,
            box,
            *(pick[k] for k in ("box_mass", "rows", "cols", "layers", "speed", "deck", "dressing", "guides")),
        )
        if key in seen:
            continue
        seen.add(key)
        spec = cell_mod.layout_for(robot, box=box)
        spec.update(pick)
        # A small arm cannot reach a full pallet's far column at two layers; the
        # layout already chose a tote for it, and a 3x3 tote is over-packed - cap
        # the pattern to what the deck holds so the cell is testable, not absurd.
        if spec.get("pallet") in ("tote", "half"):
            spec["rows"] = min(spec["rows"], 2)
            spec["cols"] = min(spec["cols"], 2)
        name = "%s b%.2f m%.1f %dx%dx%d v%.2f d%.2f%s%s" % (
            robot,
            box,
            pick["box_mass"],
            pick["rows"],
            pick["cols"],
            pick["layers"],
            pick["speed"],
            pick["deck"],
            " dressed" if pick["dressing"] else "",
            " guided" if pick["guides"] else "",
        )
        out.append((name, spec))
    return out


def sweep(
    scenarios: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    cartons: int = 2,
    scene=None,
) -> list[dict[str, Any]]:
    """Build and run each scenario, returning one row per cell.

    Each cell is built from scratch. `build` clears the stage first, which is
    load-bearing here rather than tidy: without it scenario N runs inside the
    wreckage of scenario N-1, and a leftover fixture holding the carton queue
    reads as a conveyor fault in whichever scenario happens to be next.
    """
    from demo import ur10_palletizing as cell_mod
    from simliverse_sim import Scene

    scene = scene or Scene.get()
    rows: list[dict[str, Any]] = []

    for name, spec in scenarios or SCENARIOS:
        started = time.time()
        row: dict[str, Any] = {"scenario": name, "spec": dict(spec)}
        try:
            # One more carton than slots to fill, so "no carton arrived" means
            # the belt failed to deliver rather than that the queue ran dry.
            cell = cell_mod.build(scene, boxes=cartons + 1, **spec)
            if cell.get("fouled"):
                row.update(
                    {
                        "built": False,
                        "error": "impossible cell: the pallet encloses %s"
                        % (", ".join(f["robot"] for f in cell["fouled"])),
                    }
                )
                row["wall_s"] = round(time.time() - started, 1)
                rows.append(row)
                continue
            report_ = cell_mod.palletise(cell, count=cartons)
            stack = report_.get("stack") or {}
            row.update(
                {
                    "built": True,
                    "placed": int(report_.get("placed", 0)),
                    "of": int(report_.get("of", cartons)),
                    "complete": bool(report_.get("complete")),
                    "intact": int(stack.get("placed", 0)),
                    "unreachable": list((cell.get("reach") or {}).get("unreachable", [])),
                    "clearance": cell.get("clearance"),
                    "known_gap": cell_mod.KNOWN_GAPS.get(spec.get("robot") or ""),
                    "s_per_carton": report_.get("seconds_per_carton"),
                    "per_hour": report_.get("cartons_per_hour"),
                    "errors_mm": [
                        None if c.get("error") is None else round(float(c["error"]) * 1000.0, 1)
                        for c in report_.get("cycles", [])
                    ],
                    "why": [c.get("reason") for c in report_.get("cycles", []) if c.get("reason")],
                }
            )
        except Exception as exc:  # noqa: BLE001 - a broken cell is a result
            row.update(
                {
                    "built": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "where": traceback.format_exc().strip().splitlines()[-3:],
                }
            )
        row["wall_s"] = round(time.time() - started, 1)
        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]]) -> str:
    """A table, with the baseline first so the rest can be read against it."""
    lines = ["%-14s %-7s %-7s %-22s %-11s %s" % ("scenario", "placed", "intact", "errors mm", "s/carton", "note")]
    lines.append("-" * 86)
    for row in rows:
        if not row.get("built"):
            lines.append("%-14s %-7s %-7s %-22s %-11s %s" % (row["scenario"], "-", "-", "-", "-", row["error"]))
            continue
        note = "; ".join(row["why"]) or "ok"
        if row.get("unreachable"):
            note += "; unreachable slots %s" % row["unreachable"]
        if row.get("known_gap"):
            note = "KNOWN GAP: " + row["known_gap"]
        lines.append(
            "%-14s %-7s %-7s %-22s %-11s %s"
            % (
                row["scenario"],
                "%d/%d" % (row["placed"], row["of"]),
                "%d/%d" % (row.get("intact", 0), row["of"]),
                str(row["errors_mm"]),
                row["s_per_carton"] if row["s_per_carton"] is not None else "-",
                note,
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sweep()))
