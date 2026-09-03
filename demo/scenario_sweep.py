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
    ("heavy carton", {"box_mass": 3.0}),
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

    for name, spec in (scenarios or SCENARIOS):
        started = time.time()
        row: dict[str, Any] = {"scenario": name, "spec": dict(spec)}
        try:
            # One more carton than slots to fill, so "no carton arrived" means
            # the belt failed to deliver rather than that the queue ran dry.
            cell = cell_mod.build(scene, boxes=cartons + 1, **spec)
            if cell.get("fouled"):
                row.update({
                    "built": False,
                    "error": "impossible cell: the pallet encloses %s" % (
                        ", ".join(f["robot"] for f in cell["fouled"])),
                })
                row["wall_s"] = round(time.time() - started, 1)
                rows.append(row)
                continue
            report_ = cell_mod.palletise(cell, count=cartons)
            row.update({
                "built": True,
                "placed": int(report_.get("placed", 0)),
                "of": int(report_.get("of", cartons)),
                "complete": bool(report_.get("complete")),
                "s_per_carton": report_.get("seconds_per_carton"),
                "per_hour": report_.get("cartons_per_hour"),
                "errors_mm": [
                    None if c.get("error") is None
                    else round(float(c["error"]) * 1000.0, 1)
                    for c in report_.get("cycles", [])
                ],
                "why": [c.get("reason") for c in report_.get("cycles", [])
                        if c.get("reason")],
            })
        except Exception as exc:  # noqa: BLE001 - a broken cell is a result
            row.update({
                "built": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "where": traceback.format_exc().strip().splitlines()[-3:],
            })
        row["wall_s"] = round(time.time() - started, 1)
        rows.append(row)
    return rows


def report(rows: list[dict[str, Any]]) -> str:
    """A table, with the baseline first so the rest can be read against it."""
    lines = ["%-14s %-7s %-22s %-11s %s" % (
        "scenario", "placed", "errors mm", "s/carton", "note")]
    lines.append("-" * 78)
    for row in rows:
        if not row.get("built"):
            lines.append("%-14s %-7s %-22s %-11s %s" % (
                row["scenario"], "-", "-", "-", row["error"]))
            continue
        lines.append("%-14s %-7s %-22s %-11s %s" % (
            row["scenario"],
            "%d/%d" % (row["placed"], row["of"]),
            str(row["errors_mm"]),
            row["s_per_carton"] if row["s_per_carton"] is not None else "-",
            "; ".join(row["why"]) or "ok",
        ))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sweep()))
