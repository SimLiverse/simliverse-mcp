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

"""Surfacing what Isaac said during a `run_control` call.

Three faults in one session were each diagnosed from a single line of the Kit
log, and each had already cost hours, because every one of them is silent at
the Python API — the call returns a number, the number is wrong, and the reason
sits in a file nobody opened:

    Attempted to compute inverse kinematics for an uninitialized robot
    Articulation. Cannot get joint positions          -> pose error of inf at 180 deg
    PxConstraint::setFlag() not allowed while simulation is running.
    Call will be ignored.                             -> gripper silently never latches
    ... possibly invalid inertia tensor of {1.0, 1.0, 1.0}   -> unstable wrist

The lines below are real, copied from that log.

Driven over a temp file rather than a live session, so it runs without Isaac.
"""

import importlib.util
import sys
from pathlib import Path

# Loaded by path, not by package. Importing `isaac_sim_mcp_extension` runs its
# `__init__`, which pulls in `carb` and therefore the whole simulator — and the
# point of `kit_log` is that it needs none of that.
_SPEC = importlib.util.spec_from_file_location(
    "_sl_kit_log",
    Path(__file__).resolve().parents[1] / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension" / "kit_log.py",
)
control = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = control
_SPEC.loader.exec_module(control)


REAL_LINES = [
    "2026-08-09T17:34:09Z [Warning] [articulation_subset] Attempting to access an uninitialized robot Articulation.\n",
    "2026-08-09T17:34:09Z [Error] [articulation_kinematics_solver] Attempted to compute inverse kinematics for an uninitialized robot Articulation.  Cannot get joint positions\n",
    "2026-08-09T17:34:10Z [Error] [omni.physx.plugin] PhysX error: PxConstraint::setFlag() not allowed while simulation is running. Call will be ignored.\n",
    "2026-08-09T17:34:10Z [Info] [omni.rtx] View 0 switching to 1x1 layout\n",
]


def write_log(tmp_path: Path, lines) -> Path:
    path = tmp_path / "kit.log"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_only_the_lines_this_call_produced(tmp_path: Path) -> None:
    """A call is judged on what it provoked, not the session's whole history."""
    path = write_log(tmp_path, ["2026 [Error] [old] from before the call\n"])
    offset = control.offset(str(path))

    with path.open("a", encoding="utf-8") as fh:
        fh.writelines(REAL_LINES)

    found = control.since(str(path), offset)
    joined = "\n".join(found)
    assert "from before the call" not in joined
    assert "uninitialized robot Articulation" in joined


def test_info_lines_are_not_noise_worth_carrying(tmp_path: Path) -> None:
    path = write_log(tmp_path, REAL_LINES)
    found = control.since(str(path), 0)
    assert not any("switching to 1x1 layout" in line for line in found)


def test_errors_outrank_warnings_when_truncated(tmp_path: Path) -> None:
    """A clipped list has to keep the part worth reading."""
    lines = ["2026 [Warning] [x] filler number %d\n" % i for i in range(60)]
    lines.append(REAL_LINES[2])
    path = write_log(tmp_path, lines)

    found = control.since(str(path), 0, limit=3)
    assert found[0].startswith("[ERROR]")
    assert "setFlag" in found[0]


def test_repeats_collapse_to_one_line_with_a_count(tmp_path: Path) -> None:
    """The same warning 900 times is one fact, not 900."""
    path = write_log(tmp_path, [REAL_LINES[0]] * 900)
    found = control.since(str(path), 0)
    assert len(found) == 1
    assert "(x900)" in found[0]


def test_messages_differing_only_by_number_are_the_same_message(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        [
            "2026 [Error] [physx] overlapping API read from thread 1918187328!\n",
            "2026 [Error] [physx] overlapping API read from thread 1509930688!\n",
        ],
    )
    found = control.since(str(path), 0)
    assert len(found) == 1
    assert "(x2)" in found[0]


def test_a_clean_call_carries_no_log_key(tmp_path: Path) -> None:
    """Absence is the signal. A quiet result must stay quiet."""
    path = write_log(tmp_path, ["2026 [Info] [x] nothing wrong here\n"])
    result = control.attach({"status": "success"}, str(path), 0)
    assert "isaac_log" not in result


def test_a_successful_call_can_still_report_a_complaint(tmp_path: Path) -> None:
    """The case this exists for: the code ran, and physics objected anyway."""
    path = write_log(tmp_path, REAL_LINES)
    result = control.attach({"status": "success"}, str(path), 0)
    assert result["status"] == "success"
    assert any("setFlag" in line for line in result["isaac_log"])


def test_a_missing_or_unreadable_log_is_not_an_error(tmp_path: Path) -> None:
    """Diagnostics must never take down the call they are describing."""
    assert control.since(None, 0) == []
    assert control.since(str(tmp_path / "does_not_exist.log"), 0) == []
    assert control.offset(None) == 0
    assert control.attach({"status": "success"}, None, 0) == {"status": "success"}
