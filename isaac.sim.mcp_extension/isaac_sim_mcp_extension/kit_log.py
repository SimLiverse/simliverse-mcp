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

"""What Isaac said while a call was running.

Isaac reports its real failures to the Kit log and almost nothing to the Python
API. Three separate faults in one session were each diagnosed from a single line
in that file, and each had already cost hours first:

    Attempted to compute inverse kinematics for an uninitialized robot
    Articulation. Cannot get joint positions
        -> every pose solve returns an error of `inf` at 180 degrees, for any
           target, including ones well inside the workspace

    PxConstraint::setFlag() not allowed while simulation is running.
    Call will be ignored.
        -> a constraint write is dropped, so a gripper never latches and its
           status simply stays Open

    The rigid body at /World/Arm/ee_link has a possibly invalid inertia tensor
    of {1.0, 1.0, 1.0}, small sphere approximated inertia was used.
        -> the wrist the tool mounts to is being simulated with made-up inertia

None of the three raises. The call returns a number, the number is wrong, and
the explanation sits in a file nobody opened. Attaching these to the result is
what turns that into something the caller can act on.

No Isaac imports here on purpose: this is file parsing, and it should be
testable without a simulator.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional

# Ranked, most severe first. The order is load-bearing — a truncated list has to
# keep the part worth reading.
_LEVELS = ("[Fatal]", "[Error]", "[Warning]")
_LABELS = ("FATAL", "ERROR", "WARNING")

_SEARCH = (
    "/isaac-sim/.nvidia-omniverse/logs/Kit/*/*/kit_*.log",
    "~/.nvidia-omniverse/logs/Kit/*/*/kit_*.log",
)


def active_log() -> Optional[str]:
    """The Kit log this session is writing to, or None."""
    for pattern in _SEARCH:
        try:
            paths = glob.glob(os.path.expanduser(pattern))
            if paths:
                return max(paths, key=os.path.getmtime)
        except OSError:
            continue
    return None


def offset(path: Optional[str]) -> int:
    """Where the log ends now, so a later read returns only what came after."""
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def since(path: Optional[str], start: int, limit: int = 20) -> List[str]:
    """Warnings and errors written after `start`, deduped and capped.

    Identical messages collapse to one line with a count, and messages that
    differ only in a number or an address are treated as identical — the same
    PhysX complaint from two thread ids is one fact, not two, and the same
    warning 900 times is still one fact.
    """
    if not path:
        return []
    try:
        with open(path, "r", errors="replace") as handle:
            handle.seek(start)
            fresh = handle.readlines()
    except OSError:
        return []

    counts: Dict[str, int] = {}
    first: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    for line in fresh:
        for level, severity in zip(_LEVELS, range(len(_LEVELS))):
            if level not in line:
                continue
            body = line.split(level, 1)[1].strip()
            key = re.sub(r"0x[0-9a-fA-F]+|\d+", "N", body)[:160]
            counts[key] = counts.get(key, 0) + 1
            first.setdefault(key, body[:300])
            rank[key] = min(rank.get(key, len(_LEVELS)), severity)
            break

    ordered = sorted(counts, key=lambda k: (rank[k], -counts[k]))
    out = [
        "[%s] %s%s" % (_LABELS[rank[key]], first[key], f"  (x{counts[key]})" if counts[key] > 1 else "")
        for key in ordered[:limit]
    ]
    if len(ordered) > limit:
        out.append(f"... and {len(ordered) - limit} more distinct messages")
    return out


def attach(result: Dict[str, Any], path: Optional[str], start: int) -> Dict[str, Any]:
    """Add `isaac_log` to a result, but only when there is something to say.

    Absence is the signal: a quiet call stays quiet, so the key appearing at all
    means something wants looking at. The case this exists for is a
    `status: success` that carries one — the code ran, returned a number, and
    physics objected on the way past.
    """
    entries = since(path, start)
    if entries:
        result["isaac_log"] = entries
    return result
