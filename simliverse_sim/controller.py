"""
Controllers: making a task reproducible instead of merely done.

A task driven entirely through live calls leaves nothing behind. PhysX reverts
every dynamic body to its authored pose on stop, so a scene built that way
replays as its starting state and the work is gone. Worse, the finished stage
is indistinguishable from one where the objects were simply teleported into
place — which is the shortcut every previous generation of these skills took.

A controller closes both holes at once. The scene carries a script wired to
`OnPlaybackTick`, so pressing Play *performs* the task and physics re-derives
the outcome every run. `verify()` is the check that matters and needs no model
in the loop: rewind to the authored state, play, measure, and report what
actually moved.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Any

import numpy as np

from ._compat import get_timeline, update_app

logger = logging.getLogger("simliverse_sim.controller")

# Where a controller can actually be written, most durable first.
#
# The extension directory is bind-mounted from the host and survives anything,
# but it is mounted read-only *inside* the container — so it is right for
# controllers shipped with the repo and impossible for ones authored at
# runtime. /tmp is the writable fallback: it lives in the container's own layer
# and survives a restart, though not a rebuild. Resolved per call rather than at
# import, because which of these is writable depends on how the container was
# started.
_CANDIDATE_DIRECTORIES = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controllers"),
    "/tmp/simliverse/controllers",
    os.path.join(os.path.expanduser("~"), ".simliverse", "controllers"),
)


def _writable_directory() -> str:
    problems = []
    for candidate in _CANDIDATE_DIRECTORIES:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_probe")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("")
            os.remove(probe)
            return candidate
        except OSError as exc:
            problems.append(f"{candidate}: {exc.strerror or exc}")
    raise ControllerError(
        "No writable directory for controller scripts. Tried:\n  "
        + "\n  ".join(problems)
    )


class ControllerError(RuntimeError):
    """A controller could not be authored or did not reproduce its task."""


def write(name: str, code: str, *, directory: str | None = None) -> str:
    """Validate a controller script and write it to disk.

    Rejects, rather than writes, a script that cannot work as a ScriptNode.
    Both failures below are silent at author time and produce a graph that
    simply never runs — the scene looks correctly wired and does nothing, which
    is among the most expensive things to debug from the outside.

      * a syntax error, which OmniGraph swallows into a log line
      * a missing `compute`, which leaves the node in legacy mode where exec
        scoping is broken

    Returns the absolute path, which is what `create_action_graph(script_file=)`
    expects.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ControllerError(
            f"Controller {name!r} has a syntax error on line {exc.lineno}: {exc.msg}. "
            f"Nothing was written."
        ) from exc

    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = {"setup", "compute"} - functions
    if missing:
        raise ControllerError(
            f"Controller {name!r} defines no {' and no '.join(sorted(missing))}. "
            f"A ScriptNode needs both at module level: setup(db) runs once, "
            f"compute(db) runs every frame. Without compute the node falls back "
            f"to legacy mode, where the script's scoping is broken and nothing "
            f"runs. Nothing was written."
        )

    if directory is None:
        directory = _writable_directory()
    else:
        os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.py" if not name.endswith(".py") else name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(code)
    logger.info("Wrote controller %s (%d bytes)", path, len(code))
    return path


def _sample(objects: dict[str, Any], robots: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for path, obj in objects.items():
        state[path] = {
            "position": np.asarray(obj.position, dtype=float).round(4).tolist(),
            "speed": round(float(obj.speed), 4),
            "contacts": sorted({c["body"] for c in obj.contacts()}),
        }
    for path, robot in robots.items():
        state[path] = {
            "joint_positions": np.asarray(robot.joint_positions, dtype=float).round(4).tolist()
        }
    return state


def verify(
    *,
    seconds: float = 25.0,
    objects: list[str] | None = None,
    robots: list[str] | None = None,
    settle: float = 2.0,
    scene: Any = None,
) -> dict[str, Any]:
    """Rewind, play, and report what the scene did on its own.

    This is the acceptance test for a controller, and deliberately involves no
    language model: stop the timeline so every dynamic body returns to its
    authored pose, play, let Kit tick for `seconds` of simulated time, then
    measure. Whatever moved, moved because the controller moved it.

    `moved` is the headline. A controller that is not wired, throws on its first
    tick, or was never reached leaves the scene exactly as authored — and a
    scene whose objects were teleported into their final pose reports `False`
    here too, because stop puts them back.

    Returns the before and after states so a caller can assert on real numbers
    rather than on a claim.
    """
    from .objects import RigidObject
    from .robots.base import Robot
    from .scene import Scene

    scene = scene or Scene.get()
    timeline = get_timeline()

    scene.stop()
    handles_objects = {p: RigidObject(p, scene=scene) for p in objects or []}
    handles_robots = {p: Robot(p, scene=scene) for p in robots or []}
    before = _sample(handles_objects, handles_robots)

    scene.play()
    start = timeline.get_current_time()
    ticks = 0
    # Kit's own loop drives OnPlaybackTick, so the app has to be pumped here —
    # stepping PhysX directly would advance physics without ever running the
    # graph, and the controller would never see a frame.
    while timeline.get_current_time() - start < seconds + settle:
        update_app()
        ticks += 1
        if ticks > 200_000:  # a stopped timeline would otherwise spin forever
            raise ControllerError(
                "Timeline did not advance while verifying. Is the simulation "
                "able to play — and is anything else holding it paused?"
            )

    after = _sample(handles_objects, handles_robots)
    moved = {
        path
        for path in before
        if "position" in before[path]
        and float(
            np.linalg.norm(
                np.asarray(after[path]["position"]) - np.asarray(before[path]["position"])
            )
        )
        > 0.005
    }
    at_rest = all(
        entry.get("speed", 0.0) < 0.05 for entry in after.values() if "speed" in entry
    )

    return {
        "moved": sorted(moved),
        "reproduced": bool(moved),
        "at_rest": at_rest,
        "simulated_seconds": round(float(timeline.get_current_time() - start), 2),
        "before": before,
        "after": after,
    }
