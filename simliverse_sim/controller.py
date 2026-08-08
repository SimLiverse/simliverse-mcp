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
import glob
import hashlib
import json
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


SKELETON = '''"""<what this controller does>."""

import carb
import numpy as np

WARMUP_FRAMES = 30

# Every state you need, plus WARMUP and DONE.
WARMUP, INIT, DOING, DONE = 0, 1, 2, 3

# Persistent state lives at module level, not in compute()'s locals.
_state = WARMUP
_frame = 0
_arm = None


def _go(state):
    global _state, _frame
    _state, _frame = state, 0


def _on_timeline(event):
    """Reset on STOP, or the second Play resumes mid-task instead of restarting."""
    import omni.timeline

    global _state, _frame, _arm
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _arm = WARMUP, 0, None


_timeline_sub = None


def setup(db=None):
    global _timeline_sub
    if _timeline_sub is None:
        import omni.timeline

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline)


def compute(db=None):
    """One frame. One state transition. Never loop, never step physics."""
    global _state, _frame, _arm
    _frame += 1

    if _state == WARMUP:                      # let physics settle first
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    if _state == INIT:
        from simliverse_sim import RigidObject, Scene
        from simliverse_sim.robots.manipulator import Manipulator

        _arm = Manipulator("/World/Robot", scene=Scene.get())
        _go(DOING)
        return True

    if _state == DOING:
        # servo_to advances ONE tick and returns True once converged.
        if _arm.servo_to([0.4, 0.0, 0.3]):
            _go(DONE)
        return True

    return True
'''


def skeleton() -> str:
    """The minimum correct ScriptNode shape, to fill in rather than rediscover.

    Every rule that silently produces a graph which does nothing is already
    satisfied here: both entry points defined, state at module level, a STOP
    subscription, a warmup, and one transition per call.
    """
    return SKELETON


def example() -> str:
    """A complete worked controller — the three-cube stack, as shipped.

    Cheaper and more reliable than searching the container's filesystem for it,
    which is what an agent does otherwise.
    """
    for directory in _CANDIDATE_DIRECTORIES:
        candidate = os.path.join(os.path.dirname(directory), "demo", "stack_cubes.py")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                return handle.read()
    return SKELETON


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


def attach(script_path: str, *, graph_path: str = "/World/TaskGraph") -> str:
    """Wire a controller script into the stage as OnPlaybackTick -> ScriptNode.

    Same graph the `create_action_graph` verb builds. It lives here as well
    because crossing tool types mid-task is where the deliverable kept getting
    dropped: an agent already inside `run_control` would finish the motion, mean
    to wire it, and never switch tools. One call in the tool it is already
    holding is worth the small duplication.
    """
    import omni.graph.core as og

    from ._compat import get_stage

    # Re-wiring an existing graph fails with "Failed to wrap graph in node",
    # which names neither the graph nor the cause. Any second delivery in a
    # session hits it -- a corrected controller, a second task, a re-run -- so
    # replace rather than edit in place. The graph is cheap to rebuild and this
    # makes attach() idempotent.
    stage = get_stage()
    if stage.GetPrimAtPath(graph_path).IsValid():
        logger.info("Replacing existing action graph at %s", graph_path)
        stage.RemovePrim(graph_path)

    # Trigger on the *physics* step, not the playback tick.
    #
    # OnPlaybackTick fires once per app update, which is the render rate. That
    # happens to equal the physics rate when the app is pumped headless, and does
    # not when someone presses Play and the viewport is drawing. Control code
    # written against it then runs at whatever the frame rate is: a servo loop
    # gets fewer corrections, and a trajectory gets commanded in coarse jumps
    # that the arm cuts across in joint space.
    #
    # That is not hypothetical. Driving this scene's controller at a third of
    # the physics rate knocked the obstacle 25 cm across the table and left both
    # cubes on the floor, while the report still said reproduced: True — the same
    # task passes cleanly at 1:1. Any control law that assumes a fixed timestep
    # belongs on the timestep.
    keys = og.Controller.Keys
    trigger, pulse = "isaacsim.core.nodes.OnPhysicsStep", "outputs:step"
    try:
        graph, nodes, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "push"},
            {
                keys.CREATE_NODES: [
                    ("Trigger", trigger),
                    ("ScriptNode", "omni.graph.scriptnode.ScriptNode"),
                ],
                keys.CONNECT: [(f"Trigger.{pulse}", "ScriptNode.inputs:execIn")],
            },
        )
    except Exception:
        logger.warning(
            "%s is unavailable; falling back to OnPlaybackTick. The controller "
            "will run at the render rate, so its motion depends on frame rate.",
            trigger,
        )
        stage.RemovePrim(graph_path)
        graph, nodes, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "push"},
            {
                keys.CREATE_NODES: [
                    ("Trigger", "omni.graph.action.OnPlaybackTick"),
                    ("ScriptNode", "omni.graph.scriptnode.ScriptNode"),
                ],
                keys.CONNECT: [("Trigger.outputs:tick", "ScriptNode.inputs:execIn")],
            },
        )
    if graph is None:
        raise ControllerError(f"Could not create an action graph at {graph_path!r}")

    node = graph.get_node(f"{graph_path}/ScriptNode")
    if node is None or not node.is_valid():
        raise ControllerError(f"Action graph at {graph_path!r} has no usable ScriptNode")

    # Without usePath the node ignores scriptPath and runs its inline source,
    # which is empty — a graph that is wired, valid, and does nothing.
    og.Controller.set(node.get_attribute("inputs:usePath"), True)
    og.Controller.set(node.get_attribute("inputs:scriptPath"), script_path)
    logger.info("Wired %s -> %s", script_path, graph_path)
    return graph_path


# ── Reading the stage back ────────────────────────────────────────────────────


def _script_node(graph_path: str) -> Any:
    """The ScriptNode inside an action graph, or None."""
    try:
        import omni.graph.core as og

        graph = og.get_graph_by_path(graph_path)
        if graph is None:
            return None
        node = graph.get_node(f"{graph_path}/ScriptNode")
        return node if node is not None and node.is_valid() else None
    except Exception:
        logger.debug("Could not read the graph at %s", graph_path, exc_info=True)
        return None


def _wired_script(graph_path: str) -> str | None:
    """Which controller a graph actually runs, read from the live graph.

    Read through `og.Controller`, not off the USD attribute: the USD read
    returns None for these even on a correctly wired node. A graph with
    `usePath` False runs its inline source instead, which is normally empty —
    reporting its `scriptPath` in that case would describe a controller that
    is not running.
    """
    node = _script_node(graph_path)
    if node is None:
        return None
    try:
        import omni.graph.core as og

        if not bool(og.Controller.get(node.get_attribute("inputs:usePath"))):
            return None
        return str(og.Controller.get(node.get_attribute("inputs:scriptPath")))
    except Exception:
        logger.debug("Could not read scriptPath on %s", graph_path, exc_info=True)
        return None


def graphs() -> list[dict[str, Any]]:
    """Every action graph on the stage, and the controller each one runs.

    Worth checking after any delivery that did not go cleanly the first time.
    A stage can hold several graphs pointed at the *same* script, each with its
    own copy of the module state, all commanding the same robot every frame —
    a measured session ended up with two, and the task completed roughly twice
    as fast as the controller was written to run, which looked like success.
    """
    from ._compat import get_stage

    found: list[dict[str, Any]] = []
    for prim in get_stage().Traverse():
        if prim.GetTypeName() != "OmniGraph":
            continue
        path = str(prim.GetPath())
        found.append({"graph": path, "script": _wired_script(path)})
    return found


# ── Recording a verification so it does not have to be repeated ───────────────


def _digest(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _record_path(script_path: str) -> str:
    return os.path.splitext(script_path)[0] + ".report.json"


def _record(report: dict[str, Any], *, script_path: str, graph_path: str) -> str | None:
    """Persist a verification next to the controller it verified.

    `verify` is the expensive step — it plays the scene for its full duration
    in real time — and it was being paid for twice, once by the agent that
    wrote the controller and again by the agent checking that agent's work.
    The second run is not what makes the check independent; the measurement is
    already model-free. What the checker actually needs is the numbers, plus
    enough fingerprint to know they still describe the stage in front of it.
    """
    stamped = dict(report)
    stamped["script_path"] = script_path
    stamped["script_sha256"] = _digest(script_path)
    stamped["graph_path"] = graph_path
    destination = _record_path(script_path)
    try:
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(stamped, handle, indent=2)
    except OSError:
        logger.debug("Could not record the report to %s", destination, exc_info=True)
        return None
    return destination


def _latest_record() -> str | None:
    newest, newest_time = None, -1.0
    for directory in _CANDIDATE_DIRECTORIES:
        for candidate in glob.glob(os.path.join(directory, "*.report.json")):
            try:
                stamp = os.path.getmtime(candidate)
            except OSError:
                continue
            if stamp > newest_time:
                newest, newest_time = candidate, stamp
    return newest


def audit(name: str | None = None) -> dict[str, Any]:
    """Re-check a recorded verification against the stage, without replaying it.

    Answers the question a checker actually has — *does the scene in front of
    me still do what that report says it does* — for the cost of a few file
    reads instead of a full playback.

    `current` is True only when the recorded run still describes this stage:
    the controller on disk is byte-for-byte the one that was measured, the
    graph is still present, and it is still wired to that same script through
    `usePath`. Any of those failing puts the reason in `stale_because` and the
    report has to be earned again with `verify`.

    A `current: True` record is a measurement, not a claim — `verify` runs no
    model. Read its numbers and decide for yourself whether they support what
    was asserted; that is the part worth doing twice, and it is not the part
    that costs minutes.
    """
    if name:
        path = name if name.endswith(".report.json") else None
        if path is None:
            stem = name[:-3] if name.endswith(".py") else name
            for directory in _CANDIDATE_DIRECTORIES:
                candidate = os.path.join(directory, f"{stem}.report.json")
                if os.path.isfile(candidate):
                    path = candidate
                    break
    else:
        path = _latest_record()

    if not path or not os.path.isfile(path):
        return {
            "found": False,
            "current": False,
            "stale_because": ["no recorded verification"],
            "hint": (
                "Nothing has been delivered through controller.deliver() in a way "
                "that recorded its result. Run controller.verify(...) yourself."
            ),
        }

    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    reasons: list[str] = []
    script_path = report.get("script_path") or ""
    graph_path = report.get("graph_path") or ""

    if not os.path.isfile(script_path):
        reasons.append(f"the controller {script_path} is gone")
    elif _digest(script_path) != report.get("script_sha256"):
        reasons.append(f"{script_path} has been edited since it was verified")

    wired = _wired_script(graph_path)
    if wired is None:
        reasons.append(f"no graph at {graph_path} is running a script")
    elif os.path.abspath(wired) != os.path.abspath(script_path):
        reasons.append(f"{graph_path} now runs {wired}, not the script that was verified")

    duplicates = [
        entry["graph"]
        for entry in graphs()
        if entry["script"]
        and os.path.abspath(entry["script"]) == os.path.abspath(script_path)
        and entry["graph"] != graph_path
    ]
    if duplicates:
        reasons.append(
            f"{', '.join(duplicates)} also runs this controller — the scene is "
            f"driving the robot from more than one graph"
        )

    report["found"] = True
    report["record_path"] = path
    report["current"] = not reasons
    report["stale_because"] = reasons
    return report


def deliver(
    name: str,
    code: str,
    *,
    objects: list[str] | None = None,
    robots: list[str] | None = None,
    seconds: float = 30.0,
    graph_path: str = "/World/TaskGraph",
) -> dict[str, Any]:
    """Author, wire and prove a controller in one call. This is what "done" means.

    Equivalent to `write` then `attach` then `verify`, and exists as one call
    because the three-step version kept being started and not finished.

    Re-running a task by hand after a stop is not this. That demonstrates the
    agent can repeat itself; it says nothing about the scene, which is what the
    user actually keeps. Only a controller makes the stage perform the task, and
    only `reproduced: True` here shows that it does.
    """
    path = write(name, code)
    foreign = [entry for entry in graphs() if entry["script"] and entry["graph"] != graph_path]
    if foreign:
        raise ControllerError(
            "Another action graph is already driving this scene:\n  "
            + "\n  ".join(f"{e['graph']} runs {e['script']}" for e in foreign)
            + "\n\nDelivering now would leave two controllers commanding the same "
            "robot every frame. That does not fail — it produces a scene that "
            "does roughly the right thing at roughly twice the speed, which reads "
            "as the controller working, and it has silently corrupted three "
            "measured runs.\n\n"
            "Remove the other graph first:\n"
            + "\n".join(f"    Scene.get().stage.RemovePrim({e['graph']!r})" for e in foreign)
            + "\n\nor deliver to that same graph_path to replace it."
        )
    attach(path, graph_path=graph_path)
    report = verify(seconds=seconds, objects=objects, robots=robots)
    report["controller_path"] = path
    report["graph_path"] = graph_path
    report["record_path"] = _record(report, script_path=path, graph_path=graph_path)
    if report.get("diverged"):
        report["hint"] = (
            f"{', '.join(report['diverged'])} left the world, which means the "
            f"physics setup is broken rather than the motion being wrong. The "
            f"usual cause is rigid-body physics applied to something that should "
            f"not have it — an articulation root, or a body with no valid mass. "
            f"Check the prim's applied schemas before blaming the controller."
        )
    elif not report["reproduced"]:
        report["hint"] = (
            "Nothing moved when the scene played on its own. The controller is "
            "wired but is not driving anything: check that compute() advances its "
            "state machine (one transition per call, never a loop), that it waits "
            "for physics before building handles, and that motion states call "
            "servo_to every frame rather than the blocking move_ee_to. "
            "get_isaac_logs shows anything the script printed or raised."
        )
    return report


def _is_articulation(path: str) -> bool:
    from pxr import UsdPhysics

    from ._compat import get_stage

    prim = get_stage().GetPrimAtPath(path)
    return bool(prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI))


def _split_by_kind(
    objects: list[str] | None, robots: list[str] | None
) -> tuple[list[str], list[str], list[str]]:
    """Sort requested paths into rigid bodies and articulations.

    A robot path in `objects` was originally an error, because measuring one as
    a rigid body applies RigidBodyAPI to the articulation root and throws the
    robot out of the world. Refusing was the right call and the wrong ergonomic:
    the caller cannot see the difference from the outside — both are just prims
    it wants the state of — so the error was reliably hit and reliably cost two
    turns to correct. The stage already knows which is which. Sort them here and
    report it in `rerouted` rather than making the caller guess and retry.
    """
    requested_objects = list(objects or [])
    requested_robots = list(robots or [])

    rerouted = [path for path in requested_objects if _is_articulation(path)]
    if rerouted:
        moved = set(rerouted)
        requested_objects = [p for p in requested_objects if p not in moved]
        requested_robots += [p for p in rerouted if p not in requested_robots]
        logger.info("Measuring %s as robots, not rigid bodies", ", ".join(rerouted))
    return requested_objects, requested_robots, rerouted


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
    # Sort before touching anything. Wrapping an articulation root as a rigid
    # body is silently destructive, and the resulting explosion then reads as
    # "something moved", i.e. as success — so the split has to happen before the
    # first handle is built, not as a rescue afterwards.
    body_paths, robot_paths, rerouted = _split_by_kind(objects, robots)
    handles_objects: dict[str, Any] = {}
    for path in body_paths:
        try:
            handles_objects[path] = RigidObject(path, scene=scene)
        except ValueError as exc:
            raise ControllerError(
                f"objects={path!r} cannot be measured as a rigid body.\n\n{exc}"
            ) from exc
    handles_robots = {p: Robot(p, scene=scene) for p in robot_paths}
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

    # Anything that left the world did not "move" in any sense worth reporting.
    # Motion alone was the whole test until a robot with a corrupted articulation
    # fell 14 km and counted as the scene doing its job.
    diverged = sorted(
        path
        for path, entry in after.items()
        if "position" in entry
        and (entry["position"][2] < -1.0 or max(abs(v) for v in entry["position"]) > 50.0)
    )

    report = {
        "moved": sorted(moved),
        "reproduced": bool(moved) and not diverged,
        "diverged": diverged,
        "at_rest": at_rest,
        "simulated_seconds": round(float(timeline.get_current_time() - start), 2),
        "before": before,
        "after": after,
    }
    if rerouted:
        report["rerouted"] = rerouted
        report["note"] = (
            f"Routed to robots rather than rigid bodies: {', '.join(rerouted)} "
            f"(articulation roots). A robot reports joint positions and never "
            f"counts towards `moved` — the scene has to move the objects, not "
            f"just wave the arm."
        )
    return report
