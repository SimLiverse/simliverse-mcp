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
import inspect
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

# Every state you need, plus WARMUP, DONE and FAILED. FAILED is not optional:
# without it the only way out of the machine is DONE, so a run that skipped
# half the task still ends up there. A controller reported DONE having stacked
# two of three cubes and left the third untouched on the floor -- reaching the
# last state is not the same as having done the task.
WARMUP, INIT, DOING, CHECK, DONE, FAILED = 0, 1, 2, 3, 4, 5

# Persistent state lives at module level, not in compute()'s locals.
_state = WARMUP
_frame = 0
_arm = None
_why = ""          # why it failed, readable from outside after the run


def _go(state):
    global _state, _frame
    _state, _frame = state, 0


def _fail(reason):
    """End in FAILED, saying which part did not happen."""
    global _why
    _why = reason
    carb.log_warn("controller FAILED: %s" % reason)
    _go(FAILED)


def _on_timeline(event):
    """Reset on STOP, or the second Play resumes mid-task instead of restarting."""
    import omni.timeline

    global _state, _frame, _arm, _why
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        _state, _frame, _arm, _why = WARMUP, 0, None, ""


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

    if _state in (DONE, FAILED):
        return True

    if _state == WARMUP:                      # let physics settle first
        if _frame >= WARMUP_FRAMES:
            _go(INIT)
        return True

    # Nothing below runs forever. Every state that waits on the world needs a
    # way out that is not "keep waiting" -- a servo that cannot converge, a
    # grasp that never closes, an object out of reach. Without this the run
    # neither finishes nor fails; it just stops making progress, which is the
    # hardest failure to read from outside.
    if _frame > 1500:
        _fail("state %d ran for %d frames without progressing" % (_state, _frame))
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
            _go(CHECK)
        return True

    if _state == CHECK:
        # Measure the OUTCOME, not the fact that you got here. Read the world:
        # where the objects actually ended up, whether they are at rest,
        # whether anything that had to stay put moved. Then DONE or FAILED.
        #
        # `deliver` measures the scene independently and will catch a controller
        # that lies about this -- but by then it costs you a replay, and this
        # check names which part failed while the run still knows.
        reached = _arm.ee_position
        if abs(reached[2] - 0.3) < 0.02:
            _go(DONE)
        else:
            _fail("tool ended at z=%.3f, wanted 0.300" % reached[2])
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


# In preference order. The first is the only one that has been delivered and
# measured end to end, and it is first for a reason: whatever this returns is
# what an agent copies the shape of, so a worked example that omits something
# load-bearing costs a run every time it is read.
_WORKED_EXAMPLES = (
    ("controllers", "stack_three_sizes.py"),
    ("demo", "stack_cubes.py"),
)


def example() -> str:
    """A complete worked controller — the three-cube stack, as shipped.

    Cheaper and more reliable than searching the container's filesystem for it,
    which is what an agent does otherwise.

    This used to return `demo/stack_cubes.py`, which never re-homes the arm.
    The Franka's wrist winds up across a sequence of solves, and once joint 6
    reaches its limit a demanded DOWN orientation can only be met by driving
    into the stop — so every target after the first comes back 9 to 22 cm
    short. Copying that shape is how a run reached for the last cube, missed,
    and concluded it had found a kinematic limit. It had found a starting pose.

    The controller returned instead homes before every pick and measures the
    tower in a CHECK state before it will say DONE. Reproduced three times from
    a cold Play: cubes at z=0.070 and z=0.105 against targets of 0.070 and
    0.105, base undisturbed.
    """
    for directory in _CANDIDATE_DIRECTORIES:
        root = os.path.dirname(directory)
        for subdirectory, filename in _WORKED_EXAMPLES:
            candidate = os.path.join(root, subdirectory, filename)
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
    # The physics-step trigger only works in an ON-DEMAND graph. In a push graph
    # it is evaluated with the rest of the push pipeline — once per app update —
    # and Isaac says so, at Error level, once per attach:
    #
    #   Physics OnSimulationStep node detected in a non on-demand Graph. Node
    #   will only trigger events if the parent Graph is set to compute on-demand.
    #
    # Which means a graph built this way is back to running at the render rate
    # while looking like it runs on the physics step — the failure it was added
    # to fix, now wearing a disguise. Getting the evaluator wrong is worse than
    # never having switched.
    keys = og.Controller.Keys
    trigger, pulse = "isaacsim.core.nodes.OnPhysicsStep", "outputs:step"
    try:
        graph, nodes, _, _ = og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
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
    #
    # And NOT the script itself: a shim that loads it as a module of its own.
    # Every ScriptNode on the stage executes its script in ONE shared Python
    # namespace (measured: two nodes, two scripts, one id(globals())), so the
    # second controller's module-level state overwrote the first's and both
    # nodes drove the second arm while the first stood idle. The shim keeps a
    # registry keyed by node path and dispatches setup/compute to the module
    # that belongs to this node; redefining the shim's own functions from
    # another node's shim is harmless because they are identical.
    og.Controller.set(node.get_attribute("inputs:usePath"), True)
    og.Controller.set(node.get_attribute("inputs:scriptPath"),
                      _isolating_shim(script_path, f"{graph_path}/ScriptNode"))

    # Confirm the trigger can actually fire, rather than trusting that it does.
    # A physics-step trigger in the wrong pipeline stage still builds, still
    # connects, and still looks right in the editor — it just runs at the render
    # rate instead, which is indistinguishable from working until the motion
    # depends on frame rate.
    trigger_node = graph.get_node(f"{graph_path}/Trigger")
    kind = str(trigger_node.get_type_name()) if trigger_node else ""
    if "OnPhysicsStep" in kind:
        try:
            stage_ok = (
                graph.get_pipeline_stage()
                == og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
            )
        except Exception:  # noqa: BLE001 — an unreadable stage is not a failure
            stage_ok = True
        if not stage_ok:
            raise ControllerError(
                f"{graph_path} triggers on the physics step but is not an "
                f"on-demand graph, so the trigger will not fire per step — it "
                f"will run at the render rate and the motion will depend on "
                f"frame rate. Build it with evaluator_name='execution' and "
                f"pipeline_stage=GRAPH_PIPELINE_STAGE_ONDEMAND."
            )

    logger.info("Wired %s -> %s (trigger %s)", script_path, graph_path, kind or "?")
    return graph_path


# ── Reading the stage back ────────────────────────────────────────────────────


def _isolating_shim(script_path: str, node_path: str) -> str:
    """Write the per-node loader next to `script_path`; return its path.

    The shim imports the controller under a module name derived from the
    node path, so two controllers never share globals, and registers it in a
    dict that lives in the (shared) ScriptNode namespace under the node's
    own path. `setup`/`compute` look the caller's node up in that dict.
    """
    stem = os.path.splitext(os.path.basename(script_path))[0]
    module_name = "simliverse_ctl_" + "".join(
        ch if ch.isalnum() else "_" for ch in node_path.strip("/"))
    shim_path = os.path.join(os.path.dirname(script_path), f"{stem}__node.py")
    code = SHIM_TEMPLATE.format(
        node_path=node_path, script_path=script_path, module_name=module_name,
        shim_log=os.path.join(os.path.dirname(script_path), "shim.log"))
    with open(shim_path, "w", encoding="utf-8") as handle:
        handle.write(code)
    return shim_path


#: The loader every ScriptNode runs instead of its controller. See
#: `_isolating_shim`. Module-level names here are shared with every other
#: node's shim on the stage, which is exactly why the only state kept is a
#: dict keyed by node path.
SHIM_TEMPLATE = '''# generated by simliverse_sim.controller.attach -- one module per node
#
# How a ScriptNode runs this file (measured, Isaac Sim 6.0): exec with a
# fresh LOCALS dict and one GLOBALS dict shared by every ScriptNode on the
# stage, then the locals are copied into that shared globals. So: nothing
# defined here may be called from a function during this top-level run
# (it cannot see the imports yet), every top-level name is later
# overwritten by the next node's shim (harmless: they are identical), and
# the only state that must survive is the registry, created with
# setdefault on the shared dict so all shims see one and the same.
import importlib.util as _ilu
import sys as _sys
import traceback as _tb

_SIMLIVERSE_NODE_MODULES = globals().setdefault("_SIMLIVERSE_NODE_MODULES", {{}})
_SIMLIVERSE_SHIM_LOG = {shim_log!r}

try:
    _simliverse_spec = _ilu.spec_from_file_location({module_name!r}, {script_path!r})
    _simliverse_mod = _ilu.module_from_spec(_simliverse_spec)
    _sys.modules[{module_name!r}] = _simliverse_mod
    _simliverse_spec.loader.exec_module(_simliverse_mod)
    _SIMLIVERSE_NODE_MODULES[{node_path!r}] = _simliverse_mod
    with open(_SIMLIVERSE_SHIM_LOG, "a", encoding="utf-8") as _h:
        _h.write("loaded %s for %s" % ({script_path!r}, {node_path!r}) + chr(10))
except Exception:
    with open(_SIMLIVERSE_SHIM_LOG, "a", encoding="utf-8") as _h:
        _h.write("LOAD FAILED for %s: %s" % ({node_path!r}, _tb.format_exc()) + chr(10))


def _simliverse_note(what):
    try:
        with open(_SIMLIVERSE_SHIM_LOG, "a", encoding="utf-8") as handle:
            handle.write(what + chr(10))
    except Exception:
        pass


def _simliverse_node_path(db):
    node = getattr(db, "node", None)
    for name in ("get_prim_path", "get_path"):
        getter = getattr(node, name, None)
        if getter is not None:
            try:
                return str(getter())
            except Exception:
                pass
    try:
        return str(db.abi_node.get_prim_path())
    except Exception:
        _simliverse_note("no node path on db: %s" % type(db))
        return None


def _simliverse_module(db):
    return _SIMLIVERSE_NODE_MODULES.get(_simliverse_node_path(db))


def setup(db):
    module = _simliverse_module(db)
    if module is not None and hasattr(module, "setup"):
        return module.setup(db)


def compute(db):
    module = _simliverse_module(db)
    if module is None:
        _simliverse_note("compute: no module registered for %r (have %r)"
                         % (_simliverse_node_path(db), list(_SIMLIVERSE_NODE_MODULES)))
        return True
    return module.compute(db)


def cleanup(db):
    module = _simliverse_module(db)
    if module is not None and hasattr(module, "cleanup"):
        return module.cleanup(db)
'''


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


def _stale_modules() -> list[str]:
    """simliverse_sim modules whose source on disk differs from what is running.

    Python imports a module once. Deploying a new file into a long-lived Kit
    session therefore changes nothing until something purges `sys.modules` — and
    a controller then executes the *old* library while every inspection of the
    file on disk confirms the fix is present.

    That is not a hypothetical and it was not cheap. Two experiments returned
    byte-identical results and were read as "the thing I changed is not the
    cause", when the changed code had never run. A delivered controller later
    failed on a bug that had already been fixed, and the scene simply sat still,
    which reads as the robot being stuck.
    """
    import sys

    stale = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("simliverse_sim"):
            continue
        path = getattr(module, "__file__", None)
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                on_disk = handle.read()
            running = inspect.getsource(module)
        except Exception:  # noqa: BLE001 — unreadable source is not a mismatch
            continue
        if on_disk.replace("\r\n", "\n") != running.replace("\r\n", "\n"):
            stale.append(name)
    return sorted(stale)


PURGE_SNIPPET = (
    "    import importlib, sys\n"
    "    for m in [k for k in sys.modules if k.startswith('simliverse_sim')]:\n"
    "        del sys.modules[m]\n"
    "    importlib.invalidate_caches()"
)


def deliver(
    name: str,
    code: str,
    *,
    objects: list[str] | None = None,
    robots: list[str] | None = None,
    undisturbed: list[str] | None = None,
    traveled: list[str] | None = None,
    posed: list[str] | None = None,
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
    stale = _stale_modules()
    if stale:
        raise ControllerError(
            "The running library is not the one on disk: "
            + ", ".join(stale)
            + "\n\nPython imports a module once, so a file deployed into a "
            "long-lived session changes nothing until it is re-imported. "
            "Delivering now would run the old code and fail on bugs that are "
            "already fixed, leaving a scene that does nothing.\n\n"
            "Purge and re-import first:\n\n" + PURGE_SNIPPET
        )

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
    report = verify(seconds=seconds, objects=objects, robots=robots,
                    undisturbed=undisturbed, traveled=traveled, posed=posed)
    report["controller_path"] = path
    report["graph_path"] = graph_path
    report["record_path"] = _record(report, script_path=path, graph_path=graph_path)
    if report.get("disturbed"):
        detail = "; ".join(
            "%s moved %s m, touched by %s"
            % (path, d["moved_by"], ", ".join(d["touched_by"]) or "nothing recorded")
            for path, d in report["disturbed"].items()
        )
        report["hint"] = (
            detail
            + ". These were named as things the task must not move, and the "
            "arm reached its targets and hit them on the way. That is a routing "
            "problem rather than a target problem: register them with "
            "add_obstacle before the motion that passes them, and check "
            "unavoidable_by_servo() — an obstacle the reactive policy cannot "
            "represent is avoided by plan_to and never by servo_to."
        )
    elif report.get("diverged"):
        report["hint"] = (
            f"{', '.join(report['diverged'])} left the world, which means the "
            f"physics setup is broken rather than the motion being wrong. The "
            f"usual cause is rigid-body physics applied to something that should "
            f"not have it — an articulation root, or a body with no valid mass. "
            f"Check the prim's applied schemas before blaming the controller."
        )
    elif report.get("asset_problems"):
        faults = [
            f"{path}: {fault['issue']} — {fault['consequence']}"
            for path, found in report["asset_problems"].items()
            for fault in found
        ]
        report["hint"] = (
            "The scene did not do the task, and the robot has defects that would "
            "account for it:" + "".join("\n  " + line for line in faults)
            + "\n\nThese are faults in the asset, not in the controller. Do not "
            "write code around them. Report the task as blocked, quote these "
            "measurements, and say what would have to change about the robot for "
            "it to be possible."
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


def _body_position(robot: Any) -> np.ndarray:
    """Where the robot actually is, however its base is modelled.

    `base_position` is the articulation *root*, which is right for a robot whose
    root is the thing that moves and useless for one whose root is not. A
    Ridgeback carries its base motion on a planar joint triple inside the
    articulation, so its root prim sits at the origin however far it drives —
    measured: the arm's mounting link travelled 0.975 m while `base_position`
    reported [0, 0, 0] throughout.

    Distance is therefore taken from the links rather than the root.

    DOES NOT WORK for a planar base, and the reason is worth keeping. Measured
    on a Ridgeback-Franka, commanding the base joint and reading back:

        cmd 0.0 -> joint 0.008 | link world x 0.308
        cmd 0.8 -> joint 0.783 | link world x 0.308
        cmd 0.0 -> joint 0.017 | link world x 0.308

    The joints track the command exactly; the link's USD world transform never
    changes, before or after an explicit `update_transformations`. So this reads
    the one thing that stays still. For such a robot the authoritative pose is
    the joint vector itself — the planar triple *is* the base pose — and the
    general fix is to read link poses from the physics view rather than USD.

    Kept because it is correct and exercised for every robot whose root is the
    body that moves, which is every wheeled base tested so far.
    """
    root = np.asarray(robot.base_position, dtype=float)
    try:
        from pxr import UsdGeom

        from ._compat import get_physx, get_stage

        # Push physics results into USD before reading them. Link transforms are
        # only written back when something syncs, so a bare read returns
        # whatever was there last time — measured: a joint at 1.191 while the
        # link it drives still reported the pose it had at 0.013.
        try:
            get_physx().update_transformations(False, True, True, False)
        except Exception:
            logger.debug("Could not sync transforms before reading links", exc_info=True)

        stage = get_stage()
        points = []
        for link in robot.links():
            prim = stage.GetPrimAtPath(str(link))
            if not prim.IsValid():
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            points.append(np.asarray(matrix.ExtractTranslation(), dtype=float))
        if points:
            return np.mean(np.asarray(points), axis=0)
    except Exception:
        logger.debug("Could not read link poses for %s", robot.prim_path, exc_info=True)
    return root


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


def _by_robot(bodies: set[str], robots: dict[str, Any]) -> list[str]:
    """The contacts that belong to one of the robots, links included.

    Contacts report the *link* that touched, not the articulation root, so a
    prefix match is what connects `/World/Franka/panda_link5` back to the robot
    the caller named. Falls back to every contact when none match, because "the
    post was hit by something" is still worth reporting.
    """
    hits = sorted(b for b in bodies if any(b.startswith(r) for r in robots))
    return hits or sorted(bodies)


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
    undisturbed: list[str] | None = None,
    traveled: list[str] | None = None,
    posed: list[str] | None = None,
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

    `undisturbed` is the other half, and the half that was missing. Name the
    things the task must *not* move — an obstacle, a neighbouring stack, a
    fixture — and any of them shifting makes `reproduced` False and appears in
    `disturbed`, along with which robot links were seen touching it. Without
    it, a run that transferred one cube, dropped the other on the floor and
    swept the obstacle 25 cm across the table reported `reproduced: True`,
    because something moved and nothing left the world. It did that repeatedly,
    and reading the raw positions is the only reason anyone noticed.

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
    # `attach` rather than `Robot(...)`: the plain constructor gives a base
    # handle with no gripper and no morphology-specific checks, so a robot
    # measured here reported no asset problems while `describe()` on the same
    # robot reported them plainly.
    handles_robots = {p: Robot.attach(p, scene=scene) for p in robot_paths}
    keep_still = {p: RigidObject(p, scene=scene) for p in (undisturbed or [])}
    movers = {p: Robot.attach(p, scene=scene) for p in (traveled or [])}
    posers = {p: Robot.attach(p, scene=scene) for p in (posed or [])}
    before = _sample(handles_objects, handles_robots)
    before_still = {p: np.asarray(o.position, dtype=float) for p, o in keep_still.items()}
    before_pose = {p: np.asarray(r.joint_positions, dtype=float) for p, r in posers.items()}
    before_travel = {
        p: (_body_position(r),
            np.asarray(r.joint_positions, dtype=float))
        for p, r in movers.items()
    }

    scene.play()
    start = timeline.get_current_time()
    ticks = 0
    # Kit's own loop drives OnPlaybackTick, so the app has to be pumped here —
    # stepping PhysX directly would advance physics without ever running the
    # graph, and the controller would never see a frame.
    # Who touched what, sampled throughout rather than only at the end. A body
    # that is struck and comes to rest somewhere plausible is invisible to a
    # before/after comparison of that body alone; the contact is the evidence.
    # Anything named as a robot, whether it was asked to hold still or to
    # travel. Attribution used to look only at `robots`, so a task that used
    # `traveled` got `first_touched_at: None` for every collision it caused.
    attributable = set(handles_robots) | set(movers)
    touched: dict[str, set[str]] = {p: set() for p in keep_still}
    first_touch: dict[str, float] = {}
    while timeline.get_current_time() - start < seconds + settle:
        update_app()
        for path, obj in keep_still.items():
            bodies = obj.contact_bodies()
            touched[path].update(bodies)
            # *When* it was first hit is what turns "the arm touched the post"
            # into a state you can look at. Filtered to the robot, because a
            # body resting on the ground is in contact from the first frame and
            # would otherwise stamp every obstacle at t=0.
            if path not in first_touch and any(
                b.startswith(r) for b in bodies for r in attributable
            ):
                first_touch[path] = round(timeline.get_current_time() - start, 2)
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

    # What the task promised not to move. Displacement is the fact; the robot
    # links seen touching it say how it happened.
    disturbed = {}
    for path, obj in keep_still.items():
        shift = float(np.linalg.norm(np.asarray(obj.position, dtype=float) - before_still[path]))
        if shift > 0.005 or path in first_touch:
            hits = _by_robot(touched[path], attributable)
            disturbed[path] = {
                "moved_by": round(shift, 4),
                "first_touched_at": first_touch.get(path),
                "from": before_still[path].round(4).tolist(),
                "to": np.asarray(obj.position, dtype=float).round(4).tolist(),
                "touched_by": hits,
            }

    # A robot that was supposed to go somewhere. Deliberately separate from
    # `moved`, which excludes robots on purpose: an arm waving its joints has
    # demonstrated nothing. For a base, arriving *is* the task, and without this
    # a working rover controller reports `reproduced: False` with a hint telling
    # its author to go fix a state machine that is already correct.
    #
    # Distance alone is not enough either. A base whose joints never moved was
    # repositioned rather than driven, so the joints are checked too — that is
    # the difference between locomotion and a teleport.
    travelled = {}
    for path, robot in movers.items():
        start_base, start_joints = before_travel[path]
        now_base = _body_position(robot)
        now_joints = np.asarray(robot.joint_positions, dtype=float)
        distance = float(np.linalg.norm(now_base[:2] - start_base[:2]))
        joint_delta = (
            float(np.max(np.abs(now_joints - start_joints))) if now_joints.size else 0.0
        )
        travelled[path] = {
            "distance": round(distance, 4),
            "joints_moved": round(joint_delta, 4),
            "from": start_base.round(4).tolist(),
            "to": now_base.round(4).tolist(),
            "under_own_power": bool(distance > 0.05 and joint_delta > 0.05),
        }

    # A robot that was supposed to change shape rather than place. Posture is a
    # real outcome — standing, reaching, holding a limb somewhere — and it moves
    # nothing else in the scene, so a working posture controller reported
    # `reproduced: False` with an empty `moved` and no way to say otherwise.
    posture = {}
    for path, robot in posers.items():
        # Not `start`: that name holds the timeline's start time in this scope,
        # and shadowing it turned `simulated_seconds` into an array.
        was = before_pose[path]
        now = np.asarray(robot.joint_positions, dtype=float)
        delta = float(np.max(np.abs(now - was))) if now.size else 0.0
        entry = {"joints_moved": round(delta, 4), "changed": bool(delta > 0.05)}
        # If the robot can say which way up it is, a posture that ended on the
        # floor is not a posture that was held.
        if hasattr(robot, "is_upright"):
            try:
                entry["upright"] = bool(robot.is_upright())
                entry["tilt_degrees"] = round(float(robot.tilt_degrees()), 2)
            except Exception:  # noqa: BLE001 — not every robot has an up
                pass
        posture[path] = entry

    report = {
        "moved": sorted(moved),
        "reproduced": (
            (bool(moved) or bool(travelled) or bool(posture))
            and all(t["under_own_power"] for t in travelled.values())
            and all(p["changed"] and p.get("upright", True) for p in posture.values())
            and not diverged
            and not disturbed
        ),
        "diverged": diverged,
        "disturbed": disturbed,
        "travelled": travelled,
        "posture": posture,
        "at_rest": at_rest,
        "simulated_seconds": round(float(timeline.get_current_time() - start), 2),
        "before": before,
        "after": after,
    }

    # Whatever is already known to be wrong with the robots in this scene. The
    # evidence exists at the moment of failure and was otherwise discarded,
    # leaving the author to debug a controller that was never the problem.
    faults: dict[str, Any] = {}
    for path, robot in (list(handles_robots.items()) + list(movers.items())
                        + list(posers.items())):
        try:
            found = robot.asset_problems()
        except Exception:  # noqa: BLE001 — a missing check is not itself a fault
            continue
        if found:
            faults[path] = found
    if faults:
        report["asset_problems"] = faults

    if rerouted:
        report["rerouted"] = rerouted
        report["note"] = (
            f"Routed to robots rather than rigid bodies: {', '.join(rerouted)} "
            f"(articulation roots). A robot reports joint positions and never "
            f"counts towards `moved` — the scene has to move the objects, not "
            f"just wave the arm."
        )
    return report
