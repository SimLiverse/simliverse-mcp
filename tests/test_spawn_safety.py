"""Ways that building or verifying a scene used to damage the session.

None of these were caught by the existing suite, because none of them are wrong
in isolation. Each is a correct-looking line whose effect lands somewhere else:
a constant that was never defined, a cleanup that tears down the physics view, a
fetch with no timeout on the thread that owns the application, and a second
physics scene that stops PhysX stepping at all.

The last two share a shape worth naming. Neither raises. The simulator keeps its
heartbeat and answers nothing, so from outside it looks busy rather than broken —
and on a worker with no shell, "busy rather than broken" is indistinguishable
from a task that is simply taking a while, until the instance is terminated.

All of them were found by running an agent against a live simulator, and each
cost more than a test would have.
"""

import ast
import builtins
import io
import os
import re
import urllib.error

import pytest

from simliverse_sim.robots import library
from simliverse_sim.robots.library import RobotAssetUnreachable, _check_reachable


# ── The asset fetch that could hang the simulator ─────────────────────────────


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_an_unreachable_asset_raises_instead_of_blocking(monkeypatch) -> None:
    """A timeout must become an exception, not a hung application.

    `add_reference` resolves USD synchronously on the main thread with no
    timeout of its own, so a URL that never answers does not fail — it takes
    rendering, the MCP socket and any chance of recovery with it. On a cloud
    worker with no SSH and no SSM that is the whole instance: terminate and boot
    another, ten minutes and a GPU-hour.
    """

    def never_answers(*_a: object, **_k: object) -> None:
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", never_answers)

    with pytest.raises(RobotAssetUnreachable) as caught:
        _check_reachable("https://assets.example/Isaac/6.0/Robots/Thing/thing.usd")

    # The message has to name the URL: "a robot failed to spawn" is not
    # actionable, and the catalogue is discovered at runtime so the reader
    # cannot look the path up.
    assert "thing.usd" in str(caught.value)


def test_a_404_is_reported_as_the_catalogue_being_wrong(monkeypatch) -> None:
    """Discovery lists what it finds; the server decides what it serves.

    A robot in the catalogue that 404s is a discovery bug, and saying so beats
    a generic failure — the agent can then pick a different robot rather than
    retrying the same one.
    """

    def not_found(*_a: object, **_k: object) -> None:
        raise urllib.error.HTTPError(
            "https://assets.example/x.usd", 404, "Not Found", {}, None  # type: ignore[arg-type]
        )

    monkeypatch.setattr("urllib.request.urlopen", not_found)

    with pytest.raises(RobotAssetUnreachable) as caught:
        _check_reachable("https://assets.example/x.usd")
    assert "404" in str(caught.value)


def test_a_reachable_asset_passes_quietly(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(200))
    _check_reachable("https://assets.example/ok.usd")


@pytest.mark.parametrize(
    "url",
    [
        "omniverse://localhost/NVIDIA/Assets/Isaac/6.0/Robots/x.usd",
        "/mnt/assets/robots/x.usd",
        "file:///mnt/assets/robots/x.usd",
    ],
)
def test_non_http_roots_are_left_alone(url: str, monkeypatch) -> None:
    """A check that cannot run must not become a check that refuses.

    Omniverse and local paths are perfectly valid asset roots and urllib cannot
    speak to either. Guessing would break every deployment that does not serve
    assets over HTTPS.
    """

    def must_not_be_called(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{url} should not have been fetched over HTTP")

    monkeypatch.setattr("urllib.request.urlopen", must_not_be_called)
    _check_reachable(url)


# ── The constant that made robot discovery impossible ─────────────────────────

# The extension imports `omni`, `pxr` and `isaacsim`, so it can only be imported
# inside Isaac Sim. The suite parses it instead — the same approach
# `test_handler_structure` takes.
EXTENSION_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension"
)

HANDLERS = os.path.join(EXTENSION_ROOT, "handlers")


def _module_constants_are_defined(path: str) -> list[str]:
    """Names in SCREAMING_CASE that are read but never bound in the module."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)

    read = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and re.fullmatch(r"_?[A-Z][A-Z0-9_]{2,}", node.id)
    }
    return sorted(read - bound - set(dir(builtins)))


@pytest.mark.parametrize(
    "module",
    sorted(f for f in os.listdir(HANDLERS) if f.endswith(".py")),
)
def test_no_handler_reads_a_constant_it_never_defines(module: str) -> None:
    """`_DETAIL_THRESHOLD` was referenced in `robots.py` and defined nowhere.

    Every call to `list_available_robots` raised `NameError: name
    '_DETAIL_THRESHOLD' is not defined` — including the no-argument call the
    tool's own description tells an agent to make first. Robot discovery was
    therefore impossible: an agent could not learn what it was allowed to spawn,
    only guess a name and see whether it loaded. The generalisation task that
    needs an unfamiliar robot could not be attempted at all.

    Nothing caught it because the module imports cleanly — the name is only
    resolved when that branch runs, and that branch needs a live asset server.
    This is the cheap half of that check, and it runs everywhere.
    """
    missing = _module_constants_are_defined(os.path.join(HANDLERS, module))
    assert not missing, f"{module} reads {missing} without defining or importing them"


def test_the_detail_threshold_is_a_usable_size() -> None:
    """The value decides whether a narrowed search is answerable at all.

    Set it too low and `search="ur"` — 12 hits — comes back as a bare list of
    keys, so the agent must call again for the asset paths it already asked
    for. Too high and a bare call dumps 200 descriptions into a transcript that
    is resent every turn.
    """
    source = io.open(os.path.join(HANDLERS, "robots.py"), encoding="utf-8").read()
    match = re.search(r"^_DETAIL_THRESHOLD\s*=\s*(\d+)", source, re.M)
    assert match, "robots.py no longer defines _DETAIL_THRESHOLD"
    assert 10 <= int(match.group(1)) <= 60


# ── The verifier that broke what it verified ──────────────────────────────────


class _FakeProbe:
    def __init__(self) -> None:
        self.position = [0.0, 0.0, 50.0]
        self.poses_set: list[object] = []
        self.velocities_set: list[object] = []

    def set_pose(self, position=None, orientation=None) -> None:  # noqa: ANN001
        self.poses_set.append(position)
        self.position = list(position)

    def set_velocity(self, linear=None, angular=None) -> None:  # noqa: ANN001
        self.velocities_set.append((linear, angular))


class _FakeStage:
    def __init__(self, has_probe: bool) -> None:
        self.has_probe = has_probe
        self.removed: list[str] = []

    def GetPrimAtPath(self, path: str):  # noqa: N802 - USD's own spelling
        stage = self

        class _Prim:
            def IsValid(self) -> bool:  # noqa: N802
                return stage.has_probe

        return _Prim()

    def RemovePrim(self, path: str) -> None:  # noqa: N802
        self.removed.append(path)


class _FakeScene:
    def __init__(self, has_probe: bool = False) -> None:
        self.stage = _FakeStage(has_probe)
        self.spawned: list[str] = []
        self.steps = 0

    def spawn_rigid(self, path: str, **_kwargs: object) -> _FakeProbe:
        self.spawned.append(path)
        self.stage.has_probe = True
        return _FakeProbe()

    def step(self, count: int = 1, **_kwargs: object) -> None:
        self.steps += count


def test_the_physics_probe_is_never_removed() -> None:
    """Removing the probe de-registered every articulation in the scene.

    `RemovePrim` on a body PhysX has registered tears down the physics tensor
    view. Joint drives stop being serviced, a closed gripper relaxes, and the
    held object falls. So `verify_grasp` — which calls `physics_running` first —
    destroyed the grasp it was asked to verify and then truthfully reported the
    object was on the ground. A verifier that breaks what it measures is worse
    than none, because its answer looks like evidence.
    """
    from simliverse_sim import assertions

    scene = _FakeScene()
    assertions._drop_probe(scene)  # type: ignore[arg-type]

    assert scene.stage.removed == [], (
        f"the probe removed {scene.stage.removed}; removing a registered body "
        f"invalidates the physics view and drops whatever is being held"
    )


def test_the_probe_lives_outside_world() -> None:
    """Everything the agent lists, searches or clears is scoped to `/World`.

    A probe inside it shows up in scene listings, survives a "clear the scene",
    and can be mistaken for part of the task — a stray sphere is exactly what a
    ball-grasping task is looking for.
    """
    from simliverse_sim import assertions

    scene = _FakeScene()
    assertions._drop_probe(scene)  # type: ignore[arg-type]

    assert scene.spawned == [assertions._PROBE_PATH]
    assert not assertions._PROBE_PATH.startswith("/World")


def test_a_reused_probe_is_dropped_from_rest() -> None:
    """Otherwise each probe falls further than the last.

    A reused body keeps the speed it reached last time, so the measured fall
    grows on every verification and a fixed threshold silently becomes a moving
    one.
    """
    from simliverse_sim import assertions

    scene = _FakeScene(has_probe=True)
    probe = _FakeProbe()
    monkey = lambda _s: probe  # noqa: E731
    original = assertions._existing_probe
    assertions._existing_probe = monkey  # type: ignore[assignment]
    try:
        assertions._drop_probe(scene)  # type: ignore[arg-type]
    finally:
        assertions._existing_probe = original  # type: ignore[assignment]

    assert probe.velocities_set == [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    assert probe.poses_set == [(0.0, 0.0, assertions._PROBE_HEIGHT)]


# ── The second physics scene that stalls the simulator ────────────────────────


class _FakePrim:
    def __init__(self, path: str, is_scene: bool) -> None:
        self._path = path
        self._is_scene = is_scene

    def IsA(self, _cls: object) -> bool:  # noqa: N802 - USD's own spelling
        return self._is_scene

    def GetPath(self) -> str:  # noqa: N802
        return self._path


class _TraversableStage:
    def __init__(self, prims: list[_FakePrim]) -> None:
        self._prims = prims
        self.removed: list[str] = []

    def Traverse(self):  # noqa: N802
        return list(self._prims)

    def RemovePrim(self, path: str) -> None:  # noqa: N802
        self.removed.append(path)


class _SceneWithStage:
    """Just enough of `Scene` to exercise the duplicate sweep."""

    def __init__(self, stage: _TraversableStage, playing: bool = False) -> None:
        self.stage = stage
        self._playing = playing

    def is_playing(self) -> bool:
        return self._playing


def _sweeper(stage: _TraversableStage, keep: str, *, playing: bool = False) -> list[str]:
    from simliverse_sim.scene import Scene

    holder = _SceneWithStage(stage, playing)
    return Scene._remove_duplicate_physics_scenes(holder, keep=keep)


def test_a_second_physics_scene_is_removed() -> None:
    """Two scenes stall PhysX, and nothing raises to say so.

    Kit reports it in the status bar — "per scene step is not yet supported" —
    then rendering drops to 0 FPS and every MCP call blocks forever. The process
    keeps its heartbeat, so it looks busy rather than broken. That cost a worker
    mid-task, with a built scene that could no longer be measured.
    """
    stage = _TraversableStage(
        [
            _FakePrim("/World/PhysicsScene", True),
            _FakePrim("/PhysicsScene", True),
            _FakePrim("/World/Franka", False),
        ]
    )
    removed = _sweeper(stage, "/World/PhysicsScene")

    assert removed == ["/PhysicsScene"]
    assert stage.removed == ["/PhysicsScene"]


def test_the_kept_scene_survives() -> None:
    """Removing all of them would be a different way to have no physics."""
    stage = _TraversableStage([_FakePrim("/World/PhysicsScene", True)])
    assert _sweeper(stage, "/World/PhysicsScene") == []
    assert stage.removed == []


def test_non_physics_prims_are_untouched() -> None:
    """The sweep runs on every `configure_physics`, including mid-scene."""
    stage = _TraversableStage(
        [_FakePrim("/World/Table", False), _FakePrim("/World/Ball", False)]
    )
    assert _sweeper(stage, "/World/PhysicsScene") == []
    assert stage.removed == []


def test_a_duplicate_is_left_alone_while_physics_is_running() -> None:
    """Removing a registered prim mid-run costs the session, not the prim.

    `RemovePrim` on a body PhysX has registered tears down the simulation view.
    That is the same failure `_drop_probe` used to cause, and the first version
    of this sweep reintroduced it: it removed a physics scene without checking
    the timeline, the view was invalidated, and the worker had to be destroyed
    with the task unfinished.

    A duplicate scene stalls stepping, which is bad. Tearing down physics to
    remove it is worse: a stalled simulator can still be stopped and
    reconfigured, a dead one cannot. So while the timeline is playing this
    reports and leaves it, and the removal happens on the next
    `configure_physics` — which is called at the start of essentially every
    task, with the timeline stopped.
    """
    stage = _TraversableStage(
        [_FakePrim("/World/PhysicsScene", True), _FakePrim("/PhysicsScene", True)]
    )
    removed = _sweeper(stage, "/World/PhysicsScene", playing=True)

    assert removed == []
    assert stage.removed == [], "a running simulation must not have prims removed under it"


# ── The World that could not be discarded ─────────────────────────────────────


class _StubbornWorld:
    """A `World` whose own teardown raises, which is the real behaviour."""

    _instance = object()
    cleared = False

    @classmethod
    def clear_instance(cls) -> None:
        raise RuntimeError(
            "Accessed invalid expired 'PhysicsScene' prim </World/PhysicsScene>"
        )


class _CooperativeWorld:
    _instance = object()

    @classmethod
    def clear_instance(cls) -> None:
        _CooperativeWorld._instance = None


def test_a_world_that_refuses_to_clear_is_cleared_anyway() -> None:
    """`clear_instance()` touches the world it is discarding, and can raise.

    When the world holds a deleted prim, the supported teardown fails with the
    very error it was called to escape. The singleton then survives, every later
    `Scene.get()` hits the same corpse, and the session is finished — no
    recovery from outside, on a cloud worker that means destroying it.

    This is reachable from the agent's own tool surface: `clear_scene` deletes
    `/World` wholesale and is offered under AUTHORING. One call cost a worker.
    """
    from simliverse_sim._compat import _drop_world_singleton

    _drop_world_singleton(_StubbornWorld)
    assert _StubbornWorld._instance is None


def test_the_supported_path_is_used_when_it_works() -> None:
    """Reaching into a private is the fallback, not the habit."""
    from simliverse_sim._compat import _drop_world_singleton

    _CooperativeWorld._instance = object()
    _drop_world_singleton(_CooperativeWorld)
    assert _CooperativeWorld._instance is None


# ── What the asset actually is ────────────────────────────────────────────────

import importlib.util as _il  # noqa: E402
import os as _os  # noqa: E402

# Loaded straight from the file. The extension package cannot be imported
# outside Isaac Sim -- its `__init__` reaches for `carb` -- and `asset_notes`
# has no imports of its own precisely so this works.
_spec = _il.spec_from_file_location(
    "asset_notes",
    _os.path.join(
        _os.path.dirname(__file__), "..", "isaac.sim.mcp_extension",
        "isaac_sim_mcp_extension", "handlers", "asset_notes.py",
    ),
)
_notes = _il.module_from_spec(_spec)
_spec.loader.exec_module(_notes)
_what_this_actually_is = _notes._what_this_actually_is


def test_a_gripper_only_asset_is_called_an_end_effector():
    note = _what_this_actually_is(["finger_joint1", "finger_joint2"], "lite6_gripper")

    assert "end effector" in note
    assert "lite6_gripper" in note


def test_an_arm_with_no_gripper_says_it_cannot_hold_anything():
    note = _what_this_actually_is(
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], "lite6"
    )

    assert "no gripper" in note


def test_an_arm_with_a_gripper_needs_no_note():
    cobotta = [
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
        "finger_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint",
        "right_outer_knuckle_joint", "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]

    assert _what_this_actually_is(cobotta, "cobottapro900") == ""


def test_a_franka_needs_no_note():
    franka = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]

    assert _what_this_actually_is(franka, "franka") == ""


def test_an_unknown_robot_with_no_joints_is_not_described_at_all():
    assert _what_this_actually_is([], "mystery") == ""


def test_spawn_robot_accepts_a_mounting_yaw():
    """
    A six-axis arm's base joint has a seam, and work that sits across it is
    reachable only the long way round or over the shoulder. Spawning every
    arm at yaw 0 put the seam wherever the sketch left it; one layout put a
    pallet slot at +174 degrees on a KR210 (seam at +/-185) and the set-down
    had no route the controller was allowed to drive. The builder mounts the
    arm facing the work, and needs this to do it.
    """
    import ast, os
    path = os.path.join(os.path.dirname(__file__), "..", "simliverse_sim", "robots", "library.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "spawn_robot")
    assert "yaw" in {a.arg for a in fn.args.kwonlyargs}
    src = ast.unparse(fn)
    assert "AddRotateZOp" in src, "the yaw must be applied to the prim, not just accepted"
