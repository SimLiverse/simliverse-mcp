"""`capture_view` must never silently discard the aim it was given.

An agent's only eyes on the cell are this tool. The last time an argument to
it was quietly dropped -- `eye`/`target` against a server that wanted
`position`/`look_at` -- a whole session was spent studying the default camera
and believing it was looking where it had asked to look. Nothing errored;
the pictures were simply of somewhere else.

The same door was open a second time from inside the handler: given both a
`camera_path` and a `position`/`look_at`, it took the path and threw the aim
away. On a first run the named prim does not exist yet, so the viewport was
pointed at nothing and every capture died on the frame-wait timeout -- with a
message about the capture, not about the camera.

Driven off the source, since the handler cannot be imported without Isaac.
"""

import ast
import os

HANDLER = os.path.join(
    os.path.dirname(__file__), "..", "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension", "handlers", "control.py",
)


def _capture_view():
    with open(HANDLER) as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "capture_view")
    return fn


def test_an_aim_is_honoured_even_when_a_camera_path_is_given():
    """The branch on the aim must come first, so `camera_path` cannot shadow it."""
    fn = _capture_view()
    branches = [n for n in ast.walk(fn)
                if isinstance(n, ast.If) and "viewport.camera_path" in ast.unparse(n)]
    assert branches, "capture_view no longer chooses a camera in an if/elif"
    first = min(branches, key=lambda n: n.lineno)
    names = {n.id for n in ast.walk(first.test) if isinstance(n, ast.Name)}
    assert names == {"position", "look_at"}, (
        "the first camera branch must be the one that aims. Testing "
        f"{sorted(names)} first lets a camera_path swallow the aim, which is "
        "how an agent ends up looking somewhere it did not ask to look."
    )
    # And the aim must actually be applied to the requested path.
    aim = next(n for n in ast.walk(first)
               if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_aim_camera")
    assert any(getattr(a, "id", "") == "target_path" for a in aim.args), (
        "_aim_camera must aim the camera at the path the caller named"
    )


def test_a_camera_path_that_does_not_exist_is_reported_not_timed_out():
    """
    Pointing the viewport at a missing prim renders nothing, and the failure
    then surfaces 240 frames later as a capture timeout -- which sends anyone
    debugging it to the renderer instead of to their typo.
    """
    src = ast.unparse(_capture_view())
    assert "No camera at" in src, (
        "a camera_path naming no prim must say so, not fall through to the "
        "frame-wait and report a capture timeout"
    )


def test_the_frame_wait_outlasts_a_cold_first_capture():
    """
    40 frames was not enough for the first shot through a camera created on a
    cold worker: RTX warms the new view before it hands over a frame. Waiting
    longer costs a second; a blind agent costs a session.
    """
    fn = _capture_view()
    waits = [n.args[0].value for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "range"
             and n.args and isinstance(n.args[0], ast.Constant)]
    assert max(waits) >= 200, f"frame-wait loops are {waits}, too short for a cold capture"
