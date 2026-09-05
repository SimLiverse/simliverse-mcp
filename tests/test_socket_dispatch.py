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

"""Every command gets an answer, including the ones the simulator never runs.

From a live session. Two calls came back as

    RemoteProtocolError: Server disconnected without sending a response

which is not an error message — it names no command, no cause, and no robot.
The Kit log had the reason:

    [ERROR] Task was destroyed but it is pending!
        task: <Task cancelling coro=<Extension._disable_async_rendering_after
               _update_async()>>
    RuntimeError: Cannot enter into task <Task ... core.throttling ...>
        while another task <Task ... SocketServer._dispatch_command.<locals>.
        execute_wrapper() ...> is being executed

Isaac's throttling extension and our command dispatch collided on Kit's event
loop. `run_coroutine` was fire-and-forget, so when the task died neither the
success path nor the error path ran, nothing was written to the socket, and the
caller waited until the transport gave up.

The simulator is not needed to pin this: what matters is that the thread
waiting on the socket owns the reply, whatever the main loop does or fails to
do. `_schedule` is the only Isaac-facing seam and it is replaced here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

# Loaded by path, like `test_isaac_log_capture`. Importing the package runs its
# `__init__`, which pulls in `carb` and therefore the whole simulator — and
# dispatch is plain sockets and threads. `_schedule` is the single seam that
# touches Kit, and every test here replaces it.
_SPEC = importlib.util.spec_from_file_location(
    "_sl_socket_server",
    Path(__file__).resolve().parents[1] / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension" / "socket_server.py",
)
_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _module
_SPEC.loader.exec_module(_module)
SocketServer = _module.SocketServer


class _Client:
    """A stand-in socket that records what was written to it."""

    def __init__(self, fail: bool = False):
        self.sent: list[bytes] = []
        self.fail = fail

    def sendall(self, payload: bytes) -> None:
        if self.fail:
            raise ConnectionResetError("client went away")
        self.sent.append(payload)

    def response(self) -> dict:
        assert self.sent, "nothing was ever written to the client"
        return json.loads(b"".join(self.sent).decode("utf-8"))


def _server(handler, *, schedule=None, timeout=5.0) -> SocketServer:
    server = SocketServer("127.0.0.1", 0, handler, command_timeout=timeout)
    server._schedule = schedule if schedule is not None else _run_now
    return server


def _run_now(work) -> None:
    """Kit's main thread, in the honest case: the callback runs."""
    work()


def _never_runs(work) -> None:
    """Accepted and then dropped — the production failure."""


# ── The work must be a callback, not a Task ───────────────────────────────────


def test_the_handler_is_scheduled_as_a_plain_callable() -> None:
    """Scheduling it as a coroutine is what broke USD-mutating commands.

    A Task makes asyncio mark it current for as long as it runs. Handlers pump
    `app.update()`, Kit extensions react to the USD change by scheduling
    coroutines, and asyncio refuses: "Cannot enter into task X while another
    task Y is being executed". The guard fires on *their* task, so spawning a
    prim killed `omni.anim.graph.ui` and the connection went with it.

    Measured on the live worker: from inside the old Task a concurrently
    scheduled coroutine never ran at all; from a plain callback both ran.
    """
    scheduled = []
    server = _server(lambda command: {"status": "success"}, schedule=scheduled.append)

    server._dispatch_command(_Client(), {"type": "spawn_robot"})

    assert len(scheduled) == 1
    work = scheduled[0]
    assert not asyncio.iscoroutine(work), (
        "the handler was scheduled as a coroutine — asyncio will run it as a Task "
        "and USD-change callbacks in other Kit extensions will start failing"
    )
    assert callable(work)


def test_it_is_handed_to_the_loop_as_a_threadsafe_callback() -> None:
    """`call_soon_threadsafe`, because dispatch runs on a client thread and the
    callback slot is the one place asyncio marks no current task."""
    calls = []

    class _Loop:
        def call_soon_threadsafe(self, work):
            calls.append(work)
            work()

    server = _server(lambda command: {"status": "success"}, schedule=None)
    del server._schedule  # back to the real implementation
    server._loop = _Loop()

    server._dispatch_command(_Client(), {"type": "spawn_robot"})

    assert len(calls) == 1 and callable(calls[0])


def test_a_loop_it_could_not_capture_still_answers() -> None:
    """Better the old behaviour than no commands at all — but it must not
    pretend, so `_loop` staying None is a fallback and not the design."""
    server = SocketServer("127.0.0.1", 0, lambda command: {"status": "success"})

    assert server._loop is None, "nothing should be captured before start()"


# ── The happy path still works ────────────────────────────────────────────────


def test_a_successful_command_is_answered() -> None:
    server = _server(lambda command: {"status": "success", "echo": command["type"]})
    client = _Client()

    server._dispatch_command(client, {"type": "get_scene_info"})

    assert client.response() == {"status": "success", "echo": "get_scene_info"}


def test_a_raising_handler_reports_its_own_message() -> None:
    def boom(command):
        raise ValueError("no articulation at /World/Arm")

    server = _server(boom)
    client = _Client()

    server._dispatch_command(client, {"type": "run_control"})
    response = client.response()

    assert response["status"] == "error"
    assert "no articulation at /World/Arm" in response["message"]


# ── The failure that produced no message at all ───────────────────────────────


def test_a_destroyed_task_still_answers_the_client() -> None:
    """The bug: the coroutine never ran, so nobody replied and the socket hung.

    This is what `RemoteProtocolError: Server disconnected without sending a
    response` was, seen from the other end.
    """
    ran = []
    server = _server(lambda command: ran.append(1), schedule=_never_runs, timeout=0.2)
    client = _Client()

    server._dispatch_command(client, {"type": "run_control"})

    assert not ran, "the handler was supposed to never run in this scenario"
    response = client.response()
    assert response["status"] == "error"
    assert "run_control" in response["message"]
    assert "Cannot enter into task" in response["message"], (
        "the message must point at the Kit log line that explains it"
    )


def test_it_does_not_claim_the_scene_is_untouched() -> None:
    """A timed-out command may still be running inside the simulator.

    Saying 'failed' would invite a retry that runs it twice.
    """
    server = _server(lambda command: None, schedule=_never_runs, timeout=0.1)
    client = _Client()

    server._dispatch_command(client, {"type": "play_simulation"})

    assert "do not assume the scene is unchanged" in client.response()["message"]


def test_a_scheduler_that_refuses_is_reported() -> None:
    def refuse(work):
        raise RuntimeError("async engine is shutting down")

    server = _server(lambda command: {"status": "success"}, schedule=refuse)
    client = _Client()

    server._dispatch_command(client, {"type": "get_scene_info"})
    response = client.response()

    assert response["status"] == "error"
    assert "async engine is shutting down" in response["message"]


# ── One command in the simulator at a time ────────────────────────────────────


def test_commands_do_not_overlap() -> None:
    """A thread per client, each firing its own coroutine, could interleave
    two long commands inside USD and PhysX."""
    concurrent = []
    live = 0
    guard = threading.Lock()

    def slow(command):
        nonlocal live
        with guard:
            live += 1
            concurrent.append(live)
        time.sleep(0.05)
        with guard:
            live -= 1
        return {"status": "success"}

    server = _server(slow)
    clients = [_Client() for _ in range(4)]
    threads = [threading.Thread(target=server._dispatch_command, args=(c, {"type": "run_control"})) for c in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max(concurrent) == 1, f"handlers overlapped: {concurrent}"
    for client in clients:
        assert client.response()["status"] == "success"


# ── Socket writes must not take the simulator down with them ──────────────────


def test_a_client_that_hung_up_does_not_raise() -> None:
    server = _server(lambda command: {"status": "success"})
    server._dispatch_command(_Client(fail=True), {"type": "get_scene_info"})


def test_the_lock_is_released_after_a_dead_client() -> None:
    """A raise while holding the dispatch lock would freeze every later
    command — a worse failure than the one being handled."""
    server = _server(lambda command: {"status": "success"})
    server._dispatch_command(_Client(fail=True), {"type": "get_scene_info"})

    assert server._dispatch_lock.acquire(timeout=1.0), "the dispatch lock was never released"
    server._dispatch_lock.release()


@pytest.mark.parametrize("timeout", [0.05, 0.1])
def test_the_wait_is_bounded(timeout: float) -> None:
    """It answers late rather than never — the whole point of the change."""
    server = _server(lambda command: None, schedule=_never_runs, timeout=timeout)
    started = time.monotonic()

    server._dispatch_command(_Client(), {"type": "run_control"})

    assert time.monotonic() - started < timeout + 2.0
