# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
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

"""Concurrent use of one `IsaacConnection`.

The sidecar holds a single socket to the Isaac extension and the wire format
carries no request id, so a reply belongs to whoever calls `recv` first. These
tests pin the invariant that makes that safe: one command occupies the socket at
a time.

This mattered less when the agent harness delegated sequentially. The Claude
Agent SDK can run `scene-builder` and `verifier` concurrently, and FastMCP runs
the sync tool functions on threadpool threads, so two commands really can arrive
together now.

Every test here fails without the lock in `IsaacConnection`.
"""

import json
import threading
import time

from isaac_mcp import connection as conn_mod
from isaac_mcp.connection import IsaacConnection


class SharedStreamSocket:
    """A fake socket modelling the hazard rather than a well-behaved server.

    The extension answers in the order it was asked and writes into one stream.
    `recv` pops from that same shared stream, so if two senders get their
    requests in before either reads, the first reader drains both replies and
    the second finds nothing — which is the corruption, not a timeout.

    `delay` widens the send→recv window so an unserialised caller reliably loses
    the race; the assertions themselves do not depend on timing.
    """

    def __init__(self, delay: float = 0.02) -> None:
        self._delay = delay
        self._outbuf = bytearray()
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def settimeout(self, _timeout: float) -> None:
        pass

    def close(self) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        command = json.loads(data.decode("utf-8"))
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            # The extension answers in arrival order, echoing which command it
            # served so a mismatched reply is identifiable rather than merely
            # malformed.
            reply = json.dumps(
                {"status": "success", "result": {"served": command["type"]}}
            ).encode("utf-8")
            self._outbuf.extend(reply)
        time.sleep(self._delay)

    def recv(self, buffer_size: int = 16384) -> bytes:
        deadline = time.monotonic() + 2.0
        while True:
            with self._guard:
                if self._outbuf:
                    chunk = bytes(self._outbuf[:buffer_size])
                    del self._outbuf[: len(chunk)]
                    self.active -= 1
                    return chunk
            if time.monotonic() > deadline:
                with self._guard:
                    self.active -= 1
                return b""
            time.sleep(0.001)


def _run_concurrently(connection: IsaacConnection, commands: list[str]) -> dict:
    """Fire every command from its own thread; collect results by command."""
    results: dict = {}
    errors: dict = {}
    lock = threading.Lock()

    def call(command_type: str) -> None:
        try:
            result = connection.send_command(command_type)
            with lock:
                results[command_type] = result
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            with lock:
                errors[command_type] = exc

    threads = [threading.Thread(target=call, args=(c,)) for c in commands]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors, f"commands raised: {errors}"
    return results


def test_only_one_command_occupies_the_socket_at_a_time() -> None:
    """The invariant the wire format depends on."""
    connection = IsaacConnection(host="localhost", port=1)
    fake = SharedStreamSocket()
    connection.sock = fake

    _run_concurrently(connection, [f"scene.get_{i}" for i in range(8)])

    assert fake.max_active == 1, (
        f"{fake.max_active} commands shared the socket concurrently; replies are "
        f"matched by recv order, so they can be delivered to the wrong caller"
    )


def test_each_caller_receives_its_own_reply() -> None:
    """The user-visible consequence: the verifier must not read the builder's answer."""
    connection = IsaacConnection(host="localhost", port=1)
    connection.sock = SharedStreamSocket()

    commands = ["control.run", "scene.observe", "robot.inspect", "scene.capture"]
    results = _run_concurrently(connection, commands)

    assert set(results) == set(commands)
    for command_type, result in results.items():
        assert result["served"] == command_type, (
            f"{command_type} received the reply to {result['served']}"
        )


def test_a_queued_command_is_reported() -> None:
    """A caller can wait a long time here; that must be visible in the log."""
    connection = IsaacConnection(host="localhost", port=1)
    connection.sock = SharedStreamSocket(delay=0.05)

    records: list[str] = []
    handler_lock = threading.Lock()

    class Collector:
        def handle(self, record) -> None:  # noqa: ANN001 — logging.Handler protocol
            with handler_lock:
                records.append(record.getMessage())

        level = 0

        def acquire(self) -> None: ...
        def release(self) -> None: ...
        def createLock(self) -> None: ...

    collector = Collector()
    conn_mod.logger.addHandler(collector)  # type: ignore[arg-type]
    previous_level = conn_mod.logger.level
    conn_mod.logger.setLevel(10)
    try:
        _run_concurrently(connection, ["control.run", "scene.observe"])
    finally:
        conn_mod.logger.removeHandler(collector)  # type: ignore[arg-type]
        conn_mod.logger.setLevel(previous_level)

    assert any("queued" in message for message in records), (
        f"a queued command was not reported; log was: {records}"
    )


def test_the_singleton_is_created_once_under_concurrency() -> None:
    """Two racing callers must not each build a connection and leak one."""
    built: list[IsaacConnection] = []
    build_lock = threading.Lock()

    class CountingConnection(IsaacConnection):
        def connect(self) -> bool:
            with build_lock:
                built.append(self)
            self.sock = SharedStreamSocket()
            time.sleep(0.02)
            return True

    original_cls = conn_mod.IsaacConnection
    conn_mod.reset_isaac_connection()
    conn_mod.IsaacConnection = CountingConnection  # type: ignore[misc]
    try:
        seen: list[IsaacConnection] = []
        seen_lock = threading.Lock()

        def get() -> None:
            instance = conn_mod.get_isaac_connection()
            with seen_lock:
                seen.append(instance)

        threads = [threading.Thread(target=get) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert len(built) == 1, f"{len(built)} connections were opened, expected 1"
        assert len({id(s) for s in seen}) == 1, "callers received different singletons"
    finally:
        conn_mod.IsaacConnection = original_cls  # type: ignore[misc]
        conn_mod.reset_isaac_connection()
