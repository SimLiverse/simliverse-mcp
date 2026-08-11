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

"""TCP socket server for Isaac Sim MCP — connection and dispatch logic."""

from __future__ import annotations

import json
import socket
import threading
import time
import traceback
from typing import Any, Callable, Dict


class SocketServer:
    """Manages a TCP socket server that accepts JSON commands and returns responses.

    Parameters
    ----------
    host:
        Hostname or IP address to bind to.
    port:
        Port number to listen on.
    command_handler:
        Callable invoked with the parsed command dict; must return a response dict.
    command_timeout:
        Seconds to wait for the main thread to finish one command before
        answering the client with an error instead. Generous: an asset load or a
        long physics rollout is legitimately slow, and a wrong answer here turns
        a working command into a spurious failure. The value exists so that a
        command which will *never* finish is reported rather than hung on.
    """

    def __init__(
        self,
        host: str,
        port: int,
        command_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        command_timeout: float = 600.0,
    ) -> None:
        self.host = host
        self.port = port
        self._command_handler = command_handler
        self.command_timeout = command_timeout
        self.running: bool = False
        self._socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._dispatch_lock = threading.Lock()
        # Captured on the main thread in `start()`. A client thread cannot find
        # Kit's loop for itself — `asyncio.get_event_loop()` off the main thread
        # raises "There is no current event loop in thread ...".
        self._loop: Any = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind the socket and start the background accept loop."""
        if self.running:
            return
        self.running = True
        self._loop = self._capture_loop()
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            self._socket.listen(1)
            self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self._server_thread.start()
            print(f"Isaac Sim MCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {e}")
            self.stop()

    def stop(self) -> None:
        """Signal the server to stop and close the socket."""
        self.running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        self._server_thread = None
        print("Isaac Sim MCP server stopped")

    # ── Connection handling ────────────────────────────────────────────────────

    def _server_loop(self) -> None:
        self._socket.settimeout(1.0)
        while self.running:
            try:
                client, address = self._socket.accept()
                print(f"Connected to client: {address}")
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
                    time.sleep(0.5)

    def _handle_client(self, client: socket.socket) -> None:
        client.settimeout(None)
        buffer = b""
        try:
            while self.running:
                data = client.recv(16384)
                if not data:
                    break
                buffer += data
                try:
                    command = json.loads(buffer.decode("utf-8"))
                    buffer = b""
                    self._dispatch_command(client, command)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            client.close()

    def _dispatch_command(self, client: socket.socket, command: Dict[str, Any]) -> None:
        """Run one command on Kit's loop and answer the client, always.

        The work still happens on the main thread — USD and PhysX require it —
        but three things about how it was scheduled cost a live debugging
        session each.

        **A destroyed task answered nobody.** `run_coroutine` is fire-and-forget:
        nothing held a reference to the task and nothing waited on it. When Kit
        tore it down mid-flight the `except` below never ran, so no response was
        ever written and the socket simply went quiet. The client saw

            RemoteProtocolError: Server disconnected without sending a response

        which names no command, no robot and no cause. That is what happened
        twice in one session, and what left an agent probing an API it was
        holding correctly. The waiting side now owns the reply: if the main
        thread does not produce one, *this* thread says so.

        **Nothing serialised.** A thread per client, each firing an independent
        coroutine, so two long commands could interleave inside the simulator.

        **Socket writes happened on Kit's loop.** A slow client stalled the
        simulation. They happen here now, off the main thread.

        **The work was a Task.** This is the one that mattered. A handler that
        calls `app.update()` — which `scene.play()`, `stop()` and `step()` all
        do — pumps Kit's update from inside the running command. Any Kit
        extension that reacts by scheduling a coroutine of its own then meets
        asyncio's re-entrancy guard:

            RuntimeError: Cannot enter into task <Task ... core.throttling
                Extension._disable_async_rendering_after_update_async>
              while another task <Task ... execute_wrapper()> is being executed

        The guard fires on *the other extension's* task, not ours, so the
        damage lands somewhere unrelated: `omni.anim.graph.ui` losing
        `_process_usd_change` is why spawning prims dropped connections while
        long computations were fine. It is a USD-mutation bug wearing a
        long-command disguise.

        Nothing about the handler needs a Task — it is synchronous work that
        must happen on the main thread. So it is scheduled as a plain callback.
        `call_soon_threadsafe` runs it in the loop's callback slot, where
        asyncio marks no task as current, and the guard has nothing to trip on.
        Measured on the live worker: with the handler inside a Task a
        concurrently scheduled coroutine never ran; as a plain callback both
        ran.
        """
        done = threading.Event()
        outcome: Dict[str, Any] = {}

        def execute() -> None:
            try:
                outcome["response"] = self._command_handler(command)
            except Exception as exc:
                traceback.print_exc()
                outcome["response"] = {"status": "error", "message": str(exc)}
            finally:
                done.set()

        # One command in the simulator at a time. Held across the wait, not just
        # the scheduling, because the point is that the handlers do not overlap.
        with self._dispatch_lock:
            try:
                self._schedule(execute)
            except Exception as exc:
                traceback.print_exc()
                self._reply(client, {
                    "status": "error",
                    "message": f"Could not schedule the command on Isaac's main loop: {exc}",
                })
                return

            if not done.wait(self.command_timeout):
                self._reply(client, {
                    "status": "error",
                    "message": (
                        f"Command {command.get('type', '?')!r} was scheduled on Isaac's "
                        f"main loop and did not finish within {self.command_timeout:.0f}s. "
                        f"Either it is genuinely long-running, or its task was destroyed "
                        f"by a collision with Kit's own update coroutines — check the Kit "
                        f"log for 'Cannot enter into task'. The simulator may still be "
                        f"executing it; do not assume the scene is unchanged."
                    ),
                })
                return

        self._reply(client, outcome.get("response", {
            "status": "error",
            "message": "The command completed without producing a response.",
        }))

    @staticmethod
    def _capture_loop() -> Any:
        """Kit's event loop, taken while we are still on the main thread.

        Only the main thread can answer this question, which is why it is asked
        at startup and not at dispatch: from a client thread both
        `asyncio.get_event_loop()` and the policy raise "There is no current
        event loop in thread ...".
        """
        try:
            import asyncio

            return asyncio.get_event_loop()
        except Exception as exc:  # pragma: no cover - needs a live Kit
            print(f"Could not capture Isaac's event loop at startup: {exc}")
            return None

    def _schedule(self, work: Callable[[], None]) -> None:
        """Run `work` on Kit's main thread, outside any asyncio Task.

        A plain callback, deliberately — see `_dispatch_command`. The fallback
        exists so that a loop we failed to capture degrades to the old
        behaviour instead of refusing every command; it carries the
        re-entrancy bug with it, so it says so.
        """
        loop = self._loop
        if loop is None:
            print(
                "No captured event loop — falling back to the async engine. "
                "Commands that mutate USD may drop their connection."
            )
            from omni.kit.async_engine import run_coroutine

            async def as_task() -> None:
                work()

            run_coroutine(as_task())
            return

        loop.call_soon_threadsafe(work)

    @staticmethod
    def _reply(client: socket.socket, response: Dict[str, Any]) -> None:
        """Write one response. A client that has gone is not an error here."""
        try:
            client.sendall(json.dumps(response).encode("utf-8"))
        except Exception:
            print("Failed to send response — client disconnected")
