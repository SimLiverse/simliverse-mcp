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

"""Socket connection to the Isaac Sim extension server."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("IsaacMCPServer")

DEFAULT_PORT = 8766

# How long to wait for Isaac to answer one command.
#
# This has to exceed the extension's own dispatch bound (600s), or the extension
# never gets to send the error it prepares: whoever gives up first decides what
# the caller sees, and the caller saw `Exception("No data received")` with no
# command, no cause and no elapsed time. That is what happened to a 140-second
# controller replay -- the sidecar timed out at 300s while Isaac was still
# running it, and the delivery reported a communication failure for a command
# that may well have succeeded.
#
# Long by design. A physics replay, a large USD load, or a 60-turn debugging
# session are all legitimately slow, and a short bound here does not make them
# faster -- it only removes the answer.
READ_TIMEOUT = float(os.environ.get("SIMLIVERSE_MCP_READ_TIMEOUT", "900"))


class IsaacCommandError(Exception):
    """A handler inside Isaac Sim returned an error.

    Carries the full payload — `message`, and where the handler provided them,
    `traceback`, `stdout` and `stderr`. The connection is still healthy; only the
    command failed.
    """

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.payload: Dict[str, Any] = payload or {"message": message}


@dataclass
class IsaacConnection:
    """Manages a persistent TCP socket connection to the Isaac Sim extension.

    **One socket, one command at a time.** `send_command` writes a request and
    then reads until the accumulated bytes parse as JSON — there is no request id
    in the wire format, so a reply is matched to a request purely by who calls
    `recv` first. Two threads sharing this socket therefore do not merely
    interleave, they corrupt each other: `{"a":1}{"b":2}` never parses, so the
    reader accumulates both replies and blocks until READ_TIMEOUT before
    reporting an incomplete response for a command that actually succeeded.

    That was latent while the agent harness delegated sequentially. It is not
    any more: the Claude Agent SDK can issue several `Task` calls in one turn, so
    `scene-builder` and `verifier` may hold this connection concurrently, and
    FastMCP runs these sync tool functions on real threadpool threads.

    `_lock` therefore covers the whole request/response exchange rather than the
    write alone. This costs no throughput: `socket_server.py` already serialises
    every command behind its own `_dispatch_lock`, and the work runs on Kit's
    main thread because USD and PhysX require it. Isaac was never going to
    execute two of these at once — the only thing concurrency bought here was the
    corruption.

    It is an `RLock` because `send_command` calls `connect` while holding it.
    """

    host: str = "localhost"
    port: int = 0

    def __post_init__(self):
        if self.port == 0:
            self.port = int(os.environ.get("ISAAC_MCP_PORT", DEFAULT_PORT))

    sock: Optional[socket.socket] = field(default=None, repr=False)
    # `compare=False` keeps dataclass equality on host/port/sock, where it was.
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def connect(self) -> bool:
        with self._lock:
            if self.sock:
                return True
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                logger.info(f"Connected to Isaac at {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Isaac: {e}")
                self.sock = None
                return False

    def disconnect(self) -> None:
        with self._lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception as e:
                    logger.error(f"Error disconnecting: {e}")
                finally:
                    self.sock = None

    def receive_full_response(self, sock: socket.socket, buffer_size: int = 16384) -> bytes:
        chunks = []
        sock.settimeout(READ_TIMEOUT)
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise Exception("Connection closed before receiving any data")
                        break
                    chunks.append(chunk)
                    try:
                        data = b"".join(chunks)
                        json.loads(data.decode("utf-8"))
                        return data
                    except json.JSONDecodeError:
                        continue
                except socket.timeout:
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError):
                    raise
        except socket.timeout:
            pass

        if chunks:
            data = b"".join(chunks)
            try:
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                raise Exception(
                    f"Incomplete JSON response received ({len(data)} bytes) after "
                    f"{READ_TIMEOUT:.0f}s. Isaac started answering and stopped."
                )
        raise Exception(
            f"No data received from Isaac after {READ_TIMEOUT:.0f}s. The command was "
            f"delivered — Isaac simply never answered within that window, which for a "
            f"long replay or a large asset load can mean it is still working. "
            f"DO NOT assume the scene is unchanged and DO NOT retry blindly: the same "
            f"command may already be running, and running it twice is worse than "
            f"waiting. Read the scene back first, and check the Kit log."
        )

    def send_command(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Contention is reported rather than silent. A caller can wait up to
        # READ_TIMEOUT here, and "the verifier is queued behind a 4-minute
        # controller replay" needs to be visible in the log — otherwise it
        # presents as the sidecar having hung.
        if not self._lock.acquire(blocking=False):
            logger.info(
                "Isaac connection busy; %s is queued behind a command already in "
                "flight.",
                command_type,
            )
            self._lock.acquire()
        try:
            return self._send_command_locked(command_type, params)
        finally:
            self._lock.release()

    def _send_command_locked(
        self, command_type: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """The exchange itself. Callers must hold `self._lock`."""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")

        command = {"type": command_type, "params": params or {}}
        try:
            self.sock.sendall(json.dumps(command).encode("utf-8"))
            self.sock.settimeout(READ_TIMEOUT)
            response_data = self.receive_full_response(self.sock)
            response = json.loads(response_data.decode("utf-8"))

            if response.get("status") == "error":
                # A handler-level failure (bad prim path, a NameError in control
                # code) says nothing about the socket — keep the connection and
                # carry the full payload so the caller still gets the traceback
                # and captured output.
                raise IsaacCommandError(
                    response.get("message", "Unknown error from Isaac"), response
                )
            return response.get("result", {})
        except IsaacCommandError:
            raise
        except socket.timeout:
            self.sock = None
            raise Exception("Timeout waiting for Isaac response")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {e}")
        except json.JSONDecodeError as e:
            self.sock = None
            raise Exception(f"Invalid response from Isaac: {e}")
        except Exception as e:
            self.sock = None
            raise Exception(f"Communication error with Isaac: {e}")


_isaac_connection: Optional[IsaacConnection] = None

# Guards the singleton itself, not the socket. Without it the check-then-act in
# `get_isaac_connection` lets two threads each build a connection: the loser's
# socket is dropped from the global while still open, leaking an FD and a
# half-used session on the extension side.
_singleton_lock = threading.Lock()


def get_isaac_connection() -> IsaacConnection:
    """Get or create a persistent Isaac connection singleton."""
    global _isaac_connection
    with _singleton_lock:
        if _isaac_connection is not None:
            return _isaac_connection
        connection = IsaacConnection(host="localhost")
        if not connection.connect():
            raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
        _isaac_connection = connection
        return _isaac_connection


def reset_isaac_connection() -> None:
    """Disconnect and clear the global connection (used during shutdown)."""
    global _isaac_connection
    with _singleton_lock:
        if _isaac_connection:
            _isaac_connection.disconnect()
            _isaac_connection = None
