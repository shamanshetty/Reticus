from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SERVERS: Dict[Tuple[str, int], Tuple[threading.Thread, Any]] = {}
_ERRORS: Dict[Tuple[str, int], BaseException] = {}


def _precheck_port(host: str, port: int) -> Optional[OSError]:
    """
    Try to bind the port ourselves before handing it to uvicorn. uvicorn logs
    the real bind OSError internally but re-raises a bare SystemExit(3) from
    Server.run(), which loses the actual reason ("address already in use",
    etc.) by the time it reaches our caller - so we capture it here instead.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return None
    except OSError as exc:
        return exc
    finally:
        probe.close()


def get_or_start_server(
    project: str,
    port: int,
    root_dir: Path,
    app: Any,
) -> None:
    """
    Get existing server for (project, port) or start new one.
    Server runs in non-daemon thread so it survives parent process exit.
    """
    key = (project, port)
    if key in _SERVERS:
        thread, _ = _SERVERS[key]
        if thread.is_alive():
            return  # Server already running

    _ERRORS.pop(key, None)
    bind_error = _precheck_port("127.0.0.1", port)
    if bind_error is not None:
        _ERRORS[key] = bind_error
        return

    # Start new server in non-daemon thread
    import uvicorn

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        try:
            server.run()
        except SystemExit as exc:
            # uvicorn swallows the bind OSError and re-raises a bare
            # SystemExit(code); the pre-check above handles the common case.
            _ERRORS[key] = RuntimeError(
                "dashboard server exited during startup (code %s) - the port is "
                "almost certainly already in use by another process" % exc.code
            )
        except BaseException as exc:
            _ERRORS[key] = exc

    thread = threading.Thread(target=_serve, daemon=False)
    thread.start()
    _SERVERS[key] = (thread, server)


def check_started(project: str, port: int, timeout: float = 1.2) -> Optional[BaseException]:
    """
    Wait briefly for the server to either come up or fail to bind, and return
    the bind exception if it failed. Used so callers can warn the user loudly
    instead of silently running with no reachable dashboard.
    """
    import time

    key = (project, port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if key in _ERRORS:
            return _ERRORS[key]
        thread, _ = _SERVERS.get(key, (None, None))
        if thread is not None and not thread.is_alive():
            return _ERRORS.get(key) or RuntimeError("dashboard server thread exited unexpectedly")
        time.sleep(0.05)
    return _ERRORS.get(key)


def shutdown_all() -> None:
    """
    Shutdown all running servers. Mainly for tests.
    """
    for (proj, port), (thread, server) in _SERVERS.items():
        try:
            server.should_exit = True
        except Exception:
            pass
