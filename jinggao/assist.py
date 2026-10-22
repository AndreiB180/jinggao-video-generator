"""远程协助: while the customer has it on, the developer can open this computer's 镜稿 page (and only that page)
through our relay server, to see where videos wait and run checks on the customer's own network.

tools/jinggao-assist(.exe) (source in tools/assist) holds an SSH connection to the relay as a forward-only
account; the relay listens on one of its loopback ports and passes each connection to this app's port. It
starts only from the button, stops with the button, when the app quits (its stdin closes) or after 12 hours.
The package build puts the program and its settings (assist.json, no secret) in tools/; without them there
is no button. The key is this computer's own (data/assist/assist.key, made on first use): until the
developer adds its public key on the relay the state is "waiting" and the page shows the key to send.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

from . import paths

TOOLS = paths.ROOT / "tools"
_lock = threading.Lock()
_process: subprocess.Popen | None = None
_state = {"state": "off", "port": None, "message": "", "since": None, "key": ""}


def _program():
    return TOOLS / ("jinggao-assist.exe" if os.name == "nt" else "jinggao-assist")


def available() -> bool:
    return _program().exists() and (TOOLS / "assist.json").exists()


def status() -> dict:
    """state: off, connecting, waiting (key not yet allowed on the relay), on (port = 协助码) or failed."""
    with _lock:
        running = _process is not None and _process.poll() is None
        state = dict(_state)
    if not running and state["state"] in ("connecting", "waiting", "on"):
        state.update(state="off", port=None)
    return state | {"available": available()}


def start(app_port: int) -> dict:
    global _process
    with _lock:
        if _process is None or _process.poll() is not None:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            _process = subprocess.Popen(
                [str(_program()), "-app", f"127.0.0.1:{app_port}", "-config", str(TOOLS / "assist.json"),
                 "-key", str(paths.DATA / "assist" / "assist.key")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", creationflags=flags)
            _state.update(state="connecting", port=None, message="正在连接…", since=time.time())
            threading.Thread(target=_read, args=(_process,), daemon=True).start()
    return status()


def _read(process: subprocess.Popen) -> None:
    for line in process.stdout:
        kind, _, rest = line.strip().partition(" ")
        with _lock:
            if process is not _process:
                return
            if kind == "KEY":
                _state["key"] = rest
            elif kind == "WAIT":
                _state.update(state="waiting", port=None, message=rest)
            elif kind == "LISTEN":
                _state.update(state="on", port=int(rest.split()[0]), message="已连接")
            elif kind == "DOWN":
                _state.update(state="connecting", message=rest)
            elif kind == "FAIL":
                _state.update(state="failed", port=None, message=rest)
    with _lock:
        if process is _process and _state["state"] != "failed":
            _state.update(state="off", port=None, message="已结束")


def stop() -> dict:
    global _process
    with _lock:
        process, _process = _process, None
        _state.update(state="off", port=None, message="已结束", since=None)
    if process is not None and process.poll() is None:
        try:
            process.stdin.close()    # the program ends by itself when its input closes
            process.wait(3)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    return status()
