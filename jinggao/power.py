"""Hold an idle-sleep assertion only while the queue is doing work.

This never changes saved power settings, keeps a display on, or blocks an
explicit sleep/shutdown. Assertions also disappear when the worker exits.
"""
from contextlib import contextmanager
import ctypes
import os
import subprocess
import sys


@contextmanager
def keep_awake():
    status = {"requested": False, "active": False, "method": "unsupported"}
    child = None
    restore = None
    try:
        if sys.platform == "darwin":
            status.update(requested=True, method="caffeinate_idle_system")
            # -s also holds system sleep during AC-powered maintenance wakes;
            # idle-only assertions did not prevent those on the test machine.
            # macOS ignores -s on battery; -i still prevents ordinary idle sleep.
            child = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            status["active"] = child.poll() is None
        elif sys.platform == "win32":
            status.update(requested=True, method="windows_thread_execution_state")
            setter = ctypes.windll.kernel32.SetThreadExecutionState
            setter.argtypes = [ctypes.c_uint]
            setter.restype = ctypes.c_uint
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; deliberately no display flag.
            previous = setter(0x80000001)
            if previous:
                def restore():
                    setter(previous)
                status["active"] = True
    except (OSError, AttributeError):
        status["active"] = False
    try:
        yield status
    finally:
        if restore is not None:
            restore()
        if child is not None:
            try:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
