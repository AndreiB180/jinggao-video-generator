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


def lower_priority() -> str:
    """Video work runs below normal priority, so the desktop and other programs stay responsive while every
    core is busy; FFmpeg processes started from here inherit it. Returns what was set."""
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000):   # BELOW_NORMAL_PRIORITY_CLASS
                return "below_normal"
        elif hasattr(os, "nice"):
            current = os.nice(0)
            if current < 10:
                os.nice(10 - current)
            return "nice"
    except (OSError, AttributeError):
        pass
    return "unchanged"
