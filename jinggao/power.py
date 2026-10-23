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


class _Throttling(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


throttling_off = False     # whether this process opted out of Windows power throttling (for the diagnostics)


def full_speed(handle=None) -> bool:
    """Ask Windows not to power-throttle a process (EcoQoS / "background activity"). Windows treats a process
    without a window as background work and keeps it on slow clocks, and on Intel CPUs with efficiency cores
    on those small cores only: a customer's i5-13400 (6 big + 4 small cores) ran every local step 17-46 times
    slower than our test PC while the whole of 镜稿 used 24% of its 16 threads. handle: a child process (FFmpeg);
    default this process."""
    global throttling_off
    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        # PROCESS_POWER_THROTTLING_CURRENT_VERSION 1; control EXECUTION_SPEED (1) with state 0 = never throttle.
        state = _Throttling(1, 0x1, 0x0)
        target = ctypes.c_void_p(int(handle)) if handle is not None else kernel32.GetCurrentProcess()
        done = bool(kernel32.SetProcessInformation(target, 4, ctypes.byref(state), ctypes.sizeof(state)))  # ProcessPowerThrottling
    except (OSError, AttributeError, ValueError, TypeError):
        return False
    if handle is None:
        throttling_off = done
    return done


def unthrottle_children() -> None:
    """FFmpeg runs as windowless child processes: opt each one out of power throttling right after it starts."""
    if sys.platform != "win32" or getattr(subprocess.Popen._execute_child, "_full_speed", False):
        return
    original = subprocess.Popen._execute_child

    def execute_child(self, *args, **kwargs):
        original(self, *args, **kwargs)
        full_speed(getattr(self, "_handle", None))
    execute_child._full_speed = True
    subprocess.Popen._execute_child = execute_child


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
