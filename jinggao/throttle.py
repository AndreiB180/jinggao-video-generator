"""Limits shared by every video being made at the same time.

Several videos run in separate worker processes. The launcher creates these limits once (shared()) and every
worker installs them, so however many videos run, B站 sees one polite client, the NVIDIA card never gets more
encode sessions than its driver allows, CPU video encodes never outnumber the cores (the computer stays usable)
and the 千问 speech account stays within its 3 concurrent requests.
Without install() (tests, scripts, a single worker) they are plain in-process limits of the same size.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager


def cores() -> int:
    """Logical processors this process may use (respects an affinity limit; os.cpu_count() does not)."""
    counter = getattr(os, "process_cpu_count", None)
    return (counter() if counter else None) or os.cpu_count() or 4


def cpu_slots(cores_: int | None = None) -> int:
    """CPU video encodes at once (2 threads each), leaving about two logical cores for everything else."""
    cores = cores_ or globals()["cores"]()
    return max(2, (cores - 2) // 2)


def memory_gb() -> float:
    """Installed memory (0 when unknown)."""
    try:
        import os
        import sys
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("free", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_free", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total / 2**30
            return 0.0
        if sys.platform == "darwin":
            import subprocess
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 0.0


# Each video being made needs about this much memory at its peak (text reading, speech alignment, frames);
# the rest is left for Windows, the browser and whatever else the customer runs.
VIDEO_GB, RESERVED_GB = 1.5, 7.0


def memory_videos(total: float | None = None) -> int:
    """How many videos this computer's memory carries at once (16 GB: 6; 8 GB: 1)."""
    total = memory_gb() if total is None else total
    if total <= 0:
        return 6
    return max(1, int((total - RESERVED_GB) / VIDEO_GB))


# A video being made keeps about this many logical processors busy (frame reading, text reading, watermark
# analysis, subtitles); more videos than the processors carry only make every one of them slower.
VIDEO_THREADS = 4


def cpu_videos(total: int | None = None) -> int:
    """How many videos this processor makes at once without slowing all of them (12 threads: 3, i5-13400 with 16: 4)."""
    return max(1, (total or cores()) // VIDEO_THREADS)


def max_videos() -> int:
    """Videos at once on this computer: the smaller of what its processor and its memory carry."""
    return max(1, min(MAX_VIDEOS, cpu_videos(), memory_videos()))


LIMITS = {
    "bili_api": 6,         # api.bilibili.com: search, previews (videoshot), play addresses
    "bili_download": 32,   # B站 CDN clip downloads (network only; the re-encode takes a gpu or cpu slot)
    "gpu": 3,              # NVENC sessions; the launcher sizes it from the driver (gpu.detect)
    "cpu": cpu_slots(),    # CPU video encodes (libx264, 2 threads each)
    "frames": cpu_slots() + 1,   # short FFmpeg decodes for watermark analysis and QA stills
    "qwen_tts": 3,         # 千问配音: 3 concurrent requests per Alibaba Cloud account
    "edge_tts": 4,         # free Edge voices: one IP, keep it modest
}
MAX_VIDEOS = 6             # videos made at the same time (setting video_workers)
GPU_WAIT = 3.0             # seconds a clip waits for the card before a free CPU slot may take it

_slots: dict = {name: threading.BoundedSemaphore(n) for name, n in LIMITS.items()}


class _Pause:
    """When B站 refuses a request (risk control), every worker waits before its next call."""
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def get_lock(self):
        return self.lock


_pause = _Pause()


def shared(context, sizes: dict | None = None) -> dict:
    """Cross-process versions, made by the launcher (context: a multiprocessing context); sizes overrides
    LIMITS, e.g. the card's encode sessions."""
    limits = LIMITS | (sizes or {})
    return {"slots": {name: context.BoundedSemaphore(n) for name, n in limits.items()},
            "bili_pause": context.Value("d", 0.0), "sizes": limits}


def install(objects: dict | None) -> None:
    global _slots, _pause
    if objects:
        _slots = dict(objects["slots"])
        _pause = objects["bili_pause"]
        LIMITS.update(objects.get("sizes") or {})


# Seconds this process spent waiting for each shared limit (one video per worker process): where a slow video
# queued, shown in its report and the diagnostics text.
waited: dict = {}
_waited_lock = threading.Lock()


def _note_wait(name: str, seconds: float) -> None:
    if seconds > 0.05:
        with _waited_lock:
            waited[name] = waited.get(name, 0.0) + seconds


@contextmanager
def slot(name: str, check=None):
    gate = _slots[name]
    began = time.monotonic()
    while not gate.acquire(timeout=.2):
        if check:
            check()
    _note_wait(name, time.monotonic() - began)
    try:
        yield
    finally:
        gate.release()


@contextmanager
def either(check=None, wait_for_gpu: float | None = None):
    """A video encode on the card if a session frees up soon, else on the CPU; yields "gpu" or "cpu"."""
    gpu, cpu = _slots["gpu"], _slots["cpu"]
    began = time.monotonic()
    give_up = began + (GPU_WAIT if wait_for_gpu is None else wait_for_gpu)
    while True:
        if gpu.acquire(timeout=.1):
            taken = "gpu"
            break
        if time.monotonic() >= give_up and cpu.acquire(False):
            taken = "cpu"
            break
        if check:
            check()
    _note_wait("encode", time.monotonic() - began)
    try:
        yield taken
    finally:
        _slots[taken].release()


def bili_wait(check=None) -> None:
    """Hold off while another video's request was just refused by B站."""
    began = time.monotonic()
    try:
        _bili_wait(check)
    finally:
        _note_wait("bili_paused", time.monotonic() - began)


def _bili_wait(check=None) -> None:
    while True:
        with _pause.get_lock():
            left = _pause.value - time.time()
        if left <= 0:
            return
        if check:
            check()
        time.sleep(min(left, .5))


def bili_refused(seconds: float = 3.0) -> None:
    with _pause.get_lock():
        _pause.value = max(_pause.value, time.time() + seconds)


@contextmanager
def bili_call(check=None):
    """One request to api.bilibili.com: waits out a shared pause, then takes one of the shared slots."""
    bili_wait(check)
    with slot("bili_api", check):
        yield
