"""Limits shared by every video being made at the same time.

Several videos run in separate worker processes. The launcher creates these limits once (shared()) and every
worker installs them, so however many videos run, B站 sees one polite client, the NVIDIA encoder never gets
more sessions than a GeForce card allows, and the 千问 speech account stays within its 3 concurrent requests.
Without install() (tests, scripts, a single worker) they are plain in-process limits of the same size.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

LIMITS = {
    "bili_api": 6,         # api.bilibili.com: search, previews (videoshot), play addresses
    "bili_download": 32,   # B站 CDN clip downloads (each also trims the clip with ffmpeg)
    "render": 3,           # shot encodes; older GeForce drivers allow 3 NVENC sessions
    "qwen_tts": 3,         # 千问配音: 3 concurrent requests per Alibaba Cloud account
    "edge_tts": 4,         # free Edge voices: one IP, keep it modest
}
MAX_VIDEOS = 4             # videos made at the same time (setting video_workers)

_slots: dict = {name: threading.BoundedSemaphore(n) for name, n in LIMITS.items()}


class _Pause:
    """When B站 refuses a request (risk control), every worker waits before its next call."""
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def get_lock(self):
        return self.lock


_pause = _Pause()


def shared(context) -> dict:
    """Cross-process versions, made by the launcher (context: a multiprocessing context)."""
    return {"slots": {name: context.BoundedSemaphore(n) for name, n in LIMITS.items()},
            "bili_pause": context.Value("d", 0.0)}


def install(objects: dict | None) -> None:
    global _slots, _pause
    if objects:
        _slots = dict(objects["slots"])
        _pause = objects["bili_pause"]


@contextmanager
def slot(name: str, check=None):
    gate = _slots[name]
    while not gate.acquire(timeout=.2):
        if check:
            check()
    try:
        yield
    finally:
        gate.release()


@contextmanager
def try_slot(name: str):
    """Take a slot only if one is free right now; yields whether it was taken."""
    gate = _slots[name]
    taken = gate.acquire(False)
    try:
        yield taken
    finally:
        if taken:
            gate.release()


def bili_wait(check=None) -> None:
    """Hold off while another video's request was just refused by B站."""
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
