"""Bounded look-ahead for independent work; results keep editorial order."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor as BaseThreadPoolExecutor, wait
from collections import deque
from contextlib import contextmanager
from contextvars import copy_context
from threading import BoundedSemaphore, Lock, RLock

# Three matching workers can each review two independent clip groups. Keep
# those six lanes available while bounding nested pixel/text requests globally.
MODEL_SLOTS = BoundedSemaphore(6)
NETWORK_SLOTS = BoundedSemaphore(8)
_keys_guard = Lock()
_keys = {}


class ThreadPoolExecutor(BaseThreadPoolExecutor):
    """Keep project/job budget attribution in nested worker threads."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)


@contextmanager
def network_slot(check=None):
    while not NETWORK_SLOTS.acquire(timeout=.2):
        if check:
            check()
    try:
        if check:
            check()
        yield
    finally:
        NETWORK_SLOTS.release()


@contextmanager
def keyed_lock(key, *, check=None):
    """Coalesce concurrent cache fills without accumulating permanent locks."""
    with _keys_guard:
        entry = _keys.setdefault(key, [RLock(), 0])
        entry[1] += 1
    acquired = False
    try:
        if check:
            check()
            while not entry[0].acquire(timeout=.1):
                check()
            acquired = True
            check()
        else:
            entry[0].acquire()
            acquired = True
        yield
    finally:
        if acquired:
            entry[0].release()
        with _keys_guard:
            entry[1] -= 1
            if not entry[1]:
                del _keys[key]


def ordered_map(function, items, workers=1):
    """Keep at most ``workers`` items in flight, including completed look-ahead.

    No database writes or timeline mutations belong in ``function``. Consuming
    results in order keeps interval allocation and image budgets deterministic.
    """
    workers = max(1, min(4, int(workers)))
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    sentinel = object()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jinggao-prepare")
    pending = deque()
    try:
        for _ in range(workers):
            item = next(iterator, sentinel)
            if item is sentinel:
                break
            pending.append(pool.submit(function, item))
        while pending:
            result = pending.popleft().result()
            item = next(iterator, sentinel)
            if item is not sentinel:
                pending.append(pool.submit(function, item))
            yield result
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def completed_map(function, items, workers=4):
    """Bounded independent tasks without head-of-line blocking; include the input.

    Use after timeline intervals have been allocated. Completion order must not
    decide which footage a shot owns.
    """
    iterator = iter(items)
    pool = ThreadPoolExecutor(max_workers=max(1, min(8, int(workers))), thread_name_prefix="jinggao-pool")
    pending = {}
    def submit():
        try:
            item = next(iterator)
        except StopIteration:
            return
        pending[pool.submit(function, item)] = item
    try:
        for _ in range(max(1, min(8, int(workers)))):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                result = future.result()
                submit()
                yield item, result
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
