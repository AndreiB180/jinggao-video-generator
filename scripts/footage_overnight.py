"""Grow the footage index round after round (unattended), then ship it.

Each round queues one footage_index build over every topic pack with more result pages and wider tag
widening than the last, waits for it, and exports the described index to assets/footage-index (the seed
the app installs on first start), committing only that folder and pushing the branch it started on. It
also ships once an hour while a round runs. It stops at its spending cap, on budget/account failures
(key, permission, balance), when the disk runs low, or when the server stops answering; transient
failures (risk control, rate limits, a run of failed sheets) wait ten minutes and go on.

    .venv/bin/python scripts/footage_overnight.py --budget 15
"""
from __future__ import annotations

import argparse
import json
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUNDS = [(2, 8), (3, 20), (4, 35), (5, 50), (6, 60)]     # (result pages per search, tags added per topic)
STOP_WORDS = ("预算 ¥", "模型预算", "素材库预算", "HTTP 401", "HTTP 402", "HTTP 403", "HTTP 404", "API Key 无效",
              "额度不足", "拒绝当前账号", "费用保护", "请在设置中填写")
MIN_FREE_GB = 5
STALL_SECONDS = 45 * 60       # a running build that reports nothing new for this long is treated as stuck
SHIP_EVERY = 3600


def api(base, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                                     method=method or ("GET" if body is None else "POST"))
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def spent_since(started: float) -> float:
    from jinggao import costs
    with costs._db() as con:
        value = con.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE category='footage_index' AND created>=?",
                            (started,)).fetchone()[0]
    return value / costs.MICROS


def spent_today() -> float:
    """What the per-day cap counts for the library today (open reservations included)."""
    from jinggao import costs
    with costs._db() as con:
        value = con.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE category='footage_index' AND "
                            "(day=? OR state IN ('reserved','unknown'))", (costs._day(),)).fetchone()[0]
    return value / costs.MICROS


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1024 ** 3


class Shipper:
    def __init__(self, log):
        self.log = log
        self.branch = self.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        self.last = 0.0

    def git(self, *args, check=False):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=check)

    def __call__(self, reason=""):
        from jinggao import footage_index
        self.last = time.time()
        try:
            branch = self.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            if branch != self.branch:
                self.log(f"ship skipped: branch is {branch}, started on {self.branch}")
                return
            stats = footage_index.stats()
            parts = footage_index.export_seed()
            size = sum(p.stat().st_size for p in parts) / 1048576
            self.log(f"export ({reason}): {len(parts)} parts, {size:.1f} MB, {stats['described']} described videos, "
                     f"{stats['described_moments']} frames")
            self.git("add", "-A", "--", "assets/footage-index", check=True)
            if self.git("diff", "--cached", "--quiet", "--", "assets/footage-index").returncode == 0:
                self.log("seed unchanged; nothing to commit")
                return
            message = (f"Footage index seed: {stats['described']:,} videos, {stats['described_moments']:,} described frames\n\n"
                       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
            self.git("commit", "-q", "-m", message, "--", "assets/footage-index", check=True)
            pushed = self.git("push", "-q", "origin", self.branch)
            self.log("pushed" if pushed.returncode == 0 else f"push failed: {pushed.stderr.strip()[:300]}")
        except Exception as exc:
            self.log(f"ship failed: {str(exc)[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8766/api")
    parser.add_argument("--budget", type=float, default=15.0, help="CNY this run may spend on descriptions")
    parser.add_argument("--log", default=str(ROOT / "data" / "footage-overnight.log"))
    parser.add_argument("--first-round", type=int, default=0)
    args = parser.parse_args()
    def log(message):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        with open(args.log, "a", encoding="utf-8") as out:
            out.write(line + "\n")
    from jinggao.footage_topics import PACKS
    ship = Shipper(log)
    started = time.time()
    capped_day = [None]
    def set_cap():
        """The per-day cap = today's library spend + what this run may still spend (recomputed after midnight),
        so neither stops the other early and the run cannot pass its budget."""
        from jinggao import costs
        if capped_day[0] == costs._day():
            return
        try:
            remaining = max(0.0, args.budget - spent_since(started))
            cap = min(1000.0, round(spent_today() + remaining + 0.01, 2))
            api(args.base, "/settings", {"index_daily_budget_cny": f"{cap:.2f}"}, method="PATCH")
            capped_day[0] = costs._day()
            log(f"index_daily_budget_cny set to {cap:.2f} (remaining run budget {remaining:.2f})")
        except Exception as exc:
            log(f"could not set the daily cap: {str(exc)[:200]}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        run_rounds(args, log, ship, started, set_cap, PACKS)
    finally:
        ship("exit")
        log("done")


def run_rounds(args, log, ship, started, set_cap, PACKS):
    round_index, retried, offline_waits = args.first_round, False, 0
    while round_index < len(ROUNDS):
        set_cap()
        pages, widen = ROUNDS[round_index]
        number = round_index + 1
        spent = spent_since(started)
        if spent >= args.budget * 0.95:
            log(f"stop: spent {spent:.2f} of {args.budget:.2f} CNY")
            break
        if free_gb() < MIN_FREE_GB:
            log(f"stop: only {free_gb():.1f} GB free")
            break
        job = None
        for attempt in range(30):
            try:
                job = api(args.base, "/footage-index/build", {"packs": list(PACKS), "pages": pages, "widen": widen})
                break
            except Exception as exc:
                try:      # a build already running (e.g. started by hand): follow that one
                    current = api(args.base, "/footage-index").get("job") or {}
                    if current.get("kind") == "footage_index" and current.get("status") in ("queued", "running"):
                        job = current
                        log(f"round {number}: following the build already running ({current['id']})")
                        break
                except Exception:
                    pass
                log(f"round {number}: cannot queue yet ({str(exc)[:120]})")
                time.sleep(60)
        if not job:
            log("stop: could not queue a build")
            break
        log(f"round {number}: pages={pages} widen={widen} job={job['id']}")
        failures, last_change, last_seen, current, status = 0, time.time(), None, {}, {}
        self_cancelled = ""
        while True:
            time.sleep(30)
            try:
                status = api(args.base, "/footage-index")
                failures = 0
            except Exception as exc:
                failures += 1
                log(f"poll failed ({failures}): {str(exc)[:120]}")
                if failures >= 20:
                    break
                continue
            current = status.get("job") or {}
            if current.get("id") != job["id"]:
                continue
            seen = (current.get("progress"), current.get("message"))
            if seen != last_seen:
                last_seen, last_change = seen, time.time()
            if current.get("status") not in ("queued", "running"):
                break
            set_cap()
            reason = ""
            if current.get("status") == "running" and time.time() - last_change > STALL_SECONDS:
                reason = "stall"
            elif spent_since(started) >= args.budget:
                reason = "budget"
            elif free_gb() < MIN_FREE_GB:
                reason = "disk"
            if reason and not self_cancelled:
                log(f"round {number}: cancelling ({reason})")
                try:
                    api(args.base, f"/jobs/{job['id']}/cancel", {})
                    self_cancelled = reason
                except Exception:
                    pass
                last_change = time.time()
            if time.time() - ship.last > SHIP_EVERY:
                ship("hourly")
        if failures >= 20:
            log("stop: the server stopped answering")
            break
        log(f"round {number}: {current.get('status')} · {current.get('message')} {current.get('error') or ''}".strip())
        log(f"round {number}: index {status.get('sources')} videos, {status.get('described')} described, "
            f"{status.get('described_moments')} frames; spent {spent_since(started):.2f} CNY this run")
        ship(f"round {number} {current.get('status')}")
        text = str(current.get("error") or "") + str(current.get("message") or "")
        if current.get("status") == "completed":
            if (status.get("report") or {}).get("search_offline") and offline_waits < 4:
                # Bilibili search was refused, so this round used today's cached searches only. Searching
                # again later finds new videos: wait, then run the same round once more.
                offline_waits += 1
                log(f"round {number}: search was refused; waiting 45 min before running it again ({offline_waits}/4)")
                time.sleep(45 * 60)
                continue
            round_index, retried, offline_waits = round_index + 1, False, 0
            continue
        if any(word in text for word in STOP_WORDS):
            log("stop: budget or account problem")
            break
        if current.get("status") == "cancelled":
            if self_cancelled in ("budget", "disk"):
                log(f"stop: {self_cancelled}")
                break
            if not self_cancelled:
                log("stop: cancelled by the user")
                break
        # Risk control, rate limits, a run of failed sheets, a restart: wait, then this round once more
        # (finished videos are skipped), else the next one.
        time.sleep(600)
        if retried:
            round_index, retried = round_index + 1, False
        else:
            retried = True


if __name__ == "__main__":
    main()
