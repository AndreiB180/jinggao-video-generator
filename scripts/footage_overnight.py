"""Grow the footage index round after round (unattended), then ship it.

Each round queues one footage_index build over every topic pack with more result pages and wider tag
widening than the last, waits for it, exports the described index to assets/footage-index (the seed
the app installs on first start), commits only that folder and pushes the current branch. It stops
when the spending cap for this run is reached or a build fails for account/budget reasons.

    .venv/bin/python scripts/footage_overnight.py --budget 15 --log /path/to/log.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUNDS = [(2, 8), (3, 20), (4, 35), (5, 50), (6, 60)]     # (result pages per search, tags added per topic)


def api(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def spent_since(started: float) -> float:
    from jinggao import costs
    with costs._db() as con:
        value = con.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE category='footage_index' AND created>=?",
                            (started,)).fetchone()[0]
    return value / costs.MICROS


def ship(log) -> None:
    from jinggao import footage_index
    stats = footage_index.stats()
    parts = footage_index.export_seed()
    size = sum(p.stat().st_size for p in parts) / 1048576
    log(f"exported {len(parts)} seed parts, {size:.1f} MB, {stats['described']} videos, {stats['described_moments']} frames")
    git = ["git", "-C", str(ROOT)]
    subprocess.run(git + ["add", "-A", "assets/footage-index"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode == 0:
        log("seed unchanged; nothing to commit")
        return
    message = (f"Footage index seed: {stats['described']:,} videos, {stats['described_moments']:,} described frames\n\n"
               "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    subprocess.run(git + ["commit", "-q", "-m", message], check=True)
    pushed = subprocess.run(git + ["push", "-q", "origin", "HEAD"], capture_output=True, text=True)
    log("pushed" if pushed.returncode == 0 else f"push failed: {pushed.stderr.strip()[:300]}")


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
    started = time.time()
    for number, (pages, widen) in enumerate(ROUNDS[args.first_round:], args.first_round + 1):
        spent = spent_since(started)
        if spent >= args.budget * 0.95:
            log(f"stop: spent {spent:.2f} of {args.budget:.2f} CNY")
            break
        job = None
        for attempt in range(30):
            try:
                job = api(args.base, "/footage-index/build", {"packs": list(PACKS), "pages": pages, "widen": widen})
                break
            except Exception as exc:          # a build still running, or the server restarting
                log(f"round {number}: cannot queue yet ({str(exc)[:120]})")
                time.sleep(60)
        if not job:
            log("stop: could not queue a build")
            break
        log(f"round {number}: queued pages={pages} widen={widen} job={job['id']}")
        while True:
            time.sleep(30)
            try:
                status = api(args.base, "/footage-index")
            except Exception:
                continue
            current = status.get("job") or {}
            if current.get("id") == job["id"] and current.get("status") not in ("queued", "running"):
                break
        log(f"round {number}: {current.get('status')} · {current.get('message')} {current.get('error') or ''}".strip())
        log(f"round {number}: index {status['sources']} videos, {status['described']} described, "
            f"{status['described_moments']} frames; spent {spent_since(started):.2f} CNY this run")
        try:
            ship(log)
        except Exception as exc:
            log(f"ship failed: {exc}")
        error = str(current.get("error") or "") + str(current.get("message") or "")
        if current.get("status") != "completed":
            if any(k in error for k in ("预算", "额度", "Key", "密钥", "权限", "欠费", "余额")):
                log("stop: budget or account problem")
                break
            if current.get("status") == "cancelled":
                log("stop: cancelled by the user")
                break
            time.sleep(600)    # risk control or a service hiccup: give it ten minutes, then the next round
    log("done")


if __name__ == "__main__":
    main()
