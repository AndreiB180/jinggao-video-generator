"""Grow the footage index overnight with a local vision model on a GPU machine (no model fees).

Start llama.cpp's server with a vision model first, for example on the dorm PC's RTX 5080:

    llama-server -m Qwen3.5-9B-Q6_K.gguf --mmproj mmproj-Qwen3.5-9B-BF16.gguf -ngl 99 -c 65536 -np 16
                 --port 8790 --jinja --reasoning-budget 0

then, with JINGGAO_DATA pointing at a scratch data folder:

    python scripts/footage_local.py --until 08:30 --out D:/jinggao-test/library/seed

The index starts from the seed in assets/footage-index (those videos are skipped). Each round searches
every topic pack, the packs in --first before the others, with more result pages and more tag widening
than the last, SLICE searches at a time; each new video's preview frames go to the local model one video
per request and are saved as they come back, so stopping loses nothing (searches are cached for the day). When the rounds are done before the deadline, the
seed's older videos (described by the cloud model from contact sheets) are described again. Every hour
and at the end the described index is exported into --out as seed parts for assets/footage-index.
A file named STOP next to the log ends the run at the next check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUNDS = [(2, 10), (3, 20), (4, 35), (5, 50)]     # (result pages per search, tags added per topic)
SLICE = 60                                        # searches per build: the model starts on them while later ones wait
FIRST = ["影视·家庭伦理", "影视·都市职场", "短剧", "纪实·法治案件", "纪实·档案人物", "新闻·国际现场", "新闻·国内现场",
         "人物·社会生活", "生活·家庭情感", "人物·情绪", "社会·民生", "法治·警务", "职场·教育"]
EXPORT_EVERY = 3600


class Stop(Exception):
    pass


def deadline(text: str) -> float:
    hour, minute = (int(x) for x in text.split(":"))
    now = datetime.now()
    end = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    return end.timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8790/v1")
    parser.add_argument("--until", default="08:30", help="local time to stop at")
    parser.add_argument("--out", required=True, help="folder for the exported seed parts")
    parser.add_argument("--log", default="")
    parser.add_argument("--first-round", type=int, default=0)
    parser.add_argument("--no-redo", action="store_true", help="do not describe the seed's older videos again")
    parser.add_argument("--workers", type=int, default=16, help="requests to the model at a time")
    args = parser.parse_args()
    from jinggao import footage_index as fi, footage_local, media, paths
    from jinggao.footage_topics import PACKS
    paths.prepare()
    log_path = Path(args.log) if args.log else paths.DATA / "footage-local.log"
    stop_file = log_path.with_name("STOP")
    status_path = log_path.with_name("footage-local-status.json")
    end = deadline(args.until)
    started = time.time()
    def log(message):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as out:
            out.write(line + "\n")
    def check():
        if time.time() >= end or stop_file.exists():
            raise media.Cancelled("到点停止")
    last = {"progress": 0.0, "export": time.time()}
    def progress(share, message=""):
        if time.time() - last["progress"] >= 60:
            last["progress"] = time.time()
            stats = fi.stats()
            state = {"time": time.strftime("%H:%M:%S"), "message": message, "sources": stats["sources"],
                     "described": stats["described"], "frames": stats["described_moments"],
                     "hours": round((time.time() - started) / 3600, 2)}
            status_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            log(f"{message} · 库里 {stats['described']} 个视频有描述")
        if time.time() - last["export"] >= EXPORT_EVERY:
            export("hourly")
    def export(reason):
        last["export"] = time.time()
        try:
            parts = fi.export_seed(Path(args.out))
            stats = fi.stats()
            log(f"export ({reason}): {len(parts)} parts, {sum(p.stat().st_size for p in parts) / 1048576:.1f} MB, "
                f"{stats['described']} described videos, {stats['described_moments']} frames")
        except Exception as exc:
            log(f"export failed: {str(exc)[:200]}")
    describe = footage_local.describer(args.url)
    # A night of searches from one address: slow and steady draws less risk control than bursts.
    fi.SEARCH_WORKERS, fi.SEARCH_GAP = 1, 1.5
    fi.DESCRIBE_WORKERS = args.workers          # as many as the server has slots (llama-server -np)
    log(f"start: until {datetime.fromtimestamp(end):%m-%d %H:%M}, model at {args.url}")
    seeded = fi.install_seed()
    log(f"seed installed: {seeded} videos; index has {fi.stats()['described']} described")
    order = [name for name in FIRST if name in PACKS] + [name for name in PACKS if name not in FIRST]
    queries = [(name, q) for name in order for q in PACKS[name]]
    try:
        for number in range(args.first_round, len(ROUNDS)):
            pages, widen = ROUNDS[number]
            log(f"round {number + 1}: {len(queries)} searches, pages={pages}, widen={widen}")
            for start in range(0, len(queries), SLICE):
                part = queries[start:start + SLICE]
                for attempt in range(2):
                    check()
                    try:
                        report = fi.build(part, check, progress, pages=pages, widen=widen, describer=describe)
                        log(f"round {number + 1} searches {start + 1}-{start + len(part)}: "
                            f"{json.dumps(report, ensure_ascii=False)}")
                        break
                    except media.Cancelled:
                        raise
                    except Exception as exc:
                        log(f"round {number + 1} searches {start + 1}-: stopped: {str(exc)[:300]}; waiting 10 minutes")
                        for _ in range(60):
                            check()
                            time.sleep(10)
            export(f"round {number + 1}")
        if not args.no_redo:
            redo(fi, describe, check, progress, log)
    except media.Cancelled:
        log("deadline or STOP reached")
    finally:
        export("end")
        log(f"done after {(time.time() - started) / 3600:.1f} h")


def redo(fi, describe, check, progress, log):
    """Describe the seed's older videos again, one video per request (the most used ones first)."""
    from jinggao import express_sources as es
    done_path = fi._file().with_name("footage-local-redone.txt")
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    with fi._conn() as con:
        rows = [(r["topic"], json.loads(r["row"]), r["key"]) for r in con.execute(
            "SELECT key, row, topic FROM sources WHERE described=1 ORDER BY used DESC, added")]
    todo = [(topic, row) for topic, row, key in rows if key not in done]
    log(f"redo: {len(todo)} older videos to describe again")
    bili = es.Bilibili()
    try:
        for start in range(0, len(todo), fi.CHUNK):
            check()
            chunk = todo[start:start + fi.CHUNK]
            report = {"indexed": 0, "described_frames": 0, "failed": 0}
            try:
                fi._build_chunk(bili, chunk, check, lambda share, message: progress(share, f"重新描述 {start}/{len(todo)} · {message}"),
                                report, True, describe)
            except Exception as exc:
                if type(exc).__name__ == "Cancelled":
                    raise
                log(f"redo stopped: {str(exc)[:300]}; waiting 10 minutes")
                for _ in range(60):
                    check()
                    time.sleep(10)
            with open(done_path, "a", encoding="utf-8") as out:
                out.write("\n".join(fi._key(row) for _, row in chunk) + "\n")
            if bili.shots_blocked >= 3:
                log("redo: previews refused by risk control; pausing 5 minutes")
                for _ in range(30):
                    check()
                    time.sleep(10)
                with bili.counts:
                    bili.shots_blocked = 0
    finally:
        bili.close()


if __name__ == "__main__":
    main()
