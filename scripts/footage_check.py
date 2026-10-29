"""干净体检 of the footage library on the dorm PC, for days: download a few seconds at library moments and record
whether the picture is clean (jinggao/footage_check.py), so films take checked-clean library footage first.

With JINGGAO_DATA pointing at the library's data folder (the same one scripts/footage_local.py builds) and,
optionally, the llama.cpp server of footage_local running (it then also asks whether a window is filmed at all):

    python scripts/footage_check.py --until never --log D:/jinggao-test/library/check/footage-check.log
    python scripts/footage_check.py --words "男孩 手机,父亲 摔手机,情侣 行李箱" --max-windows 300   # these first

Order: the --words sets first (the moments footage_index.find() would hand a film for them), then the topics the
client writes about (FIRST), then every other topic, largest first. Within a topic: videos films used most,
then those with the most filmed frames; within a video up to --per-video filmed moments between filmed
neighbours, spread over it, never one already checked by the current rules (a restart continues where the last
run got to; a moment checked before footage_check.RULES first ran on this index is checked again). WORKERS
videos are checked at once, one window after another each, so a video's play address and preview profile are
fetched once. B站 risk control (412 / -412 / -352, a v_voucher, or several videos in a row answered without any
video stream) pauses every worker (5, 10, 20… minutes, at most an hour) and the refused window is checked again. Results go to footage_index.record_checks in batches. A file named STOP
next to the log ends the run at the next check; so does --until. Every minute the log gets windows per hour and
the statuses so far; with --out the seed (footage_index.export_seed, which carries the checks) is exported there
every --export-hours and at the end. When everything is checked it waits an hour and looks again (new videos
from the library build, windows that failed).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The topics the client writes about (social stories, family, school, work, law, money, news), in this order.
FIRST = ["生活·家庭情感", "影视·家庭伦理", "人物·社会生活", "职场·教育", "社会·民生", "人物·情绪", "短剧", "影视·都市职场",
         "法治·警务", "纪实·法治案件", "财经·宏观", "商业·企业", "互联网·流量", "城市·中国", "乡村·三农", "国际·各国",
         "国际·外交时政", "新闻·国内现场", "新闻·国际现场", "健康·养生", "房产·家居"]
FLUSH_ITEMS, FLUSH_SECONDS = 24, 60     # results written per batch, or at least this often
MAX_PAUSE = 3600
TRIES = 2                               # a window that fails this often (not risk control) waits for the next pass
NO_STREAM_RUN = 3                       # this many windows in a row without any video stream: risk control


def deadline(text: str) -> float:
    if text == "never":
        return float("inf")
    hour, minute = (int(x) for x in text.split(":"))
    now = datetime.now()
    end = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    return end.timestamp()


def word_sets(text: str) -> list[list[str]]:
    """--words: "男孩 手机,校门" or a text file with one set per line."""
    if not text:
        return []
    path = Path(text)
    raw = path.read_text(encoding="utf-8").splitlines() if path.is_file() else text.replace("，", ",").split(",")
    return [words for words in (line.split() for line in raw) if words]


def topic_order(fi) -> list[str]:
    with fi._conn() as con:
        counts = {r["topic"]: r["n"] for r in con.execute(
            "SELECT topic, COUNT(*) AS n FROM sources WHERE described=1 AND flags!='host' GROUP BY topic")}
    return [t for t in FIRST if t in counts] + sorted((t for t in counts if t not in FIRST), key=lambda t: -counts[t])


def rules_since(fi, version: int) -> float:
    """When this version of the check rules first ran on the index (now, the first time). A check made before
    was decided by older rules: its moment is checked again (the seed carries each check's time)."""
    key = f"footage_check_rules_{version}"
    with fi._lock, fi._conn(write=True) as con:
        con.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, repr(time.time())))
        return float(con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])


def topic_videos(fi, fc, topic: str, per_video: int, seconds: float, done: dict, since: float = 0.0):
    """(row, moments) of one topic's videos still to check, in priority order. done: key -> moments checked
    in this run (not necessarily written yet); since: checks older than this are made again."""
    with fi._conn() as con:
        sources = con.execute("SELECT key, row, used FROM sources WHERE topic=? AND described=1 AND flags!='host'",
                              (topic,)).fetchall()
    ranked = []
    for start in range(0, len(sources), 500):
        part = sources[start:start + 500]
        keys = [s["key"] for s in part]
        kinds = {}
        with fi._conn() as con:
            for m in con.execute(f"SELECT source, t, kind FROM moments WHERE source IN ({','.join('?' * len(keys))}) "
                                 "AND t>=0 ORDER BY source, t", keys):
                kinds.setdefault(m["source"], []).append((float(m["t"]), m["kind"]))
        checked = fi.checks_for(keys, since=since)
        for s in part:
            try:
                row = json.loads(s["row"])
            except ValueError:
                continue
            frames = kinds.get(s["key"]) or []
            if not fc.worth_checking(row, frames):
                continue
            seen = set(checked.get(s["key"]) or ()) | done.get(s["key"], set())
            times = fc.moments_to_check(row, frames, limit=per_video, seconds=seconds, checked=seen)
            if times:
                filmed = sum(k in fc.FILMED for _, k in frames)
                ranked.append(((-int(s["used"] or 0), -filmed, -int(row.get("play") or 0)), row, times))
    ranked.sort(key=lambda item: item[0])
    for _, row, times in ranked:
        yield row, times


def word_videos(fi, sets, per_video: int, limit: int, done: dict, log, since: float = 0.0):
    """(label, row, moments): the moments find() would hand a film for each word set, best first."""
    if sets and not hasattr(fi, "find"):
        log("footage_index.find() 不可用，跳过 --words")
        return
    for words in sets:
        try:
            rows = fi.find(words, limit=limit)
        except Exception as exc:   # noqa: BLE001 — a bad set must not end the run
            log(f"find({' '.join(words)}) 失败：{str(exc)[:120]}")
            continue
        keys = [fi._key(row) for row in rows]
        checked = fi.checks_for(keys, since=since) if keys else {}
        for key, row in zip(keys, rows, strict=True):
            seen = {round(float(t), 2) for t in checked.get(key) or ()} | done.get(key, set())
            times = [float(t) for t in row.get("moments") or [] if round(float(t), 2) not in seen][:per_video]
            if times:
                yield "词：" + " ".join(words), row, times


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="never", help="local time to stop at, or never")
    parser.add_argument("--workers", type=int, default=6, help="windows checked at once (one video each)")
    parser.add_argument("--per-video", type=int, default=4)
    parser.add_argument("--short-edge", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--words", default="", help='word sets to check first: "男孩 手机,校门" or a file, one per line')
    parser.add_argument("--words-limit", type=int, default=30, help="videos per word set")
    parser.add_argument("--url", default="http://127.0.0.1:8790/v1", help="llama.cpp server of footage_local")
    parser.add_argument("--no-describe", action="store_true", help="pixels and text only, no local vision model")
    parser.add_argument("--out", default="", help="folder for the exported seed parts (none: no export)")
    parser.add_argument("--export-hours", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=0, help="stop after this many windows (trial runs)")
    parser.add_argument("--log", default="")
    args = parser.parse_args(argv)
    from jinggao import express_sources as es, footage_check as fc, footage_index as fi, media, paths, screen_text
    paths.prepare()
    log_path = Path(args.log) if args.log else paths.DATA / "footage-check.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_file = log_path.with_name("STOP")
    status_path = log_path.with_name("footage-check-status.json")
    work = paths.DATA / "footage-check-run"     # its own clip folder: cleared at start and end
    end = deadline(args.until)
    started = time.time()
    lock, flush_lock, export_lock = threading.Lock(), threading.Lock(), threading.Lock()

    def log(message):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        with lock, open(log_path, "a", encoding="utf-8") as out:
            out.write(line + "\n")

    state = {"windows": 0, "failed": 0, "pauses": 0, "stage": "", "pause_until": 0.0, "streak": 0, "no_stream": 0,
             "progress": time.time(), "export": time.time(), "flushed": time.time(), "unwritten": 0}
    counts, recent, pending, done = Counter(), [], [], {}

    def check():
        if time.time() >= end or stop_file.exists():
            raise media.Cancelled("到点停止")
        if args.max_windows and state["windows"] >= args.max_windows:
            raise media.Cancelled("已到 --max-windows")

    def flush(force=False):
        with flush_lock:
            with lock:
                if not pending or (not force and len(pending) < FLUSH_ITEMS
                                   and time.time() - state["flushed"] < FLUSH_SECONDS):
                    return
                batch = pending[:]
                pending.clear()
                state["flushed"] = time.time()
            try:
                written = fi.record_checks(batch)
            except Exception as exc:   # noqa: BLE001 — keep the results and try again with the next batch
                written = 0
                log(f"record_checks 失败：{str(exc)[:160]}")
            if not written:            # the index was busy: nothing was lost, the next flush retries
                with lock:
                    pending[:0] = batch
                    state["unwritten"] = len(pending)

    def report(force=False):
        with lock:
            if not force and time.time() - state["progress"] < 60:
                return
            state["progress"] = time.time()
        hours = max(1e-6, (time.time() - started) / 3600)
        span = min(900.0, max(60.0, time.time() - started))     # the last 15 minutes (or the run so far)
        with lock:
            recent[:] = [r for r in recent if r >= time.time() - span]
            rate_recent = round(len(recent) * 3600 / span)
            summary = dict(counts)
        try:
            totals = fi.check_stats()
        except Exception as exc:   # noqa: BLE001
            totals = {"error": str(exc)[:80]}
        good = summary.get("clean", 0) + summary.get("crop", 0)
        status = {"time": time.strftime("%H:%M:%S"), "stage": state["stage"], "windows": state["windows"],
                  "per_hour": round(state["windows"] / hours), "per_hour_recent": rate_recent,
                  "statuses": summary, "good_share": round(good / max(1, sum(summary.values())), 3),
                  "failed": state["failed"], "pauses": state["pauses"], "unwritten": state["unwritten"],
                  "index": totals, "hours": round(hours, 2)}
        try:
            status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        log(f"{state['stage']} · {state['windows']} 段（{status['per_hour']}/小时，近15分钟 {rate_recent}/小时）· "
            f"干净 {good} · {json.dumps(summary, ensure_ascii=False)} · 失败 {state['failed']} · 风控暂停 {state['pauses']} "
            f"· 库里 {json.dumps(totals, ensure_ascii=False)}")
        if args.out and time.time() - state["export"] >= args.export_hours * 3600:
            export("periodic")

    def export(reason):
        if not export_lock.acquire(blocking=reason == "end"):
            return             # another worker is exporting
        try:
            state["export"] = time.time()
            flush(force=True)
            parts = fi.export_seed(Path(args.out))
            log(f"export ({reason}): {len(parts)} parts, {sum(p.stat().st_size for p in parts) / 1048576:.1f} MB")
        except Exception as exc:   # noqa: BLE001
            log(f"export failed: {str(exc)[:200]}")
        finally:
            export_lock.release()

    def refused(message):
        with lock:
            now = time.time()
            if state["pause_until"] > now:
                return
            seconds = min(MAX_PAUSE, fi.BLOCKED_PAUSE * 2 ** state["streak"])
            state["streak"] += 1
            state["pauses"] += 1
            state["pause_until"] = now + seconds
        with bili.counts:
            bili.shots_blocked = bili.search_refused = 0
        bili.blocked = 0
        log(f"B站接口风控（{message[:60]}），全部暂停 {seconds // 60} 分钟后接着查")

    def wait_out_pause():
        while time.time() < state["pause_until"]:
            check()
            report()
            time.sleep(2)

    def record(row, t, result):
        item = {"source": f"bilibili:{row['id']}", "t": round(float(t), 2), "status": result["status"],
                "crop": result.get("crop"), "note": result.get("note", "")}
        with lock:
            pending.append(item)
            done.setdefault(item["source"], set()).add(item["t"])
            counts[result["status"]] += 1
            state["windows"] += 1
            recent.append(time.time())

    def check_video(row, times):
        try:
            for index, t in enumerate(times):
                tries = 0
                while True:
                    wait_out_pause()
                    check()
                    try:
                        result = fc.check_window(bili, row, t, short_edge=args.short_edge, seconds=args.seconds,
                                                 describe=describe, work=work, check=check)
                        break
                    except media.Cancelled:
                        raise
                    except fc.Refused as exc:
                        refused(str(exc))
                    except Exception as exc:   # noqa: BLE001 — a failed download is tried again, then left
                        if isinstance(exc, fc.NoStream):
                            # B站 under load answers every video without a stream: a run of them is a refusal.
                            with lock:
                                state["no_stream"] += 1
                                many = state["no_stream"] >= NO_STREAM_RUN
                                if many:
                                    state["no_stream"] = 0
                            if many:
                                refused(f"连续 {NO_STREAM_RUN} 段没有视频流")
                                continue
                        tries += 1
                        if tries >= TRIES:
                            with lock:
                                state["failed"] += 1
                                failed = state["failed"]
                            if failed <= 20 or failed % 100 == 0:
                                log(f"{row['id']} @{t:.1f}s 失败（第 {failed} 次）：{type(exc).__name__}: {str(exc)[:140]}")
                            result = None
                            break
                        for _ in range(5):
                            check()
                            time.sleep(1)
                if result is None:
                    continue
                with lock:
                    state["streak"] = state["no_stream"] = 0
                record(row, t, result)
                if result["status"] == "mark" and result.get("note", "").startswith(fc.CONTACT):
                    # Production drops every window of a video once one shows contact details (an uploader
                    # selling footage): its other moments are as good as checked.
                    for later in times[index + 1:]:
                        record(row, later, {"status": "mark", "crop": None, "note": f"{fc.CONTACT}：同一视频另一处有"})
                    break
                flush()
                report()
        finally:
            fc.forget(row["id"])

    def jobs():
        sets = word_sets(args.words)
        for label, row, times in word_videos(fi, sets, args.per_video, args.words_limit, done, log, since):
            state["stage"] = label
            yield row, times
        for topic in topic_order(fi):
            state["stage"] = topic
            log(f"题材：{topic}")
            yield from topic_videos(fi, fc, topic, args.per_video, args.seconds, done, since)

    if not screen_text.installed():
        try:
            screen_text.install_model(check)
        except Exception as exc:   # noqa: BLE001
            log(f"文字识别模型不可用（{str(exc)[:120]}）：花字和字幕查不出来，不能体检。把镜稿包 data/models/ocr "
                f"复制到 {screen_text.model_dir()} 后再运行")
            return 1
    describe = None
    if not args.no_describe:
        if fc.reachable(args.url):
            describe = fc.describer(args.url)
        else:
            log(f"看图服务 {args.url} 连不上：只查画面和文字（主持人/动画判断跳过）")
    fc.clear_work(work)
    seeded = fi.install_seed()
    since = rules_since(fi, fc.RULES)
    log(f"start: until {args.until if end == float('inf') else f'{datetime.fromtimestamp(end):%m-%d %H:%M}'}, "
        f"{args.workers} at once, {args.per_video}/video, {args.short_edge}p × {args.seconds:g}s, "
        f"describe={'on' if describe else 'off'}; seed added {seeded} videos; rules v{fc.RULES} since "
        f"{datetime.fromtimestamp(since):%m-%d %H:%M} (older checks are made again)")
    bili = es.Bilibili()
    passes = 0
    try:
        while True:
            passes += 1
            before = state["windows"]
            source, source_lock = jobs(), threading.Lock()

            errors = []

            def worker(source=source, source_lock=source_lock, errors=errors):
                try:
                    while True:
                        with source_lock:
                            item = next(source, None)
                        if item is None:
                            return
                        check_video(*item)
                except media.Cancelled:
                    pass
                except Exception as exc:   # noqa: BLE001 — ends the run with the error logged below
                    errors.append(exc)

            threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.workers))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            flush(force=True)
            report(force=True)
            if errors:
                log(f"error: {type(errors[0]).__name__}: {str(errors[0])[:300]}")
                raise errors[0]
            check()
            log(f"pass {passes} done: {state['windows'] - before} windows; waiting an hour for new videos")
            for _ in range(360):
                check()
                time.sleep(10)
    except media.Cancelled as exc:
        log(f"stopped: {exc}")
    finally:
        flush(force=True)
        report(force=True)
        if args.out:
            export("end")
        bili.close()
        fc.clear_work(work)
        log(f"done after {(time.time() - started) / 3600:.1f} h, {state['windows']} windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
