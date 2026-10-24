"""Grow the footage index with a local vision model on a GPU machine (no model fees), overnight or for days.

Start llama.cpp's server with a vision model first, for example on the dorm PC's RTX 5080:

    llama-server -m Qwen3.5-9B-Q6_K.gguf --mmproj mmproj-Qwen3.5-9B-BF16.gguf -ngl 99 -c 65536 -np 16
                 --port 8790 --jinja --reasoning-budget 0

then, with JINGGAO_DATA pointing at a scratch data folder:

    python scripts/footage_local.py --until 08:30 --out D:/jinggao-test/library/seed
    python scripts/footage_local.py --until never --out D:/jinggao-test/library/seed     # until a STOP file

The index starts from the seed in assets/footage-index (those videos are skipped). Each round searches
every topic pack, the packs in FIRST before the others, with more result pages and more tag widening than
the last, SLICE searches at a time; each new video's preview frames go to the local model one video per
request and are saved as they come back, so stopping loses nothing. Where it got to is kept in
footage-local-state.json, so a restart continues from there. After the rounds it snowballs: the Bilibili
tags that the described videos share most become the next searches (every search made is remembered in
footage-local-searched.txt), so the index spreads into every field its videos lead to. With a deadline and
time left after the rounds, the seed's older contact-sheet descriptions are redone instead. The described
index is exported into --out as seed parts every --export-hours and at the end. A file named STOP next to
the log ends the run at the next check.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUNDS = [(2, 10), (3, 20), (4, 35), (5, 50)]     # (result pages per search, tags added per topic)
SLICE = 60                                        # searches per build: the model starts on them while later ones wait
FIRST = ["影视·家庭伦理", "影视·都市职场", "短剧", "纪实·法治案件", "纪实·档案人物", "新闻·国际现场", "新闻·国内现场",
         "人物·社会生活", "生活·家庭情感", "人物·情绪", "社会·民生", "法治·警务", "职场·教育", "娱乐·明星", "体育·赛事现场",
         "财经·宏观"]
SNOWBALL = 600                                    # tags searched per snowball pass (re-ranked between passes)
# Tags that lead away from footage the scripts need: looks and dance channels, Bilibili's activity and
# creator tags, study/ASMR/game/music/shopping videos.
SNOWBALL_SKIP = re.compile(r"美女|小姐姐|性感|女神|福利|擦边|颜值|身材|舞蹈|翻跳|cos|二次元|白噪音|解压|助眠|asmr|学习|考试|"
                           r"网课|vlog|日常|打卡|挑战|计划|征稿|激励|新星|创作|研究所|剪刀手|名堂|投稿|活动|必剪|整活|"
                           r"鬼畜|沙雕|手书|mad|amv|游戏|手游|音乐|歌|翻唱|bgm|配乐|教程|测评|开箱|好物|带货|穿搭|美妆|"
                           r"护肤|直播|主播|up主|粉丝|抽奖|壁纸|头像|表情包|动漫|番剧|追番|知识分享官|干货|观察局|猎手|榨菜|陪伴|"
                           r"漫剧|国漫|漫画|喵星人|单机|联机|dnf|洛克王国|每天一个|一起做|人人都", re.I)
SNOWBALL_PAGES, SNOWBALL_WIDEN = 3, 5


def deadline(text: str) -> float:
    if text == "never":
        return float("inf")
    hour, minute = (int(x) for x in text.split(":"))
    now = datetime.now()
    end = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    return end.timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8790/v1")
    parser.add_argument("--until", default="08:30", help="local time to stop at, or never")
    parser.add_argument("--export-hours", type=float, default=1.0, help="export the seed this often")
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
    state_path = log_path.with_name("footage-local-state.json")
    searched_path = log_path.with_name("footage-local-searched.txt")
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
        if time.time() - last["export"] >= args.export_hours * 3600:
            export("periodic")
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
    log(f"start: until {args.until if end == float('inf') else f'{datetime.fromtimestamp(end):%m-%d %H:%M}'}, "
        f"model at {args.url}")
    seeded = fi.install_seed()
    log(f"seed installed: {seeded} videos; index has {fi.stats()['described']} described")
    order = [name for name in FIRST if name in PACKS] + [name for name in PACKS if name not in FIRST]
    queries = [(name, q) for name in order for q in PACKS[name]]
    state = {"round": args.first_round, "start": 0}
    if not args.first_round:
        try:
            state |= json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    searched = set(searched_path.read_text(encoding="utf-8").split("\n")) if searched_path.exists() else set()
    searched_lock = threading.Lock()
    def run_slices(slices, pages, widen, label, mark=None):
        """Builds over [(start, searches)], two at a time: one searches Bilibili while the other's videos are
        with the model, so the model is not left waiting for searches. mark(start) as each one begins."""
        from concurrent.futures import ThreadPoolExecutor
        def one(item):
            start, part = item
            if mark:
                mark(start)
            run_slice(part, pages, widen, f"{label} {start + 1}-{start + len(part)}")
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(one, slices):
                pass
    def run_slice(part, pages, widen, label):
        """One build over these searches; a failure waits ten minutes and tries once more."""
        for attempt in range(2):
            check()
            try:
                report = fi.build(part, check, progress, pages=pages, widen=widen, describer=describe)
                log(f"{label}: {json.dumps(report, ensure_ascii=False)}")
                break
            except media.Cancelled:
                raise
            except Exception as exc:
                log(f"{label}: stopped: {str(exc)[:300]}; waiting 10 minutes")
                for _ in range(60):
                    check()
                    time.sleep(10)
        with searched_lock:
            fresh = [q for _, q in part if q not in searched]
            searched.update(fresh)
            with open(searched_path, "a", encoding="utf-8") as out:
                out.write("".join(q + "\n" for q in fresh))
    try:
        for number in range(state["round"], len(ROUNDS)):
            pages, widen = ROUNDS[number]
            log(f"round {number + 1}: {len(queries)} searches, pages={pages}, widen={widen}")
            first = state["start"] if number == state["round"] else 0
            run_slices([(start, queries[start:start + SLICE]) for start in range(first, len(queries), SLICE)],
                       pages, widen, f"round {number + 1} searches",
                       lambda start, number=number: state_path.write_text(
                           json.dumps({"round": number, "start": max(0, start - SLICE)}), encoding="utf-8"))
            export(f"round {number + 1}")
        state_path.write_text(json.dumps({"round": len(ROUNDS), "start": 0}), encoding="utf-8")
        if end != float("inf") and not args.no_redo:
            redo(fi, describe, check, progress, log)
        passes = 0
        while True:
            check()
            tags = snowball(fi, searched)
            if not tags:
                log("snowball: no new tags; waiting an hour")
                for _ in range(360):
                    check()
                    time.sleep(10)
                continue
            passes += 1
            log(f"snowball {passes}: {len(tags)} tags, e.g. {'、'.join(q for _, q in tags[:12])}")
            run_slices([(start, tags[start:start + SLICE]) for start in range(0, len(tags), SLICE)],
                       SNOWBALL_PAGES, SNOWBALL_WIDEN, f"snowball {passes} searches")
    except media.Cancelled:
        log("deadline or STOP reached")
    finally:
        export("end")
        log(f"done after {(time.time() - started) / 3600:.1f} h")


def snowball(fi, searched: set, limit: int = SNOWBALL) -> list[tuple[str, str]]:
    """The next searches: tags that the most described videos share (Bilibili's own words for what they
    show), not searched before, each under the topic most of its videos came from."""
    from collections import Counter, defaultdict
    counts, topics = Counter(), defaultdict(Counter)
    with fi._conn() as con:
        for r in con.execute("SELECT row, topic FROM sources WHERE described=1 AND flags!='host'"):
            try:
                tags = str(json.loads(r["row"]).get("tags") or "").split(",")
            except ValueError:
                continue
            for tag in {t.strip() for t in tags}:
                if (2 <= len(tag) <= 6 and tag.lower() not in fi.GENERIC_TAGS and not fi.SKIP_TITLE.search(tag)
                        and not SNOWBALL_SKIP.search(tag) and not tag.isdigit() and tag not in searched):
                    counts[tag] += 1
                    topics[tag][r["topic"] or "扩展"] += 1
    picked = [tag for tag, n in counts.most_common() if n >= 3][:limit]
    return [(topics[tag].most_common(1)[0][0][:30], tag) for tag in picked]


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
