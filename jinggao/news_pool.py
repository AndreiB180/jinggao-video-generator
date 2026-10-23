"""新闻模式: footage from news reports about the same event, placed where they say what the line says.

Similar channels do not search footage line by line: they take a few news videos about the same event and
cut them where the narration matches what is said in them, so the journalists' own choice of people, places
and scenes comes along. For a script about a concrete news event (the planner's "event" queries) this finds
such videos on B站, downloads only their audio, transcribes it with the local speech model that aligns our
own narration, and offers each line the window of a report that says the same thing, ahead of the footage
picked by searching. Downloads and QA treat these windows like any other clip (a presenter talking to camera,
a text page or contact details are still rejected and the line falls back to its searched footage).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time

import httpx

from . import express_sources as es, local_asr, paths, throttle

MAX_SOURCES = 8
MIN_SECONDS, MAX_SECONDS = 45.0, 1200.0      # news reports and commentary, not clips or hour-long streams
MAX_AUDIO_BYTES = 40 * 2**20
MIN_SCORE = 0.06                              # share of a line's (rarity-weighted) character pairs said in the window
PER_SHOT = 3
# Words of the search queries that are not names (a line and a report "share" a name, not these).
GENERIC = {"新闻", "现场", "视频", "画面", "纪录片", "航拍", "实拍", "空镜", "素材", "电视剧片段", "电视剧", "短剧", "片段",
           "历史影像", "讲话", "会见", "发射", "袭击", "部署", "对峙", "冲突", "会议", "领导人", "民众", "街头", "士兵",
           "导弹", "弹道导弹", "巡航导弹", "无人机", "拦截", "防空", "防空系统", "军事", "目标", "武装", "部队", "军队", "战机",
           "战斗机", "军舰", "驱逐舰", "航母", "舰队", "爆炸", "空袭", "进攻", "打击", "报复", "局势", "战争", "战事", "事件",
           "消息", "政府", "官员", "发言人", "支持", "启动", "升空", "轰炸", "危机", "人道主义", "遇袭", "闯入", "增兵", "演习"}


def _plain(text: str) -> str:
    return "".join(re.findall(r"[一-鿿0-9A-Za-z]", text or "")).lower()


def _pairs(text: str) -> set:
    plain = _plain(text)
    return {plain[i:i + 2] for i in range(len(plain) - 1)}


def names(queries, script: str) -> list[str]:
    """The names a script and its search queries share (countries, places, organisations, people): what a
    line and a news report have in common even when the script is rewritten in other words."""
    found = []
    for query in queries:
        for word in query.split():
            if len(word) >= 2 and word not in GENERIC and word in script and word not in found:
                found.append(word)
    return found


def _named(text: str, wanted) -> set:
    return {name for name in wanted if name in text}


def find_sources(bili, queries, used=()) -> list[dict]:
    """News videos for the event: the first results of each query, long enough to hold a report."""
    rows, seen = [], set(used)
    for query in queries:
        for row in bili.search(query):
            if row["id"] in seen or not MIN_SECONDS <= float(row.get("duration") or 0) <= MAX_SECONDS:
                continue
            seen.add(row["id"])
            rows.append(row)
    return rows[:MAX_SOURCES]


def segments(words: list[dict]) -> list[dict]:
    """Recognised characters joined into short phrases (a pause of 0.6 s or 16 characters ends one)."""
    rows, current = [], None
    for word in words:
        text = word.get("text") or ""
        if current and (word["start"] - current["end"] > 0.6 or len(current["text"]) >= 16):
            rows.append(current)
            current = None
        if current is None:
            current = {"text": text, "start": word["start"], "end": word["end"]}
        else:
            current["text"] += text
            current["end"] = word["end"]
    if current:
        rows.append(current)
    return rows


def transcript(bili, row, check=lambda: None) -> list[dict]:
    """What a video says, as timed phrases (cached per video: a news report does not change)."""
    cache = es._cache("news-transcript") / f"{row['id']}.json"
    try:
        return json.loads(cache.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    stream = bili.audio(row["id"])
    work = es._cache("news-audio")
    source, wav = work / f"{row['id']}.m4s", work / f"{row['id']}.wav"
    try:
        failures = []
        with throttle.slot("bili_download", check):
            for url in stream["urls"]:
                try:
                    with httpx.stream("GET", url, headers=stream["headers"], timeout=httpx.Timeout(15, read=20),
                                      follow_redirects=True) as response:
                        response.raise_for_status()
                        size = 0
                        with source.open("wb") as out:
                            for chunk in response.iter_bytes(1 << 16):
                                size += len(chunk)
                                if size > MAX_AUDIO_BYTES:
                                    raise ValueError("音频过大")
                                out.write(chunk)
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    failures.append(type(exc).__name__)
                    source.unlink(missing_ok=True)
            else:
                raise ValueError("音频下载失败：" + "、".join(failures))
        check()
        subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "16000",
                        "-f", "wav", str(wav)], check=True, timeout=180, capture_output=True)
        rows = segments(local_asr.transcribe(wav, check))
    finally:
        source.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), "utf-8")
    return rows


def match(shots: list[dict], spoken: dict, wanted=()) -> dict:
    """For each shot, the windows of the reports that talk about what its line talks about:
    {shot id: [(row, t, score)]}. spoken: {video id: (row, timed phrases)}; wanted: the script's names.
    A window is one to three consecutive phrases. Scripts are rewritten commentary, so the words rarely match
    the report's: a window must name what the line names (wanted), and among those the one sharing the most
    rare character pairs comes first. A line that names nothing gets no window."""
    windows = []
    for row, phrases in spoken.values():
        for i in range(len(phrases)):
            for span in (1, 2, 3):
                part = phrases[i:i + span]
                if len(part) < span:
                    break
                text = "".join(p["text"] for p in part)
                windows.append((row, part[0]["start"], part[-1]["end"], _pairs(text)))
    if not windows:
        return {}
    freq = {}
    for _, _, _, pairs in windows:
        for pair in pairs:
            freq[pair] = freq.get(pair, 0) + 1
    rarity = {pair: math.log(1 + len(windows) / count) for pair, count in freq.items()}
    texts = {}
    for row, phrases in spoken.values():
        for i in range(len(phrases)):
            for span in (1, 2, 3):
                part = phrases[i:i + span]
                if len(part) == span:
                    texts[(row["id"], part[0]["start"], part[-1]["end"])] = "".join(p["text"] for p in part)
    found = {}
    for shot in shots:
        want = _pairs(shot["narration"])
        total = sum(rarity.get(p, math.log(1 + len(windows))) for p in want)
        if not want or total <= 0:
            continue
        line_names = _named(shot["narration"], wanted)
        scored = []
        for row, start, end, pairs in windows:
            score = sum(rarity[p] for p in want & pairs) / total
            # A line that names nothing gets no news window: by wording alone a rewritten line meets the
            # report's filler ("和以色列，自己躲在后面" -> "弹兵在洞窟里") more often than its subject.
            shared = line_names & _named(texts[(row["id"], start, end)], wanted)
            if shared and score >= MIN_SCORE:
                scored.append((len(shared) + score, row, start, end))
        scored.sort(key=lambda item: -item[0])
        picked, sources = [], {}
        for score, row, start, end in scored:
            if sources.get(row["id"], 0) >= 2 or any(abs(start - s) < 8 and row["id"] == r["id"] for r, s, _ in picked):
                continue
            sources[row["id"]] = sources.get(row["id"], 0) + 1
            # Footage under these words: from the start of the phrase, as long as the shot.
            picked.append((row, start + min(float(shot.get("duration") or 4) / 2, max(0.5, (end - start) / 2)), round(score, 3)))
            if len(picked) >= PER_SHOT:
                break
        if picked:
            found[shot["id"]] = picked
    return found


def build(bili, queries, shots, check=lambda: None, notes=None, used=(), extra_queries=(), searched=()) -> dict:
    """The whole step: sources, their transcripts (in parallel, at most two at a time), and the matches.
    searched: B站 rows the footage search already found; those whose titles carry the script's names and run
    as long as a report come first (a search for the event alone finds few real reports).
    Returns {"sources": n, "matched": {shot id: [(row, t, score)]}, "seconds": s}."""
    began = time.monotonic()
    notes = notes if notes is not None else []
    script = "".join(s["narration"] for s in shots)
    # The event queries name the parties and places (the group queries also carry generic scene words).
    wanted = names(queries, script) or names(extra_queries, script)
    titled, seen = [], set(used)
    for row in searched:
        hits = len(_named(row.get("title") or "", wanted))
        if hits and row["id"] not in seen and MIN_SECONDS <= float(row.get("duration") or 0) <= MAX_SECONDS:
            seen.add(row["id"])
            titled.append((-hits, len(titled), row))
    rows = [row for *_, row in sorted(titled)][:MAX_SOURCES]
    try:
        rows += find_sources(bili, queries, seen)[:max(0, MAX_SOURCES - len(rows))]
    except Exception as exc:  # noqa: BLE001 — news mode is an extra; the searched footage still works
        notes.append(f"新闻模式检索失败：{type(exc).__name__}")
    if not rows:
        return {"sources": 0, "matched": {}, "seconds": round(time.monotonic() - began, 1)}
    spoken = {}
    from concurrent.futures import ThreadPoolExecutor

    def run(row):
        try:
            return row, transcript(bili, row, check)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"新闻转写失败 {row['id']}: {str(exc)[:80] or type(exc).__name__}")
            return row, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row, phrases in pool.map(run, rows):
            said = "".join(p["text"] for p in phrases)
            # Search results include music, games and vlogs: keep reports that name the event's names.
            if phrases and len(_named(said, wanted)) >= min(2, len(wanted) or 1):
                spoken[row["id"]] = (row, phrases)
    matched = match(shots, spoken, wanted)
    return {"sources": len(spoken), "matched": matched, "seconds": round(time.monotonic() - began, 1)}
