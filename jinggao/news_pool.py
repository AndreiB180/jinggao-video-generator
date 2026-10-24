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

try:
    from pypinyin import lazy_pinyin
except ImportError:      # an older install without it: names match by characters only
    lazy_pinyin = None

MAX_SOURCES = 5
HEARD_SECONDS = 300                           # transcribe a report's first minutes: the event is told up front
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


def _sounds(text: str) -> str:
    """Toneless pinyin as "|hu|sai|": what a name sounds like."""
    plain = _plain(text)
    return "|" + "|".join(lazy_pinyin(plain) if lazy_pinyin else plain) + "|"


def _named(text: str, wanted, sounds: str | None = None) -> set:
    """The names in text. Transcripts are matched by sound too: the speech model writes a name it does not
    know with other characters of the same sound (胡塞 as 胡赛, 泽连斯基 as 泽联斯基). sounds: _sounds(text)."""
    found = {name for name in wanted if name in text}
    rest = [name for name in wanted if name not in found]
    if rest and sounds is not None:
        found |= {name for name in rest if _sounds(name) in sounds}
    return found


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
        subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(source), "-t", str(HEARD_SECONDS),
                        "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)], check=True, timeout=180, capture_output=True)
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
        heard = [_sounds(p["text"]) for p in phrases]
        for i in range(len(phrases)):
            for span in (1, 2, 3):
                part = phrases[i:i + span]
                if len(part) == span:
                    texts[(row["id"], part[0]["start"], part[-1]["end"])] = (
                        "".join(p["text"] for p in part), "|".join(h.strip("|") for h in heard[i:i + span]).join("||"))
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
            said, heard = texts[(row["id"], start, end)]
            shared = line_names & _named(said, wanted, heard)
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


ONSITE_AHEAD = 14.0       # seconds after the words are said in which the report usually cuts to the scene
ONSITE_TILES = 4          # preview frames looked at per window
KIND_SYSTEM = """图片是新闻视频预览帧的联系表，每格左上角有黄色编号。判断每一格是哪种画面，只能是：
现场（事件现场、街道、人群、军队、武器、建筑、会场等实拍）、主播（演播室、主持人或记者正对镜头讲话、人物正脸讲话）、文字（图表、地图、截图、整屏标题为主）、其他（黑屏、模糊、片头、动画）。
只输出 JSON：{"1":"现场","2":"主播",...}，每个编号都要有。"""


def onsite(bili, matched: dict, check=lambda: None, notes=None) -> dict:
    """Moves each window onto the report's own footage of the scene. The words are often said by the
    presenter on camera and the report cuts to the scene a few seconds later: Bilibili's preview frames from
    the moment the words are said to ONSITE_AHEAD later are shown to the vision model (one contact sheet per
    report) and the window starts at the first one that is on-site footage; a window with none (presenter,
    text pages) is dropped, so no download is spent on it. A report whose previews cannot be read keeps its
    windows unchecked. Returns {shot id: [(row, t, score)]}."""
    from PIL import Image
    from . import express_match as em, footage_index, planner
    notes = notes if notes is not None else []
    by_video = {}
    for found in matched.values():
        for row, t, _ in found:
            by_video.setdefault(row["id"], (row, set()))[1].add(round(float(t), 2))
    kinds = {}           # (video id, t) -> kind, for the preview frames looked at
    near = {}            # (video id, window t) -> preview times in order
    for vid, (row, starts) in by_video.items():
        check()
        try:
            previews = bili.previews(vid, float(row.get("duration") or 0))
            picked = {}
            for t in sorted(starts):
                inside = [p for p in previews if t - 1 <= p["t"] <= t + ONSITE_AHEAD]
                step = max(1, len(inside) / ONSITE_TILES)
                chosen = [inside[int(i * step)] for i in range(min(ONSITE_TILES, len(inside)))]
                near[(vid, t)] = [p["t"] for p in chosen]
                picked.update({p["t"]: p for p in chosen})
            frames = [{"t": tt, "image": es.bilibili_tile(bili, p)} for tt, p in sorted(picked.items())]
            for start in range(0, len(frames), 24):
                part = frames[start:start + 24]
                sheet = footage_index._sheet(part, 6)
                content = [{"type": "text", "text": f"共 {len(part)} 格。"},
                           {"type": "image_url", "image_url": {"url": em._jpeg(sheet)}}]
                answer, _ = planner.chat(KIND_SYSTEM, content, vision=True, max_tokens=40 + 12 * len(part), timeout=60,
                                         task="express_pick")
                for key, value in (answer or {}).items() if isinstance(answer, dict) else []:
                    if str(key).isdigit() and 1 <= int(key) <= len(part):
                        kinds[(vid, part[int(key) - 1]["t"])] = str(value)
        except (planner.ModelUnavailable, Image.DecompressionBombError):
            raise
        except Exception as exc:  # noqa: BLE001 — this report's windows stay as they are
            notes.append(f"新闻预览检查失败 {vid}: {str(exc)[:60] or type(exc).__name__}")
            for t in starts:
                near.pop((vid, t), None)
    out, moved, dropped = {}, 0, 0
    for sid, found in matched.items():
        kept = []
        for row, t, score in found:
            times = near.get((row["id"], round(float(t), 2)))
            if times is None:              # not checked
                kept.append((row, t, score))
                continue
            field = [tt for tt in times if "现场" in kinds.get((row["id"], tt), "")]
            if not field:
                dropped += 1
                continue
            moved += field[0] > t + 1
            kept.append((row, max(float(t), field[0]), score))
        if kept:
            out[sid] = kept
    if near:
        notes.append(f"新闻模式：看了 {len(near)} 个报道片段的预览，{moved} 个挪到现场画面，{dropped} 个只有主播或文字已放弃")
    return out


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
            if phrases and len(_named(said, wanted, _sounds(said))) >= min(2, len(wanted) or 1):
                spoken[row["id"]] = (row, phrases)
    matched = match(shots, spoken, wanted)
    checked = False
    if matched:
        try:
            matched, checked = onsite(bili, matched, check, notes), True
        except Exception as exc:  # noqa: BLE001 — unchecked windows still go through QA
            notes.append(f"新闻预览检查未完成：{str(exc)[:80] or type(exc).__name__}")
    return {"sources": len(spoken), "matched": matched, "onsite": checked, "seconds": round(time.monotonic() - began, 1)}
