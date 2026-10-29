"""干净体检: fetch a few seconds of a library video at one described moment and judge, with production's own
checks, whether a film could use that picture clean.

The library (footage_index) knows what its preview frames show, not whether the picture itself is usable: the
client's worst complaints (a vertical video, black or blurred side bands, 花字 or a watermark across the
picture, burned subtitles that go with any crop he makes in 剪映) live in the stream, not in the description,
and 88% of the library windows the dorm runs downloaded failed. A check takes the DASH segments around the
moment at a low resolution (express_sources.dash_window, as production fetches a window), runs the same pixel
analysis (express_clean.analyse: bars, an inner picture, blurred sides, still or dark frames, a layout jump,
logos, subtitle lines; the source-wide marks from the preview sprites, as Matcher.analyse does), reads the
words burned into it (screen_text: contact details, 花字, scrolling lines; as Matcher.read_text) and, given a
local vision model, asks whether the window is filmed footage at all (a presenter, a text page, a cartoon, a
meme). One status per (video, moment) goes to footage_index.record_checks; find() then leaves the bad moments
out and ranks the clean ones first. scripts/footage_check.py runs it for days on the dorm PC; production
reuses check_window for library windows it has not seen checked.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from PIL import Image

from . import express_clean, express_sources as es, media, paths, screen_text, throttle

# Statuses (footage_index.record_checks): GOOD ones may be used, everything else keeps the moment out of films.
# "gone" and "lowres" are said of the whole video (footage_index.CHECK_VIDEO): only a definite answer of B站
# gives them. "outside" is one moment past the end of the stream (the library's length covers every part of a
# multi-part video, the stream only the first): that moment, not the video.
GOOD = {"clean", "crop"}
STATUSES = ("clean", "crop", "text", "mark", "narrow", "presenter", "cartoon", "still", "dark", "jump", "gone",
            "lowres", "outside")
# The version of the rules that decide a status. Bump it whenever one changes (screen_text.classify, the
# mapping below, the model's prompt): scripts/footage_check.py notes when a version first ran on an index and
# checks every moment checked before that again, so a fix reaches the verdicts already made.
# 2 (10-28): 花字 by character size, account names and platform marks, overlays by how they behave.
RULES = 2
OUT_ASPECT = 16 / 9     # the client's films are landscape: "narrow" is judged against 16:9, as Matcher.out_aspect
FILMED = {"实拍", "影视剧"}
BAD_SHARE = 0.35        # videos with more presenter/text/black frames than this are commentary (library_jobs)
CONTACT = "联系方式"     # note prefix: production drops every window of a video once one shows contact details
STREAM_TTL = 15 * 60    # play addresses expire after a while; a refetch costs two signed API calls
KEPT_STREAMS = 64
KEPT_SHEETS = 12        # decoded preview sheets kept (grey; about 1.4-13 MB each)
DESCRIBE_FRACTIONS = (0.15, 0.5, 0.85)


class Refused(RuntimeError):
    """B站 risk control refused a request (412/-412/-352…, a v_voucher): nothing was learnt about the window.
    Pause, then check it again; never record it."""


class NoStream(ValueError):
    """The play address came without any video stream and without a reason: under load B站 answers so for
    every video, so it is no verdict on this one (a failed check; several in a row are risk control)."""


_lock = threading.Lock()
_streams: OrderedDict = OrderedDict()     # (bvid, short_edge) -> (when, stream or a verdict)
_sheets: OrderedDict = OrderedDict()      # (bvid, sheet) -> [(t, grey PIL image)]
_profiles: OrderedDict = OrderedDict()    # (bvid, analysis size) -> express_clean.profile()


def worth_checking(row: dict, kinds: list) -> bool:
    """Videos the library would offer a film at all (footage_index.find's rules): long enough, not hours of a
    stream recording, no skipped title, and not a commentary video (mostly presenter, text or black frames)."""
    from .footage_index import BAD_KINDS, SKIP_TITLE
    duration = float(row.get("duration") or 0)
    if not 20 <= duration <= 3 * 3600 or SKIP_TITLE.search(str(row.get("title") or "")):
        return False
    return bool(kinds) and sum(k in BAD_KINDS for _, k in kinds) <= BAD_SHARE * len(kinds)


def moments_to_check(row: dict, kinds: list, *, limit: int = 4, seconds: float = 7.0, checked=()) -> list[float]:
    """Up to `limit` moments of one video worth a check, spread over it: filmed frames whose previous and next
    described frames are filmed too (a filmed frame between presenter or text frames is a commentary video
    cutting away for a second; production's library lane takes only these), where a window fits before the
    end, not checked already. kinds: [(t, kind)] of the video's described frames."""
    kinds = sorted((float(t), k) for t, k in kinds if float(t) >= 0)
    done = {round(float(t), 2) for t in checked}
    length = float(row.get("duration") or 0)
    good = []
    for i, (t, kind) in enumerate(kinds):
        if kind not in FILMED or any(k not in FILMED for _, k in kinds[max(0, i - 1):i + 2]):
            continue
        if round(t, 2) in done or (length and t > length - 1.0):
            continue
        good.append(t)
    if len(good) <= limit:
        return good
    if limit <= 1:
        return good[len(good) // 2:][:max(0, limit)]
    step = (len(good) - 1) / (limit - 1)
    return [good[round(i * step)] for i in range(limit)]


def fetch_window(stream: dict, start: float, seconds: float, out: Path, *, short_edge: int, check=None) -> None:
    """The download step (tests replace it): the DASH segments covering the window, trimmed locally."""
    es.dash_window(stream, start, seconds, out, check=check, short_edge=short_edge)


def _risky(exc) -> bool:
    return "风控" in str(exc)


def _stream(bili, bvid: str, short_edge: int) -> dict:
    """The play address (cached for every window of the video), or a verdict {"available": False, "gone"|…}."""
    key = (bvid, short_edge)
    with _lock:
        hit = _streams.get(key)
        if hit and time.time() - hit[0] < STREAM_TTL:
            _streams.move_to_end(key)
            return hit[1]
    try:
        stream = bili.stream(bvid, short_edge)
    except media.Cancelled:
        raise
    except ValueError as exc:
        if _risky(exc):
            raise Refused(str(exc)[:120]) from None
        if "不可用" not in str(exc):
            raise
        # The view or play-address answer says no: deleted, hidden, under review, paid or region-locked.
        stream = {"available": False, "gone": str(exc)[:80]}
    stream = stream or {"available": False, "gone": "没有播放地址"}
    if stream.get("available") or stream.get("gone") or 0 < int(stream.get("best_short_edge") or 0) < short_edge:
        # Only answers about the video are kept for its other windows; a refusal is asked again.
        with _lock:
            _streams[key] = (time.time(), stream)
            while len(_streams) > KEPT_STREAMS:
                _streams.popitem(last=False)
    return stream


def _drop_stream(bvid: str) -> None:
    with _lock:
        for key in [k for k in _streams if k[0] == bvid]:
            _streams.pop(key, None)


def _sprites(bili, row: dict, start: float) -> list:
    """Preview frames of the video in grey, as Matcher.sprite_frames: the sheets already on disk (at most
    four), else the one around the window. A refused preview call is a pause; any other failure leaves the
    check to the clip alone (no source-wide marks)."""
    bvid = row["id"]
    try:
        previews = bili.previews(bvid, float(row.get("duration") or 0))
        if not previews:
            return []
        meta, root = previews[0]["meta"], previews[0]["root"]
        sheets = sorted({p["sheet"] for p in previews if (Path(root) / f"sprite-{p['sheet']}.jpg").exists()})[:4]
        if not sheets:
            sheets = [min((abs(p["t"] - start), p["sheet"]) for p in previews)[1]]
        frames = []
        for sheet in sheets:
            key = (bvid, sheet)
            with _lock:
                cached = _sheets.get(key)
            if cached is None:
                cached = []
                with Image.open(bili.sprite(root, meta, sheet)) as im:
                    im = im.convert("L")
                    for p in previews:
                        if p["sheet"] != sheet:
                            continue
                        x, y = (p["cell"] % meta["x"]) * meta["w"], (p["cell"] // meta["x"]) * meta["h"]
                        if y + meta["h"] <= im.height:
                            cached.append((p["t"], im.crop((x, y, x + meta["w"], y + meta["h"]))))
                with _lock:
                    _sheets[key] = cached
                    while len(_sheets) > KEPT_SHEETS:
                        _sheets.popitem(last=False)
            frames += cached
        return frames
    except media.Cancelled:
        raise
    except Exception as exc:   # noqa: BLE001 — previews only add source-wide marks
        if _risky(exc):
            raise Refused(str(exc)[:120]) from None
        return []


def _analyse(bili, row: dict, path: Path, start: float, duration: float) -> tuple[dict, bool]:
    """express_clean.analyse of the window with the video's preview profile (Matcher.analyse); (result, sprites
    were available)."""
    from . import express_match
    info = next(s for s in media.probe(path)["streams"] if s.get("codec_type") == "video")
    size = (int(info["width"]), int(info["height"]))
    sprites = _sprites(bili, row, start)
    if sprites and size[1] > size[0] and sprites[0][1].width > sprites[0][1].height:
        # Portrait videos sit pillarboxed inside landscape preview cells: keep only the picture.
        cw, ch = sprites[0][1].size
        inner = max(2, round(ch * size[0] / size[1]))
        left = (cw - inner) // 2
        sprites = [(t, im.crop((left, 0, left + inner, ch))) for t, im in sprites]
    analysis = (express_clean.WIDTH, max(2, round(express_clean.WIDTH * size[1] / size[0])))
    key = (row["id"], analysis)
    with _lock:
        profile = _profiles.get(key)
    if profile is None:
        profile = express_clean.profile([im for _, im in sprites], analysis)
        with _lock:
            _profiles[key] = profile
            while len(_profiles) > KEPT_STREAMS:
                _profiles.popitem(last=False)
    broadcast = bool(express_match.BROADCAST.search(" ".join(str(row.get(k, "")) for k in ("title", "tags", "author"))))
    return express_clean.analyse(path, duration, size, OUT_ASPECT, profile, sprites, start,
                                 channel_logo=broadcast), bool(sprites)


def _read_text(path: Path, duration: float, clean: dict, check) -> tuple[dict, dict, set]:
    """Matcher.read_text without the side effects: (classify() result, the analysis re-cropped around the
    caption and title marks, soft flags). The frames go to classify too: text that pops in and out over one
    steady shot, or stays put over a cut, is an overlay however short it is."""
    with throttle.slot("frames", check):
        frames = screen_text.frames_of(path, duration)
        lines = [screen_text.read(frame) for frame in frames]
    aspect = frames[0].width / max(1, frames[0].height) if frames else OUT_ASPECT
    found = screen_text.classify(lines, clean.get("content") or (0.0, 0.0, 1.0, 1.0), aspect, frames)
    soft = set()
    if found["dense"]:
        soft |= {"wm", "center"}
    kinds = {m["kind"] for m in found["marks"]}
    if found["marks"]:
        if clean.get("crop"):
            clean = express_clean.recrop(clean, found["marks"], OUT_ASPECT)
        elif kinds - {"logo"}:
            soft.add("wm")
    return found, clean, soft


def _flags(clean: dict, soft: set) -> set:
    """Matcher.flags of one window."""
    flags = set(soft)
    motion = clean.get("motion") or {}
    if motion.get("still"):
        flags.add("still")
    if motion.get("dark"):
        flags.add("dark")
    if clean.get("layout_jump"):
        flags.add("jump")
    if express_clean.dirty(clean):
        flags.add("wm")
    if clean.get("narrow"):
        flags.add("narrow")
    return flags


def _unpainted(clean: dict) -> set:
    """Kinds of the marks left in view that neither the crop nor painting out removes."""
    return {r["kind"] for r in clean.get("residual") or []
            if not (clean.get("size") and express_clean.paintable(r, clean["crop"]))}


NAMES = {"logo": "角标", "subtitle": "字幕", "caption": "字幕", "banner": "横条", "center": "画面中间的字"}


def verdict(clean: dict, found: dict | None, soft: set) -> tuple[str, str]:
    """(status, note) from production's findings, most fundamental first. Production (Matcher.run's settle and
    _fetch_job) treats: dark and jump as QA failures; contact details and scrolling text as a failed download
    (contact: every window of the video); wm (花字 across the middle, marks the crop cannot clear) and narrow
    as "used only after pictures and borrowing"; still as a last resort. Only a window none of that applies to
    is used as it comes."""
    flags = _flags(clean, soft)
    found = found or {"contact": [], "moving": False, "dense": False, "marks": []}
    if "dark" in flags:
        return "dark", "黑屏或淡入黑场"
    if "narrow" in flags:
        return "narrow", "竖屏或两边黑边/模糊边"
    if "jump" in flags:
        return "jump", "画面版式中途跳变（片头或画中画切换）"
    if "still" in flags:
        return "still", "近似静态画面（图片或定格）"
    if found["contact"]:
        return "mark", f"{CONTACT}：" + "、".join(found["contact"])[:40]
    if found["moving"]:
        return "text", "滚动字幕"
    if found["dense"]:
        return "text", "画面中间有花字或大段文字"
    if clean.get("central"):
        return "mark", "画面中间有去不掉的水印"
    if "wm" in flags:
        left = _unpainted(clean)
        if left & {"subtitle", "caption", "center"}:
            return "text", "字幕或文字裁不掉"
        return "mark", "有去不掉的水印或台标"
    shown = sorted({NAMES.get(o["kind"], o["kind"]) for o in clean.get("overlays") or []})
    painted = [r for r in clean.get("residual") or [] if express_clean.paintable(r, clean["crop"])]
    if shown or painted:
        return "crop", "裁掉" + "、".join(shown or ["角标"]) + ("，抹掉小角标" if painted else "")
    return "clean", ""


def _crop(clean: dict) -> list | None:
    rect = clean.get("crop")
    if not rect or (rect[0] <= 0.005 and rect[1] <= 0.005 and rect[2] >= 0.995 and rect[3] >= 0.995):
        return None
    return [round(float(v), 4) for v in rect]


def check_window(bili, row: dict, t: float, *, short_edge: int = 480, seconds: float = 7.0, describe=None,
                 work: Path | None = None, check=None) -> dict:
    """Download [t-1, t-1+seconds] of the video at a low resolution and judge it like production does:
    {"status": one of STATUSES, "crop": [x,y,w,h] (fractions of the source picture) | None, "note": short
    reason, "seconds": wall time the check took}. The clip is deleted afterwards.

    describe(row, images) -> "filmed" | "presenter" | "text" | "cartoon" | "meme" | "overlay" (describer()) looks
    at three frames of a window that is otherwise clean; without it that part is skipped. Raises Refused when
    B站's risk control refused a request (pause and retry; nothing to record), NoStream when the answer had no
    video stream and no reason, and other errors for a failed download: none of them is a verdict."""
    began = time.monotonic()
    def done(status, crop=None, note=""):
        return {"status": status, "crop": crop, "note": note, "seconds": round(time.monotonic() - began, 2)}
    if row.get("provider", "bilibili") != "bilibili" or not row.get("id"):
        raise ValueError("只检查B站视频")
    if not express_clean.available():
        raise RuntimeError("缺少 numpy，无法分析画面")
    if check:
        check()
    stream = _stream(bili, row["id"], short_edge)
    if stream.get("gone"):
        return done("gone", note=stream["gone"])
    if not stream.get("available"):
        best = int(stream.get("best_short_edge") or 0)
        if stream.get("voucher"):
            raise Refused("B站没有给出视频流（v_voucher，风控）")
        if 0 < best < short_edge:
            return done("lowres", note=f"最高只有 {best}p")
        # No video stream at all (B站 under load answers so with code 0), or only ones larger than asked for.
        raise NoStream(f"B站没有给出可用的视频流（最高 {best}p）" if best else "B站没有给出视频流")
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if w and h and w < h:
        # A portrait stream is a vertical video: production never downloads it for a landscape film.
        return done("narrow", note="竖屏视频")
    length = float(stream.get("duration") or row.get("duration") or 0)
    if length and t > length - 0.5:
        return done("outside", note=f"时间点超出播放的视频长度（{length:.0f} 秒，分P视频的后几P）")
    start = max(0.0, float(t) - 1.0)
    if length and start + seconds > length - 0.1:
        start = max(0.0, length - 0.1 - seconds)   # near the end: the same moment, in a window that fits
    work = Path(work) if work else paths.DATA / "footage-check"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"check-{row['id']}-{int(float(t) * 100)}-{uuid.uuid4().hex[:6]}.mp4"
    try:
        try:
            fetch_window(stream, start, seconds, out, short_edge=short_edge, check=check)
        except media.Cancelled:
            raise
        except Exception:
            _drop_stream(row["id"])   # the address may have expired: the next try asks for a fresh one
            raise
        got = es.probe_duration(out)
        if got < min(seconds, 2.5) - 0.2:
            raise ValueError(f"下载片段时长不足（{got:.1f}/{seconds:.1f} 秒）")
        clean, sprites = _analyse(bili, row, out, start, got)
        if not clean:
            raise ValueError("画面分析没有结果（解码到的帧太少）")
        found, soft, notes = None, set(), []
        if _flags(clean, set()) & {"dark", "narrow", "jump", "still"}:
            pass   # the pixels decide it (verdict's first rules): no text to read, no model to ask
        elif screen_text.installed():
            found, clean, soft = _read_text(out, got, clean, check)
        else:
            notes.append("未识别文字")
        if not sprites:
            notes.append("无预览帧")
        status, note = verdict(clean, found, soft)
        if status in GOOD and describe is not None:
            try:
                with throttle.slot("frames", check):
                    images = screen_text.frames_of(out, got, fractions=DESCRIBE_FRACTIONS, width=640)
                kind = describe(row, images)
            except media.Cancelled:
                raise
            except Exception as exc:   # noqa: BLE001 — the local model is an extra; pixels and text still decided
                kind = None
                notes.append(f"看图失败：{type(exc).__name__}")
            judged = {"presenter": ("presenter", "主持人或博主对镜头讲话"), "text": ("text", "文字页、截图或图表为主"),
                      "cartoon": ("cartoon", "动画、游戏或CG画面"), "meme": ("text", "表情包或梗图"),
                      "overlay": ("text", "画面上叠加了花字、大字或水印（看图）")}.get(kind)
            if judged:
                status, note = judged
        note = "；".join([note] + notes) if note and notes else note or "；".join(notes)
        return done(status, _crop(clean) if status in GOOD else None, note[:80])
    finally:
        for leftover in (out, out.with_suffix(".src.mp4"), out.with_name(out.stem + ".part.mp4")):
            leftover.unlink(missing_ok=True)


def forget(bvid: str) -> None:
    """A video is done: drop its cached address and preview frames and delete its sprite sheets from disk (a
    run over the whole library would otherwise keep gigabytes of them); its preview metadata stays."""
    _drop_stream(bvid)
    with _lock:
        for key in [k for k in _sheets if k[0] == bvid]:
            _sheets.pop(key, None)
        for key in [k for k in _profiles if k[0] == bvid]:
            _profiles.pop(key, None)
    for sprite in (es._cache("bili-shots") / bvid).glob("sprite-*.jpg"):
        try:
            sprite.unlink(missing_ok=True)
        except OSError:
            pass   # open elsewhere (Windows): a production film may be cropping it


def clear_work(work: Path) -> None:
    """Clips left behind by a killed run."""
    if work.is_dir():
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ local vision model
DESCRIBE_PROMPT = """这三张图是同一段视频几秒之内按时间顺序截的画面。判断这段画面是哪一类，只能选一个：
实拍（真实拍摄的人物、场景、新闻现场、纪录片画面）、影视剧（电视剧、电影、短剧画面）、主持人出镜（有人正对镜头讲话：主播、演播室、UP主口播、对着镜头的采访）、文字页（以文字、图表、截图、聊天记录、PPT、地图为主）、动画（动画、CG、游戏、AI生成的画面）、表情包（梗图、贴纸、对镜头做鬼脸的夸张反应）、花字（画面中间或上面叠加了后期加上去的大字、花字、标题字、贴纸字或水印，挡住了画面）。
画面底部一行普通字幕、角落里的小台标或账号名不算花字；画面里本来就有的招牌、标语、证件上的字也不算。
三张里只要有一张是主持人出镜、文字页、动画、表情包或花字，就选那一类。只输出 JSON：{"类型":"实拍"}"""
KINDS = (("表情", "meme"), ("梗图", "meme"), ("花字", "overlay"), ("水印", "overlay"), ("大字", "overlay"),
         ("主持", "presenter"), ("口播", "presenter"), ("文字", "text"), ("图表", "text"), ("截图", "text"),
         ("动画", "cartoon"), ("游戏", "cartoon"), ("CG", "cartoon"), ("实拍", "filmed"), ("影视", "filmed"))


def parse_kind(answer: str) -> str | None:
    """The model's answer -> filmed / presenter / text / cartoon / meme / overlay (None when it named none)."""
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S)
    try:
        start, end = text.find("{"), text.rfind("}")
        value = json.loads(text[start:end + 1]).get("类型") if 0 <= start < end else None
    except (ValueError, AttributeError):
        value = None
    for source in (str(value or ""), text):
        for word, kind in KINDS:
            if word in source:
                return kind
    return None


def describer(url: str, model: str = "local", timeout: float = 120):
    """describe(row, images) -> kind through llama.cpp's server at url (…/v1), as footage_local.describer."""
    import httpx
    from .footage_local import _jpeg
    client = httpx.Client(timeout=timeout, trust_env=False)
    def describe(row, images):
        content = [{"type": "text", "text": f"视频标题：{str(row.get('title') or '')[:80]}"}]
        for n, image in enumerate(images, 1):
            content += [{"type": "text", "text": f"图{n}"}, {"type": "image_url", "image_url": {"url": _jpeg(image)}}]
        body = {"model": model, "temperature": 0.1, "max_tokens": 40,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": DESCRIBE_PROMPT}, {"role": "user", "content": content}]}
        response = client.post(url.rstrip("/") + "/chat/completions", json=body)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"].get("content") or ""
        kind = parse_kind(answer)
        if kind is None:
            raise ValueError(f"本地模型没有给出画面类型：{answer[:60]}")
        return kind
    return describe


def reachable(url: str) -> bool:
    import httpx
    try:
        return httpx.get(url.rstrip("/") + "/models", timeout=5, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False
