"""对标采集: benchmark authors on 好看视频, their recent popular videos, and their narration rewritten into scripts.

1. Authors' homepages (haokan.baidu.com/author/<id>) are read as a visitor opens them: the page carries the
   newest 16 videos (title, when, plays, length). A page that asks for a browser check (it does from abroad) is
   reported and left alone; the app never works around one.
2. The customer saves the videos he picks into 对标采集/1-放视频 himself (or a .txt of their narration).
3. Each file there is recognised locally (local_asr, as 音频批量 does), punctuated (script_norm), rewritten by the
   model (rewrite.rewrite, with its fact check against the original), and written to 2-待审 as <title>.txt, with
   the original narration beside it in 原文案; the file moves to 已转写.
4. The customer reads and edits 2-待审, then sends the scripts he keeps to a batch; they move to 3-已出片.

State lives in DATA/benchmark.json: {"authors": [...], "videos": {vid: {...}}}."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
from html import unescape
from pathlib import Path

from . import config, media, network, paths

PAGE = "https://haokan.baidu.com/author/{id}"
VIDEO = "https://haokan.baidu.com/v?vid={vid}"
AGENT = "Jinggao/0.7 (local video editor; benchmark list)"
GAP = 3.0                    # seconds between two authors' pages: a person opening tabs, not a crawler
STALE = 3 * 3600             # the list refreshes itself when opened after this long
MAX_AUTHORS = 50
MAX_VIDEOS = 3000
MEDIA = {".mp4", ".m4a", ".mp3", ".wav", ".aac", ".flv", ".mov", ".webm", ".mkv", ".ts", ".m4v"}
FOLDERS = ("1-放视频", "2-待审", "3-已出片", "原文案", "已转写", "封面")

_lock = threading.RLock()
_work = {"running": False, "done": 0, "total": 0, "current": "", "errors": [], "finished": 0.0}


def _file() -> Path:
    paths.prepare()
    return paths.DATA / "benchmark.json"


def _load() -> dict:
    try:
        data = json.loads(_file().read_text("utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"authors": list(data.get("authors") or []), "videos": dict(data.get("videos") or {}),
            "scanned": float(data.get("scanned") or 0)}


def _save(data: dict) -> None:
    videos = sorted(data["videos"].values(), key=lambda v: -float(v.get("published") or v.get("seen") or 0))
    data["videos"] = {v["vid"]: v for v in videos[:MAX_VIDEOS]}
    target = _file()
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    tmp.replace(target)


# ------------------------------------------------------------------ authors and their videos
def author_id(text: str) -> str:
    """The author id from a homepage link (haokan.baidu.com/author/1872764043922508, …?app_id=…) or the id."""
    text = str(text or "").strip()
    m = re.search(r"haokan\.baidu\.com/author/(\d{6,25})", text) or re.search(r"[?&]app_id=(\d{6,25})", text) \
        or re.fullmatch(r"(\d{6,25})", text)
    if not m:
        raise ValueError(f"认不出好看视频作者主页：{text[:60]}（应像 https://haokan.baidu.com/author/1872764043922508）")
    return m.group(1)


def add_authors(lines: str) -> dict:
    links = [x for x in re.split(r"[\s,，;；]+", str(lines or "")) if x.strip()]
    if not links:
        raise ValueError("请粘贴作者主页链接，一行一个")
    ids = [author_id(x) for x in links]
    with _lock:
        data = _load()
        known = {a["id"] for a in data["authors"]}
        for i in dict.fromkeys(ids):
            if i not in known:
                if len(data["authors"]) >= MAX_AUTHORS:
                    raise ValueError(f"最多 {MAX_AUTHORS} 个对标作者")
                data["authors"].append({"id": i, "name": "", "url": PAGE.format(id=i), "added": time.time(),
                                        "scanned": 0, "fans": "", "count": "", "error": ""})
                known.add(i)
        _save(data)
    return state()


def remove_author(aid: str) -> dict:
    with _lock:
        data = _load()
        data["authors"] = [a for a in data["authors"] if a["id"] != aid]
        data["videos"] = {k: v for k, v in data["videos"].items() if v.get("author") != aid}
        _save(data)
    return state()


def _state_json(page: str) -> dict:
    m = re.search(r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\})\s*;\s*(?:document|window|\(function|</script>)", page, re.S)
    if not m:
        raise ValueError("页面里没有作品列表")
    return json.loads(m.group(1))


def published(text: str, now: float | None = None) -> float:
    """A timestamp for 好看's relative dates: 刚刚, 5分钟前, 3小时前, 昨天, 2天前, 10月28日, 2025年10月28日, 2025-10-28."""
    now = time.time() if now is None else now
    text = str(text or "").strip()
    day = 86400
    if not text:
        return 0.0
    if text in ("刚刚", "今天"):
        return now
    if m := re.fullmatch(r"(\d+)\s*分钟前", text):
        return now - 60 * int(m.group(1))
    if m := re.fullmatch(r"(\d+)\s*小时前", text):
        return now - 3600 * int(m.group(1))
    if text == "昨天":
        return now - day
    if text == "前天":
        return now - 2 * day
    if m := re.fullmatch(r"(\d+)\s*天前", text):
        return now - day * int(m.group(1))
    if m := re.fullmatch(r"(\d+)\s*(?:周|星期)前", text):
        return now - 7 * day * int(m.group(1))
    local = time.localtime(now)
    m = re.fullmatch(r"(?:(\d{4})[年\-/.])?(\d{1,2})[月\-/.](\d{1,2})日?", text)
    if m:
        year = int(m.group(1) or local.tm_year)
        try:
            stamp = time.mktime((year, int(m.group(2)), int(m.group(3)), 12, 0, 0, 0, 0, -1))
        except (OverflowError, ValueError):
            return 0.0
        if not m.group(1) and stamp > now + day:          # 12月30日 seen in January: last year
            stamp = time.mktime((year - 1, int(m.group(2)), int(m.group(3)), 12, 0, 0, 0, 0, -1))
        return stamp
    return 0.0


def _seconds(text: str) -> int:
    parts = [int(p) for p in re.findall(r"\d+", str(text or ""))][-3:]
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def _plays(value) -> int:
    text = str(value or "").replace(",", "").strip()
    if m := re.fullmatch(r"([\d.]+)\s*万", text):
        return int(float(m.group(1)) * 10000)
    if m := re.fullmatch(r"([\d.]+)\s*亿", text):
        return int(float(m.group(1)) * 100000000)
    try:
        return int(float(text))
    except ValueError:
        return 0


def parse_author(page: str, now: float | None = None) -> tuple[dict, list[dict]]:
    """(author info, videos) from an author's homepage."""
    if "mkdcheck" in page and "__PRELOADED_STATE__" not in page:
        raise PermissionError("好看视频要求先在浏览器里验证（在国外网络或访问太频繁时会这样），镜稿不绕过，稍后再试")
    data = _state_json(page)
    info = data.get("author_info") or {}
    author = {"name": unescape(str(info.get("name") or ""))[:40], "fans": str(info.get("fansCntText") or ""),
              "count": str(info.get("videoCntText") or "")}
    videos = []
    for item in (data.get("video") or {}).get("results") or []:
        c = item.get("content") if isinstance(item, dict) else None
        if not isinstance(c, dict) or not c.get("vid"):
            continue
        vid = str(c["vid"])
        videos.append({"vid": vid, "title": unescape(str(c.get("title") or "")).strip()[:120],
                       "published_text": str(c.get("publish_time") or ""), "published": published(c.get("publish_time"), now),
                       "plays": _plays(c.get("playcnt")), "duration": _seconds(c.get("duration")),
                       "cover": str(c.get("cover_src") or c.get("poster") or ""), "url": VIDEO.format(vid=vid)})
    return author, videos


def _fetch(aid: str) -> str:
    with network.client(timeout=20, headers={"User-Agent": AGENT}, follow_redirects=True) as client:
        r = client.get(PAGE.format(id=aid))
        r.raise_for_status()
        return r.text


def scan(check=lambda: None) -> dict:
    """Every author's homepage once, a few seconds apart; new videos join the list, known ones update plays."""
    with _lock:
        authors = [a["id"] for a in _load()["authors"]]
    for n, aid in enumerate(authors):
        check()
        if n:
            time.sleep(GAP)
        now = time.time()
        error, author, found = "", {}, []
        try:
            author, found = parse_author(_fetch(aid), now)
        except PermissionError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — one author failing leaves the others
            error = f"读取失败：{type(exc).__name__}"
        with _lock:
            data = _load()
            for a in data["authors"]:
                if a["id"] == aid:
                    a.update({k: v for k, v in author.items() if v}, scanned=now, error=error)
            for v in found:
                old = data["videos"].get(v["vid"]) or {}
                stamp = v["published"] if not old.get("published") else old["published"]
                data["videos"][v["vid"]] = old | v | {"author": aid, "seen": old.get("seen") or now,
                                                       "published": stamp, "status": old.get("status", "")}
            data["scanned"] = now
            _save(data)
        if error.startswith("好看视频要求"):
            break       # the site wants a browser check from this network: the others would ask too
    return state()


def popular(days: float = 3, plays: int = 10000, include_done: bool = False) -> list[dict]:
    data = _load()
    names = {a["id"]: a.get("name") or a["id"] for a in data["authors"]}
    since = time.time() - days * 86400 if days else 0
    rows = [v | {"author_name": names.get(v.get("author"), "")} for v in data["videos"].values()
            if v.get("author") in names and int(v.get("plays") or 0) >= plays
            and (not since or float(v.get("published") or v.get("seen") or 0) >= since)
            and (include_done or v.get("status") != "copied")]
    return sorted(rows, key=lambda v: -int(v.get("plays") or 0))


def mark(vids: list[str], status: str) -> None:
    if status not in ("", "copied"):
        raise ValueError("状态无效")
    with _lock:
        data = _load()
        for vid in vids:
            if vid in data["videos"]:
                data["videos"][vid]["status"] = status
        _save(data)


def save_covers(vids: list[str]) -> int:
    """The picked videos' covers into 对标采集/封面 as <title>.jpg: the full picture when the address's resize
    suffix ("@s_0,w_660,…") can be left off, else the size the homepage shows."""
    data = _load()
    folder = root() / "封面"
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    with network.client(timeout=20, headers={"User-Agent": AGENT}, follow_redirects=True) as client:
        for vid in vids:
            video = data["videos"].get(vid) or {}
            address = str(video.get("cover") or "")
            if not address.startswith("https://"):
                continue
            target = folder / f"{_safe(video.get('title') or vid)}.jpg"
            if target.exists():
                saved += 1
                continue
            for url in dict.fromkeys([address.split("@", 1)[0], address]):
                try:
                    r = client.get(url)
                    if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") \
                            and len(r.content) > 2000:
                        target.write_bytes(r.content)
                        saved += 1
                        break
                except Exception:  # noqa: BLE001 — the next address, then the next video
                    continue
    return saved


def state(days: float = 3, plays: int = 10000, include_done: bool = False) -> dict:
    data = _load()
    return {"authors": data["authors"], "scanned": data["scanned"], "stale": time.time() - data["scanned"] > STALE,
            "videos": popular(days, plays, include_done), "known": len(data["videos"]), "folders": folders(),
            "work": dict(_work), "pending": [p.name for p in waiting()], "review": review()}


# ------------------------------------------------------------------ files: recognise, rewrite, review
def root() -> Path:
    return media.output_folder() / "对标采集"


def folders() -> dict:
    base = root()
    out = {}
    for name in FOLDERS:
        folder = base / name
        folder.mkdir(parents=True, exist_ok=True)
        out[name] = str(folder)
    return out


def waiting() -> list[Path]:
    folder = root() / FOLDERS[0]
    folder.mkdir(parents=True, exist_ok=True)
    return sorted((p for p in folder.iterdir() if p.is_file() and (p.suffix.lower() in MEDIA or p.suffix.lower() == ".txt")),
                  key=lambda p: p.stat().st_mtime)


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", str(name or "")).strip(" .")[:60] or "对标文案"


def _unique(folder: Path, stem: str, suffix: str) -> Path:
    target = folder / f"{stem}{suffix}"
    n = 2
    while target.exists():
        target = folder / f"{stem}（{n}）{suffix}"
        n += 1
    return target


def _title_for(path: Path) -> str:
    """The benchmark video's title when the file is named after it (download tools usually are), else the name."""
    stem = re.sub(r"[_\-\s]*(?:\d{3,4}p|hd|高清|超清|无水印)$", "", path.stem, flags=re.I).strip() or path.stem
    data = _load()
    flat = lambda s: re.sub(r"[\W_]+", "", s)
    for v in data["videos"].values():
        if v.get("title") and (flat(v["title"])[:20] == flat(stem)[:20] or flat(stem) in flat(v["title"])):
            return v["title"]
    return stem


def narration(path: Path, check=lambda: None) -> str:
    """The original narration of one file: its text, or what local recognition hears in it."""
    if path.suffix.lower() == ".txt":
        raw = path.read_text("utf-8", errors="ignore")
        if len(re.sub(r"\s+", "", raw)) < 30:
            raise ValueError("文案太短（不到 30 个字）")
        return raw.strip()
    from . import local_asr, script_norm
    with tempfile.TemporaryDirectory(prefix="benchmark-") as directory:
        wav = Path(directory) / "speech.wav"
        media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-vn", "-ar", "16000", "-ac", "1",
                   "-c:a", "pcm_s16le", str(wav)], check=check)
        words = local_asr.transcribe(wav, check)
    raw = "".join(str(w.get("text", "")) for w in words).strip()
    if len(raw) < 30:
        raise ValueError("没听出口播（不到 30 个字），可能是纯音乐或没有人声")
    script, _ = script_norm.normalize(raw, check)
    return script


def process_one(path: Path, check=lambda: None) -> Path:
    from . import rewrite
    title = _title_for(path)
    original = narration(path, check)
    cfg = config.read()
    minutes = float(cfg.get("benchmark_minutes") or 0) or None
    result = rewrite.rewrite(original, title=title, minutes=minutes, project_id="benchmark", check=check)
    base = root()
    stem = _safe(result.get("title") or title)
    (_unique(base / "原文案", _safe(title), ".txt")).write_text(original + "\n", "utf-8")
    out = _unique(base / "2-待审", stem, ".txt")
    out.write_text(result["script"].strip() + "\n", "utf-8")
    shutil.move(str(path), str(_unique(base / "已转写", path.stem, path.suffix)))
    return out


def process_all() -> dict:
    """Recognise and rewrite every file waiting in 1-放视频, one after another, in the background."""
    with _lock:
        if _work["running"]:
            return dict(_work)
        files = waiting()
        if not files:
            raise ValueError("「1-放视频」里还没有视频、音频或文案")
        if not config.read().get("model_api_key"):
            raise ValueError("请先在服务与设置中填写模型 API Key")
        _work.update(running=True, done=0, total=len(files), current="", errors=[], finished=0.0)

    def run():
        try:
            for path in files:
                _work["current"] = path.name
                try:
                    if path.exists():
                        process_one(path)
                except Exception as exc:  # noqa: BLE001 — the next file still goes
                    _work["errors"].append(f"{path.name}：{exc}"[:160])
                _work["done"] += 1
        finally:
            _work.update(running=False, current="", finished=time.time())

    threading.Thread(target=run, name="benchmark", daemon=True).start()
    return dict(_work)


def review() -> list[dict]:
    folder = root() / FOLDERS[1]
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(folder.glob("*.txt"), key=lambda p: p.stat().st_mtime):
        text = p.read_text("utf-8", errors="ignore")
        out.append({"name": p.name, "title": p.stem, "chars": len(re.sub(r"\s+", "", text)), "text": text[:4000],
                    "modified": p.stat().st_mtime})
    return out


def save_review(name: str, text: str) -> None:
    path = _review_path(name)
    if not path.exists():
        raise ValueError("这篇稿子已经不在「2-待审」里了")
    path.write_text(str(text or "").strip() + "\n", "utf-8")


def drop_review(name: str) -> None:
    _review_path(name).unlink(missing_ok=True)


def _review_path(name: str) -> Path:
    folder = (root() / FOLDERS[1]).resolve()
    name = str(name or "")
    path = (folder / name).resolve()
    if Path(name).name != name or path.parent != folder or path.suffix.lower() != ".txt":
        raise ValueError("文件名无效")
    return path


def take_reviewed(names: list[str]) -> list[dict]:
    """The scripts sent to a batch: [{title, script}], their files moved to 3-已出片."""
    out = []
    for name in names:
        path = _review_path(name)
        if not path.exists():
            continue
        text = path.read_text("utf-8", errors="ignore").strip()
        if len(re.sub(r"\s+", "", text)) < 30:
            continue
        out.append({"title": path.stem[:100], "script": text})
        shutil.move(str(path), str(_unique(root() / FOLDERS[2], path.stem, ".txt")))
    return out
