"""做封面: covers made on their own, without making a film (0.7.13).

The customer had some 70 films without covers and made them by hand (11-07), and reads drafts to a digital-human tool
whose videos need covers too. One run takes either a folder or titles pasted one a line:
- a video in the folder (镜稿's films or any other MP4): title from its file name, picture from its own frames, the
  covers saved next to it as <name>-封面-横版.jpg / -竖版.jpg (the names 镜稿 gives covers made with a film);
- a .txt draft (a 对标采集 batch folder: its 改写好的文案): title from its "标题：" line or file name, covers in a
  新封面 folder beside the drafts;
- a pasted title: covers in <成片>/做封面/<"11月7日 10点20分 封面12条">.
The way each is drawn is the one picked on the page (cover.py: 排版 about ¥0.001, AI 背景 + 排版 about ¥0.02,
AI 整张 gpt-image-2 about ¥0.03 a picture). Without a video there is no frame: the AI draws the picture, or the
排版 cover gets a plain dark background. Covers already there are left alone unless 重做 is ticked.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import config, cover, media, paths

VIDEOS = {".mp4", ".mov", ".mkv", ".flv", ".webm", ".m4v", ".avi", ".wmv"}
WORKERS = 4                 # covers at once: an AI one waits a minute or two on the image service
FRAMES = (0.05, 0.13, 0.22, 0.32, 0.42, 0.53, 0.65, 0.78)   # where in a video the candidate pictures are taken
MAX_ITEMS = 1000
NEW_COVERS = "新封面"
MODES = ("design", "ai_bg", "ai")

_lock = threading.Lock()
_stop = threading.Event()
_run: dict = {"running": False, "done": 0, "total": 0, "rows": [], "folder": "", "out": "", "mode": "", "sizes": "",
              "started": 0.0, "finished": 0.0, "note": ""}


def state() -> dict:
    with _lock:
        return {k: [dict(r) for r in v] if k == "rows" else v for k, v in _run.items()} | {
            "key": bool((config.read().get("cover_api_key") or "").strip()),
            "default_sizes": config.read().get("cover_sizes") or "both"}


def _stem_title(stem: str) -> str:
    """A film's name as its title: without the （2） a second film of one name gets."""
    return re.sub(r"(?:[（(](?:\d+|无字幕|带字幕)[）)])+$", "", stem).strip() or stem


def _draft_title(path: Path) -> str:
    try:
        head = path.read_text("utf-8", errors="ignore").strip().splitlines()[:1]
    except OSError:
        head = []
    if head and re.match(r"^\s*(标题|题目)\s*[:：]", head[0]):
        return cover.headline(head[0])
    return _stem_title(path.stem)


def film_titles(folder: Path) -> dict:
    """{film file name: what its cover should say} for 镜稿's films saved in folder: a film is named after its first
    sentence (the hook it opens with), while its project keeps the draft's title. Projects tidied away after a week
    leave the file name."""
    from . import store
    try:
        with store.connection() as con:
            rows = con.execute("SELECT body FROM projects WHERE body LIKE ?", (f"%{folder.name}%",)).fetchall()
    except Exception:   # noqa: BLE001 — no record: the file names
        return {}
    out = {}
    for row in rows:
        try:
            project = json.loads(row[0])
        except (TypeError, ValueError):
            continue
        title = cover.title_of(project)
        for export in project.get("exports") or []:
            saved = str(export.get("saved_to") or "").replace("\\", "/")
            if title and saved and saved.rsplit("/", 2)[-2:-1] == [folder.name]:
                out[saved.rsplit("/", 1)[-1]] = title
    return out


def items_in(folder: Path) -> list[dict]:
    """What a folder holds to make covers for: its videos, else its drafts (a 对标采集 batch folder: the ones in
    改写好的文案). Covers made earlier (…-封面-横版.jpg) are not videos or drafts, so they are never listed."""
    if not folder.is_dir():
        raise ValueError(f"文件夹不在了：{folder}")
    files = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name)
    videos = [p for p in files if p.suffix.lower() in VIDEOS]
    if videos:
        known = film_titles(folder)
        return [{"name": p.name, "title": known.get(p.name) or cover.title_of({"title": _stem_title(p.stem)})
                 or _stem_title(p.stem), "video": str(p), "out": str(p.parent), "stem": p.stem}
                for p in videos][:MAX_ITEMS]
    drafts_dir = folder / "改写好的文案" if (folder / "改写好的文案").is_dir() else folder
    drafts = sorted(drafts_dir.glob("*.txt"), key=lambda p: p.name)
    out = drafts_dir / NEW_COVERS
    return [{"name": p.name, "title": _draft_title(p), "video": "", "out": str(out), "stem": p.stem}
            for p in drafts][:MAX_ITEMS]


def items_from_titles(text: str) -> tuple[list[dict], Path]:
    titles = [t.strip() for t in str(text or "").splitlines() if len(re.sub(r"\s+", "", t)) >= 2]
    titles = list(dict.fromkeys(titles))[:MAX_ITEMS]
    out = media.output_folder() / "做封面" / media.batch_folder(len(titles), kind="封面")
    rows = []
    for title in titles:
        stem = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", title).strip(" .")[:60] or "封面"
        rows.append({"name": title, "title": title, "video": "", "out": str(out), "stem": stem})
    return rows, out


def _targets(item: dict, sizes: str) -> list[Path]:
    shapes = {"both": ["landscape", "portrait"], "landscape": ["landscape"], "portrait": ["portrait"]}[sizes]
    return [Path(item["out"]) / f"{item['stem']}-封面-{cover.SHAPES[s][1]}.jpg" for s in shapes]


def video_frames(video: Path) -> list[Image.Image]:
    """Candidate pictures from across a video (the film's own frames: 镜稿 films carry no burned-in subtitles unless
    the customer chose them), dark ones left out."""
    try:
        length = float(media.probe(video).get("format", {}).get("duration") or 0)
    except Exception:   # noqa: BLE001 — unreadable: no frames, the cover gets a plain background
        return []
    if length <= 0:
        return []
    out = []
    for at in FRAMES:
        picture = cover._frame(video, max(0.5, length * at), (1920, 1080))
        if picture is not None and cover._score(picture) > 0:
            out.append(picture)
    return out


TEXT_PLAN = ("要为一条视频做百家号封面。" + cover.PLAN_SYSTEM.split("2. ", 1)[1].split("只输出 JSON")[0]
             + '只输出 JSON：{"tag":"","main":"","sub":"","ask":""}')


def split_title(title: str) -> dict:
    """The title's parts for the 排版 cover when there is no picture to show the model: the low-cost model splits it
    (text only), else commas do."""
    from . import planner
    try:
        answer, _ = planner.chat(TEXT_PLAN, f"标题：{cover.headline(title)}", max_tokens=200, timeout=60,
                                 task="express_pick")
    except Exception:   # noqa: BLE001 — no key or budget: split at the commas
        return cover.plan_text(title)
    if not isinstance(answer, dict):
        return cover.plan_text(title)
    parts = cover.tidy(answer, title)
    return parts if parts["main"] else cover.plan_text(title)


def backdrop() -> Image.Image:
    """A plain dark red-to-black picture for a 排版 cover with no video behind it."""
    w, h = 1920, 1080
    base = Image.new("RGB", (w, h), (12, 10, 14))
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse((int(w * 0.45), int(-h * 0.3), int(w * 1.3), int(h * 1.1)), fill=150)
    base.paste((120, 18, 24), (0, 0), glow.filter(ImageFilter.GaussianBlur(160)))
    return base


def make_one(item: dict, mode: str, sizes: str) -> dict:
    """Covers for one item; returns {"files", "error"}. AI ways fall back to 排版 for whatever fails (cover.render)."""
    from . import costs
    cfg = config.read() | {"cover_mode": mode, "cover_sizes": sizes}
    title = item["title"]
    with costs.scope(f"cover-{uuid.uuid4().hex[:12]}", "", "cover"):
        frames = video_frames(Path(item["video"])) if item["video"] else []
        if frames:
            picture, parts = cover.choose(title, frames)
        else:
            picture, parts = None, split_title(title)
        if picture is None:
            picture, cfg = backdrop(), cfg | {"cover_reference": "0"}
        folder = Path(item["out"])
        folder.mkdir(parents=True, exist_ok=True)
        return cover.render(cfg, picture, parts, title, folder, item["stem"])


def start(folder: str = "", titles: str = "", mode: str = "ai", sizes: str = "", redo: bool = False) -> dict:
    if mode not in MODES:
        raise ValueError("封面方式无效")
    sizes = sizes or config.read().get("cover_sizes") or "both"
    if sizes not in ("both", "landscape", "portrait"):
        raise ValueError("尺寸无效")
    if mode in ("ai", "ai_bg") and not (config.read().get("cover_api_key") or "").strip():
        raise ValueError("AI 封面要先填 GrsAI 的 API Key")
    with _lock:
        if _run["running"]:
            raise ValueError("上一批封面还在做：等它做完，或先点「停止」")
    if folder.strip():
        source = Path(folder.strip().strip('"')).expanduser()
        items = items_in(source)
        where = str(source)
        if not items:
            raise ValueError("这个文件夹里没有视频，也没有 .txt 稿子")
    elif titles.strip():
        items, out = items_from_titles(titles)
        where = str(out)
        if not items:
            raise ValueError("先贴标题，一行一个")
    else:
        raise ValueError("先选文件夹，或贴标题")
    rows = []
    for item in items:
        targets = _targets(item, sizes)
        have = all(t.exists() for t in targets)
        skip = have and not redo
        # a skipped one shows the cover it already has
        rows.append(item | {"state": "skipped" if skip else "queued", "files": [str(t) for t in targets] if skip else [],
                            "error": ""})
    todo = [r for r in rows if r["state"] == "queued"]
    with _lock:
        _run.update(running=bool(todo), done=0, total=len(todo), rows=rows, folder=where, mode=mode, sizes=sizes,
                    out=items[0]["out"] if items else where, started=time.time(), finished=0.0 if todo else time.time(),
                    note="" if todo else "都已经有封面了：要重新做就勾上「重做已有的」")
        _stop.clear()
    if todo:
        threading.Thread(target=_work, args=(todo, mode, sizes), name="covers", daemon=True).start()
    return state()


def _work(todo: list[dict], mode: str, sizes: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def one(row: dict) -> None:
        if _stop.is_set():
            with _lock:
                row["state"] = "stopped"
            return
        with _lock:
            row["state"] = "working"
        try:
            result = make_one(row, mode, sizes)
            files = [f for f in result.get("files") or [] if Path(f).exists()]
            with _lock:
                row.update(state="done" if files else "failed", files=files,
                           error=("用了排版封面：" + result["error"]) if files and result.get("error") else
                           ("" if files else result.get("error") or "没做出来"))
            if result.get("error") in (cover.NO_CREDIT, cover.BAD_KEY):
                # every AI cover after it would fail the same way: stop instead of drawing them all as 排版 covers
                _stop.set()
                with _lock:
                    _run["note"] = result["error"] + "：弄好后勾「重做已有的」再点开始"
        except Exception as exc:  # noqa: BLE001 — the next one still goes
            with _lock:
                row.update(state="failed", error=str(exc)[:160])
        finally:
            with _lock:
                _run["done"] += 1

    try:
        with ThreadPoolExecutor(WORKERS, thread_name_prefix="cover") as pool:
            list(pool.map(one, todo))
    finally:
        with _lock:
            _run.update(running=False, finished=time.time())
            if _run["note"] and any(r["state"] == "stopped" for r in _run["rows"]):
                _run["note"] += f"（还有 {sum(r['state'] == 'stopped' for r in _run['rows'])} 条没做）"


def stop() -> dict:
    """停止: the covers being drawn finish, the rest are not started."""
    _stop.set()
    return state()


def open_folder() -> str:
    target = Path(_run.get("out") or _run.get("folder") or media.output_folder())
    if not target.exists():
        target = Path(_run.get("folder") or media.output_folder())
    if not target.exists():
        raise ValueError("文件夹不在了")
    media.open_path(target)
    return str(target)


def preview(path: str) -> Path:
    """A cover this run made (the page shows it small): only files listed in the current run's rows."""
    with _lock:
        made = {f for r in _run["rows"] for f in r.get("files") or []}
    if path not in made or not Path(path).is_file():
        raise ValueError("封面不在了")
    return Path(path)
