"""本地素材索引：B站视频的标题、标签、预览帧和画面描述，不下载视频本身。

Each indexed video keeps its search row (title, tags, length, uploader) and 8-16 "moments": preview
frames from Bilibili's own thumbnail sprites, each with a one-line description written by the vision
model from a contact sheet (about 0.00003 元 a frame). Text is searched with Chinese character
bigrams in SQLite FTS5, so a shot's query finds the video and the moment in it that shows the thing.
Production downloads only the chosen few seconds, as before.

The index grows two ways: bulk builds from topic packs or keywords (build()), and every production run's
live search results (add_seen()). The described part ships with the app as gzip JSON lines
(assets/footage-index, export_seed/install_seed).

Three processes use the file at once (the server, the production worker, the index-building worker):
every write is its own short BEGIN IMMEDIATE transaction, reads need no lock (WAL), and nothing holds
a connection open between calls.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager

from . import costs, execution, paths
from .footage_topics import PACKS   # topic -> Bilibili searches

MOMENTS = 10            # preview frames kept (and described) per video when the length is unknown
SHEET = (6, 4)          # describe sheet: columns x rows of frames
SEARCH_WORKERS = 3      # Bilibili's search API answers bursts with 412 risk control
SEARCH_GAP = 0.4        # seconds between one search's result pages
SHOT_WORKERS = 8        # videoshot metadata (same API gateway as search) and its CDN sprites
DESCRIBE_WORKERS = 10   # the background worker allows 10 model calls at a time (execution.MODEL_SLOTS)
MAX_TRIES = 2           # a video whose sheets failed this often is left out of later builds
SKIP_TITLE = re.compile(r"直播|回放|教程|PPT|课件|歌曲|MV|翻唱|游戏|手游|原神|王者荣耀|和平精英|鬼畜|reaction|开箱|测评|"
                        r"带货|广告|招聘|网课|考研|公务员|英语|单词")
BAD_KINDS = {"主持人出镜", "图表文字", "黑屏或模糊"}
DESCRIBE_SYSTEM = """图片是视频预览帧的联系表，每格左上角有黄色编号。为每一格写一句画面描述（最多16个字，不要标点结尾）：主体、动作、场景，能看出时代的写上（古代/近代/现代），能认出地名或事物的写上；再给画面类型，只能是：实拍、影视剧、动画、图表文字、主持人出镜、黑屏或模糊。
只输出 JSON：{"1":["描述","类型"],"2":["描述","类型"],...}，每个编号都要有。"""
SEED = paths.ROOT / "assets" / "footage-index"
SEED_PART_BYTES = 40 * 1024 * 1024     # GitHub refuses files over 100 MB and warns over 50 MB
BATCH = 500                            # videos per write transaction when installing or merging

_lock = threading.Lock()               # one writer at a time inside a process (SQLite serialises processes)
_ready, _ready_lock = set(), threading.Lock()
CJK = re.compile(r"[㐀-鿿]+")
WORD = re.compile(r"[A-Za-z0-9]{2,}")
BV = re.compile(r"BV[0-9A-Za-z]{10}")


def grams(text: str) -> list[str]:
    """Searchable terms: two-character pieces of every Chinese run (one character alone), plus words."""
    out = []
    for run in CJK.findall(text or ""):
        out += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    out += [w.lower() for w in WORD.findall(text or "")]
    return list(dict.fromkeys(out))


def _file():
    return paths.DATA / "footage-index.sqlite"


def _schema(con):
    paths.wal(con)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY, provider TEXT NOT NULL, id TEXT NOT NULL, row TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '', added REAL NOT NULL, described INTEGER NOT NULL DEFAULT 0,
            used INTEGER NOT NULL DEFAULT 0, flags TEXT NOT NULL DEFAULT '', tries INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS moments (
            source TEXT NOT NULL, t REAL NOT NULL, hash TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, t));
        CREATE VIRTUAL TABLE IF NOT EXISTS terms USING fts5(source UNINDEXED, t UNINDEXED, words);
        CREATE TABLE IF NOT EXISTS term_rows (source TEXT NOT NULL, rid INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS term_rows_source ON term_rows(source);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    con.execute("BEGIN IMMEDIATE")
    try:
        try:   # indexes made before 'tries' existed
            con.execute("ALTER TABLE sources ADD COLUMN tries INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        # term_rows lets a video's terms be deleted by rowid (an UNINDEXED FTS column would scan the table).
        if (not con.execute("SELECT 1 FROM term_rows LIMIT 1").fetchone()
                and con.execute("SELECT 1 FROM terms LIMIT 1").fetchone()):
            con.execute("INSERT INTO term_rows (source, rid) SELECT source, rowid FROM terms")
        con.commit()
    except BaseException:
        con.rollback()
        raise


@contextmanager
def _conn(write: bool = False, timeout: float = 30):
    """A short-lived connection: commits on success, rolls back on error, always closes."""
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        with _ready_lock:
            if str(path) not in _ready:
                _schema(con)
                _ready.add(str(path))
        if write:
            con.execute("BEGIN IMMEDIATE")     # the guard in _add and its writes see the same state
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def _key(row) -> str:
    return f"{row['provider']}:{row['id']}"


def known(keys) -> set:
    keys = list(keys)
    found = set()
    with _conn() as con:
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            found |= {r["key"] for r in con.execute(
                f"SELECT key FROM sources WHERE key IN ({','.join('?' * len(part))})", part)}
    return found


def add(row: dict, moments: list[dict], *, topic: str = "", described: bool = False) -> None:
    """Save (or refresh) one video: its search row and moments [{t, hash, text, kind}]."""
    with _lock, _conn(write=True) as con:
        _add(con, row, moments, topic=topic, described=described)


def _add(con, row, moments, *, topic="", described=False) -> bool:
    """One video, inside the caller's write transaction; False when a described copy is already there."""
    key = _key(row)
    old = con.execute("SELECT described, used, topic, tries FROM sources WHERE key=?", (key,)).fetchone()
    if old and old["described"] and not described:
        return False      # never replace described moments with bare ones
    keep = {k: row.get(k) for k in ("provider", "id", "url", "title", "tags", "description", "duration", "author",
                                    "typename", "play", "pubdate", "query")}
    usable = [m for m in moments if m.get("kind") not in BAD_KINDS]
    hosted = sum(1 for m in moments if m.get("kind") == "主持人出镜")
    flags = "host" if moments and hosted * 2 >= len(moments) else ""
    con.execute("INSERT OR REPLACE INTO sources (key,provider,id,row,topic,added,described,used,flags,tries) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (key, row["provider"], row["id"], json.dumps(keep, ensure_ascii=False),
                 topic or (old["topic"] if old else ""), time.time(), int(described), old["used"] if old else 0, flags,
                 0 if described or not old else old["tries"]))
    if old:
        con.execute("DELETE FROM moments WHERE source=?", (key,))
        con.execute("DELETE FROM terms WHERE rowid IN (SELECT rid FROM term_rows WHERE source=?)", (key,))
        con.execute("DELETE FROM term_rows WHERE source=?", (key,))
    con.executemany("INSERT OR REPLACE INTO moments (source,t,hash,text,kind) VALUES (?,?,?,?,?)",
                    [(key, round(float(m["t"]), 2), str(m.get("hash", "")), str(m.get("text") or ""),
                      str(m.get("kind") or "")) for m in moments])
    head = " ".join(grams(" ".join(str(row.get(k) or "") for k in ("title", "tags", "query", "typename"))))
    rows = [(-1.0, head)] + [(round(float(m["t"]), 2), " ".join(grams(m["text"]))) for m in usable if m.get("text")]
    rids = []
    for t, words in rows:
        rids.append((key, con.execute("INSERT INTO terms (source,t,words) VALUES (?,?,?)", (key, t, words)).lastrowid))
    con.executemany("INSERT INTO term_rows (source, rid) VALUES (?,?)", rids)
    return True


def add_seen(rows: list[dict], tiles: list[dict], topic: str = "") -> int:
    """Production found these videos live: keep the new ones (title text and frame times; described in a
    later build). One short transaction; a busy index is skipped rather than holding up the film."""
    by_source = {}
    for tile in tiles:
        source = tile["source"]
        if source.get("provider") != "bilibili" or source.get("indexed"):
            continue
        by_source.setdefault(_key(source), []).append({"t": tile["t"], "hash": format(tile.get("hash") or 0, "x")})
    wanted = {}
    for row in rows:
        if row.get("provider") == "bilibili" and not row.get("indexed") and _key(row) in by_source:
            wanted.setdefault(_key(row), row)
    if not wanted:
        return 0
    have = known(wanted)
    new = 0
    with _lock, _conn(write=True, timeout=3) as con:
        for key, row in wanted.items():
            if key not in have and _add(con, row, by_source[key], topic=topic):
                new += 1
    return new


def search(query: str, limit: int = 8, *, exclude: set | None = None) -> list[dict]:
    """Videos whose title or described frames match the query, best first. Each row is shaped like a
    live Bilibili search row plus "indexed", "moments" (matching frame times), "avoid" (frames showing a
    presenter, a text page or black) and "index_score"."""
    words = grams(query)[:60]
    if not words:
        return []
    expression = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
    with _conn() as con:
        try:
            hits = con.execute("SELECT source, t, bm25(terms) AS rank FROM terms WHERE terms MATCH ? "
                               "ORDER BY rank LIMIT 600", (expression,)).fetchall()
        except sqlite3.OperationalError:
            return []
        by = {}
        for hit in hits:
            entry = by.setdefault(hit["source"], {"title": 0.0, "moments": []})
            if float(hit["t"]) < 0:
                entry["title"] = -float(hit["rank"])
            else:
                entry["moments"].append((-float(hit["rank"]), float(hit["t"])))
        scored = []
        for key, entry in by.items():
            if exclude and key in exclude:
                continue
            best = sorted(entry["moments"], reverse=True)
            score = entry["title"] * 0.6 + (best[0][0] if best else 0) + 0.3 * min(3, len(best))
            scored.append((score, key, [t for _, t in best[:6]]))
        scored.sort(reverse=True)
        out = []
        for score, key, moments in scored[:limit * 3]:
            src = con.execute("SELECT row, flags, used FROM sources WHERE key=?", (key,)).fetchone()
            if not src or src["flags"] == "host":
                continue
            row = json.loads(src["row"])
            avoid = [r["t"] for r in con.execute(
                f"SELECT t FROM moments WHERE source=? AND kind IN ({','.join('?' * len(BAD_KINDS))})",
                (key, *sorted(BAD_KINDS)))]
            out.append(row | {"indexed": True, "moments": moments, "avoid": avoid, "index_score": round(score, 3),
                              "index_used": src["used"]})
            if len(out) >= limit:
                break
        return out


def _valid(row) -> bool:
    return isinstance(row, dict) and row.get("provider") == "bilibili" and bool(BV.fullmatch(str(row.get("id", ""))))


def _write_batch(items) -> int:
    """[(row, moments, topic, described)] in one write transaction; returns how many were saved."""
    if not items:
        return 0
    with _lock, _conn(write=True) as con:
        return sum(1 for row, moments, topic, described in items
                   if _add(con, row, moments, topic=topic, described=described))


def merge(other_file) -> int:
    """Take in another machine's index: videos this one lacks, or has without descriptions."""
    from pathlib import Path
    try:
        # A read-only URI (percent-encoded, forward slashes) also works for Windows paths with Chinese names.
        other = sqlite3.connect(Path(other_file).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        raise ValueError("素材库文件无法读取") from None
    other.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in other.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"sources", "moments"} <= tables:
            raise ValueError("不是镜稿素材库文件")
        with _conn() as con:
            mine = {r["key"]: r["described"] for r in con.execute("SELECT key, described FROM sources")}
        added, batch = 0, []
        for src in other.execute("SELECT key, row, topic, described FROM sources"):
            if src["key"] in mine and (mine[src["key"]] or not src["described"]):
                continue
            try:
                row = json.loads(src["row"])
            except (TypeError, ValueError):
                continue
            if not _valid(row):
                continue
            moments = [dict(m) for m in other.execute("SELECT t, hash, text, kind FROM moments WHERE source=?",
                                                     (src["key"],))]
            batch.append((row, moments, str(src["topic"] or "")[:30], bool(src["described"])))
            if len(batch) >= BATCH:
                added += _write_batch(batch)
                batch = []
        return added + _write_batch(batch)
    except sqlite3.DatabaseError:
        raise ValueError("素材库文件无法读取") from None
    finally:
        other.close()


def export_seed(folder=SEED, part_bytes: int = SEED_PART_BYTES) -> list:
    """The described part of the index as gzip JSON lines in numbered parts (shipped with the app). Each
    part is written beside its final name and moved into place; parts the new export no longer has go last,
    so an interrupted export never leaves the folder empty or a part cut short."""
    import gzip
    import io
    folder.mkdir(parents=True, exist_ok=True)
    for leftover in folder.glob("*.tmp"):
        leftover.unlink()
    with _conn() as con:
        sources = con.execute("SELECT key, row, topic FROM sources WHERE described=1 ORDER BY added").fetchall()
        moments = {}
        for m in con.execute("SELECT source, t, hash, text, kind FROM moments WHERE source IN "
                             "(SELECT key FROM sources WHERE described=1) ORDER BY source, t"):
            moments.setdefault(m["source"], []).append(
                {"t": m["t"], "hash": m["hash"], "text": m["text"], "kind": m["kind"]})
    # gzip holds output back while writing, so parts are cut by raw size: this JSON shrinks 3-4 times.
    parts, lines, raw = [], [], 0
    def close():
        path = folder / f"part-{len(parts) + 1:02d}.jsonl.gz"
        temporary = path.with_name(path.name + ".tmp")
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as writer:
            writer.write(b"".join(lines))
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, path)
        parts.append(path)
        lines.clear()
    for src in sources:
        line = json.dumps({"row": json.loads(src["row"]), "topic": src["topic"],
                           "moments": moments.get(src["key"], [])}, ensure_ascii=False, separators=(",", ":"))
        lines.append((line + "\n").encode("utf-8"))
        raw += len(lines[-1])
        if raw >= part_bytes * 3:
            close()
            raw = 0
    if lines or not parts:
        close()
    for stale in folder.glob("part-*.jsonl.gz"):
        if stale not in parts:
            stale.unlink()
    return parts


def _rich(moments) -> bool:
    """Described frame by frame by the local model, with search words ("描述｜关键词", footage_local)."""
    return any("｜" in str(m.get("text") or "") for m in moments)


def install_seed(folder=SEED, progress=lambda *_: None) -> int:
    """Take in the index shipped with the app once per seed version: videos this machine lacks, and better
    descriptions (frame by frame, with search words) of videos it only has contact-sheet lines for; its
    own videos stay otherwise. Written in batches, so closing the app part-way loses little and the next
    start continues; the installed version is recorded inside the index, so a new or reset index file
    is seeded again."""
    import gzip
    import hashlib
    parts = sorted(folder.glob("part-*.jsonl.gz")) if folder.is_dir() else []
    if not parts:
        return 0
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.read_bytes())
    fingerprint = digest.hexdigest()
    with _conn() as con:
        installed = con.execute("SELECT value FROM meta WHERE key='seed'").fetchone()
        if installed and installed["value"] == fingerprint and con.execute(
                "SELECT 1 FROM sources WHERE described=1 LIMIT 1").fetchone():
            return 0
        described = {r["key"] for r in con.execute("SELECT key FROM sources WHERE described=1")}
        rich = {r["source"] for r in con.execute("SELECT DISTINCT source FROM moments WHERE text LIKE '%｜%'")}
    added, batch, broken = 0, [], []
    for index, part in enumerate(parts):
        try:
            with gzip.open(part, "rt", encoding="utf-8") as lines:
                for line in lines:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    row = item.get("row") if isinstance(item, dict) else None
                    if not _valid(row):
                        continue
                    moments = [m for m in item.get("moments") or [] if isinstance(m, dict) and "t" in m]
                    if _key(row) in described and (_key(row) in rich or not _rich(moments)):
                        continue
                    batch.append((row, moments, str(item.get("topic", ""))[:30], True))
                    if len(batch) >= BATCH:
                        added += _write_batch(batch)
                        batch = []
                        progress(index + 1, len(parts), added)
        except (OSError, EOFError, ValueError, zlib.error):
            broken.append(part.name)        # a damaged part: keep the rest, retry this one next start
        added += _write_batch(batch)
        batch = []
        progress(index + 1, len(parts), added)
    if broken:
        raise ValueError(f"素材库种子文件损坏：{'、'.join(broken)}（其余部分已导入 {added} 个视频）")
    with _lock, _conn(write=True) as con:
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seed', ?)", (fingerprint,))
    return added


def mark_used(keys) -> None:
    keys = list(keys)
    if not keys:
        return
    with _lock, _conn(write=True, timeout=3) as con:
        con.executemany("UPDATE sources SET used=used+1 WHERE key=?", [(k,) for k in keys])


def _note_failed(items) -> None:
    """Videos whose sheets could not be described: keep them (bare) and count the try."""
    with _lock, _conn(write=True) as con:
        for topic, row, frames in items:
            _add(con, row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in frames], topic=topic)
            con.execute("UPDATE sources SET tries=tries+1 WHERE key=? AND described=0", (_key(row),))


def stats() -> dict:
    with _conn() as con:
        sources = con.execute("SELECT COUNT(*), COALESCE(SUM(described),0) FROM sources").fetchone()
        moments = con.execute("SELECT COUNT(*), COALESCE(SUM(text!=''),0) FROM moments").fetchone()
        topics = [dict(r) for r in con.execute(
            "SELECT topic, COUNT(*) AS sources FROM sources WHERE topic!='' GROUP BY topic ORDER BY sources DESC")]
    return {"sources": sources[0], "described": sources[1], "moments": moments[0], "described_moments": moments[1],
            "topics": topics, "packs": {name: len(queries) for name, queries in PACKS.items()},
            "file_mb": round(_file().stat().st_size / 1048576, 1) if _file().exists() else 0}


# ------------------------------------------------------------------ building
def _frames(bili, row, count=MOMENTS):
    """Evenly spread preview frames of one video (not dark, not repeats): [{t, image, hash}]. The sprite
    sheets are deleted once cropped (production fetches the ones it needs again), so long builds do not
    fill the disk."""
    from pathlib import Path
    from . import express_match as em, express_sources as es
    previews = bili.previews(row["id"], row["duration"])
    if not previews:
        return []
    try:
        duration = row["duration"] or previews[-1]["t"]
        inner = [p for p in previews if duration * 0.04 <= p["t"] <= duration * 0.96] or previews
        step = max(1, len(inner) / count)
        picked = [inner[min(len(inner) - 1, int(i * step + step / 2))] for i in range(min(count, len(inner)))]
        frames, hashes = [], []
        for p in picked:
            image = es.bilibili_tile(bili, p)
            if em._dark(image):
                continue
            h = em._dhash(image)
            if any(bin(h ^ other).count("1") <= 5 for other in hashes):
                continue
            hashes.append(h)
            frames.append({"t": p["t"], "image": image, "hash": h})
        return frames
    finally:
        for sprite in Path(previews[0]["root"]).glob("sprite-*.jpg"):
            try:
                sprite.unlink(missing_ok=True)
            except OSError:
                pass        # open elsewhere (Windows): production may still be cropping it


def _sleep(seconds, check):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check()
        time.sleep(min(2.0, max(0.0, end - time.monotonic())))


def _describe(frames, check):
    """One vision request for a sheet of frames: {index: (text, kind)}. Rate limits are waited out
    (about three minutes at most); a refused key, permission or balance stops at once."""
    from . import express_match as em, planner
    sheet = _sheet(frames, SHEET[0])
    content = [{"type": "text", "text": f"共 {len(frames)} 格。"},
               {"type": "image_url", "image_url": {"url": em._jpeg(sheet)}}]
    result = None
    for attempt in range(6):
        check()
        try:
            result, _ = planner.chat(DESCRIBE_SYSTEM, content, vision=True, max_tokens=80 + 48 * len(frames),
                                     timeout=60, task="footage_index")
            break
        except planner.ModelUnavailable as exc:
            if "429" not in str(exc):
                raise          # key, permission or balance: stop the whole build
            if attempt == 5:
                raise RuntimeError("看图接口持续限流") from None
            _sleep(min(60, 5 * 2 ** attempt), check)
        except ValueError as exc:
            if "输出长度上限" in str(exc) and len(frames) > 1:
                # The answer was cut off: half the sheet at a time.
                half = len(frames) // 2
                first, second = _describe(frames[:half], check), _describe(frames[half:], check)
                return first | {half + i: value for i, value in second.items()}
            if attempt >= 3 or not any(k in str(exc) for k in ("超时", "HTTP 5", "JSON")):
                raise
            _sleep(2 * (attempt + 1), check)
    out = {}
    for key, value in (result or {}).items() if isinstance(result, dict) else []:
        if not str(key).isdigit() or not value:
            continue
        value = value if isinstance(value, list) else [str(value)]
        text, kind = str(value[0]), (str(value[1]) if len(value) > 1 else "")
        if not kind and re.search(r"[”\"]\s*[，,]\s*[“\"]", text):    # '描述”，“类型”]' inside one string
            text, kind = re.split(r"[”\"]\s*[，,]\s*[“\"]", text, maxsplit=1)
        text = re.sub(r"[“”\"\[\]]", "", text).strip(" ，,。")[:40]
        if text and 1 <= int(key) <= len(frames):
            out[int(key) - 1] = (text, _kind(kind))
    return out


def _kind(raw: str) -> str:
    """The answer's picture type, reduced to the fixed list (the model sometimes writes two or decorates it)."""
    for word, kind in (("主持", "主持人出镜"), ("黑屏", "黑屏或模糊"), ("模糊", "黑屏或模糊"), ("图表", "图表文字"),
                       ("文字", "图表文字"), ("动画", "动画"), ("影视", "影视剧"), ("实拍", "实拍")):
        if word in raw:
            return kind
    return ""


def _sheet(frames, cols):
    """Frames in a grid, numbered from 1 in yellow corner labels (the numbers the answer uses)."""
    from PIL import Image, ImageDraw
    from . import express_match as em
    w, h = em.TILE_W, em.TILE_H
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h))
    draw = ImageDraw.Draw(sheet)
    font = em._font(26)
    for i, frame in enumerate(frames):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(frame["image"].resize((w, h)), (x, y))
        box = draw.textbbox((x + 4, y + 2), str(i + 1), font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 214, 0))
        draw.text((x + 4, y + 2), str(i + 1), font=font, fill=(0, 0, 0))
    return sheet


GENERIC_TAGS = {"生活记录", "日常", "vlog", "搞笑", "知识", "科普", "历史", "纪录片", "视频", "记录", "分享", "必剪创作",
                "打卡挑战", "生活", "自制", "原创", "剪辑", "混剪", "合集", "教程", "高清", "4k", "中国", "bilibili", "哔哩哔哩",
                "全能打卡挑战", "热点", "新闻", "社会", "美食", "旅行", "旅游", "风景", "科技", "军事", "财经", "人文", "趣味",
                "涨知识", "干货", "热门", "推荐", "震撼", "经典", "解说", "影视", "电影", "电视剧", "音乐", "动画"}
CHUNK = 160             # videos framed and described together; each sheet's videos are saved as it returns
BLOCKED_PAUSE = 300     # seconds to wait when Bilibili's risk control blocks previews, before trying again


def frame_count(duration: float) -> int:
    """Longer videos hold more distinct scenes: 8 frames under 5 minutes, 12 under 20, else 16."""
    return 8 if duration < 300 else 12 if duration < 1200 else 16


def expand(found: dict, used: set, per_topic: int = 8) -> list[tuple[str, str]]:
    """New searches from the tags a topic's videos share (Bilibili's own words for the subject).
    found: topic -> distinct videos."""
    from collections import Counter
    out = []
    for topic, rows in found.items():
        counts = Counter()
        for row in {_key(r): r for r in rows}.values():
            for tag in {t.strip() for t in str(row.get("tags", "")).split(",")}:
                if 2 <= len(tag) <= 10 and tag.lower() not in GENERIC_TAGS and not SKIP_TITLE.search(tag):
                    counts[tag] += 1
        picked = [tag for tag, n in counts.most_common() if n >= 3 and tag not in used][:per_topic]
        out += [(topic, tag) for tag in picked]
        used |= set(picked)
    return out


def _risk_paused(bili, check, progress, report, pauses, message):
    """When Bilibili's risk control refuses previews or searches: wait, reset the breakers, go on.
    Returns the new pause count; three pauses in one build stop it (finished videos stay saved)."""
    if bili.shots_blocked < 3 and bili.search_refused < 3:
        return pauses
    report["blocked"] = True
    if pauses >= 3:
        raise RuntimeError("B站接口持续风控，已停止扩库；已完成的视频已保存，稍后重新扩充会接着做")
    progress(message + f" · B站接口风控，暂停 {BLOCKED_PAUSE // 60} 分钟后继续")
    _sleep(BLOCKED_PAUSE, check)
    with bili.counts:
        bili.shots_blocked = bili.search_refused = 0
    return pauses + 1


def build(queries: list[tuple[str, str]], check, progress, *, pages: int = 1, describe: bool = True,
          limit: int = 100000, widen: int = 8, describer=None) -> dict:
    """Index the Bilibili results of [(topic, query)]: search, preview frames, descriptions. Each sheet's
    videos are saved as soon as it is described; videos already described (or failed twice on their own)
    are skipped, so a stopped or failed build continues where it was. describer(row, images) -> {frame
    index: (text, kind)} replaces the contact-sheet model with one that looks at a video at a time (a local
    model, scripts/footage_local.py)."""
    from . import express_sources as es
    bili = es.Bilibili()
    report = {"queries": len(queries), "found": 0, "new": 0, "indexed": 0, "described_frames": 0, "failed": 0,
              "blocked": False, "expanded": 0}
    pauses = 0
    try:
        rows, lock, by_topic = {}, threading.Lock(), {}
        def find(item):
            topic, query = item
            found = []
            for page in range(1, pages + 1):
                check()
                if bili.search_refused >= 3 and not bili.offline_search:
                    break                  # risk control: stop paging (the build pauses after this batch)
                try:
                    asked = not bili.cached(query, page)
                    got = bili.search(query, page)
                except Exception:
                    break
                found += got
                if not got:
                    break
                if asked and not bili.offline_search:
                    time.sleep(SEARCH_GAP)    # stay under the search API's burst limit
            with lock:
                for row in found:
                    if 20 <= row.get("duration", 0) <= 3 * 3600 and not SKIP_TITLE.search(row.get("title", "")):
                        rows.setdefault(_key(row), (topic, row))
                        by_topic.setdefault(topic, {})[_key(row)] = row
        search_pauses = 0
        def search_all(items, lo, hi, label):
            nonlocal search_pauses
            done = 0
            for start in range(0, len(items), 30):
                with execution.ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
                    for _ in pool.map(find, items[start:start + 30]):
                        done += 1
                        progress(lo + (hi - lo) * done / max(1, len(items)), f"{label} {done}/{len(items)} · 已找到 {len(rows)} 个视频")
                if bili.search_refused >= 3 and not bili.offline_search:
                    report["blocked"] = True
                    if search_pauses:
                        # Still refused after a pause: go on with the searches cached today and describe those.
                        bili.offline_search = True
                        report["search_offline"] = True
                        progress(lo + (hi - lo) * done / max(1, len(items)), f"{label} · B站搜索风控，本轮改用今天已缓存的搜索结果")
                    else:
                        progress(lo + (hi - lo) * done / max(1, len(items)),
                                 f"{label} · B站搜索风控，暂停 {BLOCKED_PAUSE // 60} 分钟后继续")
                        _sleep(BLOCKED_PAUSE, check)
                        search_pauses += 1
                    with bili.counts:
                        bili.search_refused = 0
        search_all(queries, 1, 12, "检索关键词")
        if widen:
            more = expand({topic: list(found.values()) for topic, found in by_topic.items()},
                          {q for _, q in queries}, per_topic=widen)
            report["expanded"] = len(more)
            search_all(more, 12, 18, "按标签扩展检索")
        report["found"] = len(rows)
        done = known(rows)
        with _conn() as con:
            skip = {r["key"] for r in con.execute(f"SELECT key FROM sources WHERE described=1 OR tries>={MAX_TRIES}")}
        todo = [rows[k] for k in rows if k not in skip][:limit]
        report["new"] = len([k for k in rows if k not in done])
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        index = 0
        while index < len(chunks):
            lo = 18 + 81 * index / max(1, len(chunks))
            hi = 18 + 81 * (index + 1) / max(1, len(chunks))
            label = f"第 {index + 1}/{len(chunks)} 批（共 {len(todo)} 个视频）"
            refused = _build_chunk(bili, chunks[index], check,
                                   lambda share, message: progress(lo + (hi - lo) * share, f"{label} · {message}"),
                                   report, describe, describer)
            if bili.shots_blocked >= 3:
                # Videos whose previews were refused go round again after the pause.
                pauses = _risk_paused(bili, check, lambda m: progress(hi, m), report, pauses, label)
                if refused:
                    chunks.insert(index + 1, refused)
            index += 1
        return report
    finally:
        bili.close()


class _Halted(Exception):
    """Another sheet stopped the build: do not start more model requests."""


def _build_chunk(bili, chunk, check, progress, report, describe, describer=None) -> list:
    """Frames and descriptions for up to CHUNK videos. Returns the videos whose previews were refused by
    risk control (to retry after a pause)."""
    from concurrent.futures import as_completed, wait
    from . import media
    from .planner import ModelUnavailable
    stops = (media.Cancelled, costs.BudgetExceeded, costs.PricingUnavailable, ModelUnavailable)
    def frames(item):
        topic, row = item
        check()
        try:
            return topic, row, _frames(bili, row, frame_count(float(row.get("duration") or 0))), False
        except stops:
            raise
        except Exception as exc:
            return topic, row, None, "风控" in str(exc)      # this video only
    refused = []
    if describer:
        # Each video goes to the model as soon as its frames are in, so the model is not left waiting
        # for the whole chunk's previews.
        def stream():
            with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
                for topic, row, got, blocked in pool.map(frames, chunk):
                    if got:
                        yield topic, row, got
                    elif blocked:
                        refused.append((topic, row))
                    else:
                        report["failed"] += 1
        _describe_videos(stream(), describer, check, progress, report, len(chunk))
        return refused
    framed = []
    with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
        for n, (topic, row, got, blocked) in enumerate(pool.map(frames, chunk), 1):
            if got:
                framed.append((topic, row, got))
            elif blocked:
                refused.append((topic, row))
            else:
                report["failed"] += 1
            if n % 20 == 0 or n == len(chunk):
                progress(0.4 * n / max(1, len(chunk)), f"预览帧 {n}/{len(chunk)}")
    if not describe:
        _write_batch([(row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in got], topic, False)
                      for topic, row, got in framed])
        report["indexed"] += len(framed)
        return refused
    # Whole videos to a sheet, so each answer completes its videos and they can be saved at once.
    per = SHEET[0] * SHEET[1]
    sheets, current, count = [], [], 0
    for item in framed:
        if current and count + len(item[2]) > per:
            sheets.append(current)
            current, count = [], 0
        current.append(item)
        count += len(item[2])
    if current:
        sheets.append(current)
    halt = threading.Event()
    streak = {"failed": 0}
    def guarded():
        check()
        if halt.is_set():
            raise _Halted()
    def describe_sheet(sheet):
        """{frame index: (text, kind)} for one sheet; a sheet that fails as a whole raises."""
        flat = [frame for _, _, got in sheet for frame in got]
        answer = _describe(flat, guarded)
        missing = [i for i in range(len(flat)) if i not in answer]
        if missing:          # cut-off or partial answer: ask once more for the skipped frames, keep what came
            try:
                again = _describe([flat[i] for i in missing], guarded)
            except stops + (_Halted,):
                raise
            except Exception:
                again = {}
            for j, i in enumerate(missing):
                if j in again:
                    answer[i] = again[j]
        return answer
    def run(sheet):
        """(sheet, kind, payload): "sheet" with one answer for the whole sheet, "videos" with an answer
        (or None when refused) per video, or "skip" when the service, not these videos, failed."""
        guarded()
        try:
            answer = describe_sheet(sheet)
        except stops + (_Halted,):
            raise
        except Exception as exc:
            check()
            with _lock:
                streak["failed"] += 1
                stop = streak["failed"] >= 8
                alone = streak["failed"] <= 1
            if stop:
                raise RuntimeError(f"连续 8 张联系表描述失败（{str(exc)[:80]}），已停止扩库；已完成的视频已保存，"
                                   "稍后重新扩充会接着做") from None
            # A refusal that may belong to these videos (HTTP 400 content check, unusable answer) is worked
            # out video by video; service-wide trouble (limits, timeouts, 5xx, several failures) counts nothing.
            if not (alone and isinstance(exc, ValueError) and "HTTP 400" in str(exc)):
                return sheet, "skip", None
            answers = []
            for item in sheet:
                try:
                    answers.append(describe_sheet([item]) if len(sheet) > 1 else None)
                except stops + (_Halted,):
                    raise
                except Exception:
                    answers.append(None)
            return sheet, "videos", answers
        with _lock:
            streak["failed"] = 0
        return sheet, "sheet", answer
    def save(sheet, kind, payload):
        if kind == "skip":
            return
        items, failed, position = [], [], 0
        for n, item in enumerate(sheet):
            topic, row, got = item
            if kind == "sheet":
                own, offset = payload, position
            else:
                own, offset = payload[n], 0
            position += len(got)
            if own is None:
                failed.append(item)
                continue
            moments = []
            for k, frame in enumerate(got):
                text, kind_ = own.get(offset + k, ("", ""))
                moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind_})
            if any(m["text"] for m in moments):
                items.append((row, moments, topic, True))
                report["described_frames"] += sum(1 for m in moments if m["text"])
            else:
                failed.append(item)         # answered, but nothing usable for this video
        _write_batch(items)
        report["indexed"] += len(items)
        if failed:
            _note_failed(failed)
    with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        futures = [pool.submit(run, sheet) for sheet in sheets]
        saved = set()
        try:
            for n, future in enumerate(as_completed(futures), 1):
                save(*future.result())
                saved.add(future)
                progress(0.4 + 0.59 * n / max(1, len(sheets)), f"描述画面 {n}/{len(sheets)} 张联系表")
        except BaseException:
            halt.set()                   # running sheets finish their current request and stop
            for future in futures:
                future.cancel()
            wait(futures)
            for future in futures:       # answers that came back are paid for: keep them
                if future not in saved and not future.cancelled() and future.exception() is None:
                    try:
                        save(*future.result())
                    except Exception:
                        pass
            raise
    progress(1.0, "已保存")
    return refused


def _describe_videos(framed, describer, check, progress, report, total: int) -> None:
    """(topic, row, frames) items, as they come, described one video per request and each saved as it
    returns. A video the describer could not answer counts a try; eight failures in a row mean the model
    itself is down."""
    from concurrent.futures import as_completed, wait
    from . import media
    halt = threading.Event()
    def run(item):
        check()
        if halt.is_set():
            raise _Halted()
        try:
            return item, describer(item[1], [frame["image"] for frame in item[2]])
        except media.Cancelled:
            raise
        except Exception as exc:
            return item, exc
    state = {"streak": 0, "done": 0}
    def save(future):
        (topic, row, got), answer = future.result()
        moments = []
        if isinstance(answer, dict):
            for k, frame in enumerate(got):
                text, kind = answer.get(k, ("", ""))
                moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind})
        if any(m["text"] for m in moments):
            state["streak"] = 0
            _write_batch([(row, moments, topic, True)])
            report["indexed"] += 1
            report["described_frames"] += sum(1 for m in moments if m["text"])
        else:
            state["streak"] += 1
            _note_failed([(topic, row, got)])
            if state["streak"] >= 8:
                raise RuntimeError(f"连续 8 个视频描述失败（{str(answer)[:80]}），已停止扩库")
        state["done"] += 1
        if state["done"] % 10 == 0:
            progress(0.99 * state["done"] / max(1, total), f"描述画面 {state['done']}/{total} 个视频")
    futures, pending = [], set()
    with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        try:
            for item in framed:
                future = pool.submit(run, item)
                futures.append(future)
                pending.add(future)
                for finished in [f for f in pending if f.done()]:
                    pending.discard(finished)
                    save(finished)
            for finished in as_completed(pending):
                save(finished)
        except BaseException:
            halt.set()
            for future in futures:
                future.cancel()
            wait(futures)
            raise
    progress(1.0, "已保存")
