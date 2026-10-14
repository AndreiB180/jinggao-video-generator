"""本地素材索引：B站视频的标题、标签、预览帧和画面描述，不下载视频本身。

Each indexed video keeps its search row (title, tags, length, uploader) and 8-12 "moments": preview
frames from Bilibili's own thumbnail sprites, each with a one-line description written by the vision
model from a contact sheet (about 0.00005 元 a frame). Text is searched with Chinese character
bigrams in SQLite FTS5, so a shot's query finds the video and the moment in it that shows the thing.
Production downloads only the chosen few seconds, as before.

The index grows two ways: bulk builds from topic packs or keywords (build()), and every production run's
live search results (add_seen()). The whole index is one SQLite file that can be copied between machines.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time

from . import costs, execution, paths

MOMENTS = 10            # preview frames kept (and described) per video
SHEET = (6, 4)          # describe sheet: columns x rows of frames
SEARCH_WORKERS = 3      # Bilibili's search API answers bursts with 412 risk control
SHOT_WORKERS = 8        # videoshot metadata (same API gateway as search) and its CDN sprites
SPRITE_WORKERS = 16     # sprite images come from the CDN
DESCRIBE_WORKERS = 10    # the background worker allows 10 model calls at a time (execution.MODEL_SLOTS)
SKIP_TITLE = re.compile(r"直播|回放|教程|PPT|课件|歌曲|MV|翻唱|游戏|手游|原神|王者荣耀|和平精英|鬼畜|reaction|开箱|测评|"
                        r"带货|广告|招聘|网课|考研|公务员|英语|单词")
BAD_KINDS = {"主持人出镜", "图表文字", "黑屏或模糊"}
DESCRIBE_SYSTEM = """图片是视频预览帧的联系表，每格左上角有黄色编号。为每一格写一句画面描述（最多16个字，不要标点结尾）：主体、动作、场景，能看出时代的写上（古代/近代/现代），能认出地名或事物的写上；再给画面类型，只能是：实拍、影视剧、动画、图表文字、主持人出镜、黑屏或模糊。
只输出 JSON：{"1":["描述","类型"],"2":["描述","类型"],...}，每个编号都要有。"""

from .footage_topics import PACKS   # noqa: E402  (topic -> Bilibili searches)

_lock = threading.Lock()
CJK = re.compile(r"[㐀-鿿]+")
WORD = re.compile(r"[A-Za-z0-9]{2,}")


def grams(text: str) -> list[str]:
    """Searchable terms: two-character pieces of every Chinese run (one character alone), plus words."""
    out = []
    for run in CJK.findall(text or ""):
        out += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    out += [w.lower() for w in WORD.findall(text or "")]
    return list(dict.fromkeys(out))


def _db() -> sqlite3.Connection:
    paths.DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(paths.DATA / "footage-index.sqlite", timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY, provider TEXT NOT NULL, id TEXT NOT NULL, row TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '', added REAL NOT NULL, described INTEGER NOT NULL DEFAULT 0,
            used INTEGER NOT NULL DEFAULT 0, flags TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS moments (
            source TEXT NOT NULL, t REAL NOT NULL, hash TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, t));
        CREATE VIRTUAL TABLE IF NOT EXISTS terms USING fts5(source UNINDEXED, t UNINDEXED, words);
    """)
    return con


def _key(row) -> str:
    return f"{row['provider']}:{row['id']}"


def known(keys) -> set:
    keys = list(keys)
    if not keys:
        return set()
    with _lock, _db() as con:
        found = set()
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            found |= {r["key"] for r in con.execute(
                f"SELECT key FROM sources WHERE key IN ({','.join('?' * len(part))})", part)}
        return found


def add(row: dict, moments: list[dict], *, topic: str = "", described: bool = False) -> None:
    """Save (or refresh) one video: its search row and moments [{t, hash, text, kind}]."""
    with _lock, _db() as con:
        _add(con, row, moments, topic=topic, described=described)


def _add(con, row, moments, *, topic="", described=False) -> bool:
    """One video into the open connection; False when a described copy is already there."""
    key = _key(row)
    old = con.execute("SELECT described, used, topic FROM sources WHERE key=?", (key,)).fetchone()
    if old and old["described"] and not described:
        return False      # never replace described moments with bare ones
    keep = {k: row.get(k) for k in ("provider", "id", "url", "title", "tags", "description", "duration", "author",
                                    "typename", "play", "pubdate", "query")}
    usable = [m for m in moments if m.get("kind") not in BAD_KINDS]
    hosted = sum(1 for m in moments if m.get("kind") == "主持人出镜")
    flags = "host" if moments and hosted * 2 >= len(moments) else ""
    con.execute("INSERT OR REPLACE INTO sources (key,provider,id,row,topic,added,described,used,flags) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (key, row["provider"], row["id"], json.dumps(keep, ensure_ascii=False),
                 topic or (old["topic"] if old else ""), time.time(), int(described), old["used"] if old else 0, flags))
    con.execute("DELETE FROM moments WHERE source=?", (key,))
    con.execute("DELETE FROM terms WHERE source=?", (key,))
    con.executemany("INSERT OR REPLACE INTO moments (source,t,hash,text,kind) VALUES (?,?,?,?,?)",
                    [(key, round(float(m["t"]), 2), str(m.get("hash", "")), m.get("text", ""), m.get("kind", ""))
                     for m in moments])
    head = " ".join(grams(" ".join(str(row.get(k) or "") for k in ("title", "tags", "query", "typename"))))
    con.execute("INSERT INTO terms (source,t,words) VALUES (?,?,?)", (key, -1, head))
    con.executemany("INSERT INTO terms (source,t,words) VALUES (?,?,?)",
                    [(key, round(float(m["t"]), 2), " ".join(grams(m["text"]))) for m in usable if m.get("text")])
    return True


def add_seen(rows: list[dict], tiles: list[dict], topic: str = "") -> int:
    """Production found these videos live: keep them (title text and frame times; described later)."""
    by_source = {}
    for tile in tiles:
        source = tile["source"]
        if source.get("provider") != "bilibili" or source.get("indexed"):
            continue
        by_source.setdefault(_key(source), []).append({"t": tile["t"], "hash": format(tile.get("hash") or 0, "x")})
    new = 0
    for row in rows:
        if row.get("provider") == "bilibili" and not row.get("indexed") and _key(row) in by_source:
            add(row, by_source[_key(row)], topic=topic)
            new += 1
    return new


def search(query: str, limit: int = 8, *, exclude: set | None = None) -> list[dict]:
    """Videos whose title or described frames match the query, best first. Each row is shaped like a
    live Bilibili search row plus "indexed", "moments" (matching frame times) and "index_score"."""
    words = grams(query)
    if not words:
        return []
    expression = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
    with _lock, _db() as con:
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


def merge(other_file) -> int:
    """Take in another machine's index: videos this one lacks, or has without descriptions."""
    other = sqlite3.connect(other_file)
    other.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in other.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"sources", "moments"} <= tables:
            raise ValueError("不是镜稿素材库文件")
        rows = other.execute("SELECT key, row, topic, described FROM sources").fetchall()
        mine = {}
        with _lock, _db() as con:
            mine = {r["key"]: r["described"] for r in con.execute("SELECT key, described FROM sources")}
        added = 0
        for src in rows:
            if src["key"] in mine and (mine[src["key"]] or not src["described"]):
                continue
            moments = [dict(m) for m in other.execute("SELECT t, hash, text, kind FROM moments WHERE source=?",
                                                     (src["key"],))]
            add(json.loads(src["row"]), moments, topic=src["topic"], described=bool(src["described"]))
            added += 1
        return added
    except sqlite3.DatabaseError:
        raise ValueError("素材库文件无法读取") from None
    finally:
        other.close()


SEED = paths.ROOT / "assets" / "footage-index"
SEED_PART_BYTES = 40 * 1024 * 1024     # GitHub refuses files over 100 MB and warns over 50 MB


def export_seed(folder=SEED, part_bytes: int = SEED_PART_BYTES) -> list:
    """The described part of the index as gzip JSON lines in numbered parts (shipped with the app)."""
    import gzip
    import io
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("part-*.jsonl.gz"):
        old.unlink()
    with _lock, _db() as con:
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
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as writer:
            writer.write(b"".join(lines))
        path.write_bytes(buffer.getvalue())
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
    return parts


def install_seed(folder=SEED, progress=lambda *_: None) -> int:
    """Take in the index shipped with the app once per seed version (videos this machine lacks)."""
    import gzip
    import hashlib
    parts = sorted(folder.glob("part-*.jsonl.gz")) if folder.is_dir() else []
    if not parts:
        return 0
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.read_bytes())
    fingerprint = digest.hexdigest()
    marker = paths.DATA / "footage-seed.sha256"
    if marker.exists() and marker.read_text().strip() == fingerprint:
        return 0
    added = 0
    with _lock, _db() as con:
        described = {r["key"] for r in con.execute("SELECT key FROM sources WHERE described=1")}
        for index, part in enumerate(parts):
            with gzip.open(part, "rt", encoding="utf-8") as lines:
                for line in lines:
                    item = json.loads(line)
                    row = item["row"]
                    if row.get("provider") != "bilibili" or not re.fullmatch(r"BV[0-9A-Za-z]{10}", str(row.get("id", ""))):
                        continue
                    if _key(row) in described:
                        continue
                    if _add(con, row, item.get("moments") or [], topic=str(item.get("topic", ""))[:30], described=True):
                        added += 1
            con.commit()
            progress(index + 1, len(parts))
    marker.write_text(fingerprint)
    return added


def mark_used(keys) -> None:
    with _lock, _db() as con:
        con.executemany("UPDATE sources SET used=used+1 WHERE key=?", [(k,) for k in keys])


def stats() -> dict:
    with _lock, _db() as con:
        sources = con.execute("SELECT COUNT(*), COALESCE(SUM(described),0) FROM sources").fetchone()
        moments = con.execute("SELECT COUNT(*), COALESCE(SUM(text!=''),0) FROM moments").fetchone()
        topics = [dict(r) for r in con.execute(
            "SELECT topic, COUNT(*) AS sources FROM sources WHERE topic!='' GROUP BY topic ORDER BY sources DESC")]
    return {"sources": sources[0], "described": sources[1], "moments": moments[0], "described_moments": moments[1],
            "topics": topics, "packs": {name: len(queries) for name, queries in PACKS.items()},
            "file_mb": round((paths.DATA / "footage-index.sqlite").stat().st_size / 1048576, 1)
            if (paths.DATA / "footage-index.sqlite").exists() else 0}


# ------------------------------------------------------------------ building
def _frames(bili, row, count=MOMENTS):
    """Evenly spread preview frames of one video (not dark, not repeats): [{t, image, hash}]."""
    from . import express_match as em, express_sources as es
    previews = bili.previews(row["id"], row["duration"])
    if not previews:
        return []
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


def _describe(frames, check):
    """One vision request for a sheet of frames: {index: (text, kind)}."""
    from . import express_match as em, planner
    sheet = _sheet(frames, SHEET[0])
    content = [{"type": "text", "text": f"共 {len(frames)} 格。"},
               {"type": "image_url", "image_url": {"url": em._jpeg(sheet)}}]
    for attempt in range(4):
        check()
        try:
            result, _ = planner.chat(DESCRIBE_SYSTEM, content, vision=True, max_tokens=80 + 48 * len(frames),
                                     timeout=60, task="footage_index")
            break
        except planner.ModelUnavailable as exc:
            if "429" not in str(exc) or attempt == 3:
                raise          # key, permission or balance: stop the whole build (see _build_chunk)
            time.sleep(5 * (attempt + 1))     # rate limited: wait, then the same sheet again
        except ValueError as exc:
            if attempt == 3 or not any(k in str(exc) for k in ("超时", "HTTP 5", "JSON")):
                raise
            time.sleep(2 * (attempt + 1))
    out = {}
    for key, value in (result or {}).items() if isinstance(result, dict) else []:
        if not str(key).isdigit() or not value:
            continue
        value = value if isinstance(value, list) else [str(value)]
        text, kind = str(value[0]), (str(value[1]) if len(value) > 1 else "")
        if not kind and re.search(r"[”\"]\s*[，,]\s*[“\"]", text):    # '描述”，“类型”]' inside one string
            text, kind = re.split(r"[”\"]\s*[，,]\s*[“\"]", text, maxsplit=1)
        text = re.sub(r"[“”\"\[\]]", "", text).strip(" ，,。")[:40]
        if text:
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
CHUNK = 160             # videos framed, described and saved together (a stopped build keeps finished chunks)


def frame_count(duration: float) -> int:
    """Longer videos hold more distinct scenes: 8 frames under 5 minutes, 12 under 20, else 16."""
    return 8 if duration < 300 else 12 if duration < 1200 else 16


def expand(found: dict, used: set, per_topic: int = 8) -> list[tuple[str, str]]:
    """New searches from the tags a topic's results share (Bilibili's own words for the subject)."""
    from collections import Counter
    out = []
    for topic, rows in found.items():
        counts = Counter()
        for row in rows:
            for tag in {t.strip() for t in str(row.get("tags", "")).split(",")}:
                if 2 <= len(tag) <= 10 and tag.lower() not in GENERIC_TAGS and not SKIP_TITLE.search(tag):
                    counts[tag] += 1
        picked = [tag for tag, n in counts.most_common() if n >= 3 and tag not in used][:per_topic]
        out += [(topic, tag) for tag in picked]
        used |= set(picked)
    return out


def build(queries: list[tuple[str, str]], check, progress, *, pages: int = 1, describe: bool = True,
          limit: int = 100000, widen: int = 8) -> dict:
    """Index the Bilibili results of [(topic, query)]: search, preview frames, descriptions. Saved chunk
    by chunk; videos already described are skipped, so a stopped or failed build continues where it was."""
    from . import express_sources as es
    bili = es.Bilibili()
    report = {"queries": len(queries), "found": 0, "new": 0, "indexed": 0, "described_frames": 0, "failed": 0,
              "blocked": False, "expanded": 0}
    try:
        rows, lock, by_topic = {}, threading.Lock(), {}
        def find(item):
            topic, query = item
            found = []
            for page in range(1, pages + 1):
                check()
                try:
                    found += bili.search(query, page)
                except Exception:
                    break
                time.sleep(0.4)        # stay under the search API's burst limit
            with lock:
                for row in found:
                    if 20 <= row.get("duration", 0) <= 3 * 3600 and not SKIP_TITLE.search(row.get("title", "")):
                        rows.setdefault(_key(row), (topic, row))
                        by_topic.setdefault(topic, []).append(row)
        def search_all(items, lo, hi, label):
            with execution.ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
                for n, _ in enumerate(pool.map(find, items), 1):
                    progress(lo + (hi - lo) * n / max(1, len(items)), f"{label} {n}/{len(items)} · 已找到 {len(rows)} 个视频")
        search_all(queries, 1, 12, "检索关键词")
        if widen:
            more = expand(by_topic, {q for _, q in queries}, per_topic=widen)
            report["expanded"] = len(more)
            search_all(more, 12, 18, "按标签扩展检索")
        report["found"] = len(rows)
        done = known(rows)
        with _lock, _db() as con:
            described = {r["key"] for r in con.execute("SELECT key FROM sources WHERE described=1")}
        todo = [rows[k] for k in rows if k not in described][:limit]
        report["new"] = len([k for k in rows if k not in done])
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        for index, chunk in enumerate(chunks):
            lo = 18 + 81 * index / max(1, len(chunks))
            hi = 18 + 81 * (index + 1) / max(1, len(chunks))
            label = f"第 {index + 1}/{len(chunks)} 批（共 {len(todo)} 个视频）"
            _build_chunk(bili, chunk, check, lambda share, message: progress(lo + (hi - lo) * share, f"{label} · {message}"),
                         report, describe)
            report["blocked"] = report["blocked"] or bili.blocked >= 3
        return report
    finally:
        bili.close()


def _build_chunk(bili, chunk, check, progress, report, describe):
    def frames(item):
        topic, row = item
        check()
        for attempt in range(3):
            try:
                return topic, row, _frames(bili, row, frame_count(float(row.get("duration") or 0)))
            except (costs.BudgetExceeded, KeyboardInterrupt):
                raise
            except Exception as exc:
                if "风控" not in str(exc):
                    return topic, row, None
                time.sleep(10 * (attempt + 1))   # Bilibili risk control: back off, then try again
        return topic, row, None
    framed = []
    with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
        for n, (topic, row, got) in enumerate(pool.map(frames, chunk), 1):
            if got:
                framed.append((topic, row, got))
            else:
                report["failed"] += 1
            if n % 20 == 0 or n == len(chunk):
                progress(0.4 * n / max(1, len(chunk)), f"预览帧 {n}/{len(chunk)}")
    if not describe:
        for topic, row, got in framed:
            add(row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in got], topic=topic)
            report["indexed"] += 1
        return
    per = SHEET[0] * SHEET[1]
    items = [(topic, row, frame) for topic, row, got in framed for frame in got]
    batches = [items[i:i + per] for i in range(0, len(items), per)]
    results = {}
    failures = {"row": 0}
    def run(batch):
        from .planner import ModelUnavailable
        check()
        try:
            answer = _describe([f for _, _, f in batch], check)
        except (costs.BudgetExceeded, ModelUnavailable):
            raise                  # budget, key, permission or balance: no later sheet would fare better
        except Exception as exc:
            with _lock:
                failures["row"] += 1
                stop = failures["row"] >= 8
            if stop:
                raise RuntimeError(f"连续 8 张联系表描述失败（{str(exc)[:80]}），已停止扩库；已完成的批次已保存，"
                                   "检查阿里云账户余额和服务状态后重新扩充会接着做") from None
            return batch, {}
        with _lock:
            failures["row"] = 0
        return batch, answer
    missed = []
    with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        for n, (batch, answer) in enumerate(pool.map(run, batches), 1):
            for i, item in enumerate(batch):
                if i in answer:
                    results[id(item[2])] = answer[i]
                else:
                    missed.append(item)
            progress(0.4 + 0.5 * n / max(1, len(batches)), f"描述画面 {n}/{len(batches)} 张联系表")
        # Frames an answer skipped (cut-off output, refused sheet): one more try in new sheets.
        again = [missed[i:i + per] for i in range(0, len(missed), per)]
        for batch, answer in pool.map(run, again):
            for i, item in enumerate(batch):
                if i in answer:
                    results[id(item[2])] = answer[i]
    for topic, row, got in framed:
        moments = []
        for frame in got:
            text, kind = results.get(id(frame), ("", ""))
            moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind})
            report["described_frames"] += bool(text)
        add(row, moments, topic=topic, described=any(m["text"] for m in moments))
        report["indexed"] += 1
    progress(1.0, "已保存")
