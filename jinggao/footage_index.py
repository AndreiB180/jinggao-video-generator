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
DESCRIBE_WORKERS = 6
SKIP_TITLE = re.compile(r"直播|回放|教程|PPT|课件|歌曲|MV|翻唱|游戏|手游|原神|王者荣耀|和平精英|鬼畜|reaction|开箱|测评|"
                        r"带货|广告|招聘|网课|考研|公务员|英语|单词")
BAD_KINDS = {"主持人出镜", "图表文字", "黑屏或模糊"}
DESCRIBE_SYSTEM = """图片是视频预览帧的联系表，每格左上角有黄色编号。为每一格写一句画面描述（最多16个字，不要标点结尾）：主体、动作、场景，能看出时代的写上（古代/近代/现代），能认出地名或事物的写上；再给画面类型，只能是：实拍、影视剧、动画、图表文字、主持人出镜、黑屏或模糊。
只输出 JSON：{"1":["描述","类型"],"2":["描述","类型"],...}，每个编号都要有。"""

# Starting points for a general marketing-video library; each line is one Bilibili search.
PACKS = {
    "历史·宋朝": ["宋朝 纪录片 市井生活", "北宋 汴京 繁华 场景", "宋代 官员 朝堂", "清明上河图 动态", "宋朝 江南 水乡",
                 "古代 粮仓 漕运", "古代 灾荒 饥民 赈灾", "古代 码头 商船", "宋朝 寺庙 僧人", "古代 酒楼 宴席"],
    "历史·通用": ["古代 战争 场面 纪录片", "古代 皇宫 宫殿 航拍", "古代 农耕 农民 劳作", "古代 集市 街道", "古代 书生 读书",
                 "古代 城墙 城门", "古代 衙门 县衙", "唐朝 长安 复原", "明朝 紫禁城 纪录片", "秦朝 兵马俑 纪录片"],
    "军事": ["航母 编队 航行", "战斗机 起飞 着舰", "坦克 演习 实拍", "导弹 发射 实拍", "阅兵 方阵", "潜艇 下潜",
            "士兵 训练 实拍", "军舰 海上 航拍", "无人机 作战", "炮兵 射击 演习"],
    "科普·自然": ["太空 星系 宇宙", "地球 卫星 视角", "火山 喷发 实拍", "海洋 鱼群 水下", "森林 航拍 4K", "草原 动物 迁徙",
                 "细胞 显微镜 动画", "极光 延时摄影", "暴雨 洪水 实拍", "沙漠 航拍"],
    "城市·经济": ["港口 集装箱 航拍", "城市 夜景 延时", "工厂 生产线 实拍", "股市 交易所 大屏", "高铁 行驶 航拍",
                 "工地 施工 塔吊", "商场 人流", "写字楼 上班族 通勤", "物流 仓库 分拣", "农村 田野 航拍"],
    "美食·生活": ["街头 小吃 制作", "炒菜 特写 慢镜头", "火锅 沸腾", "面条 手工 制作", "早餐 摊 实拍", "菜市场 买菜",
                 "家庭 吃饭 团圆", "老人 公园 散步", "学生 教室 上课", "医院 医生 病房"],
}

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
    key = _key(row)
    keep = {k: row.get(k) for k in ("provider", "id", "url", "title", "tags", "description", "duration", "author",
                                    "typename", "play", "pubdate", "query")}
    usable = [m for m in moments if m.get("kind") not in BAD_KINDS]
    hosted = sum(1 for m in moments if m.get("kind") == "主持人出镜")
    flags = "host" if moments and hosted * 2 >= len(moments) else ""
    with _lock, _db() as con:
        old = con.execute("SELECT described, used, topic FROM sources WHERE key=?", (key,)).fetchone()
        if old and old["described"] and not described:
            return      # never replace described moments with bare ones
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
            out.append(row | {"indexed": True, "moments": moments, "index_score": round(score, 3),
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
    for attempt in range(3):
        check()
        try:
            result, _ = planner.chat(DESCRIBE_SYSTEM, content, vision=True, max_tokens=80 + 48 * len(frames),
                                     timeout=60, task="footage_index")
            break
        except ValueError as exc:
            if attempt == 2 or not any(k in str(exc) for k in ("超时", "HTTP 5", "JSON", "HTTP 429")):
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


def build(queries: list[tuple[str, str]], check, progress, *, pages: int = 1, describe: bool = True,
          limit: int = 2000) -> dict:
    """Index the Bilibili results of [(topic, query)]: search, preview frames, descriptions.
    Videos already described are skipped; at most `limit` new videos per run."""
    from . import express_sources as es
    bili = es.Bilibili()
    report = {"queries": len(queries), "found": 0, "new": 0, "indexed": 0, "described_frames": 0, "failed": 0,
              "blocked": False}
    try:
        progress(2, f"检索 {len(queries)} 个关键词")
        rows, lock = {}, threading.Lock()
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
        with execution.ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
            for n, _ in enumerate(pool.map(find, queries), 1):
                progress(2 + 18 * n / max(1, len(queries)), f"检索关键词 {n}/{len(queries)} · 已找到 {len(rows)} 个视频")
        report["found"] = len(rows)
        done = known(rows)
        with _lock, _db() as con:
            described = {r["key"] for r in con.execute("SELECT key FROM sources WHERE described=1")}
        todo = [rows[k] for k in rows if k not in described][:limit]
        report["new"] = len([k for k in rows if k not in done])
        progress(20, f"读取 {len(todo)} 个视频的预览帧")
        framed = []
        def frames(item):
            topic, row = item
            check()
            try:
                return topic, row, _frames(bili, row)
            except Exception:
                return topic, row, None
        with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
            for n, (topic, row, got) in enumerate(pool.map(frames, todo), 1):
                if got:
                    framed.append((topic, row, got))
                else:
                    report["failed"] += 1
                if n % 10 == 0 or n == len(todo):
                    progress(20 + 40 * n / max(1, len(todo)), f"预览帧 {n}/{len(todo)}")
        report["blocked"] = bili.blocked >= 3
        if not describe:
            for topic, row, got in framed:
                add(row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in got], topic=topic)
            report["indexed"] = len(framed)
            return report
        per = SHEET[0] * SHEET[1]
        batches, current = [], []
        for topic, row, got in framed:
            for frame in got:
                current.append((topic, row, frame))
                if len(current) == per:
                    batches.append(current)
                    current = []
        if current:
            batches.append(current)
        results = {}
        def run(batch):
            check()
            try:
                return batch, _describe([f for _, _, f in batch], check)
            except costs.BudgetExceeded:
                raise
            except Exception:
                return batch, {}
        missed = []
        with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
            for n, (batch, answer) in enumerate(pool.map(run, batches), 1):
                for i, item in enumerate(batch):
                    if i in answer:
                        results[id(item[2])] = (item, answer[i])
                    else:
                        missed.append(item)
                progress(60 + 33 * n / max(1, len(batches)), f"描述画面 {n}/{len(batches)} 张联系表")
            # Frames an answer skipped (cut-off output, refused sheet): one more try in new sheets.
            again = [missed[i:i + per] for i in range(0, len(missed), per)]
            for n, (batch, answer) in enumerate(pool.map(run, again), 1):
                for i, item in enumerate(batch):
                    if i in answer:
                        results[id(item[2])] = (item, answer[i])
                progress(93 + 5 * n / max(1, len(again)), f"补充描述 {n}/{len(again)} 张联系表")
        by_video = {}
        for topic, row, got in framed:
            moments = []
            for frame in got:
                text, kind = results.get(id(frame), (None, ("", "")))[1]
                moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind})
                report["described_frames"] += bool(text)
            by_video[_key(row)] = (topic, row, moments)
        for topic, row, moments in by_video.values():
            add(row, moments, topic=topic, described=any(m["text"] for m in moments))
            report["indexed"] += 1
        return report
    finally:
        bili.close()
