"""Persistent public source and actual-clip index, shared between projects.

Retrieval scores are suggestions. Editorial approvals and person assertions are
never copied into another project. File blobs and reviewed time windows are separate.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from contextlib import contextmanager
from urllib.parse import urlparse

from . import domestic, execution, media, paths, sources, store

FIELDS = ("title", "description", "source_url", "provider", "publisher", "publisher_url",
          "published_at", "source_evidence", "source_verified", "license", "license_url", "author",
          "context_sources", "thumbnail_url", "video_id", "publisher_verified", "discovery_evidence", "platform_provider", "privacy_redaction")
# Search metadata is kept separately from decoded clip dimensions/duration.
INDEX_FIELDS = FIELDS + ("duration", "collection_url")
MAX_BYTES = 8 * 1024**3


def root():
    folder = paths.DATA / "library"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def file_path(relative):
    folder = root().resolve()
    target = (folder / relative).resolve()
    if not target.is_relative_to(folder) or target == folder:
        raise ValueError("素材库文件路径无效")
    return target


@contextmanager
def connection():
    con = sqlite3.connect(root() / "index.sqlite3", timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, url TEXT UNIQUE NOT NULL,
            body TEXT NOT NULL, updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS blobs (id TEXT PRIMARY KEY, path TEXT NOT NULL, thumbnail TEXT NOT NULL,
            bytes INTEGER NOT NULL, duration REAL NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
            blob_id TEXT NOT NULL, start REAL NOT NULL, offset REAL NOT NULL, duration REAL NOT NULL,
            description TEXT NOT NULL, visible_text TEXT NOT NULL, updated REAL NOT NULL,
            FOREIGN KEY(source_id) REFERENCES sources(id), FOREIGN KEY(blob_id) REFERENCES blobs(id));
        CREATE INDEX IF NOT EXISTS clips_source ON clips(source_id, start);
        CREATE TABLE IF NOT EXISTS clip_observation_methods (clip_id TEXT PRIMARY KEY, method TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS clip_terms USING fts5(clip_id UNINDEXED, terms);
        CREATE TABLE IF NOT EXISTS source_catalog (source_id TEXT PRIMARY KEY, provider TEXT NOT NULL,
            terms TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'indexed', checked REAL NOT NULL DEFAULT 0,
            reason TEXT NOT NULL DEFAULT '');
        CREATE INDEX IF NOT EXISTS catalog_provider ON source_catalog(provider);
        CREATE VIRTUAL TABLE IF NOT EXISTS source_terms USING fts5(terms,
            content='source_catalog', content_rowid='rowid');
        CREATE TRIGGER IF NOT EXISTS source_terms_insert AFTER INSERT ON source_catalog BEGIN
            INSERT INTO source_terms(rowid,terms) VALUES (new.rowid,new.terms); END;
        CREATE TRIGGER IF NOT EXISTS source_terms_delete AFTER DELETE ON source_catalog BEGIN
            INSERT INTO source_terms(source_terms,rowid,terms) VALUES ('delete',old.rowid,old.terms); END;
        CREATE TRIGGER IF NOT EXISTS source_terms_update AFTER UPDATE OF terms ON source_catalog BEGIN
            INSERT INTO source_terms(source_terms,rowid,terms) VALUES ('delete',old.rowid,old.terms);
            INSERT INTO source_terms(rowid,terms) VALUES (new.rowid,new.terms); END;
        CREATE TABLE IF NOT EXISTS source_topics (source_id TEXT NOT NULL,topic TEXT NOT NULL,
            query TEXT NOT NULL,PRIMARY KEY(source_id,topic,query));
        CREATE INDEX IF NOT EXISTS topics_lookup ON source_topics(topic,source_id);
        CREATE TABLE IF NOT EXISTS index_pages (provider TEXT NOT NULL,query TEXT NOT NULL,
            page INTEGER NOT NULL,status TEXT NOT NULL,count INTEGER NOT NULL,added INTEGER NOT NULL,
            updated REAL NOT NULL,error TEXT NOT NULL,PRIMARY KEY(provider,query,page));
        CREATE TABLE IF NOT EXISTS catalog_entities (id TEXT PRIMARY KEY,name TEXT NOT NULL,
            kind TEXT NOT NULL,domain TEXT NOT NULL,aliases TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS source_entities (source_id TEXT NOT NULL,entity_id TEXT NOT NULL,
            mention TEXT NOT NULL,PRIMARY KEY(source_id,entity_id,mention));
        CREATE INDEX IF NOT EXISTS entities_lookup ON source_entities(entity_id,source_id);
    """)
    try:
        if con.execute("PRAGMA user_version").fetchone()[0] < 3:
            for row in con.execute("SELECT id,body FROM sources").fetchall():
                _index_source(con, row["id"], json.loads(row["body"]))
            con.execute("""UPDATE source_catalog SET state='cached',checked=(
                SELECT MAX(updated) FROM clips WHERE source_id=source_catalog.source_id)
                WHERE source_id IN (SELECT source_id FROM clips)""")
            con.execute("PRAGMA user_version=3")
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def source_url(value):
    parsed = urlparse(value or "")
    if parsed.scheme not in ("https", "http") or parsed.username or parsed.password or not domestic.allowed(value):
        raise ValueError("索引库只接受已接入的公开素材来源")
    if parsed.hostname in ("www.bilibili.com", "bilibili.com"):
        found = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?", parsed.path)
        if found:
            return "https://www.bilibili.com/video/" + found[1]
    return parsed._replace(fragment="").geturl()


def digest_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_record(candidate):
    url = source_url(candidate.get("source_url", ""))
    sid = hashlib.sha256(url.encode()).hexdigest()[:32]
    body = {k: candidate[k] for k in INDEX_FIELDS if k in candidate}
    body.update(source_url=url, title=str(candidate.get("title") or "公开视频")[:500])
    if body.get("thumbnail_url") and not domestic.allowed(body["thumbnail_url"]):
        body.pop("thumbnail_url")
    # NASA exposes a stable, public JSON manifest. Keep that locator so an
    # indexed row can be inspected later, but never retain signed media URLs.
    if "collection_url" in body:
        manifest = urlparse(str(body["collection_url"]))
        if (body.get("provider") != "nasa" or manifest.scheme != "https"
                or manifest.hostname != "images-assets.nasa.gov" or manifest.username
                or manifest.password or manifest.query or manifest.fragment
                or not manifest.path.endswith("/collection.json")):
            body.pop("collection_url")
    return sid, body


def observed_text(review):
    """Keep frame OCR as retrieval evidence, independently of the verdict."""
    observations = (review or {}).get("observations") or {}
    if observations.get("observation_method") == "blind_actual_frames_v1":
        # An empty neutral OCR result is meaningful. Do not fall back to the
        # earlier script-aware review, which may have invented a caption.
        texts = []
        for frame in observations.get("frame_observations") or []:
            if (not isinstance(frame, dict) or type(frame.get("frame_index")) is not int
                    or frame["frame_index"] < 0 or not isinstance(frame.get("text"), list)):
                continue
            texts.extend(t for t in frame["text"] if isinstance(t, str) and t.strip())
        return " · ".join(dict.fromkeys(texts))[:2000]
    rows = observations.get("visual_evidence") or []
    return " · ".join(dict.fromkeys(str(r["quote"]) for r in rows if isinstance(r, dict)
        and r.get("kind") in ("caption", "nameplate", "scoreboard", "sign", "document")
        and isinstance(r.get("frame_index"), int) and r.get("quote")))[:2000]


def _index_source(con, sid, body):
    text = " ".join(str(body.get(k) or "") for k in ("title", "description", "publisher", "author"))
    terms = " ".join(sorted(sources.tokens(text)))
    con.execute("""INSERT INTO source_catalog(source_id,provider,terms) VALUES (?,?,?)
        ON CONFLICT(source_id) DO UPDATE SET provider=excluded.provider,terms=excluded.terms""",
        (sid, body.get("provider", "web_video"), terms))


def remember_sources(candidates, *, topic="", query=""):
    from . import catalog_taxonomy
    rows = []
    for candidate in candidates:
        if candidate.get("kind") != "video":
            continue
        try:
            sid, body = source_record(candidate)
            rows.append((sid, body))
        except ValueError:
            continue
    if not rows:
        return 0
    added = 0
    with connection() as con:
        for sid, body in rows:
            old = con.execute("SELECT body FROM sources WHERE id=?", (sid,)).fetchone()
            if old:
                # Search can fill missing metadata, but cannot overwrite page/frame evidence.
                existing = json.loads(old["body"])
                body.update({k: v for k, v in existing.items() if v not in (None, "", [], {})})
            else:
                added += 1
            con.execute("""INSERT INTO sources VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                body=excluded.body,updated=excluded.updated""",
                (sid, body["source_url"], json.dumps(body, ensure_ascii=False), time.time()))
            _index_source(con, sid, body)
            con.executemany("INSERT OR IGNORE INTO source_entities VALUES (?,?,?)",
                [(sid, eid, mention) for eid, mention in catalog_taxonomy.mentions(
                    str(body.get("title", ""))+" "+str(body.get("description", "")))])
            if topic:
                con.execute("INSERT OR IGNORE INTO source_topics VALUES (?,?,?)", (sid, topic, query))
    return added


def remember_window(pid, asset, start=0, duration=None, *, description="", visible_text="",
                    replace_observations=False, check=lambda: None):
    """Save bytes once, then index the exact observed window inside those bytes."""
    if asset.get("kind") != "video":
        return None
    sid, body = source_record(asset)
    original = paths.file_path(pid, asset["path"])
    duration = float(duration if duration is not None else asset["duration"] - start)
    start = float(start)
    if start < 0 or duration < .5 or start + duration > float(asset["duration"]) + .04:
        raise ValueError("索引片段时间超出实际文件")
    check()
    key = digest_file(original)
    relative = "blobs/" + key + ".mp4"
    thumb_relative = "blobs/" + key + ".jpg"
    dest, thumb = file_path(relative), file_path(thumb_relative)
    with execution.keyed_lock(("library-write", str(root()))):
        if not dest.exists():
            with connection() as con:
                used = con.execute("SELECT COALESCE(SUM(bytes),0) FROM blobs").fetchone()[0]
            if used + original.stat().st_size > MAX_BYTES:
                raise ValueError("素材库已达到 8 GB 缓存上限；现有索引仍可使用")
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix("." + store.uid() + ".part")
            try:
                shutil.copyfile(original, temp)
                check()
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
        if not thumb.exists() and asset.get("thumbnail"):
            old_thumb = paths.file_path(pid, asset["thumbnail"])
            if old_thumb.is_file():
                shutil.copyfile(old_thumb, thumb)
        source_start = round(float(asset.get("source_clip_start") or 0) + start, 3)
        cid = hashlib.sha256(f"{sid}:{key}:{source_start:.3f}:{duration:.3f}".encode()).hexdigest()[:32]
        description, visible_text = str(description)[:1500], str(visible_text)[:2000]
        with connection() as con:
            old_source = con.execute("SELECT body FROM sources WHERE id=?", (sid,)).fetchone()
            previous_body = json.loads(old_source[0]) if old_source else {}
            # Decoded blob duration is not the original video's duration.
            body.pop("duration", None)
            body = previous_body | {k: v for k, v in body.items() if v not in (None, "", [], {})}
            previous = con.execute("SELECT description,visible_text FROM clips WHERE id=?", (cid,)).fetchone()
            if previous and not replace_observations:
                method = con.execute("SELECT method FROM clip_observation_methods WHERE clip_id=?", (cid,)).fetchone()
                if method and method["method"] == "blind_actual_frames_v1":
                    # Rebuilding an older project must not overwrite corrected
                    # observations, including an intentionally empty OCR result.
                    description, visible_text = previous["description"], previous["visible_text"]
                else:
                    description = description or previous["description"]
                    visible_text = visible_text or previous["visible_text"]
            con.execute("INSERT INTO sources VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,updated=excluded.updated",
                        (sid, body["source_url"], json.dumps(body, ensure_ascii=False), time.time()))
            _index_source(con, sid, body)
            con.execute("UPDATE source_catalog SET state='cached',checked=?,reason='' WHERE source_id=?",
                        (time.time(), sid))
            con.execute("INSERT OR IGNORE INTO blobs VALUES (?,?,?,?,?,?,?)", (key, relative, thumb_relative,
                        dest.stat().st_size, asset["duration"], asset["width"], asset["height"]))
            con.execute("INSERT OR REPLACE INTO clips VALUES (?,?,?,?,?,?,?,?,?)",
                        (cid, sid, key, source_start, start, duration, description, visible_text, time.time()))
            if replace_observations:
                con.execute("INSERT OR REPLACE INTO clip_observation_methods VALUES (?,?)",
                            (cid, "blind_actual_frames_v1"))
            terms = sources.tokens(" ".join([body["title"], body.get("description", ""), description, visible_text]))
            con.execute("DELETE FROM clip_terms WHERE clip_id=?", (cid,))
            con.execute("INSERT INTO clip_terms VALUES (?,?)", (cid, " ".join(sorted(terms))))
    return cid


def stats():
    with connection() as con:
        return {"sources": con.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "clips": con.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
                "described": con.execute("SELECT COUNT(*) FROM clips WHERE description!=''").fetchone()[0],
                "bytes": con.execute("SELECT COALESCE(SUM(bytes),0) FROM blobs").fetchone()[0],
                "blob_count": con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0], "max_bytes": MAX_BYTES}


CLIP_SELECT = """SELECT c.*, b.path,b.thumbnail,b.bytes,b.width,b.height,b.duration AS file_duration,
                       s.body FROM clips c JOIN blobs b ON b.id=c.blob_id JOIN sources s ON s.id=c.source_id"""


def unpack(row):
    r = dict(row)
    r["source"] = json.loads(r.pop("body"))
    return r


def get(clip_id):
    with connection() as con:
        row = con.execute(CLIP_SELECT + " WHERE c.id=?", (clip_id,)).fetchone()
    if row is None:
        raise KeyError("索引片段不存在")
    return unpack(row)


def observed_terms(description, visible_text=""):
    """Do not rank an explicitly absent activity as observed activity.

    This affects retrieval scores only. Keep the original description and text
    unchanged for the grounded reviewer. Non-matching/ambiguous clauses remain.
    """
    clauses = re.split(r"[。；;\n]", description)
    positive = [clause for clause in clauses if not re.match(
        r"^.{0,12}?(?:未出现|未观察到|未展示|没有出现|未见|看不到)", clause)]
    return sources.tokens(" ".join(positive) + " " + visible_text)


def search(query="", *, limit=40, min_duration=0, min_height=0):
    tokens = sorted(sources.tokens(query))[:60]
    if query.strip() and not tokens:
        return []
    where = " WHERE c.duration>=? AND MIN(b.width,b.height)>=?"
    parameters = [max(0, min_duration), max(0, min_height)]
    if tokens:
        where += " AND c.id IN (SELECT clip_id FROM clip_terms WHERE clip_terms MATCH ?)"
        parameters.append(" OR ".join('"'+t.replace('"', '""')+'"' for t in tokens))
    with connection() as con:
        rows = con.execute(CLIP_SELECT + where, parameters).fetchall()
    # Score every FTS match before the result cap. Recent title-only matches
    # must not hide an older window with stronger actual-frame evidence.
    ranked, query_terms = [], set(tokens)
    for row in rows:
        item = unpack(row)
        observed = observed_terms(item["description"], item["visible_text"])
        source_terms = sources.tokens(item["source"]["title"] + " " + item["source"].get("description", ""))
        item["score"] = 2 * len(query_terms & observed) + len(query_terms & source_terms)
        ranked.append(item)
    ranked.sort(key=lambda r: (r["score"], bool(r["description"]), r["updated"]), reverse=True)
    result = []
    for item in ranked:
        try:
            path = file_path(item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                continue
            # Source eligibility is checked by the current provider policy.
            source_url(item["source"]["source_url"])
        except (OSError, ValueError):
            continue
        result.append(item)
        if len(result) >= min(100, max(1, limit)):
            break
    return result


def materialize(pid, clip, check=lambda: None, *, internal=True):
    """Copy an indexed blob into the project; do not copy any old approvals."""
    check()
    project = store.get_project(pid)
    base = clip["start"] - clip["offset"]
    old = next((a for a in project["assets"] if a.get("library_blob") == clip["blob_id"]
                and a.get("source_url") == clip["source"]["source_url"]
                and abs(float(a.get("source_clip_start", 0))-base) < .001), None)
    if old and paths.file_path(pid, old["path"]).is_file():
        return old
    src = file_path(clip["path"])
    if src.stat().st_size != clip["bytes"] or digest_file(src) != clip["blob_id"]:
        raise ValueError("缓存视频已变更，请重新采集来源")
    target = paths.project_dir(pid) / "assets" / (store.uid() + ".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    try:
        asset = media.inspect_asset(pid, target, clip["source"]["title"])
        if asset["kind"] != "video":
            raise ValueError("索引文件不是视频")
        asset.update({k: clip["source"][k] for k in FIELDS if k in clip["source"]})
        asset.update(library_blob=clip["blob_id"], library_clip=clip["id"], downloaded_bytes=0,
                     download_mode="library_cache", source_clip_start=base)
        check()
        store.mutate(pid, lambda p: p["assets"].append(asset), internal=internal)
        return asset
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def rebuild(check=lambda: None, progress=lambda *a: None):
    """Backfill existing decoded project clips, retaining only observed descriptions."""
    projects = store.list_projects()
    count, skipped = 0, 0
    for n, summary in enumerate(projects):
        check()
        p = store.get_project(summary["id"])
        assets = {a["id"]: a for a in p["assets"]}
        for s in p["shots"]:
            preview = s.get("match_preview") or {}
            a = assets.get(preview.get("asset_id")) or assets.get(s.get("selected_asset"))
            if not a or a.get("kind") != "video":
                continue
            # A failed later candidate's preview may coexist with an earlier
            # selected asset. Never label that candidate using the old review.
            review = (s.get("match_result") if preview else s.get("verification")) or {}
            observations = review.get("observations") or {}
            try:
                remember_window(p["id"], a, float(preview.get("clip_start", s.get("clip_start", 0))),
                                float(preview.get("duration", s["duration"])),
                                description=observations.get("scene_description") or "",
                                visible_text=observed_text(review),
                                replace_observations=observations.get("observation_method") == "blind_actual_frames_v1",
                                check=check)
                count += 1
            except (ValueError, OSError, KeyError):
                skipped += 1
        progress((n+1)/max(1, len(projects))*95, f"整理已有项目 {n+1}/{len(projects)} · 索引 {count} 个时间段")
    return stats() | {"processed": count, "skipped": skipped}
