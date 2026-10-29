from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from collections.abc import Callable

from . import paths


def uid() -> str:
    return uuid.uuid4().hex


_wal = set()     # database files this process has already switched to WAL (the mode is kept in the file)


@contextmanager
def connection():
    paths.prepare()
    path = paths.DATA / "jinggao.sqlite3"
    if not path.exists():
        _wal.discard(str(path))
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    if str(path) not in _wal:
        paths.wal(con)
        _wal.add(str(path))
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def init() -> None:
    with connection() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY, body TEXT NOT NULL, updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
            kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
            progress REAL NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '等待执行',
            error TEXT NOT NULL DEFAULT '', created REAL NOT NULL, updated REAL NOT NULL,
            cancel INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id, created);
        CREATE TABLE IF NOT EXISTS used_footage (
            source TEXT NOT NULL, project_id TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY (source, project_id)
        );
        CREATE INDEX IF NOT EXISTS used_footage_at ON used_footage(at);
        """)


def get_project(project_id: str) -> dict:
    with connection() as con:
        row = con.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
    if not row:
        raise KeyError("项目不存在")
    return defaults(json.loads(row["body"]))


def defaults(p: dict) -> dict:
    # Old image-based projects keep their original editing/export behavior.
    p.setdefault("media_policy", "mixed")
    p.setdefault("min_video_ratio", 0.8)
    p.setdefault("min_video_height", 720)
    return p


def list_projects() -> list[dict]:
    with connection() as con:
        rows = con.execute("SELECT body FROM projects ORDER BY updated DESC").fetchall()
        latest = {row["project_id"]: dict(row) for row in con.execute(
            "SELECT project_id,kind,status,progress,message FROM jobs ORDER BY created")}
    result = []
    for row in rows:
        p = json.loads(row["body"])
        if p.get("system_library"):
            continue
        video = (p.get("exports") or [None])[0]
        result.append({key: p.get(key) for key in ("id", "title", "updated", "aspect", "revision")}
                      | {"shot_count": len(p["shots"]), "script_length": len(p["script"]),
                         "video": video and {k: video.get(k) for k in ("id", "path", "duration", "created", "saved_to")},
                         "audio_source": bool(p.get("source_audio")),
                         "job": latest.get(p["id"]) and {k: latest[p["id"]][k] for k in ("kind", "status", "progress", "message")}})
    return result


def new_project(title: str, script: str = "", aspect: str = "16:9", *,
                media_policy: str = "video_preferred", min_video_ratio: float = 0.8,
                min_video_height: int = 720, burn_subtitles: bool = True) -> dict:
    now = time.time()
    p = dict(id=uid(), title=title, script=script, aspect=aspect, style="真实资料与清晰解说",
             requirements="", shots=[], assets=[], exports=[], music=None, full_audio=None,
             media_policy=media_policy, min_video_ratio=min_video_ratio, min_video_height=min_video_height,
             plan_info=None, created=now, updated=now, revision=1, burn_subtitles=burn_subtitles)
    with connection() as con:
        con.execute("INSERT INTO projects VALUES (?,?,?)", (p["id"], json.dumps(p, ensure_ascii=False), now))
    paths.project_dir(p["id"])
    return p


def mutate(project_id: str, action: Callable[[dict], None], *, internal: bool = False) -> dict:
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        if not internal and con.execute(
            "SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')", (project_id,)
        ).fetchone():
            raise ValueError("本项目有任务进行中，请等待完成或取消后修改")
        row = con.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise KeyError("项目不存在")
        p = defaults(json.loads(row["body"]))
        action(p)
        p["updated"] = time.time()
        p["revision"] += 1
        con.execute("UPDATE projects SET body=?,updated=? WHERE id=?",
                    (json.dumps(p, ensure_ascii=False), p["updated"], project_id))
    return p


def discard_project(project_id: str) -> None:
    """Remove a project that was never used (an import that failed before any job ran)."""
    import shutil
    with connection() as con:
        if con.execute("SELECT 1 FROM jobs WHERE project_id=?", (project_id,)).fetchone():
            return
        con.execute("DELETE FROM projects WHERE id=?", (project_id,))
    shutil.rmtree(paths.project_dir(project_id), ignore_errors=True)


def enqueue(project_id: str, kind: str, payload: dict | None = None) -> dict:
    get_project(project_id)
    now, job_id = time.time(), uid()
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')",
                       (project_id,)).fetchone():
            raise ValueError("本项目已有任务，请等待完成或取消")
        con.execute("INSERT INTO jobs (id,project_id,kind,payload,created,updated) VALUES (?,?,?,?,?,?)",
                    (job_id, project_id, kind, json.dumps(payload or {}), now, now))
    return get_job(job_id)


def get_job(job_id: str) -> dict:
    with connection() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise KeyError("任务不存在")
    d = dict(row)
    d["payload"] = json.loads(d["payload"])
    return d


def jobs(project_id: str) -> list[dict]:
    with connection() as con:
        rows = con.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY created DESC LIMIT 30",
                           (project_id,)).fetchall()
    return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]


# Long background work (building the footage index) runs in its own worker so videos never queue behind it.
BACKGROUND_KINDS = ("footage_index",)


def _lane(lane: str) -> str:
    marks = ",".join("'" + k + "'" for k in BACKGROUND_KINDS)
    return f"kind IN ({marks})" if lane == "background" else f"kind NOT IN ({marks})"


def claim(lane: str = "main") -> dict | None:
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        # Batch videos wait behind everything else, so a single video or a quick edit never queues behind 100.
        # A video waiting out a Bilibili refusal (payload not_before) is not taken before its time.
        row = con.execute(f"SELECT id FROM jobs WHERE status='queued' AND {_lane(lane)} "
                          "AND COALESCE(json_extract(payload, '$.not_before'), 0) <= ? "
                          "ORDER BY json_extract(payload, '$.batch') IS NOT NULL, created LIMIT 1",
                          (time.time(),)).fetchone()
        if not row:
            return None
        con.execute("UPDATE jobs SET status='running',message='开始执行',updated=? WHERE id=?",
                    (time.time(), row["id"]))
    return get_job(row["id"])


def job_update(job_id: str, **changes) -> None:
    allowed = {"status", "progress", "message", "error", "cancel"}
    if not set(changes) <= allowed:
        raise ValueError("无效的任务更新")
    changes["updated"] = time.time()
    with connection() as con:
        con.execute(f"UPDATE jobs SET {','.join(k+'=?' for k in changes)} WHERE id=?",
                    (*changes.values(), job_id))


def batch_folder(batch: str) -> str | None:
    """The dated output subfolder of a batch's earlier films (an audio batch arrives one file at a time)."""
    with connection() as con:
        rows = con.execute("SELECT p.body FROM jobs j JOIN projects p ON p.id=j.project_id "
                           "WHERE json_extract(j.payload, '$.batch')=? ORDER BY j.created LIMIT 5", (batch,)).fetchall()
    for row in rows:
        try:
            folder = json.loads(row[0]).get("save_folder")
        except (ValueError, TypeError):
            continue
        if folder:
            return folder
    return None


def cancel_all(batch: str | None = None) -> dict:
    """Cancel every queued and running video job at once (or one batch's): queued ones stop now, running
    ones at their next check. The footage index build (background lane) is left alone."""
    where = f"{_lane('main')}" + (" AND json_extract(payload, '$.batch')=?" if batch else "")
    args = (batch,) if batch else ()
    now = time.time()
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        queued = con.execute(f"UPDATE jobs SET status='cancelled',cancel=1,message='已取消（全部取消）',updated=? "
                             f"WHERE status='queued' AND {where}", (now, *args)).rowcount
        running = con.execute(f"UPDATE jobs SET cancel=1,message='正在取消，等待当前操作结束',updated=? "
                              f"WHERE status='running' AND {where}", (now, *args)).rowcount
    return {"queued": queued, "running": running}


def recover(lane: str = "main") -> None:
    with connection() as con:
        con.execute("UPDATE jobs SET status='interrupted',message='程序上次退出时任务被中断，可手动重试',"
                    f"updated=? WHERE status='running' AND {_lane(lane)}", (time.time(),))


def footage_used_elsewhere(project_id: str, days: float = 7) -> dict:
    """{"provider:id": number of other videos that used it lately}: a batch on similar topics should not show
    the same clips in every video."""
    with connection() as con:
        rows = con.execute("SELECT source, COUNT(*) AS n FROM used_footage WHERE project_id<>? AND at>? GROUP BY source",
                           (project_id, time.time() - days * 86400)).fetchall()
    return {row["source"]: row["n"] for row in rows}


def mark_footage_used(project_id: str, sources) -> None:
    now = time.time()
    with connection() as con:
        con.execute("DELETE FROM used_footage WHERE project_id=?", (project_id,))
        con.executemany("INSERT OR REPLACE INTO used_footage VALUES (?,?,?)", [(s, project_id, now) for s in set(sources)])
