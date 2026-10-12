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


@contextmanager
def connection():
    paths.prepare()
    con = sqlite3.connect(paths.DATA / "jinggao.sqlite3", timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
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
    result = []
    for row in rows:
        p = json.loads(row["body"])
        if p.get("system_library"):
            continue
        result.append({key: p.get(key) for key in ("id", "title", "updated", "aspect", "revision")}
                      | {"shot_count": len(p["shots"]), "script_length": len(p["script"])})
    return result


def new_project(title: str, script: str = "", aspect: str = "16:9", *,
                media_policy: str = "video_preferred", min_video_ratio: float = 0.8,
                min_video_height: int = 720) -> dict:
    now = time.time()
    p = dict(id=uid(), title=title, script=script, aspect=aspect, style="真实资料与清晰解说",
             requirements="", shots=[], assets=[], exports=[], music=None, full_audio=None,
             media_policy=media_policy, min_video_ratio=min_video_ratio, min_video_height=min_video_height,
             plan_info=None, created=now, updated=now, revision=1)
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


def claim() -> dict | None:
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
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


def recover() -> None:
    with connection() as con:
        con.execute("UPDATE jobs SET status='interrupted',message='程序上次退出时任务被中断，可手动重试',"
                    "updated=? WHERE status='running'", (time.time(),))
