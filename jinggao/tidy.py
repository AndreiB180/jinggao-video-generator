"""Keeps a busy machine's disk from filling up.

A customer's data folder passed 200 GB (11-05): every film left its downloaded clips (~60% of it), a copy of the
film and a delivery zip in data/projects, and nothing ever removed them (the app has no delete button). Now:

- a project whose newest film is saved in the output folder (镜稿成片) and that nobody has touched for
  project_keep_days (default 7; "0" keeps everything) is removed, clips and record; the film, its subtitles and
  cover stay in 镜稿成片;
- the big download caches (B站 preview sheets, Pixabay files) drop what has not been fetched for two weeks.

Runs in the app process a few minutes after start and every six hours."""
from __future__ import annotations

import json
import os
import shutil
import stat
import threading
import time
from pathlib import Path

from . import config, paths, store

FIRST = 10 * 60           # seconds after start: the app and its workers come up first
EVERY = 6 * 3600
CACHE_DAYS = 14
BIG_CACHES = ("bili-shots", "pixabay-files", "pixabay-thumbs")


def keep_days() -> float:
    try:
        return max(0.0, float(config.read().get("project_keep_days") or 0))
    except (TypeError, ValueError):
        return 0.0


def _saved(exports) -> bool:
    """The newest film of the project is in the output folder."""
    films = [e for e in exports or [] if isinstance(e, dict)]
    if not films:
        return False
    newest = max(films, key=lambda e: float(e.get("created") or 0))
    saved = str(newest.get("saved_to") or "")
    return bool(saved) and Path(saved).is_file()


def old_projects(now: float | None = None, days: float | None = None) -> list[dict]:
    """[{id, title, saved_to}] of the projects that may go."""
    now = time.time() if now is None else now
    days = keep_days() if days is None else days
    if days <= 0:
        return []
    cutoff = now - days * 86400
    with store.connection() as con:
        busy = {r[0] for r in con.execute("SELECT DISTINCT project_id FROM jobs WHERE status IN ('queued','running')")}
        touched = {r[0]: r[1] or 0 for r in con.execute("SELECT project_id, MAX(updated) FROM jobs GROUP BY project_id")}
        rows = con.execute("SELECT id, json_extract(body,'$.title'), json_extract(body,'$.system_library'), "
                           "json_extract(body,'$.exports') FROM projects WHERE updated < ?", (cutoff,)).fetchall()
    out = []
    for pid, title, system, exports in rows:
        if pid in busy or system or touched.get(pid, 0) >= cutoff:
            continue
        try:
            films = json.loads(exports) if exports else []
        except ValueError:
            continue
        if _saved(films):
            newest = max(films, key=lambda e: float(e.get("created") or 0))
            out.append({"id": pid, "title": title or "", "saved_to": newest.get("saved_to")})
    return out


def _writable(function, path, error):
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    function(path)


def _size(target: Path) -> int:
    if target.is_file():
        return target.stat().st_size
    total = 0
    for base, _, files in os.walk(target):
        for name in files:
            try:
                total += os.stat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def _folder(project_id: str) -> Path:
    return paths.DATA / "projects" / project_id     # paths.project_dir would make it again


def remove_project(project_id: str) -> int:
    """The project's record, jobs and folder; bytes freed (0 when it got a job meanwhile)."""
    with store.connection() as con:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')",
                       (project_id,)).fetchone():
            return 0
        con.execute("DELETE FROM jobs WHERE project_id=?", (project_id,))
        con.execute("DELETE FROM projects WHERE id=?", (project_id,))
    freed = 0
    for folder in (_folder(project_id), paths.DATA / "cache" / "rejected-windows" / project_id,
                   paths.DATA / "diagnostics" / project_id):
        if folder.is_dir():
            freed += _size(folder)
            try:
                shutil.rmtree(folder, onexc=_writable)
            except OSError:
                shutil.rmtree(folder, ignore_errors=True)   # a file held open by a player stays; the rest goes
    return freed


def prune_caches(now: float | None = None, days: float = CACHE_DAYS) -> int:
    """Entries (a file, or a video's folder) of the big caches not written for `days`; bytes freed."""
    cutoff = (time.time() if now is None else now) - days * 86400
    freed = 0
    for name in BIG_CACHES:
        folder = paths.DATA / "cache" / "express" / name
        if not folder.is_dir():
            continue
        for entry in list(folder.iterdir()):
            try:
                if entry.is_file():
                    newest, size = entry.stat().st_mtime, entry.stat().st_size
                else:
                    files = [f for f in entry.rglob("*") if f.is_file()]
                    newest = max((f.stat().st_mtime for f in files), default=entry.stat().st_mtime)
                    size = sum(f.stat().st_size for f in files)
                if newest >= cutoff:
                    continue
                if entry.is_file():
                    entry.unlink()
                else:
                    shutil.rmtree(entry, ignore_errors=True)
                freed += size
            except OSError:
                continue
    return freed


def run_once(now: float | None = None) -> dict:
    removed, freed = [], 0
    for project in old_projects(now):
        try:
            got = remove_project(project["id"])
        except OSError:
            continue
        if got or not _folder(project["id"]).exists():
            removed.append(project["title"])
            freed += got
    cache = prune_caches(now)
    report = {"at": time.time(), "projects": len(removed), "project_bytes": freed, "cache_bytes": cache}
    try:
        (paths.DATA / "tidy.json").write_text(json.dumps(report, ensure_ascii=False), "utf-8")
    except OSError:
        pass
    if removed or cache:
        print(f"自动清理：删了 {len(removed)} 个 {keep_days():g} 天没动、成片已保存的项目，"
              f"共腾出 {(freed + cache) / 1024**3:.1f} GB", flush=True)
    return report


def start() -> threading.Thread:
    def loop():
        time.sleep(FIRST)
        while True:
            try:
                run_once()
            except Exception as exc:  # noqa: BLE001 — a failed pass must not take the app down
                print(f"自动清理出错：{exc}", flush=True)
            time.sleep(EVERY)
    thread = threading.Thread(target=loop, name="tidy", daemon=True)
    thread.start()
    return thread
