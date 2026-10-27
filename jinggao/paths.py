from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("JINGGAO_DATA", ROOT / "data")).resolve()


# Small things a new copy of the app must not ask for again: the client unpacks every version into a new
# folder and had to type the API key and scan the Bilibili code each time (and the remote-assist code
# changed). A copy lives in the user's own folder (Windows %APPDATA%\\Jinggao); a new data folder starts from
# it, or from the newest copy of the app found next to this one.
KEPT = ("settings.json", "bilibili-cookies.json", "assist/assist.key")
KEEP = (Path(os.environ["JINGGAO_KEEP"]) if os.environ.get("JINGGAO_KEEP") else
        Path(os.environ["APPDATA"]) / "Jinggao" if os.name == "nt" and os.environ.get("APPDATA")
        and not os.environ.get("JINGGAO_DATA") else None)


def keep_sync() -> None:
    """Copy each kept file to KEEP when this data folder has a newer one; fill a missing one from KEEP, or (a
    first start of a new copy before any was kept) from the most recently used copy of the app nearby."""
    if KEEP is None:
        return
    older = None
    if not any((DATA / name).exists() for name in KEPT) and not any((KEEP / name).exists() for name in KEPT):
        older = _previous_copy()
    for name in KEPT:
        here, kept = DATA / name, KEEP / name
        try:
            if here.exists():
                if not kept.exists() or (kept.read_bytes() != here.read_bytes() and here.stat().st_mtime >= kept.stat().st_mtime):
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(here, kept)
            else:
                source = kept if kept.exists() else (older / name if older and (older / name).exists() else None)
                if source:
                    here.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, here)
        except OSError:
            continue    # a locked or unreadable file: the next start tries again


def _previous_copy() -> Path | None:
    """The data folder of another copy of the app next to this one (unpacked side by side), newest first."""
    found = []
    for base, pattern in ((ROOT.parent, "*/data/settings.json"), (ROOT.parent.parent, "*/*/data/settings.json")):
        try:
            found += [f for f in base.glob(pattern) if f.parent.resolve() != DATA]
        except OSError:
            continue
    return max(found, key=lambda f: f.stat().st_mtime).parent if found else None


_kept_once = []


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "projects").mkdir(exist_ok=True)
    if not _kept_once:     # once per process; settings and login writes sync again themselves
        _kept_once.append(True)
        keep_sync()


def project_dir(project_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise ValueError("项目编号无效")
    path = DATA / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_path(project_id: str, relative: str) -> Path:
    root = project_dir(project_id).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("文件路径无效")
    return path


def binary(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    local = ROOT / "runtime" / "ffmpeg" / f"{name}{suffix}"
    if local.exists():
        return str(local)
    located = shutil.which(name)
    if not located:
        raise RuntimeError(f"没有找到 {name}，请使用完整发行包或将其放入 runtime/ffmpeg")
    return located


def font_path() -> Path:
    path = ROOT / "assets" / "fonts" / "NotoSansSC-Regular.otf"
    if not path.exists():
        raise RuntimeError("缺少中文字幕字体，请运行 scripts/fetch_font.py 或使用完整发行包")
    return path


def temporary(target: Path) -> Path:
    """A temporary name next to target that no other process or thread is writing."""
    import threading
    return target.with_name(f".{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")


def publish(temporary_file: Path, target: Path) -> None:
    """Move a finished temporary file into place. Another worker may be reading or replacing the same cache
    file (Windows refuses to replace an open file); if it already holds a finished copy, keep that one."""
    import time
    for attempt in range(5):
        try:
            os.replace(temporary_file, target)
            return
        except PermissionError:
            if attempt == 4:
                if target.exists():
                    temporary_file.unlink(missing_ok=True)
                    return
                raise
            time.sleep(.1 * (attempt + 1))


def wal(con) -> None:
    """Put a SQLite file in WAL mode. Switching a new file does not wait for the busy timeout: another
    process doing the same at that moment gets "database is locked" at once, so try again briefly."""
    import sqlite3
    import time
    for attempt in range(50):
        try:
            con.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == 49:
                raise
            time.sleep(0.1)
