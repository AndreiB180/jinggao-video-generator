from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("JINGGAO_DATA", ROOT / "data")).resolve()


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "projects").mkdir(exist_ok=True)


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
