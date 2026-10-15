"""NVIDIA video encoding: what this computer's card can do, checked with the bundled FFmpeg itself.

Three ways to make a video clip, fastest first:
  full   NVDEC decodes, CUDA scales, NVENC encodes: the CPU only moves bytes
  nvenc  the CPU decodes and scales, NVENC encodes
  cpu    libx264, always works
The launcher checks the card once at start (detect) and saves the answer in data/gpu-check.json, which every
worker reads. A clip that fails on the GPU is made again on the CPU; three GPU failures in a row stop this
process from trying that way again until it restarts. Nothing here is required: without a card everything is
the CPU path, as before.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from . import config, paths

CHECK_FILE = "gpu-check.json"
NVENC_CLIP = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0"]
NVENC_RENDER = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-b:v", "0"]
X264_CLIP = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2"]
X264_RENDER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "2"]
BREAK_AFTER = 3          # GPU failures in a row before this process stops trying that way

_lock = threading.Lock()
_cache: dict = {"mtime": None, "value": None}
_failures: dict = {"full": 0, "nvenc": 0}
_disabled: dict = {}     # mode -> last error, for this process only
_counts: dict = {}       # mode -> clips/shots made that way in this process


def _smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if not found and os.name == "nt":
        legacy = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        found = str(legacy) if legacy.exists() else None
    return found


def card() -> dict:
    """Name, driver and memory from nvidia-smi; empty when there is no NVIDIA driver."""
    from . import media
    smi = _smi()
    if not smi:
        return {}
    try:
        line = media.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         timeout=15).strip().splitlines()[0]
    except (ValueError, IndexError, OSError):
        return {}
    parts = [p.strip() for p in line.split(",")]
    return {"gpu": parts[0], "driver": parts[1] if len(parts) > 1 else "", "memory": parts[2] if len(parts) > 2 else ""}


def sessions_for(driver: str) -> int:
    """Encode sessions to use at once. GeForce drivers allowed 3 sessions, 5 from 530 and 8 from 551;
    one or two stay free for other programs (screen recorders, browsers)."""
    try:
        major = float(re.match(r"\d+(\.\d+)?", driver or "").group(0))
    except (AttributeError, ValueError):
        return 3
    return 6 if major >= 551 else 4 if major >= 530 else 3


def advice(error: str, driver: str = "") -> str:
    """What the user can do about a failed check, in plain words."""
    text = error or ""
    needed = re.search(r"minimum required Nvidia driver for nvenc is ([\d.]+)", text)
    if needed or "required nvenc API version" in text:
        want = needed.group(1) if needed else "更新的版本"
        return (f"显卡驱动版本{(' ' + driver + ' ') if driver else ''}太旧，显卡编码需要 {want} 以上。"
                "请到 NVIDIA 官网或 GeForce Experience 更新显卡驱动，装好后重启镜稿。")
    if re.search(r"Cannot load (nvcuda|nvEncodeAPI)", text):
        return "没有找到 NVIDIA 驱动的编码组件，请重新安装显卡驱动后重启镜稿。"
    if re.search(r"No capable devices found|unsupported device|No NVENC capable", text, re.I):
        return "这块显卡不支持硬件编码（NVENC），只能用 CPU。"
    if re.search(r"incompatible client key|out of memory|OpenEncodeSessionEx failed", text):
        return "显卡编码通道被其他程序占满（例如录屏、直播软件），关掉它们后点「重新检测」。"
    return "显卡编码测试没有通过，已自动改用 CPU。"


def _tail(error: Exception | str) -> str:
    text = str(error).replace("媒体处理失败：", "").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-4:])[:600]


def detect(write: bool = True) -> dict:
    """Run the real checks with the bundled FFmpeg (a few seconds with an NVIDIA card, instant without)."""
    from . import express_sources, media
    result = {"checked": time.time(), "gpu": "", "driver": "", "memory": "", "nvidia": False,
              "nvenc": False, "full": False, "sessions": 0, "reason": "", "advice": ""}
    try:
        ffmpeg = paths.binary("ffmpeg")
    except RuntimeError as exc:
        return _save(result | {"reason": str(exc)}, write)
    info = card()
    result.update(info, nvidia=bool(info))
    try:
        media.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25", "-t", "1",
                   *NVENC_CLIP, "-f", "null", "-"], timeout=40)
        result["nvenc"] = True
    except ValueError as exc:
        if not info:
            return _save(result | {"reason": "没有检测到 NVIDIA 显卡，使用 CPU 合成"}, write)
        return _save(result | {"reason": _tail(exc), "advice": advice(str(exc), info.get("driver", ""))}, write)
    result["sessions"] = sessions_for(result["driver"])
    with tempfile.TemporaryDirectory(prefix="gpu-check-", dir=paths.DATA) as folder:
        sample, out = Path(folder) / "sample.mp4", Path(folder) / "clip.mp4"
        try:
            media.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30", "-t", "2",
                       "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(sample)], timeout=40)
            express_sources.cut_window(str(sample), 0.5, 1.0, out, short_edge=720, mode="full", timeout=40)
            stream = next(s for s in media.probe(out)["streams"] if s.get("codec_type") == "video")
            if (int(stream["width"]), int(stream["height"])) != (1280, 720):
                raise ValueError(f"尺寸不对：{stream['width']}x{stream['height']}")
            result["full"] = True
        except (ValueError, StopIteration, KeyError, OSError) as exc:
            # NVENC alone still takes the encoding off the CPU.
            result["reason"] = "显卡解码未通过，片段改为 CPU 解码、显卡编码：" + _tail(exc)
    return _save(result, write)


def _save(result: dict, write: bool) -> dict:
    if write:
        target = paths.DATA / CHECK_FILE
        temporary = paths.temporary(target)
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        paths.publish(temporary, target)
    with _lock:
        _cache.update(mtime=None, value=result)
        _disabled.clear()
        _failures.update(full=0, nvenc=0)
    return result


def status() -> dict:
    """The saved check (re-read when the file changes); checked here if no launcher has written one."""
    target = paths.DATA / CHECK_FILE
    try:
        mtime = target.stat().st_mtime
    except OSError:
        mtime = None
    with _lock:
        if _cache["value"] is not None and (mtime is None or mtime == _cache["mtime"]):
            return _cache["value"]
    if mtime is None:
        return detect()
    try:
        value = json.loads(target.read_text("utf-8"))
    except (OSError, ValueError):
        return detect()
    with _lock:
        if _cache["mtime"] is not None:      # a new check (「重新检测」) starts this process afresh too
            _disabled.clear()
            _failures.update(full=0, nvenc=0)
        _cache.update(mtime=mtime, value=value)
    return value


def mode(kind: str = "clip") -> str:
    """How to encode now: clip -> full | nvenc | cpu; render -> nvenc | cpu (the encoder setting decides first)."""
    setting = config.read().get("encoder", "auto")
    if setting == "cpu":
        return "cpu"
    state = status()
    with _lock:
        if kind == "clip" and state.get("full") and "full" not in _disabled:
            return "full"
        if state.get("nvenc") and "nvenc" not in _disabled:
            return "nvenc"
    return "cpu"


def worked(way: str) -> None:
    with _lock:
        _counts[way] = _counts.get(way, 0) + 1
        if way in _failures:
            _failures[way] = 0


def failed(way: str, error: Exception | str) -> None:
    """A GPU attempt failed (the caller redoes it on the CPU)."""
    with _lock:
        _counts[way + "_failed"] = _counts.get(way + "_failed", 0) + 1
        if way not in _failures:
            return
        _failures[way] += 1
        if _failures[way] >= BREAK_AFTER:
            _disabled[way] = _tail(error)


def process_notes() -> dict:
    with _lock:
        return {"counts": dict(_counts), "disabled": dict(_disabled)}


def summary(state: dict | None = None) -> str:
    """One line for the launcher window and the settings page."""
    state = state or status()
    if state.get("full"):
        return f"显卡加速已开启：{state.get('gpu') or 'NVIDIA'}（驱动 {state.get('driver') or '未知'}），片段转码和合成都用显卡"
    if state.get("nvenc"):
        return f"显卡编码已开启：{state.get('gpu') or 'NVIDIA'}（驱动 {state.get('driver') or '未知'}），解码仍用 CPU"
    if state.get("nvidia"):
        return f"显卡加速未开启：{state.get('gpu')}（驱动 {state.get('driver') or '未知'}）。{state.get('advice') or ''}"
    return "没有检测到 NVIDIA 显卡，使用 CPU 合成"
