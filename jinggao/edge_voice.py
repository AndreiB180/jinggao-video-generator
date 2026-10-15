"""免费配音：微软 Edge「大声朗读」的神经网络音色（通过 edge-tts 库，LGPLv3）。

No key and no charge: the same neural voices many explainer videos use (云希, 云扬, 晓晓...). It is the
service behind Edge's read-aloud button, not a published API, so Microsoft may limit or change it; the
paid 千问配音 stays available in Settings. The script is cut at sentence ends into parts, voiced a few
at a time, and joined into one track; speed is applied by the service (rate), not by stretching audio.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from . import config, paths
from .execution import ThreadPoolExecutor

PART = 800                  # characters per request (the service cuts at 4096 bytes)
WORKERS = 3
DEFAULT_VOICE = "zh-CN-YunxiNeural"
SAMPLE = "大家好，欢迎收看今天的节目。接下来，我用三分钟给你讲清楚这件事。"
SPEEDS = ("1.0", "1.1", "1.2", "1.3")

# (voice, 名字, 说明, 分组)
VOICES = [
    ("zh-CN-YunxiNeural", "云希", "男声 · 阳光自然，解说常用", "解说"),
    ("zh-CN-YunyangNeural", "云扬", "男声 · 新闻播报", "解说"),
    ("zh-CN-YunjianNeural", "云健", "男声 · 激情解说", "解说"),
    ("zh-CN-XiaoxiaoNeural", "晓晓", "女声 · 温暖自然", "解说"),
    ("zh-CN-XiaoyiNeural", "晓伊", "女声 · 活泼", "解说"),
    ("zh-CN-YunxiaNeural", "云夏", "男童", "角色"),
    ("zh-CN-liaoning-XiaobeiNeural", "晓北", "女声 · 东北话", "方言"),
    ("zh-CN-shaanxi-XiaoniNeural", "晓妮", "女声 · 陕西话", "方言"),
    ("zh-HK-HiuMaanNeural", "曉曼", "女声 · 粤语", "方言"),
    ("zh-HK-WanLungNeural", "雲龍", "男声 · 粤语", "方言"),
    ("zh-TW-HsiaoChenNeural", "曉臻", "女声 · 台湾腔", "方言"),
    ("zh-TW-YunJheNeural", "雲哲", "男声 · 台湾腔", "方言"),
]
NAMES = {voice for voice, *_ in VOICES}
_gate = threading.BoundedSemaphore(WORKERS)


def voices() -> list[dict]:
    return [{"id": voice, "name": name, "note": note, "group": group} for voice, name, note, group in VOICES]


def voice(cfg: dict) -> str:
    chosen = str(cfg.get("edge_tts_voice") or DEFAULT_VOICE)
    return chosen if chosen in NAMES else DEFAULT_VOICE


def rate(cfg: dict) -> str:
    speed = cfg.get("tts_speed", "1.0")
    speed = speed if speed in SPEEDS else "1.0"
    return f"+{round((float(speed) - 1) * 100)}%"


def split(text: str) -> list[str]:
    from . import qwen_tts
    # qwen_tts.split counts a Chinese character as two; PART * 2 keeps about PART characters a part.
    return qwen_tts.split(text, limit=PART * 2)


def _proxy() -> str | None:
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None


def _voice_part(text: str, target: Path, name: str, speed: str, check) -> None:
    import edge_tts
    last = None
    for attempt in range(4):
        with _gate:
            if check:
                check()
            try:
                asyncio.run(edge_tts.Communicate(text, name, rate=speed, proxy=_proxy()).save(str(target)))
                if target.exists() and target.stat().st_size > 0:
                    return
                last = "没有返回音频"
            except Exception as exc:          # network, the service refusing, a changed protocol
                last = f"{type(exc).__name__}: {str(exc)[:120]}"
        time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"免费配音（微软 Edge 朗读）暂时不可用：{last}。请稍后重试，或在设置里改用千问配音")


def synthesize(text: str, target: Path, check=None) -> None:
    from . import media
    cfg = config.read()
    parts = split(text)
    if not parts:
        raise ValueError("没有可配音的文字")
    name, speed = voice(cfg), rate(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-", dir=target.parent) as directory:
        folder = Path(directory)
        files = [folder / f"{i:03d}.mp3" for i in range(len(parts))]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for future in [pool.submit(_voice_part, part, file, name, speed, check)
                           for part, file in zip(parts, files, strict=True)]:
                future.result()
        if check:
            check()
        joined = folder / "joined.wav"
        args = [paths.binary("ffmpeg"), "-v", "error", "-y"]
        for file in files:
            args += ["-i", str(file)]
        media.run(args + ["-filter_complex", f"concat=n={len(files)}:v=0:a=1[a]", "-map", "[a]", "-ar", "24000",
                          "-ac", "1", "-c:a", "pcm_s16le", str(joined)], check=check, timeout=600)
        if float(media.probe(joined).get("format", {}).get("duration", 0)) <= 0:
            raise ValueError("免费配音合成的音轨没有有效时长")
        joined.replace(target)
    target.with_suffix(".txt").write_text(text, "utf-8")


def preview(name: str) -> Path:
    """A short sample of one voice: bundled with the app, otherwise made once and kept."""
    if name not in NAMES:
        raise ValueError("没有这个音色")
    bundled = paths.ROOT / "assets" / "voices" / f"{name}.mp3"
    if bundled.exists():
        return bundled
    key = hashlib.sha1(f"edge|{name}|{SAMPLE}".encode()).hexdigest()[:10]
    cached = paths.DATA / "voice-samples" / f"{name}-{key}.mp3"
    if not cached.exists():
        paths.DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sample-", dir=paths.DATA) as directory:
            file = Path(directory) / "sample.mp3"
            _voice_part(SAMPLE, file, name, "+0%", None)
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), cached)
    return cached
