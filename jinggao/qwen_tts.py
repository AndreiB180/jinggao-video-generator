"""千问配音 (qwen3-tts-flash on Alibaba Cloud Bailian): the same key as the text and vision models.

One request carries at most 600 billed characters (a Chinese character counts as two), so a script is
cut at sentence ends into parts, voiced a few at a time (the account allows 180 requests a minute and
answers bursts with HTTP 429), downloaded and joined in order. List price 0.8 元 per 10,000 billed
characters, i.e. about 0.16 元 for 1,000 Chinese characters (4 minutes of narration).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import config, paths
from .execution import ThreadPoolExecutor

MODEL = "qwen3-tts-flash"
RATE_PER_MILLION = "80"     # 元 per million billed characters (0.8 元 / 10,000)
PART = 480                  # billed characters per request; the service limit is 600
WORKERS = 3
SPEEDS = ("1.0", "1.1", "1.2", "1.3")
DEFAULT_VOICE = "Ethan"
SAMPLE = "大家好，欢迎收看今天的节目。接下来，我用三分钟给你讲清楚这件事。"

# (voice, 名字, 说明, 分组). The service accepts every voice below with Mandarin text; dialect voices
# read it with their accent.
VOICES = [
    ("Ethan", "晨煦", "男声 · 标准普通话，阳光温暖", "解说"),
    ("Neil", "阿闻", "男声 · 新闻主播，平稳清晰", "解说"),
    ("Elias", "墨讲师", "女声 · 知识讲解，娓娓道来", "解说"),
    ("Cherry", "芊悦", "女声 · 亲切自然", "解说"),
    ("Maia", "四月", "女声 · 知性温柔", "解说"),
    ("Serena", "苏瑶", "女声 · 温柔", "解说"),
    ("Moon", "月白", "男声 · 率性帅气", "解说"),
    ("Andre", "安德雷", "男声 · 磁性沉稳", "解说"),
    ("Vincent", "田叔", "男声 · 沙哑烟嗓，史诗感", "故事"),
    ("Eldric Sage", "沧明子", "男声 · 沉稳睿智的长者", "故事"),
    ("Arthur", "徐大爷", "男声 · 不紧不慢讲故事的大爷", "故事"),
    ("Bellona", "燕铮莺", "女声 · 字正腔圆，说书感", "故事"),
    ("Katerina", "卡捷琳娜", "女声 · 御姐，韵律感强", "故事"),
    ("Ryan", "甜茶", "男声 · 节奏快，戏感强", "故事"),
    ("Kai", "凯", "男声 · 舒缓放松", "故事"),
    ("Dylan", "北京-晓东", "男声 · 北京话", "方言"),
    ("Peter", "天津-李彼得", "男声 · 天津话", "方言"),
    ("Jada", "上海-阿珍", "女声 · 上海话", "方言"),
    ("Li", "南京-老李", "男声 · 南京话", "方言"),
    ("Marcus", "陕西-秦川", "男声 · 陕西话", "方言"),
    ("Sunny", "四川-晴儿", "女声 · 四川话", "方言"),
    ("Eric", "四川-程川", "男声 · 四川话", "方言"),
    ("Roy", "闽南-阿杰", "男声 · 闽南腔", "方言"),
    ("Rocky", "粤语-阿强", "男声 · 粤语", "方言"),
    ("Kiki", "粤语-阿清", "女声 · 粤语", "方言"),
    ("Chelsie", "千雪", "女声 · 二次元", "角色"),
    ("Momo", "茉兔", "女声 · 俏皮", "角色"),
    ("Vivian", "十三", "女声 · 飒爽可爱", "角色"),
    ("Mia", "乖小妹", "女声 · 乖巧", "角色"),
    ("Bella", "萌宝", "女声 · 小萝莉", "角色"),
    ("Bunny", "萌小姬", "女声 · 萌系", "角色"),
    ("Nini", "邻家妹妹", "女声 · 软糯", "角色"),
    ("Seren", "小婉", "女声 · 助眠", "角色"),
    ("Stella", "少女阿月", "女声 · 少女", "角色"),
    ("Mochi", "沙小弥", "男声 · 小大人", "角色"),
    ("Pip", "顽屁小孩", "男声 · 调皮小孩", "角色"),
    ("Nofish", "不吃鱼", "男声 · 不卷舌", "角色"),
]
NAMES = {voice for voice, *_ in VOICES}
_gate = threading.BoundedSemaphore(WORKERS)


def voices() -> list[dict]:
    return [{"id": voice, "name": name, "note": note, "group": group} for voice, name, note, group in VOICES]


def voice(cfg: dict) -> str:
    chosen = str(cfg.get("qwen_tts_voice") or DEFAULT_VOICE)
    return chosen if chosen in NAMES else DEFAULT_VOICE


def endpoint(cfg: dict) -> str:
    host = (urlparse(cfg.get("model_base_url", "")).hostname or "").lower()
    if host in ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"):
        return f"https://{host}/api/v1/services/aigc/multimodal-generation/generation"
    raise ValueError("千问配音需要阿里云百炼（DashScope）接口：请在设置里填写百炼的服务地址和密钥")


def billed(text: str) -> int:
    """An upper bound of the service's count: every non-ASCII character counts as two."""
    return sum(1 if ord(ch) < 128 else 2 for ch in text)


def split(text: str, limit: int = PART) -> list[str]:
    """Parts of at most `limit` billed characters, cut at sentence ends (then commas, then anywhere)."""
    def pieces(chunk, marks):
        if billed(chunk) <= limit:
            return [chunk]
        if not marks:
            out, current = [], ""
            for ch in chunk:
                if billed(current + ch) > limit:
                    out.append(current)
                    current = ""
                current += ch
            return out + ([current] if current else [])
        parts = [p for p in re.split(rf"(?<=[{re.escape(marks[0])}])", chunk) if p]
        return [q for p in parts for q in pieces(p, marks[1:])]

    text = re.sub(r"\s*\n\s*", "\n", str(text or "").strip())
    sentences = [s for s in pieces(text, ["。！？!?；;…\n", "，,、：:", " "]) if s.strip()]
    parts, current = [], ""
    for sentence in sentences:
        if current and billed(current + sentence) > limit:
            parts.append(current)
            current = ""
        current += sentence
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _download(client: httpx.Client, url: str, target: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not (parsed.hostname or "").endswith(".aliyuncs.com"):
        raise ValueError("千问配音返回了无法识别的音频地址")
    for attempt in range(3):
        try:
            r = client.get(url, timeout=60)
            r.raise_for_status()
            target.write_bytes(r.content)
            return
        except httpx.HTTPError:
            if attempt == 2:
                raise ValueError("千问配音音频下载失败，请检查网络后重试") from None
            time.sleep(1 + attempt)


def _voice_part(client, cfg, text, target, check):
    from .tts_costs import speech_request
    cfg = cfg | {"tts_provider": "qwen_tts"}
    url = endpoint(cfg)
    payload = {"model": MODEL, "input": {"text": text, "voice": voice(cfg), "language_type": "Chinese"}}
    for attempt in range(6):
        retry = 0.0
        with _gate:
            if check:
                check()
            with speech_request(cfg, text, check=check) as charge:
                charge.mark_sent()
                try:
                    r = client.post(url, headers={"Authorization": "Bearer " + cfg["model_api_key"]},
                                    json=payload, timeout=120)
                except httpx.HTTPError:
                    if attempt == 5:
                        raise ValueError("千问配音请求失败，请检查网络后重试") from None
                    retry = 2.0 * (attempt + 1)
                else:
                    charge.response(r)
                    if r.status_code == 429 or r.status_code >= 500:
                        retry = 2.0 * (attempt + 1)
                    elif r.status_code in (401, 403):
                        raise ValueError("千问配音被拒绝：请检查百炼密钥是否有效、是否开通了语音合成")
                    elif r.status_code != 200:
                        message = ""
                        try:
                            message = r.json().get("message", "")
                        except ValueError:
                            pass
                        raise ValueError(f"千问配音失败（HTTP {r.status_code}）{message}".strip())
        if retry:
            if attempt == 5:
                raise ValueError("千问配音请求过于频繁或服务繁忙，请稍后重试")
            time.sleep(retry)
            continue
        audio = ((r.json().get("output") or {}).get("audio") or {}).get("url")
        if not audio:
            raise ValueError("千问配音没有返回音频")
        _download(client, audio, target)
        return


def synthesize(text: str, target: Path, check=None) -> None:
    from . import media
    cfg = config.read()
    if not cfg.get("model_api_key"):
        raise ValueError("请先在设置里填写阿里云百炼密钥（千问配音与文案、画面检查共用这一个密钥）")
    parts = split(text)
    if not parts:
        raise ValueError("没有可配音的文字")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-", dir=target.parent) as directory:
        folder = Path(directory)
        files = [folder / f"{i:03d}.wav" for i in range(len(parts))]
        with httpx.Client(follow_redirects=False) as client, ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for future in [pool.submit(_voice_part, client, cfg, part, file, check)
                           for part, file in zip(parts, files, strict=True)]:
                future.result()
        if check:
            check()
        joined = folder / "joined.wav"
        args = [paths.binary("ffmpeg"), "-v", "error", "-y"]
        for file in files:
            args += ["-i", str(file)]
        speed = cfg.get("tts_speed", "1.0")
        chain = f"concat=n={len(files)}:v=0:a=1"
        if speed in SPEEDS and speed != "1.0":
            chain += f",atempo={speed}"
        media.run(args + ["-filter_complex", chain + "[a]", "-map", "[a]", "-ar", "24000", "-ac", "1",
                          "-c:a", "pcm_s16le", str(joined)], check=check, timeout=600)
        if float(media.probe(joined).get("format", {}).get("duration", 0)) <= 0:
            raise ValueError("千问配音合成的音轨没有有效时长")
        joined.replace(target)


def preview(name: str) -> Path:
    """A short sample of one voice: bundled with the app, otherwise made once and kept."""
    if name not in NAMES:
        raise ValueError("没有这个音色")
    bundled = paths.ROOT / "assets" / "voices" / f"{name}.mp3"
    if bundled.exists():
        return bundled
    key = hashlib.sha1(f"{MODEL}|{name}|{SAMPLE}".encode()).hexdigest()[:10]
    cached = paths.DATA / "voice-samples" / f"{name}-{key}.wav"
    if not cached.exists():
        cfg = config.read() | {"qwen_tts_voice": name, "tts_speed": "1.0"}
        with tempfile.TemporaryDirectory(prefix=".sample-", dir=paths.DATA) as directory:
            file = Path(directory) / "sample.wav"
            with httpx.Client(follow_redirects=False) as client:
                if not cfg.get("model_api_key"):
                    raise ValueError("请先在设置里填写阿里云百炼密钥")
                _voice_part(client, cfg, SAMPLE, file, None)
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), cached)
    return cached
