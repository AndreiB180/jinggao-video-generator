"""Bounded public source-audio timestamps, used only to locate video frames.

No transcript is a picture verdict. This module accepts one already inspected
anonymous Bilibili source, never cookies, alternate entitlements or paid ASR.
Temporary audio is removed; the cache contains only bound transcript records.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlencode, urlparse

from . import execution, local_asr, media, paths, segment_discovery

VERSION = 1
MAX_SECONDS = 30
MAX_BYTES = 6 * 1024 * 1024
CACHE_TTL = 7 * 86400
_MODEL_CACHE = {}
_FLAGS = (
    "is_drm", "has_drm", "is_preview", "is_ugc_pay_preview", "is_pay", "pay",
    "is_upower_exclusive", "is_upower_play", "is_upower_preview", "is_stein_gate",
    "need_login", "need_vip", "drmFamilies", "drm_infos", "drm_info",
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("来源音轨缺少有效数值")
    return float(value)


def _restricted(row):
    for name in _FLAGS:
        value = row.get(name)
        if value not in (None, False, 0, "", "0", "false", [], {}):
            return True
    return False


def _source(candidate):
    source = candidate.get("source_url", "")
    match = re.fullmatch(r"https://(?:www\.)?bilibili\.com/video/(BV[0-9A-Za-z]{10})/?", source)
    evidence = candidate.get("source_evidence") or {}
    if (not match or candidate.get("metadata_checked") is not True
            or candidate.get("source_verified") is not True
            or candidate.get("download_status") != "available"
            or candidate.get("media_extractor") != "bilibili-public"
            or evidence.get("readable") is not True
            or evidence.get("evidence_origin") != "platform_metadata"
            or evidence.get("video_id") != match[1]
            or evidence.get("url", "").rstrip("/") != source.rstrip("/")
            or _restricted(candidate) or _restricted(evidence)
            or _restricted(candidate.get("rights") or {})):
        raise ValueError("音轨定位只接受已核对的普通公开 B站来源")
    aid, cid = evidence.get("aid"), evidence.get("cid")
    if (type(aid) is not int or type(cid) is not int or aid <= 0 or cid <= 0
            or any(key in candidate and candidate[key] != value for key, value in (("aid", aid), ("cid", cid)))):
        raise ValueError("来源音轨的视频编号不一致")
    duration = _number(candidate.get("duration"))
    if not 30 <= duration <= 420:
        raise ValueError("来源音轨定位仅处理 30–420 秒的公开视频")
    return {"source_url": source.rstrip("/"), "video_id": match[1], "aid": aid, "cid": cid,
            "source_duration": round(duration, 3)}


def eligible(candidate):
    """Cheap source-only check, with no filesystem, model or network access."""
    try:
        _source(candidate)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def _digest(path, check):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            check()
            value.update(chunk)
    check()
    return value.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _model_identity(check):
    """Bind actual installed weights and code; never install missing weights."""
    root = local_asr.model_dir()
    try:
        signature = tuple((name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns,
                           (root / name).stat().st_ctime_ns) for name in sorted(local_asr.FILES))
    except OSError:
        raise ValueError("本地字幕模型未安装，已跳过音轨定位") from None
    key = (str(root.resolve()), signature, tuple(sorted(local_asr.FILES.items())))
    check()
    if key not in _MODEL_CACHE:
        for name, expected in local_asr.FILES.items():
            if _digest(root / name, check) != expected:
                raise ValueError("本地字幕模型校验失败，已跳过音轨定位")
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = True
    return {"model_id": local_asr.MODEL_ID, "files": dict(local_asr.FILES),
            "implementation_sha256": _digest(local_asr.__file__, check)}


def _near(actual, expected):
    return abs(_number(actual) - expected) <= 1.5


def _audio_url(value, cid):
    if not isinstance(value, str):
        raise ValueError("公开音轨没有有效地址")
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    filename = parsed.path.rsplit("/", 1)[-1]
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not any(host == domain or host.endswith("." + domain) for domain in ("bilivideo.com", "bilivideo.cn"))
            or f"/{cid}/" not in parsed.path or not filename.startswith(f"{cid}_")
            or not filename.endswith(".m4s") or parsed.fragment):
        raise ValueError("公开音轨地址与已检查视频编号不一致")
    return value


def _read(url, budget, referer, *, limit, target=None):
    """Interrupt header reads independently of HTTPX's per-read timeout.

    A peer can keep sending partial headers before each socket timeout without
    yielding a response to the caller. The existing trace hook registers the
    connected socket before those headers, allowing the watcher to shut it down.
    Keep the original cancellation even if socket shutdown raises an I/O error.
    """
    done, failures = threading.Event(), []

    def watch():
        while not done.is_set():
            try:
                budget.check()
            except Exception as exc:
                failures.append(exc)
                budget.stop()
                return
            done.wait(min(.01, max(0, budget.deadline - time.monotonic())))

    watcher = threading.Thread(target=watch, name="source-audio-deadline", daemon=True)
    watcher.start()
    failure = None
    try:
        return segment_discovery._read(url, budget, referer, limit=limit, target=target,
                                       deadline_supervised=True)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        done.set()
        watcher.join()
        if failures and (failure is None or isinstance(failure, Exception)) and not isinstance(failure, media.Cancelled):
            raise failures[0]


def _metadata(binding, budget):
    url = "https://api.bilibili.com/x/player/playurl?" + urlencode({
        "avid": binding["aid"], "cid": binding["cid"], "qn": 64, "fnval": 16, "fourk": 0})
    raw, final = _read(url, budget, binding["source_url"], limit=128 * 1024)
    if final != url:
        raise ValueError("公开音轨元数据发生了不一致的跳转")
    body = json.loads(raw)
    data = body.get("data") or {}
    if (body.get("code") != 0 or not isinstance(data, dict) or _restricted(data)
            or _restricted(data.get("rights") or {})):
        raise ValueError("来源没有返回普通公开音轨，已停止")
    for key, expected in (("aid", binding["aid"]), ("cid", binding["cid"]), ("bvid", binding["video_id"])):
        if key in data and data[key] != expected:
            raise ValueError("公开音轨元数据的视频编号不一致")
    dash = data.get("dash") or {}
    expected = binding["source_duration"]
    if (not _near(_number(data.get("timelength")) / 1000, expected)
            or not isinstance(dash, dict) or _restricted(dash) or not _near(dash.get("duration"), expected)):
        raise ValueError("公开音轨时长与原视频不一致")
    choices = []
    for row in dash.get("audio") or []:
        if not isinstance(row, dict) or _restricted(row):
            continue
        mime = row.get("mimeType") or row.get("mime_type")
        bandwidth = row.get("bandwidth")
        if (mime != "audio/mp4" or row.get("codecs") != "mp4a.40.2"
                or type(bandwidth) not in (int, float) or not math.isfinite(bandwidth)
                or not 32000 <= bandwidth <= 160000
                or bandwidth * expected / 8 > MAX_BYTES - budget.bytes):
            continue
        try:
            target = _audio_url(row.get("baseUrl") or row.get("base_url"), binding["cid"])
        except ValueError:
            continue
        choices.append((bandwidth, target))
    if not choices:
        raise ValueError("来源没有预算内可直接读取的普通 AAC 音轨")
    return min(choices, key=lambda row: row[0])[1]


def _decode(source, target, expected, budget):
    budget.check()
    metadata = json.loads(media.run([
        paths.binary("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-f", "mov",
        "-show_entries", "format=duration:stream=codec_type,codec_name,duration", "-of", "json", str(source),
    ], timeout=max(.1, budget.deadline - time.monotonic()), check=budget.check))
    tracks = metadata.get("streams") or []
    if (len(tracks) != 1 or tracks[0].get("codec_type") != "audio" or tracks[0].get("codec_name") != "aac"
            or not _near(float((metadata.get("format") or {}).get("duration", 0)), expected)):
        raise ValueError("下载音轨的实际格式或时长与来源不一致")
    media.run([
        paths.binary("ffmpeg"), "-v", "error", "-nostdin", "-y", "-threads", "1",
        "-protocol_whitelist", "file", "-f", "mov", "-i", str(source), "-map", "0:a:0", "-vn",
        "-t", str(expected + 1.5), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
    ], timeout=max(.1, budget.deadline - time.monotonic()), check=budget.check)
    budget.check()
    with wave.open(str(target), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("来源音轨没有解码成标准本地语音格式")
        duration = audio.getnframes() / 16000
    if not _near(duration, expected):
        raise ValueError("来源音轨解码时长与视频不一致")
    return duration


def _words(rows, duration):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8000:
        raise ValueError("来源音轨没有可用的语音时间索引")
    result, previous_start, previous_end = [], 0.0, 0.0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not 1 <= len(row["text"].strip()) <= 64:
            raise ValueError("来源音轨的识别文字无效")
        start, end = _number(row.get("start")), _number(row.get("end"))
        if (not 0 <= start < end <= duration + .08 or start < previous_start
                or end < previous_end or start < previous_end - .05):
            raise ValueError("来源音轨的识别时间戳无效")
        result.append({"text": row["text"].strip(), "start": start, "end": end})
        previous_start, previous_end = start, end
    return result


def _cached(file, binding):
    try:
        if file.stat().st_size > 2 * 1024 * 1024:
            return None
        record = json.loads(file.read_text("utf-8"))
        payload = record["payload"]
        if (record["version"] != VERSION or record["binding"] != binding
                or not 0 <= time.time() - record["created_at"] <= CACHE_TTL
                or record["payload_sha256"] != _json_digest(payload)
                or not re.fullmatch(r"[0-9a-f]{64}", payload["audio_sha256"])
                or payload["cache_key"] != _json_digest({"binding": binding, "audio_sha256": payload["audio_sha256"]})
                or payload["source_url"] != binding["source_url"]
                or not _near(payload["duration"], binding["source_duration"])):
            return None
        return {"words": _words(payload["words"], payload["duration"]), "duration": payload["duration"],
                "source_url": binding["source_url"], "audio_sha256": payload["audio_sha256"],
                "cache_key": payload["cache_key"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def transcript(candidate, check=lambda: None):
    """Return bound word timestamps or ValueError; propagate user cancellation.

    The shared deadline covers metadata, bytes, decoding, ASR-lock waiting and
    every ASR chunk boundary. One active ONNX chunk cannot be pre-empted midway.
    """
    began = time.monotonic()
    budget = segment_discovery._Budget(MAX_SECONDS, MAX_BYTES, check)
    try:
        budget.check()
        binding = _source(candidate) | {"version": VERSION, "asr": _model_identity(budget.check)}
        root = paths.DATA / "cache" / "source-audio"
        root.mkdir(parents=True, exist_ok=True)
        cache_file = root / (_json_digest(binding) + ".json")
        with execution.keyed_lock(("source-audio", str(cache_file)), check=budget.check):
            cached = _cached(cache_file, binding)
            if cached:
                budget.check()
                return cached | {"cache_hit": True, "elapsed_seconds": round(time.monotonic() - began, 3),
                                 "downloaded_bytes": 0, "model_requests": 0, "network_requests": 0,
                                 "sampling_only": True, "transcript_origin": "local_asr",
                                 "cache_path": str(cache_file)}
            target_url = _metadata(binding, budget)
            with tempfile.TemporaryDirectory(prefix=".audio-", dir=root) as work:
                compressed, wav = Path(work) / "source.m4s", Path(work) / "source.wav"
                raw, final = _read(target_url, budget, binding["source_url"],
                                   limit=MAX_BYTES - budget.bytes, target=compressed)
                _audio_url(final, binding["cid"])
                audio_sha256 = hashlib.sha256(raw).hexdigest()
                del raw
                duration = _decode(compressed, wav, binding["source_duration"], budget)
                rows = local_asr.transcribe(wav, check=budget.check, progress=lambda *_: budget.check())
                budget.check()
                words = _words(rows, duration)
                if _model_identity(budget.check) != binding["asr"]:
                    raise ValueError("本地语音模型发生变化，未缓存来源索引")
                payload = {"words": words, "duration": duration, "source_url": binding["source_url"],
                           "audio_sha256": audio_sha256,
                           "cache_key": _json_digest({"binding": binding, "audio_sha256": audio_sha256})}
                record = {"version": VERSION, "binding": binding, "created_at": time.time(),
                          "payload": payload, "payload_sha256": _json_digest(payload)}
                temporary = Path(work) / "transcript.json"
                temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), "utf-8")
                budget.check()
                temporary.replace(cache_file)
            return payload | {"cache_hit": False, "elapsed_seconds": round(time.monotonic() - began, 3),
                              "downloaded_bytes": budget.bytes, "model_requests": 0, "network_requests": 2,
                              "sampling_only": True, "transcript_origin": "local_asr",
                              "cache_path": str(cache_file)}
    except media.Cancelled:
        raise
    except Exception as exc:
        # Preserve cancellation over a concurrent decoder/network failure and
        # keep temporary signed CDN addresses out of user-visible errors.
        check()
        if isinstance(exc, ValueError) and len(str(exc)) < 180 and "http" not in str(exc).lower():
            raise
        raise ValueError("来源音轨定位失败，已停止本次读取和本地识别") from None
    finally:
        budget.stop()
