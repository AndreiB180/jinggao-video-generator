"""Bounded, standalone public-footage supply probe; never approves a scene.

Run in a separate process with ``python -m jinggao.public_supply_probe``. This
does not update projects or the production source index. A recorded URL is not
automatically retried, including after interruption. Baidu/Haokan and protected
routes are deliberately not implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

MAX_QUERIES = 3
MAX_CANDIDATES = 6
MAX_MEDIA_BYTES = 24 * 1024 * 1024
CANDIDATE_SECONDS = 75
PROVIDERS = ("tencent", "bilibili", "sohu")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    pending.replace(path)


def _safe_error(exc: Exception) -> str:
    return re.sub(r"https?://[^\s\"'<>]+", "[URL omitted]", str(exc))[:600]


def candidate_metadata(candidate: dict) -> dict:
    """Keep canonical publisher evidence, never playback URLs or auth fields."""
    from .baidu_videos import _original
    from .video_sources import _duration
    source, provider, ident = _original(str(candidate.get("source_url", "")))
    if not source or provider not in PROVIDERS:
        raise ValueError("仅接受已接入的腾讯、B站或搜狐单个公开视频原页")
    return {"source_url": source, "provider": provider, "video_id": ident,
            "kind": "video", "title": str(candidate.get("title", ""))[:300],
            "description": str(candidate.get("description", ""))[:1200],
            "publisher": str(candidate.get("publisher", ""))[:200],
            "duration": _duration(candidate.get("duration")),
            "license": "公开可读不等于转载许可；本记录仅供诊断"}


def search_candidates(provider: str, queries: list[str], root: Path) -> list[dict]:
    """One normal first-page request per explicit query; stop on route failure."""
    from . import platform_videos
    if provider not in ("tencent", "bilibili"):
        raise ValueError("在线检索仅支持腾讯和B站；搜狐使用已取得的公开目录候选")
    queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if not 1 <= len(queries) <= MAX_QUERIES or any(len(q) > 60 for q in queries):
        raise ValueError("每次需要 1–3 个不超过 60 字的明确通用主题")
    result = []
    fetch = getattr(platform_videos, provider + "_search")
    for query in queries:
        ident = hashlib.sha256((provider + "\n" + query).encode()).hexdigest()[:16]
        record = root / "search" / (ident + ".json")
        if record.exists():
            row = json.loads(record.read_text("utf-8"))
        else:
            row = {"provider": provider, "query": query, "status": "started", "candidates": []}
            _write(record, row)
            began = time.monotonic()
            try:
                candidates = fetch(query)
                rows = []
                for c in candidates:
                    try:
                        rows.append(candidate_metadata(c))
                    except ValueError:
                        continue
                row.update(status="ok", returned=len(candidates), candidates=rows)
            except Exception as exc:
                row.update(status="failed", error=_safe_error(exc))
            row["seconds"] = round(time.monotonic() - began, 3)
            _write(record, row)
        if row.get("status") != "ok":
            break  # Includes an interrupted previous request; no automatic retry.
        result.extend(row.get("candidates", []))
    return result


def _probe_one(candidate: dict, root: Path, clip_seconds: float) -> dict:
    from PIL import Image, ImageChops, ImageStat
    from . import media, paths, video_sources as video
    began = time.monotonic()
    row = dict(candidate, status="inspecting", scene_verified=False, license_verified=False)

    def check():
        if time.monotonic() - began > 65:
            raise ValueError("单候选诊断达到取消期限")

    inspect_start = time.monotonic()
    inspected = video._inspect_video_candidate(candidate, check)
    row.update(inspect_seconds=round(time.monotonic() - inspect_start, 3),
               duration=inspected.get("duration", 0), width=inspected.get("width", 0),
               height=inspected.get("height", 0),
               inspection_status=inspected.get("download_status"),
               inspection_reason=_safe_error(ValueError(inspected.get("download_reason", ""))),
               stream_hosts=sorted({urlsplit(s.get("url", "")).hostname or ""
                                    for s in inspected.get("media_streams", [])}))
    if inspected.get("download_status") != "available":
        return dict(row, status="unavailable", seconds=round(time.monotonic() - began, 3))
    if float(inspected.get("duration", 0)) < max(30, clip_seconds):
        return dict(row, status="too_short", seconds=round(time.monotonic() - began, 3))
    inspected.update(prefer_efficient_stream=True, min_video_height=720)
    # This function only runs in the isolated worker; production globals stay intact.
    video.CLIP_LIMIT = MAX_MEDIA_BYTES
    start = round((float(inspected["duration"]) - clip_seconds) * .35, 3)
    ident = hashlib.sha256(candidate["source_url"].encode()).hexdigest()[:16]
    clip = root / "clips" / (ident + ".mp4")
    clip_start = time.monotonic()
    try:
        info = video.download_video_candidate(inspected, clip, start, clip_seconds, check)
    except Exception as exc:
        return dict(row, status="clip_failed", error=_safe_error(exc),
                    clip_seconds=round(time.monotonic() - clip_start, 3),
                    seconds=round(time.monotonic() - began, 3))
    row["clip"] = {k: info[k] for k in ("path", "source_clip_start", "duration", "width", "height",
                                       "download_mode", "downloaded_bytes")}
    row["clip_seconds"] = round(time.monotonic() - clip_start, 3)
    frame_paths, small = [], []
    for i, position in enumerate((.1, .5, .9)):
        check()
        frame = root / "frames" / f"{ident}-{i}.jpg"
        frame.parent.mkdir(exist_ok=True)
        media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", str(clip_seconds * position),
                   "-i", str(clip), "-frames:v", "1", str(frame)], timeout=10, check=check)
        frame_paths.append(str(frame))
        with Image.open(frame) as image:
            small.append(image.convert("L").resize((64, 64)))
    row["frame_paths"] = frame_paths
    # Changes can be edits, overlays, or camera motion; this is not a scene verdict.
    row["frame_change_mae"] = [round(ImageStat.Stat(ImageChops.difference(a, b)).mean[0], 3)
                               for a, b in zip(small, small[1:], strict=False)]
    return dict(row, status="clip_decoded", seconds=round(time.monotonic() - began, 3))


def _bounded_worker(candidate: dict, root: Path, clip_seconds: float) -> dict:
    """A real process deadline, with only this diagnostic's child tree stopped."""
    from . import video_sources as video
    ident = hashlib.sha256(candidate["source_url"].encode()).hexdigest()[:16]
    task = root / "workers" / (ident + ".json")
    result = task.with_suffix(".result.json")
    _write(task, {"candidate": candidate, "root": str(root), "clip_seconds": clip_seconds})
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
        "SSL_CERT_FILE", "SSL_CERT_DIR"}}
    env["JINGGAO_DATA"] = str(root / "isolated-data")
    proc = subprocess.Popen([sys.executable, "-m", "jinggao.public_supply_probe", "_worker", str(task)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env, cwd=Path(__file__).resolve().parent.parent,
                            start_new_session=os.name != "nt",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        proc.wait(timeout=CANDIDATE_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            video._stop_public_extractor(proc)
        return dict(candidate, status="deadline", scene_verified=False, license_verified=False)
    except BaseException:
        video._stop_public_extractor(proc)
        raise
    if proc.returncode or not result.exists():
        return dict(candidate, status="worker_failed", scene_verified=False, license_verified=False)
    return json.loads(result.read_text("utf-8"))


def probe_candidates(candidates: list[dict], root: Path, *, limit=3, clip_seconds=10, runner=None) -> dict:
    if not 1 <= limit <= MAX_CANDIDATES or not 6 <= clip_seconds <= 12:
        raise ValueError("每次最多 6 个候选，每个片段 6–12 秒")
    root = root.resolve()
    report_path = root / "report.json"
    report = json.loads(report_path.read_text("utf-8")) if report_path.exists() else {"attempts": []}
    attempted = {row["source_url"] for row in report["attempts"]}
    selected, seen = [], set()
    for candidate in candidates:
        try:
            row = candidate_metadata(candidate)
        except ValueError:
            continue
        source = row["source_url"]
        if source in seen or source in attempted or 0 < row["duration"] < 30:
            continue
        seen.add(source)
        selected.append(row)
        if len(selected) == limit:
            break
    report.update(input_count=len(candidates), selection="原始搜索顺序；已知时长至少30秒；未知时长先核查",
                  limits={"candidates_per_run": limit, "clip_seconds": clip_seconds,
                          "media_bytes_per_candidate": MAX_MEDIA_BYTES,
                          "worker_seconds": CANDIDATE_SECONDS, "workers": 1},
                  scene_verified=False, production_index_modified=False)
    run = runner or _bounded_worker
    for candidate in selected:
        report["attempts"].append(dict(candidate, status="started", scene_verified=False))
        _write(report_path, report)
        row = run(candidate, root, clip_seconds)
        report["attempts"][-1] = row
        _write(report_path, report)
    report["decoded_count"] = sum(r.get("status") == "clip_decoded" for r in report["attempts"])
    _write(report_path, report)
    return report


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        task = Path(argv[1])
        args = json.loads(task.read_text("utf-8"))
        if os.name != "nt":
            def cancelled(_sig, _frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, cancelled)
        result = _probe_one(args["candidate"], Path(args["root"]), args["clip_seconds"])
        _write(task.with_suffix(".result.json"), result)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path, help="JSON list or {candidates: [...]} from an existing search")
    source.add_argument("--provider", choices=("tencent", "bilibili"))
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, choices=range(1, MAX_CANDIDATES + 1))
    parser.add_argument("--clip-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    from . import paths
    # CLI execution only; this does not change a running service's data directory.
    paths.DATA = root / "isolated-data"
    if args.candidates:
        payload = json.loads(args.candidates.read_text("utf-8"))
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else payload
    else:
        candidates = search_candidates(args.provider, args.query, root)
    report = probe_candidates(candidates, root, limit=args.limit, clip_seconds=args.clip_seconds)
    print(json.dumps({"report": str(root / "report.json"), "attempted": len(report["attempts"]),
                      "decoded_720p": report["decoded_count"], "scene_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
