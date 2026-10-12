"""Index project video bytes once for retrieval; never transfer an approval.

Uploaded files can serve several script segments. Downloaded blocks already
enter the shared library through their ordinary review. Sparse observations
locate opportunities only. The selected full
window still goes through the ordinary content, privacy and quality review.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

from . import execution, library, media, neutral_observer, paths, sources, store

VERSION = "project_video_pool_v1"
MAX_WINDOWS = 32
WINDOW_SECONDS = 16
FPS = 25
SOURCE_FIELDS = tuple(dict.fromkeys((*library.FIELDS, "candidate_id", "source_clip_start")))


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _source(asset):
    return {k: asset[k] for k in SOURCE_FIELDS if k in asset} | {
        "source_url": asset.get("source_url", ""), "title": asset.get("title", "导入视频")}


def source_key(clip):
    return clip["source"].get("source_url") or clip["source"].get("candidate_id") or clip["project_asset_id"]


def ranges(duration):
    if not math.isfinite(duration) or duration < 2:
        return []
    # A short tail is included in the preceding window, not silently discarded.
    count = max(1, math.floor(duration / WINDOW_SECONDS))
    size = duration / count
    return [(round(i * size, 6), round((i + 1) * size - i * size, 6)) for i in range(count)]


def _tasks(project, check):
    eligible, seen = [], set()
    for asset in project.get("assets", []):
        check()
        if (asset.get("kind") != "video" or asset.get("source_parts")
                or asset.get("privacy_redaction") or asset.get("library_blob")
                or asset.get("download_mode") or asset.get("candidate_id")
                or min(asset.get("width", 0), asset.get("height", 0)) < project.get("min_video_height", 720)):
            continue
        file = paths.file_path(project["id"], asset["path"])
        if not file.is_file():
            continue
        digest = library.digest_file(file)
        check()
        key = (digest, _digest(_source(asset)))
        if key in seen:
            continue
        seen.add(key)
        windows = ranges(float(asset.get("duration", 0)))
        if windows:
            eligible.append((asset, file, digest, windows))
    if not eligible:
        return [], 0
    total = sum(len(row[3]) for row in eligible)
    # Share the finite indexing allowance; never let a long first upload hide
    # every other source. Report omitted windows rather than calling this full.
    quotas = [0] * len(eligible)
    for _ in range(min(total, MAX_WINDOWS)):
        choices = [i for i, row in enumerate(eligible) if quotas[i] < len(row[3])]
        index = min(choices, key=lambda i: (quotas[i], i))
        quotas[index] += 1
    result = []
    for row, quota in zip(eligible, quotas, strict=True):
        if not quota:
            continue
        asset, file, digest, windows = row
        indices = [min(len(windows)-1, int((i+.5)*len(windows)/quota)) for i in range(quota)]
        result.append((asset, file, digest, [windows[i] for i in indices]))
    return result, max(0, total-MAX_WINDOWS)


def _frames(file, digest, windows, check):
    indices = sorted({int((start + duration*f) * FPS) for start, duration in windows for f in (.25, .75)})
    return _frames_at_indices(file, digest, indices, check)


def _frames_at_indices(file, digest, indices, check):
    if (not indices or indices != sorted(set(indices))
            or any(type(n) is not int or n < 0 for n in indices)):
        raise ValueError("项目视频采样位置无效")
    key = _digest([VERSION, digest, indices])
    folder = paths.DATA / "cache" / "project-pool-frames" / key
    receipt = folder / "frames.json"
    with execution.keyed_lock(("project-pool-frames", key), check=check):
        check()
        files = [folder / f"{i:04d}.jpg" for i in range(1, len(indices)+1)]
        try:
            saved = json.loads(receipt.read_text("utf-8"))
            if saved == {"version": VERSION, "media_sha256": digest, "indices": indices,
                         "frame_sha256": [library.digest_file(f) for f in files]}:
                return dict(zip(indices, files, strict=True)), True
        except (OSError, ValueError, KeyError, TypeError):
            pass
        folder.mkdir(parents=True, exist_ok=True)
        receipt.unlink(missing_ok=True)
        for frame in files:
            frame.unlink(missing_ok=True)
        scale = "scale=1280:1280:force_original_aspect_ratio=decrease:force_divisible_by=2"
        if max(indices)/FPS > 600:
            # A long source must not be decoded from the beginning simply to
            # inspect a few distant frames. Bound both seeks and concurrency.
            def seek(pair):
                index, target = pair
                check()
                media.run([paths.binary("ffmpeg"), "-v", "error", "-nostdin", "-xerror", "-y",
                    "-threads", "1", "-ss", str(index/FPS), "-i", str(file), "-an",
                    "-vf", scale, "-frames:v", "1", "-q:v", "3", str(target)], timeout=20, check=check)
            for _ in execution.completed_map(seek, list(zip(indices, files, strict=True)), 2):
                check()
        else:
            select = "+".join(f"eq(n\\,{i})" for i in indices)
            media.run([paths.binary("ffmpeg"), "-v", "error", "-nostdin", "-xerror", "-y",
                "-threads", "1", "-i", str(file), "-an", "-vf",
                f"fps={FPS},select='{select}',{scale}", "-fps_mode", "vfr",
                "-frames:v", str(len(indices)), "-q:v", "3", str(folder / "%04d.jpg")],
                timeout=45, check=check)
        check()
        if library.digest_file(file) != digest or any(not f.is_file() for f in files):
            raise ValueError("项目视频已变更或索引抽帧不完整")
        body = {"version": VERSION, "media_sha256": digest, "indices": indices,
                "frame_sha256": [library.digest_file(f) for f in files]}
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(body), "utf-8")
        temporary.replace(receipt)
        return dict(zip(indices, files, strict=True)), False


def collect(project, check, progress, workers=2):
    began = time.monotonic()
    stats = {"sources": 0, "windows": 0, "model_requests": 0, "frame_cache_hits": 0,
             "failures": [], "index_only": True}
    tasks, deferred = _tasks(project, check)
    stats["deferred_windows"] = deferred
    records, clips = [], {}
    for asset, file, digest, windows in tasks:
        check()
        try:
            frames, cached = _frames(file, digest, windows, check)
        except (ValueError, OSError) as exc:
            stats["failures"].append(str(exc))
            continue
        stats["sources"] += 1
        stats["frame_cache_hits"] += int(cached)
        source = _source(asset)
        base = float(asset.get("source_clip_start") or 0)
        for offset, duration in windows:
            cid = _digest([VERSION, project["id"], asset["id"], digest, offset, duration, source])[:32]
            indices = sorted({int((offset + duration*f)*FPS) for f in (.25, .75)})
            records.append({"shot": {"id": cid}, "frame_paths": [frames[i] for i in indices]})
            clips[cid] = {"id": cid, "project_id": project["id"], "project_asset_id": asset["id"],
                "source": source, "source_id": (hashlib.sha256(source["source_url"].encode()).hexdigest()[:32]
                    if source["source_url"] else _digest([project["id"], asset["id"]])[:32]),
                "blob_id": digest, "path": asset["path"], "offset": offset,
                "start": round(base+offset, 6), "duration": duration,
                "width": asset["width"], "height": asset["height"], "bytes": file.stat().st_size,
                "index_frame_times": [i/FPS for i in indices], "updated": time.time()}
    observations = {}
    def observe(batch):
        try:
            return neutral_observer.observe(batch, check), ""
        except ValueError as exc:
            check()  # Cancellation and the project budget still stop the job.
            return ({}, getattr(exc, "request_count", 0)), str(exc)
    batches = [records[i:i+4] for i in range(0, len(records), 4)]
    for i, (_, ((result, requests), error)) in enumerate(execution.completed_map(observe, batches, min(2, workers))):
        check()
        observations.update(result)
        stats["model_requests"] += requests
        if error:
            stats["failures"].append(error)
        progress(i+1, len(batches))
    ready = []
    for cid, observation in observations.items():
        clip = clips[cid]
        # This current-call blind observation is the only route into the
        # ranking shortlist. Titles and user-entered descriptions cannot mark it.
        clip.update(description=observation["scene_description"],
            visible_text="\n".join(dict.fromkeys(t for f in observation["frames"] for t in f["text"])),
            frame_observations=observation["frames"],
            observation_method="blind_actual_frames_v1")
        ready.append(clip)
    stats.update(windows=len(ready), elapsed_seconds=round(time.monotonic()-began, 3))
    return ready, stats


def resolve(clip, check=lambda: None, *, digests=None):
    """Resolve a code-owned clip against current project bytes and attribution."""
    check()
    project = store.get_project(clip["project_id"])
    asset = next((a for a in project["assets"] if a["id"] == clip["project_asset_id"]), None)
    if (not asset or asset.get("kind") != "video" or _source(asset) != clip["source"]
            or asset["path"] != clip["path"]):
        raise ValueError("项目素材或来源标注已改变，请重新建立索引")
    start, duration = float(clip["offset"]), float(clip["duration"])
    if (not all(math.isfinite(n) for n in (start, duration)) or start < 0 or duration <= 0
            or start+duration > float(asset["duration"])+.001
            or not math.isfinite(float(clip["start"]))
            or abs(clip["start"]-start-float(asset.get("source_clip_start") or 0)) > .001):
        raise ValueError("项目素材字节或索引时间已改变")
    file = paths.file_path(project["id"], asset["path"])
    def fingerprint():
        stat = file.stat()
        return (file, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = fingerprint()
    # Shared only within one ranking snapshot, not between jobs or reviews.
    # Reusing a video for several windows must not hash the whole file per row.
    digests = {} if digests is None else digests
    if key not in digests:
        digests[key] = library.digest_file(file)
    check()
    if key != fingerprint() or digests[key] != clip["blob_id"]:
        raise ValueError("项目素材字节或索引时间已改变")
    return asset


def search(clips, query, minimum):
    terms = sources.tokens(query)
    result = []
    for clip in clips:
        if clip["duration"]+.02 < minimum:
            continue
        seen = set()
        for frame, timestamp in zip(clip["frame_observations"], clip["index_frame_times"], strict=True):
            description = frame["objects"] + "；" + frame["action"]
            visible_text = "\n".join(frame["text"])
            observed = library.observed_terms(description, visible_text)
            score = 2 * len(terms & observed)
            # Center an opportunity on the actual observed frame. Always
            # testing the beginning of a 16s block would miss later actions.
            offset = round(max(clip["offset"], min(timestamp-minimum/2,
                clip["offset"]+clip["duration"]-minimum)), 6)
            if score and offset not in seen:
                seen.add(offset)
                result.append(clip | {"id": _digest([clip["id"], offset, minimum])[:32],
                    "score": score, "offset": offset, "duration": minimum,
                    "start": round(clip["start"]+offset-clip["offset"], 6),
                    "description": description, "visible_text": visible_text})
    return sorted(result, key=lambda c: (-c["score"], c["start"]))
