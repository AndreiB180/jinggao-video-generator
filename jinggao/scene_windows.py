"""Keep a sampled source location inside one locally decoded scene.

This changes retrieval windows only. A cut boundary says nothing about content,
identity or quality; every chosen window still needs the ordinary full review.
"""

from __future__ import annotations

import hashlib
import json
import math

from . import execution, frame_sampling, library, media, paths

VERSION = "scene_anchor_v1"
MAX_PADDING = 8.0
DECODE_SECONDS = 8
FPS = frame_sampling.FPS


def _anchored(row):
    proposal = row.get("discovery") or {}
    anchor = proposal.get("sample_at")
    return (
        bool(row["candidate"].get("source_url"))
        and proposal.get("source_url") == row["candidate"]["source_url"]
        and type(anchor) in (int, float)
        and math.isfinite(anchor)
        and row["start"] <= anchor < row["start"] + row["shot"]["duration"]
    )


def download_bounds(block):
    """Read a small neighbourhood once, within the source's actual duration."""
    start, end = block["start"], block["end"]
    source = block["candidate"]
    if source.get("asset_id") or not any(_anchored(r) for r in block["items"]):
        return start, end
    length = float(source.get("duration") or 0)
    if not math.isfinite(length) or length < end:
        return start, end
    padding = min(MAX_PADDING, max(r["size"] for r in block["items"]))
    return round(max(0, start - padding), 3), round(min(length, end + padding), 3)


def _scenes(project, asset, begin, end, check):
    file = paths.file_path(project["id"], asset["path"])
    digest = library.digest_file(file)
    base = float(asset.get("source_clip_start", 0))
    begin, end = max(begin, base), min(end, base + float(asset["duration"]))
    duration = end - begin
    if not 0 < duration <= 150:
        raise ValueError("候选窗口的本地转场检测范围无效")
    # Ignore a sub-frame tail; do not ask the decoder to invent its last frame.
    count = math.floor(duration * FPS + 1e-7)
    if count < 1:
        raise ValueError("候选窗口不足一帧")
    duration = count / FPS
    end = begin + duration
    binding = [VERSION, digest, begin - base, duration, frame_sampling.policy_hash()]
    key = hashlib.sha256(json.dumps(binding).encode()).hexdigest()
    receipt = paths.DATA / "cache" / "scene-windows" / (key + ".json")
    with execution.keyed_lock(("scene-windows", key), check=check):
        check()
        try:
            saved = json.loads(receipt.read_text())
            starts = saved["starts"]
            if (
                saved["binding"] != binding
                or not isinstance(starts, list)
                or not starts
                or starts[0] != 0
                or any(type(n) is not int or not 0 <= n < count for n in starts)
                or starts != sorted(set(starts))
            ):
                raise ValueError("Invalid scene cache")
        except (OSError, ValueError, KeyError, TypeError):
            output = media.run(
                [
                    paths.binary("ffmpeg"),
                    "-hide_banner",
                    "-v",
                    "info",
                    "-nostdin",
                    "-xerror",
                    "-threads",
                    "1",
                    "-ss",
                    str(begin - base),
                    "-t",
                    str(duration),
                    "-i",
                    str(file),
                    "-an",
                    "-vf",
                    f"setpts=PTS-STARTPTS,fps={FPS},trim=end_frame={count},scale={frame_sampling.ANALYSIS_WIDTH}:-2,"
                    "select='gte(scene,0)',metadata=print",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=DECODE_SECONDS,
                check=check,
            )
            starts = frame_sampling.parse_scores(output, count)
            if library.digest_file(file) != digest:
                raise ValueError("转场检测期间实际视频发生变化") from None
            receipt.parent.mkdir(parents=True, exist_ok=True)
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps({"binding": binding, "starts": starts}))
            temporary.replace(receipt)
        check()
        if library.digest_file(file) != digest:
            raise ValueError("转场检测期间实际视频发生变化")
    # Leave two frames around internal hard cuts to avoid transition residue.
    return [
        (
            begin + (left + (2 if left else 0)) / FPS,
            min(end, begin + (right - (2 if right < count else 0)) / FPS),
        )
        for left, right in zip(starts, [*starts[1:], count], strict=True)
    ]


def _gaps(left, right, occupied):
    for start, stop in sorted(occupied):
        if stop <= left:
            continue
        if start >= right:
            break
        if start > left:
            yield left, min(start, right)
        left = max(left, stop)
    if left < right:
        yield left, right


def adjust(block, asset, scenes, used):
    """Move reservations atomically; retain the anchor and avoid every neighbour."""
    rows, changes = [], []
    base = float(asset.get("source_clip_start", 0))
    for row in block["items"]:
        proposal = row.get("discovery") or {}
        anchor = proposal.get("sample_at")
        source = row["candidate"].get("source_url") or row["candidate"]["id"]
        if not _anchored(row):
            rows.append(row)
            continue
        occupied = list(used[source])
        own = next(
            (
                pair
                for pair in occupied
                if abs(pair[0] - row["start"]) < 0.002 and abs(pair[1] - row["start"] - row["size"]) < 0.002
            ),
            None,
        )
        if own is None:
            rows.append(row)
            continue
        occupied.remove(own)
        size, duration = row["size"], row["shot"]["duration"]
        choices = []
        for left, right in scenes:
            if not left <= anchor < right:
                continue
            for a, b in _gaps(left, right, occupied):
                # Grid coordinates are relative to the actual downloaded file.
                lo = math.ceil((max(a, anchor - duration + 1 / FPS) - base) * FPS - 1e-7)
                hi = math.floor((min(b - size, anchor) - base) * FPS + 1e-7)
                if lo <= hi:
                    index = min(hi, max(lo, round((row["start"] - base) * FPS)))
                    at = round(base + index / FPS, 3)
                    if (
                        at >= a
                        and at + size <= b
                        and at <= anchor < at + duration
                        and not any(at < y and at + size > x for x, y in occupied)
                    ):
                        choices.append((abs(at - row["start"]), at, left, right))
        if not choices:
            # A short scene is not stretched or automatically declared usable.
            # Keep the original candidate for the ordinary content review.
            rows.append(row)
            continue
        _, at, left, right = min(choices)
        if abs(at - row["start"]) <= 0.002:
            rows.append(row)
            continue
        used[source].remove(own)
        used[source].append((at, at + size))
        hint = {
            "method": VERSION,
            "shot_id": row["shot"]["id"],
            "sample_at": anchor,
            "original_start": row["start"],
            "start": at,
            "duration": duration,
            "scene_start": round(left, 3),
            "scene_end": round(right, 3),
            "content_approved": False,
        }
        rows.append(row | {"start": at, "scene_window_hint": hint})
        changes.append(hint)
    return rows, changes


def align(project, block, asset, used, check):
    if block["candidate"].get("asset_id") or not any(_anchored(r) for r in block["items"]):
        return block["items"], {"adjusted": [], "error": ""}
    begin, end = download_bounds(block)
    try:
        scenes = _scenes(project, asset, begin, end, check)
        rows, changes = adjust(block, asset, scenes, used)
        return rows, {"adjusted": changes, "error": "", "scene_count": len(scenes)}
    except (OSError, ValueError) as exc:
        check()
        return block["items"], {"adjusted": [], "error": str(exc)}
