"""Bounded inspection of public sources with genuinely reviewed local windows.

An old verdict authorizes only another read. Every new interval still needs
actual sampling, allocation without overlap, and a fresh full-window review.
"""
from __future__ import annotations

import hashlib
import math

from . import library

SOURCE_LIMIT = 4
POINT_LIMIT = 4
PUBLIC_FIELDS = ("title", "provider", "publisher", "publisher_url", "published_at", "author",
                 "license", "license_url", "description", "source_evidence", "source_verified")


def quality_seed_allowed(shot, asset):
    """Require independent pixel-quality evidence before expanding a source."""
    quality = (shot.get("verification") or {}).get("visual_quality") or {}
    hashes = quality.get("frame_sha256")
    return (quality.get("quality_status") == "not_detected"
            and quality.get("representation") in ("live_action", "screen_capture", "recorded_mix")
            and isinstance(hashes, list) and len(hashes) >= 5)


def seed_candidates(project, *, seed_allowed=None, limit=SOURCE_LIMIT):
    """Reopen selected source URLs even when fresh search did not return them.

    ``seed_allowed(shot, asset)`` is the independent pixel-quality integration
    hook. No title/keyword inference about animation or obstruction occurs here.
    Local clip duration, file identity and prior approval never become public
    full-source metadata or approval for a different window.
    """
    assets = {a["id"]: a for a in project.get("assets", [])}
    result = {}
    for shot in project.get("shots", []):
        asset = assets.get(shot.get("selected_asset"), {})
        verdict = shot.get("verification") or {}
        observation = verdict.get("observations") or {}
        preview = shot.get("match_preview") or {}
        if (asset.get("kind") != "video" or verdict.get("verdict") != "match"
                or verdict.get("selection_blocked", True) or verdict.get("needs_source_check", True)
                or verdict.get("frame_count", 0) < 5
                or observation.get("observation_method") != "blind_actual_frames_v1"
                or len(observation.get("frame_observations") or []) < 5
                or preview.get("asset_id") != asset.get("id")
                or min(asset.get("width") or 0, asset.get("height") or 0) < project.get("min_video_height", 720)
                or seed_allowed is not None and not seed_allowed(shot, asset)):
            continue
        try:
            url = library.source_url(asset.get("source_url"))
            start = float(asset.get("source_clip_start", 0)) + float(shot.get("clip_start", 0))
            duration = float(shot.get("duration") or 0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            continue
        if url not in result:
            if len(result) >= min(SOURCE_LIMIT, max(0, limit)):
                continue
            result[url] = {k: asset[k] for k in PUBLIC_FIELDS if k in asset}
            result[url].update(id=asset.get("candidate_id") or "verified-" + hashlib.sha256(url.encode()).hexdigest()[:20],
                               kind="video", source_url=url, url=url, duration=0, width=0, height=0,
                               metadata_checked=False, download_status="unchecked", status="unverified",
                               source_expansion={"seed_windows": [], "observed_context": [], "sampling_only": True})
        seed = result[url]["source_expansion"]
        seed["seed_windows"].append({"shot_id": shot["id"], "asset_id": asset["id"],
                                      "start": round(start, 3), "end": round(start + duration, 3)})
        scene = str(observation.get("scene_description") or "")[:1200]
        if scene and scene not in seed["observed_context"]:
            seed["observed_context"].append(scene)
    return list(result.values())


def topic_record(candidate):
    """Expose previous *observations* only to retrieval opportunity matching."""
    scenes = (candidate.get("source_expansion") or {}).get("observed_context", [])
    return {"title": candidate.get("title") or "",
            "description": (candidate.get("description") or "") + " " + " ".join(scenes)}


def neighbour_windows(candidate, verified, occupied, durations, *, probed=(), limit=POINT_LIMIT):
    """Plan distinct free neighbouring probes, not presumed continuous footage.

    The second pass can look beyond the first pass's four planned positions,
    including when a seed is near the source boundary. At most eight steps
    are considered per side; at most four actual probes are returned. Failed
    probes are not repeated and the per-source time/byte budget is unchanged.
    """
    length = float(candidate.get("duration") or 0)
    if not math.isfinite(length) or length <= 0 or limit <= 0:
        return []
    sizes = sorted({float(d) for d in durations if math.isfinite(float(d)) and 0 < float(d) <= length}, reverse=True)
    result = []
    previous = list(probed)
    blocked = list(occupied) + previous
    steps = POINT_LIMIT + min(POINT_LIMIT, len(previous))
    for duration in sizes:
        size = math.ceil(duration * 25) / 25 + .12
        for step in range(steps):
            for begin, end in verified:
                for start in (end + .2 + step * (size + .2), begin - .2 - (step + 1) * size - step * .2):
                    stop = start + size
                    if (start < 0 or stop > length - .04
                            or any(start < b and stop > a for a, b in blocked)):
                        continue
                    result.append({"at": round(start + duration / 2, 3), "start": round(start, 3),
                                   "end": round(stop, 3), "duration": duration})
                    blocked.append((start, stop))
                    if len(result) >= min(POINT_LIMIT, limit):
                        return result
    return result
