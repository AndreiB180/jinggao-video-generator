"""Recover observed but undownloaded source positions after an interrupted run.

These are retrieval hints only: native source metadata, fresh decoded samples
and full-window review are still required. No verdict is copied from a log.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import domestic, paths, segment_discovery


def candidates(project, shots, check, *, limit=3):
    pending = {s["id"]: s for s in shots}
    result = {sid: {} for sid in pending}
    root = paths.DATA / "cache" / "scene-ranks"
    frames_root = (paths.DATA / "cache" / "scene-samples").resolve()
    # Bound legacy-log recovery. Newly decoded windows are available through
    # the ordinary clip library; do not keep reviewing those failed files here.
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime)[-160:]
    for path in files:
        check()
        try:
            if path.stat().st_size > 2_000_000:
                continue
            saved = json.loads(path.read_text("utf-8"))
            if saved.get("status") != "complete" or not saved.get("frame_observations"):
                continue
            request = saved["request"]
            if request.get("schema_version") != segment_discovery.RANK_VERSION:
                continue
            records = {s["id"]: s for s in request["shots"] if s.get("id") in pending}
            for sid, record in records.items():
                shot = pending[sid]
                if record.get("narration") != shot["narration"] or record.get("duration") != shot["duration"]:
                    continue
                for proposal in saved.get("proposals", {}).get(sid, []):
                    url, at, start = proposal["source_url"], float(proposal["sample_at"]), float(proposal["clip_start"])
                    if not domestic.allowed(url) or not all(math.isfinite(n) and n >= 0 for n in (at, start)):
                        continue
                    source = next(c for c in request["candidates"] if c["source_url"] == url)
                    if not start <= at < start+shot["duration"] or at >= float(source["duration"]):
                        continue
                    if any(a.get("kind") == "video" and a.get("source_url") == url
                           and float(a.get("source_clip_start", 0)) <= start+.08
                           and float(a.get("source_clip_start", 0))+float(a.get("duration", 0)) >= start+shot["duration"]-.08
                           for a in project.get("assets", [])):
                        continue
                    frame = next(f for f in request["frames"] if f["source_url"] == url
                                 and f["frame_id"] == proposal["frame_id"] and abs(float(f["at"])-at) < .001)
                    image = Path(frame["path"]).resolve()
                    if (not image.is_relative_to(frames_root) or not image.is_file()
                            or hashlib.sha256(image.read_bytes()).hexdigest() != frame["sha256"]):
                        continue
                    if url not in result[sid] and len(result[sid]) >= limit:
                        continue
                    candidate = result[sid].setdefault(url, {
                        "id": "sampled-"+hashlib.sha256(url.encode()).hexdigest()[:20], "kind": "video",
                        "provider": "web_video", "source_url": url, "url": url, "title": source["title"],
                        "duration": source["duration"], "width": frame["width"], "height": frame["height"],
                        "download_status": "unchecked", "resume_points": [], "sample_windows": [],
                        "source_evidence": source.get("source_evidence", {}), "sampling_only": True})
                    if at not in candidate["resume_points"] and len(candidate["resume_points"]) < 4:
                        candidate["resume_points"].append(at)
                        candidate["sample_windows"].append({"at": at, "start": start, "end": start+shot["duration"]})
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            continue
    return {sid: list(rows.values()) for sid, rows in result.items()}
