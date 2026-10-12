"""Locate unused scenes in downloaded bytes before another remote search.

This pool owns neither approvals nor new downloads. Source-time reservations
are shared across overlapping files; each allocated view receives the normal
full content, quality and privacy review.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
import math
import time

from . import execution, library, paths, project_pool, scene_windows, segment_discovery, source_expansion

VERSION = "downloaded_pool_v1"
MAX_ASSETS = 12
MAX_WINDOWS = 6  # At most 12 location frames: one existing observation batch.
WINDOWS_PER_SHOT = 2
FPS = scene_windows.FPS


def _gaps(left, right, occupied):
    for begin, end in sorted(occupied):
        if end <= left or begin >= right:
            continue
        if begin > left:
            yield left, min(begin, right)
        left = max(left, end)
    if left < right:
        yield left, right


def _candidate(asset):
    return asset | {"id": asset.get("candidate_id") or asset["id"], "asset_id": asset["id"]}


def _available(shot, asset, begin, end, tried):
    base = float(asset.get("source_clip_start", 0))
    size = math.ceil(shot["duration"]*FPS)/FPS+.12
    occupied = [(a, b) for url, a, b in tried.get(shot["id"], []) if url == asset["source_url"]]
    return any(math.ceil((a-base)*FPS-1e-7)/FPS+base+size <= b+.000001
               for a, b in _gaps(begin, end, occupied))


def _placement(shot, asset, begin, end, anchor, desired, occupied, tried):
    base = float(asset.get("source_clip_start", 0))
    size = math.ceil(shot["duration"]*FPS)/FPS+.12
    blocked = [*occupied, *((a, b) for url, a, b in tried.get(shot["id"], []) if url == asset["source_url"])]
    choices = []
    for left, right in _gaps(begin, end, blocked):
        lo = math.ceil((max(left, anchor-shot["duration"]+1/FPS)-base)*FPS-1e-7)
        hi = math.floor((min(right-size, anchor)-base)*FPS+1e-7)
        if lo > hi:
            continue
        frame = min(hi, max(lo, round((desired-base)*FPS)))
        start = round(base+frame/FPS, 6)
        if start >= left-.000001 and start+size <= right+.000001 and start <= anchor < start+shot["duration"]:
            choices.append(start)
    return min(choices, key=lambda at: abs(at-desired)) if choices else None


def _anchors(asset, targets, begin, end, tried):
    """Pick actual grid points which can each anchor an untried full window."""
    base = float(asset.get("source_clip_start", 0))
    preferred = [(begin+(end-begin)*f-base)*FPS for f in (.25, .75)]
    candidates = {int(n) for n in preferred}
    for shot in targets:
        blocked = [(a, b) for url, a, b in tried.get(shot["id"], []) if url == asset["source_url"]]
        candidates.update(int(((a+b)/2-base)*FPS) for a, b in _gaps(begin, end, blocked)
                          if _available(shot, asset, a, b, tried))
    coverage = {n: {s["id"] for s in targets
                   if _placement(s, asset, begin, end, base+n/FPS, base+n/FPS-s["duration"]/2, [], tried) is not None}
                for n in candidates if begin <= base+n/FPS < end}
    coverage = {n: ids for n, ids in coverage.items() if ids}
    picked, covered = [], set()
    for preferred_index in preferred:
        if not coverage:
            break
        index = min(coverage, key=lambda n: (-len(coverage[n]-covered), abs(n-preferred_index), n))
        picked.append(index)
        covered.update(coverage.pop(index))
    return sorted(picked), covered


def plan(project, shots, used, check, *, allowed=lambda s, c: True, tried=None, workers=2):
    """No network/model calls: find distinct, in-file, continuous free spans."""
    stats = {"assets": 0, "windows": 0, "frames": 0, "model_requests": 0,
             "failures": [], "sampling_only": True}
    tried = tried or {}
    successful = defaultdict(int)
    assets = {a["id"]: a for a in project["assets"]}
    for shot in project["shots"]:
        asset = assets.get(shot.get("selected_asset"))
        verdict = shot.get("verification") or {}
        if (asset and verdict.get("verdict") == "match" and not verdict.get("selection_blocked", True)
                and not verdict.get("needs_source_check", True) and source_expansion.quality_seed_allowed(shot, asset)):
            successful[asset.get("source_url")] += 1
    def source_order(item):
        return -successful[item[0]], item[0]
    by_source = defaultdict(list)
    for asset in project["assets"]:
        check()
        if (asset.get("kind") != "video" or not asset.get("source_url")
                or not (asset.get("download_mode") or asset.get("candidate_id"))
                or any(asset.get(k) for k in ("source_parts", "privacy_redaction", "library_blob"))
                or min(asset.get("width", 0), asset.get("height", 0)) < max(720, project.get("min_video_height", 720))):
            continue
        candidate = _candidate(asset)
        targets = [s for s in shots if allowed(s, candidate)]
        if not targets:
            continue
        base, duration = float(asset.get("source_clip_start", 0)), float(asset.get("duration", 0))
        minimum = min(math.ceil(s["duration"]*FPS)/FPS+.12 for s in targets)
        if (not all(math.isfinite(v) for v in (base, duration)) or base < 0 or not minimum <= duration <= 150
                or not any(_available(s, asset, a, b, tried) for s in targets
                           for a, b in _gaps(base, base+duration, used.get(asset["source_url"], [])))):
            continue
        file = paths.file_path(project["id"], asset["path"])
        if file.is_file():
            by_source[asset["source_url"]].append((asset, targets, minimum))
    # Round-robin sources, longest files first. Repeated overlapping downloads
    # cannot exhaust the finite pool before another source has been examined.
    def capacity(row):
        asset, targets, _ = row
        begin = float(asset.get("source_clip_start", 0))
        return max((b-a for a, b in _gaps(begin, begin+asset["duration"], used.get(asset["source_url"], []))
                    if any(_available(s, asset, a, b, tried) for s in targets)), default=0)
    queues = [deque(sorted(rows, key=lambda r: (-capacity(r), r[0]["id"])))
              for _, rows in sorted(by_source.items(), key=source_order)]
    tasks = []
    while any(queues) and len(tasks) < MAX_ASSETS:
        for queue in queues:
            if queue and len(tasks) < MAX_ASSETS:
                tasks.append(queue.popleft())

    def scenes(task):
        asset, _, _ = task
        file = paths.file_path(project["id"], asset["path"])
        try:
            digest = library.digest_file(file)
            base = float(asset.get("source_clip_start", 0))
            spans = scene_windows._scenes(project, asset, base, base+asset["duration"], check)
            if library.digest_file(file) != digest:
                raise ValueError("本地素材在建立区间池时改变")
            return digest, spans, ""
        except (OSError, ValueError) as exc:
            check()
            return "", [], str(exc)

    results = {}
    for task, result in execution.completed_map(scenes, tasks, min(2, workers)):
        results[task[0]["id"]] = result
    occupied = defaultdict(list, {key: list(spans) for key, spans in used.items()})
    windows, opportunities, opportunities_per_shot = [], defaultdict(list), defaultdict(int)
    for asset, targets, minimum in tasks:
        digest, spans, error = results[asset["id"]]
        if error:
            stats["failures"].append(error)
            continue
        stats["assets"] += 1
        base = float(asset.get("source_clip_start", 0))
        source = asset["source_url"]
        for left, right in spans:
            for begin, end in _gaps(left, right, used.get(source, [])):
                opportunities[source].append((asset, targets, minimum, digest, begin, end))
    # Take the most broadly usable span of each source before its second span.
    queues = [deque(sorted(rows, key=lambda r: (-sum(_available(s, r[0], r[4], r[5], tried)
                    for s in r[1]), r[4]-r[5], r[0]["id"])))
              for _, rows in sorted(opportunities.items(), key=source_order)]
    while any(queues) and len(windows) < MAX_WINDOWS:
        for queue in queues:
            if not queue or len(windows) >= MAX_WINDOWS:
                continue
            asset, targets, minimum, digest, left, right = queue.popleft()
            base, source = float(asset.get("source_clip_start", 0)), asset["source_url"]
            for begin, end in list(_gaps(left, right, occupied[source])):
                # Align to the decoded file's 25 fps grid; never round a start
                # backwards into an occupied interval or beyond a scene cut.
                offset = math.ceil((begin-base)*FPS-1e-7)/FPS
                duration = math.floor((end-base-offset)*FPS+1e-7)/FPS
                if duration+.000001 < minimum or len(windows) >= MAX_WINDOWS:
                    continue
                begin, end = base+offset, base+offset+duration
                shot_ids = [s["id"] for s in targets if _available(s, asset, begin, end, tried)]
                # A lone short shot must not buy a dozen tiny alternatives.
                # For broader windows, relevance is still unknown before
                # observation; do not let the first two unrelated sources
                # remove every other shot's opportunity.
                if len(shot_ids) == 1 and opportunities_per_shot[shot_ids[0]] >= WINDOWS_PER_SHOT:
                    continue
                if not shot_ids:
                    continue
                indices, covered = _anchors(asset, [s for s in targets if s["id"] in shot_ids], begin, end, tried)
                shot_ids = [sid for sid in shot_ids if sid in covered]
                if not shot_ids:
                    continue
                identity = [VERSION, project["id"], asset["id"], digest, offset, duration, project_pool._source(asset)]
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
                clip = {"id": key, "project_id": project["id"], "project_asset_id": asset["id"],
                        "source": project_pool._source(asset), "path": asset["path"], "blob_id": digest,
                        "offset": offset, "start": begin, "duration": duration}
                windows.append({"clip": clip, "asset": asset, "shot_ids": shot_ids, "indices": indices})
                for sid in shot_ids:
                    opportunities_per_shot[sid] += 1
                occupied[source].append((begin, end))
    stats.update(windows=len(windows), free_seconds=round(sum(w["clip"]["duration"] for w in windows), 3))
    return windows, stats


def samples(windows, check, *, workers=2):
    """Two location frames per free window, decoded once per existing file."""
    tasks = defaultdict(list)
    for window in windows:
        tasks[window["asset"]["id"]].append(window)

    def extract(rows):
        clip = rows[0]["clip"]
        file = paths.file_path(clip["project_id"], clip["path"])
        # A frame-cache hit must not conceal changed video bytes or attribution.
        digests = {}
        for window in rows:
            project_pool.resolve(window["clip"], check, digests=digests)
        indices = sorted({n for window in rows for n in window["indices"]})
        frames, _ = project_pool._frames_at_indices(file, clip["blob_id"], indices, check)
        result = []
        for window in rows:
            current, asset = window["clip"], window["asset"]
            indices = window["indices"]
            result.append({"candidate_id": current["id"], "source_url": asset["source_url"],
                "title": asset.get("title", ""), "source_evidence": asset.get("source_evidence", {}),
                "duration": current["duration"], "sampling_only": True,
                "frames": [{"id": f"f{n}", "at": round(index/FPS-current["offset"], 6),
                            "path": str(frames[index])} for n, index in enumerate(indices)]})
        return result

    def safe_extract(rows):
        try:
            return extract(rows), ""
        except (OSError, ValueError) as exc:
            check()
            return [], str(exc)

    prepared, failures = [], []
    for _, (rows, error) in execution.completed_map(safe_extract, tasks.values(), min(2, workers)):
        check()
        prepared.extend(rows)
        if error:
            failures.append(error)
    return prepared, failures


def allocate(project, shots, windows, proposals, used, check, *, allowed=lambda s, c: True, tried=None):
    """Translate window-local sample times to original-source reservations."""
    by_id = {w["clip"]["id"]: w for w in windows}
    ready, digests = [], {}
    tried = tried or {}
    offered = {sid for window in windows for sid in window["shot_ids"]}
    for shot in (s for s in shots if s["id"] in offered):
        for proposal in proposals.get(shot["id"], []):
            window = by_id.get(proposal["candidate_id"])
            if not window or shot["id"] not in window["shot_ids"]:
                continue
            clip = window["clip"]
            asset = project_pool.resolve(clip, check, digests=digests)
            candidate, source = _candidate(asset), asset["source_url"]
            if not allowed(shot, candidate) or proposal.get("source_url") != source:
                continue
            size = math.ceil(shot["duration"]*FPS)/FPS+.12
            anchor = clip["start"]+proposal["sample_at"]
            if not clip["start"] <= anchor < clip["start"]+clip["duration"]:
                continue
            base = float(asset.get("source_clip_start", 0))
            if not any(abs(base+n/FPS-anchor) <= .000002 for n in window["indices"]):
                continue
            start = _placement(shot, asset, clip["start"], clip["start"]+clip["duration"], anchor,
                               clip["start"]+proposal["clip_start"], used.get(source, []), tried)
            if start is None:
                continue
            used.setdefault(source, []).append((start, start+size))
            ready.append({"candidate": candidate, "asset": asset,
                "shot": shot | {"clip_start": round(start-base, 6)}, "start": start, "size": size,
                "discovery": proposal | {"clip_start": start, "sample_at": anchor},
                "downloaded_pool": {"version": VERSION, "clip": deepcopy(clip), "sampling_only": True}})
            break
    return ready


def propose(project, shots, used, check, progress, *, allowed=lambda s, c: True, tried=None, workers=2):
    """One bounded local rescue; no project writes and no automatic approval."""
    began = time.monotonic()
    windows, stats = plan(project, shots, used, check, allowed=allowed, tried=tried, workers=workers)
    if not windows:
        return [], stats
    progress("检查已经下载的空余片段")
    actual, failures = samples(windows, check, workers=workers)
    stats["failures"].extend(failures)
    stats["frames"] = sum(len(s["frames"]) for s in actual)
    if not actual:
        return [], stats
    observed, info = segment_discovery.observe_samples(actual, check, workers)
    stats["model_requests"] += info["model_requests"]
    stats["failures"].extend(info["failures"])
    bound = segment_discovery.shared_rank_input_bound(actual, observed, check)
    # Preserve the existing cheap-tier text input bound and batch limit.
    batches, batch = [], []
    available_ids = {sample["candidate_id"] for sample in actual}
    offered = {sid for window in windows if window["clip"]["id"] in available_ids for sid in window["shot_ids"]}
    for shot in (s for s in shots if s["id"] in offered):
        if batch and (len(batch) >= 32 or bound([*batch, shot]) > 32000):
            batches.append(batch)
            batch = []
        if bound([shot]) <= 32000:
            batch.append(shot)
    if batch:
        batches.append(batch)
    proposals = {}
    for batch in batches:
        check()
        try:
            ranked = segment_discovery.rank_windows_batch(batch, actual, check,
                prepared_observations=observed, shared_pool=True)
        except ValueError as exc:
            check()
            stats["model_requests"] += getattr(exc, "model_requests", 0)
            stats["failures"].append(str(exc))
            continue
        stats["model_requests"] += ranked.get("model_requests", 0)
        proposals.update(ranked["proposals"])
    ready = allocate(project, shots, windows, proposals, used, check, allowed=allowed, tried=tried)
    stats.update(proposed=len(ready), elapsed_seconds=round(time.monotonic()-began, 3))
    return ready, stats
