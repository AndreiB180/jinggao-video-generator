"""Use an actual source frame within the project's existing still-image budget.

Some relevant local footage is shorter than a narration unit. A source frame
can be used as an explicitly recorded image with motion, never as extra video
seconds. It gets a new pixel/content review and retains its exact source time.
"""
from __future__ import annotations

from collections import defaultdict
import json

from . import curation, editorial_contract, execution, library, library_match, media, paths, store

MAX_SHOTS = 8
ATTEMPTS = 2


def materialize_frame(pid, clip, fraction, check):
    original = library.materialize(pid, clip, check)
    offset = float(clip["offset"]) + float(clip["duration"]) * fraction
    at = float(clip["start"]) + float(clip["duration"]) * fraction
    target = paths.project_dir(pid) / "assets" / (store.uid() + ".jpg")
    media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-y", "-ss", str(offset),
               "-i", str(paths.file_path(pid, original["path"])), "-frames:v", "1", "-q:v", "2", str(target)],
              timeout=20, check=check)
    check()
    asset = media.inspect_asset(pid, target, "资料截帧 · " + original["title"])
    if asset["kind"] != "image" or min(asset.get("width", 0), asset.get("height", 0)) < 720:
        raise ValueError("原视频截帧清晰度不足，未加入图片补充")
    asset.update({key: original[key] for key in library.FIELDS if key in original and key != "title"})
    asset.update(source_clip_start=round(at, 6), derived_from_video={
        "asset_id": original["id"], "source_url": original["source_url"],
        "source_at": round(at, 6), "clip_id": clip["id"], "image_only": True})
    store.mutate(pid, lambda p: p["assets"].append(asset), internal=True)
    return asset


def commit_reviewed(pid, row):
    """Commit a freshly reviewed image under an atomic, current-project budget."""
    shot, asset, result = row["shot"], row["asset"], row["verification"]
    if (asset["kind"] != "image" or result.get("verdict") != "match"
            or result.get("scene_alignment") != "match" or result.get("selection_blocked", True)
            or result.get("needs_source_check", True)):
        return False
    committed = False
    def choose(project):
        nonlocal committed
        current = next(s for s in project["shots"] if s["id"] == shot["id"])
        if (current.get("locked") or project.get("media_policy") == "video_only"
                or curation.image_in_use(project, asset, shot["id"])
                or any(current.get(k) != shot.get(k) for k in ("narration", "duration", "editorial_contract"))):
            return
        if project.get("media_policy") == "video_preferred":
            report = curation.composition(project)
            old = next((a for a in project["assets"] if a["id"] == current.get("selected_asset")), {})
            replaced = current["duration"] if old.get("kind") == "image" else 0
            if report["image_seconds"]-replaced+current["duration"] > report["total_seconds"]*(1-project.get("min_video_ratio", .8))+.05:
                return
        current.update(selected_asset=asset["id"], clip_start=0, fit=shot.get("fit", "contain"), approved=False,
            verification=result, match_result=result, source_review_note="", source_review_url="",
            search_note="已复核原视频截帧，按图片计入成片并添加运镜；"+result["reason"],
            match_preview=row["preview"] | {"asset_id": asset["id"], "clip_start": 0, "duration": current["duration"]})
        current.update(curation.reviewed_label_fields(result))
        committed = True
    store.mutate(pid, choose, internal=True)
    return committed


def match(project, shots, check, *, review_batch, commit=None, workers=4):
    from .pooled_match import intervals, relevance
    report = curation.composition(project)
    budget = (report["total_seconds"] * (1-project.get("min_video_ratio", .8)) - report["image_seconds"]
              if project.get("media_policy") == "video_preferred" else report["missing_seconds"])
    stats = {"model_requests": 0, "images_selected": 0, "image_seconds": 0, "failures": []}
    if commit is None:
        def commit(row):
            return commit_reviewed(project["id"], row)
    if project.get("media_policy") == "video_only" or budget < 1:
        return set(), stats
    pending = sorted((shot for shot in shots if not curation.needs_identity(shot)
                      and not shot.get("locked") and 0 < shot["duration"] <= budget+.05),
                     key=lambda s: (s.get("preferred_media") != "image", s["duration"]))[:MAX_SHOTS]
    options = {}
    for shot in pending:
        check()
        fields = editorial_contract.retrieval_fields(shot)
        query = " ".join(fields["query_hints"]) if fields else shot.get("query", "")
        options[shot["id"]] = [row for row in library.search(query, limit=100, min_duration=2, min_height=720)
                                if not relevance(shot, row["source"])[0]]
    pending, options = library_match.quick_candidates(project, pending, options, stats)
    def started():
        stats["model_requests"] += 1
    library_match.rank_windows(pending, options, check, diagnostics=stats["failures"],
                              request_started=started, workers=min(2, workers))
    used = intervals(project)
    for shot in project["shots"]:
        asset = next((a for a in project["assets"] if a["id"] == shot.get("selected_asset")), {})
        origin = asset.get("derived_from_video") or {}
        if origin.get("source_url") and isinstance(origin.get("source_at"), (int, float)):
            at = origin["source_at"]
            used[origin["source_url"]].append((at-1.5, at+1.5))
    attempted, passed = defaultdict(set), set()
    for _ in range(ATTEMPTS):
        assignments = []
        for shot in pending:
            if shot["id"] in passed or shot["duration"] > budget+.05:
                continue
            for clip in options[shot["id"]]:
                if clip["id"] in attempted[shot["id"]]:
                    continue
                attempted[shot["id"]].add(clip["id"])
                key = clip["source"]["source_url"]
                fraction = next((f for f in (.5, .25, .75) if not any(
                    begin-1.5 <= clip["start"] + clip["duration"]*f <= end+1.5 for begin, end in used[key])), None)
                if fraction is None:
                    continue
                at = clip["start"] + clip["duration"]*fraction
                used[key].append((at, at))
                assignments.append((shot, clip, fraction))
                break
        if not assignments:
            break
        def prepare(item):
            shot, clip, fraction = item
            check()
            try:
                asset = materialize_frame(project["id"], clip, fraction, check)
                return {"shot": shot | {"clip_start": 0}, "asset": asset,
                        "candidate": {"id": "frame-" + asset["id"], "kind": "image",
                                      "source_url": asset["source_url"], "title": asset["title"]}}, None
            except (ValueError, OSError) as exc:
                return None, str(exc)
        ready = []
        for _, (row, error) in execution.completed_map(prepare, assignments, workers):
            check()
            if error:
                stats["failures"].append(error)
            else:
                ready.append(row)
        def review(items):
            try:
                return review_batch(project, items, check)
            except ValueError as exc:
                return [], [(row, str(exc)) for row in items], {}
        for _, (rows, errors, info) in execution.completed_map(review, [ready[i:i+4] for i in range(0, len(ready), 4)], workers):
            check()
            stats["model_requests"] += info.get("model_requests", 0)
            stats["failures"].extend(message for _, message in errors)
            folder = paths.DATA/"cache"/"frame-reviews"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder/(store.uid()+".json")
            target.write_text(json.dumps({"project_id": project["id"], "reviews": [
                {"shot_id": row["shot"]["id"], "asset_id": row["asset"]["id"],
                 "verification": row["verification"], "preview": row.get("preview")} for row in rows]},
                ensure_ascii=False, indent=2), "utf-8")
            stats.setdefault("review_summaries", []).append(str(target))
            for row in rows:
                result = row["verification"]
                duration = row["shot"]["duration"]
                if (duration <= budget+.05 and result.get("verdict") == "match"
                        and result.get("scene_alignment") == "match" and not result.get("selection_blocked", True)
                        and not result.get("needs_source_check", True) and commit(row)):
                    budget -= duration
                    passed.add(row["shot"]["id"])
                    stats["images_selected"] += 1
                    stats["image_seconds"] += duration
    return passed, stats
