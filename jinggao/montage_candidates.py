"""Compose remaining local short clips, reviewing the finished pixels once.

Indexed descriptions rank opportunities only. All parts remain unapproved until
the ordinary full-window content, privacy and quality checks inspect the actual
composite. Failed composites do not become selected or approved shots.
"""

from __future__ import annotations

from collections import defaultdict
import time

from . import (
    curation,
    editorial_contract,
    execution,
    library,
    library_match,
    montage_match,
    plan_reselection,
    store,
)


def eligible(shot):
    return (
        bool(montage_match.piece_lengths(shot["duration"]))
        and not any(shot.get(k) for k in ("locked", "approved", "selected_asset", "location"))
        and not curation.needs_identity(shot)
        and (shot.get("visual_plan") or {}).get("mode") != "explicit_sequence"
    )


def match(project, shots, check, progress, *, review_batch, workers=2):
    from .pooled_match import intervals, relevance

    started_at = time.monotonic()
    stats = {
        "model_requests": 0,
        "selected": 0,
        "failures": [],
        "candidates": 0,
        "piece_review_requests": 0,
        "whole_review_batches": 0,
        "content_approved_by_index": False,
    }
    pending = [s for s in shots if eligible(s)]
    options, lengths = {}, {}
    for shot in pending:
        check()
        fields = editorial_contract.retrieval_fields(shot)
        query = " ".join(fields["query_hints"]) if fields else shot.get("query", "")
        lengths[shot["id"]] = montage_match.piece_lengths(shot["duration"])
        options[shot["id"]] = [
            c
            for c in library.search(
                query, limit=100, min_duration=2, min_height=project.get("min_video_height", 720)
            )
            if not relevance(shot, c["source"])[0]
        ]
    pending, options = library_match.quick_candidates(project, pending, options, stats)
    existing = intervals(project)
    compact = montage_match._shared_sources(pending, options)
    # A dry allocation avoids even a ranking request when the source time is
    # insufficient. It cannot approve content or reserve another shot's footage.
    pending = [
        s
        for s in pending
        if montage_match._plan(
            options[s["id"]], lengths[s["id"]], existing, set(), check, compact_sources=compact
        )[0]
        is not None
    ]
    if not pending:
        return set(), stats | {"elapsed_seconds": round(time.monotonic() - started_at, 3)}

    def requested():
        stats["model_requests"] += 1

    library_match.rank_windows(
        pending, options, check, diagnostics=stats["failures"], request_started=requested
    )
    compact = montage_match._shared_sources(pending, options)
    used = defaultdict(list)
    for key, spans in existing.items():
        used[montage_match._source_key(key)].extend(spans)
    prepared, reservations = [], {}
    for shot in pending:
        check()
        plan, _ = montage_match._plan(
            options[shot["id"]], lengths[shot["id"]], used, set(), check, compact_sources=compact
        )
        if plan is None:
            continue
        parts = []
        try:
            for allocation, duration in zip(plan, lengths[shot["id"]], strict=True):
                check()
                clip = allocation["clip"]
                asset = library.materialize(project["id"], clip, check)
                offset = clip["offset"] + allocation["reserved"][1][0] - clip["start"]
                parts.append({"asset": asset, "shot": shot | {"clip_start": offset, "duration": duration}})
            asset = montage_match.assemble_candidate(project, shot, parts, check)
        except (ValueError, OSError) as exc:
            stats["failures"].append(str(exc))
            continue
        reservations[shot["id"]] = [a["reserved"] for a in plan]
        for key, span in reservations[shot["id"]]:
            used[key].append(span)
        prepared.append({"shot": shot | {"clip_start": 0, "fit": "contain"}, "asset": asset})
    stats["candidates"] = len(prepared)
    passed = set()
    context = tuple(project.get(k) for k in ("script", "requirements", "aspect", "media_policy"))

    def review(batch):
        check()
        return review_batch(store.get_project(project["id"]), batch, check)

    batches = [prepared[i : i + 4] for i in range(0, len(prepared), 4)]
    for _batch, (rows, errors, info) in execution.completed_map(review, batches, min(2, workers)):
        check()
        stats["whole_review_batches"] += 1
        stats["model_requests"] += info.get("model_requests", 0)
        stats["failures"].extend(error for _, error in errors)
        for row in rows:
            result, shot, asset = row["verification"], row["shot"], row["asset"]
            if (
                result.get("verdict") != "match"
                or result.get("scene_alignment") != "match"
                or result.get("selection_blocked", True)
                or result.get("needs_source_check", True)
            ):
                stats["failures"].append(result.get("reason", "完整候选未通过审核"))
                continue
            committed = False

            def choose(p, shot=shot, asset=asset, result=result, row=row):
                nonlocal committed
                current = next(s for s in p["shots"] if s["id"] == shot["id"])
                if (
                    tuple(p.get(k) for k in ("script", "requirements", "aspect", "media_policy")) != context
                    or not eligible(current)
                    or any(
                        current.get(k) != shot.get(k)
                        for k in ("narration", "duration", "editorial_contract", "visual_plan")
                    )
                ):
                    return
                if not plan_reselection.window_available(p, shot["id"], asset, 0, shot["duration"]):
                    return
                current.update(
                    selected_asset=asset["id"],
                    clip_start=0,
                    fit="contain",
                    approved=False,
                    verification=result,
                    match_result=result,
                    source_review_note="",
                    source_review_url="",
                    search_note="本地短片段组合已完成整段画面审核；" + result["reason"],
                    match_preview=row["preview"]
                    | {"asset_id": asset["id"], "clip_start": 0, "duration": shot["duration"]},
                )
                current.update(curation.reviewed_label_fields(result))
                committed = True

            store.mutate(project["id"], choose, internal=True)
            if committed:
                passed.add(shot["id"])
        progress(1, f"本地短片段完整审核 · 已配 {len(passed)}/{len(prepared)} 镜")
    stats.update(selected=len(passed), elapsed_seconds=round(time.monotonic() - started_at, 3))
    return passed, stats
