"""Fill a long illustration with several separately reviewed real video cuts."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import json
import math

from . import context_labels, curation, editorial_contract, execution, library, library_match, media, paths, store


def piece_lengths(duration):
    frames = round(duration*25)
    count = max(2, math.ceil(frames/100))
    if not 4 <= duration <= 12 or count > 3:
        return []
    base, extra = divmod(frames, count)
    return [(base+(i < extra))/25 for i in range(count)]


_SEARCH_LIMIT = 4096


def _source_key(value):
    try:
        return library.source_url(value)
    except ValueError:
        # Indexed URLs are already validated. Keep non-URL legacy source IDs
        # usable for occupancy accounting without inventing URL equivalence.
        return str(value or "").split("#", 1)[0]


def _frame(value, rounding):
    return int((Decimal(str(value))*25).to_integral_value(rounding=rounding))


def _free(left, right, occupied):
    for start, stop in sorted(occupied):
        if stop <= left:
            continue
        if start >= right:
            break
        if start > left:
            yield left, min(start, right)
        left = max(left, stop)
        if left >= right:
            return
    if left < right:
        yield left, right


def _starts(left, right, length, occupied, compact=False):
    gaps = [(a, b-length) for a, b in _free(left, right, occupied) if b-a >= length]
    center = (left+right-length)//2
    seen = set()
    edges = [edge for gap in gaps for edge in gap]
    for at in [*edges, center] if compact else [center, *edges]:
        if at not in seen and any(a <= at <= b for a, b in gaps):
            seen.add(at)
            yield at
    # Boundary choices usually suffice. The finite frame-grid fallback makes
    # the search complete unless its explicit, cancellable work limit is hit.
    for a, b in gaps:
        for at in range(a, b+1):
            if at not in seen:
                yield at


def _near_attempt(window, attempted, reusable=()):
    """Deduplicate opportunities, not evidence: both cuts overlap at least 80%."""
    if window in reusable:
        return False  # Only an exact successful in-call review can be reused.
    key, start, stop = window
    return any(key == old_key and 5*(min(stop, end)-max(start, begin)) >= 4*max(stop-start, end-begin)
               for old_key, begin, end in attempted)


def _plan(clips, durations, used, failed, check, *, compact_sources=(), attempted=(), reusable=()):
    """Find one entire fixed-length plan without changing caller reservations."""
    check()
    def finish(plan, status):
        check()
        return plan, status
    occupied = defaultdict(list)
    for key, windows in used.items():
        occupied[_source_key(key)].extend(
            (_frame(a, ROUND_FLOOR), _frame(b, ROUND_CEILING)) for a, b in windows)
    choices, available = [], defaultdict(list)
    for clip in clips:
        check()
        try:
            start, duration = Decimal(str(clip["start"])), Decimal(str(clip["duration"]))
            if not start.is_finite() or not duration.is_finite() or start < 0 or duration <= 0:
                continue
            key = _source_key(clip["source"]["source_url"])
            left, right = _frame(start, ROUND_CEILING), _frame(start+duration, ROUND_FLOOR)
        except (KeyError, ValueError, InvalidOperation):
            continue
        if key and left < right:
            choices.append((clip, key, left, right))
            available[key].extend(_free(left, right, occupied[key]))
    frames = [round(length*25) for length in durations]
    if not frames:
        return finish([], "feasible")
    # Union capacity counts source aliases/overlapping clips once, and never
    # treats a single observed frame or a title as a continuous usable window.
    capacity = 0
    for spans in available.values():
        end = -1
        for a, b in sorted(spans):
            capacity += max(0, b-max(a, end))
            end = max(end, b)
    if capacity < sum(frames):
        return finish(None, "infeasible")
    visits = 0

    class SearchLimit(Exception):
        pass

    def place(index):
        nonlocal visits
        if index == len(frames):
            return []
        length = frames[index]
        for clip, key, left, right in choices:
            for at in _starts(left, right, length, occupied[key], key in compact_sources):
                if visits >= _SEARCH_LIMIT:
                    raise SearchLimit
                visits += 1
                if visits % 32 == 0:
                    check()
                window = (key, at, at+length)
                if window in failed or _near_attempt(window, attempted, reusable):
                    continue
                occupied[key].append((at, at+length))
                try:
                    tail = place(index+1)
                finally:
                    occupied[key].pop()
                if tail is not None:
                    return [{"clip": clip, "reserved": (key, (at/25, (at+length)/25))}, *tail]
        return None

    try:
        plan = place(0)
    except SearchLimit:
        return finish(None, "search_limit")
    return finish(plan, "feasible" if plan is not None else "infeasible")


def _window_id(reservation):
    key, (start, stop) = reservation
    return _source_key(key), round(start*25), round(stop*25)


def _shared_sources(shots, options):
    owners = defaultdict(set)
    for shot in shots:
        for clip in options[shot["id"]]:
            owners[_source_key(clip["source"]["source_url"])].add(shot["id"])
    return {key for key, ids in owners.items() if len(ids) > 1}


def reserve(clip, duration, used):
    plan, _ = _plan([clip], [duration], used, set(), lambda: None)
    if plan:
        key, window = plan[0]["reserved"]
        used.setdefault(key, []).append(window)
        return key, window
    return None


def assemble(project, shot, parts, check):
    """No loops, stills or padding: concatenate exact, source-traced video cuts."""
    return _assemble(project, shot, parts, check, require_review=True)


def assemble_candidate(project, shot, parts, check):
    """Build unapproved retrieval material for one complete final-frame review."""
    if (curation.needs_identity(shot) or shot.get("location") or shot.get("locked")
            or shot.get("approved") or shot.get("selected_asset")
            or (shot.get("visual_plan") or {}).get("mode") == "explicit_sequence"):
        raise ValueError("当前镜头不能使用未审核短片段组合候选")
    return _assemble(project, shot, parts, check, require_review=False)


def _assemble(project, shot, parts, check, *, require_review):
    pid = project["id"]
    folder = paths.project_dir(pid)/"montages"/store.uid()
    folder.mkdir(parents=True)
    width, height = (720, 1280) if project.get("aspect") == "9:16" else (1280, 720)
    sources, files = [], []
    for i, row in enumerate(parts):
        check()
        asset, part = row["asset"], row["shot"]
        result = row.get("verification") or {}
        if (asset.get("source_parts") or not asset.get("source_url") or require_review and (
                result.get("verdict") != "match" or result.get("scene_alignment") != "match"
                or result.get("selection_blocked", True) or result.get("needs_source_check", True))):
            raise ValueError("拼接必须使用逐段新审核通过且来源可追踪的原视频")
        duration, start = float(part["duration"]), float(part["clip_start"])
        if (not all(math.isfinite(v) for v in (duration, start)) or start < 0 or duration <= 0
                or asset["kind"] != "video" or asset["duration"] < start+duration-.04
                or min(asset["width"], asset["height"]) < project.get("min_video_height", 720)):
            raise ValueError("拼接来源不满足真实视频时长或清晰度")
        filename = f"part-{i:02}.mp4"
        media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-y", "-ss", str(start),
            "-i", str(paths.file_path(pid, asset["path"])), "-an", "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x111318,setsar=1,fps=25",
            "-frames:v", str(round(duration*25)), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-threads", "2", str(folder/filename)], check=check, timeout=45)
        files.append(f"file '{filename}'")
        sources.append({"asset_id": asset["id"], "path": asset["path"], "title": asset["title"],
            "source_url": asset.get("source_url", ""), "clip_start": start, "duration": duration,
            "source_at": float(asset.get("source_clip_start", 0))+start,
            "source_evidence": asset.get("source_evidence"), "privacy_redaction": asset.get("privacy_redaction"),
            **({"verification": row["verification"]} if require_review else {"review_status": "unreviewed_candidate"})})
    if abs(sum(p["duration"] for p in sources)-shot["duration"]) > .02:
        raise ValueError("拼接片段未覆盖完整口播时长")
    (folder/"concat.txt").write_text("\n".join(files), "utf-8")
    target = paths.project_dir(pid)/"assets"/(store.uid()+".mp4")
    target.parent.mkdir(exist_ok=True)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-y", "-f", "concat", "-safe", "0",
               "-i", "concat.txt", "-c", "copy", "-movflags", "+faststart", str(target)], cwd=folder, check=check)
    asset = media.inspect_asset(pid, target, "资料片段组合 · "+parts[0]["asset"]["title"])
    if abs(asset.get("duration", 0)-shot["duration"]) > .08:
        raise ValueError("拼接结果实际时长与口播不符")
    asset.update(source_parts=sources, provider="local_montage", source_url="", source_verified=False)
    store.mutate(pid, lambda p: p["assets"].append(asset), internal=True)
    return asset


def match(project, shots, check, progress, *, review_batch, workers=3):
    from .pooled_match import intervals, relevance
    stats = {"model_requests": 0, "selected": 0, "failures": [], "review_summaries": [],
             "allocation_diagnostics": []}
    pending = [s for s in shots if piece_lengths(s["duration"]) and not s.get("locked")
        and not curation.needs_identity(s) and not s.get("location")
        and (s.get("visual_plan") or {}).get("mode") != "explicit_sequence"]
    if not pending:
        return set(), stats
    options = {}
    for shot in pending:
        check()
        fields = editorial_contract.retrieval_fields(shot)
        query = " ".join(fields["query_hints"]) if fields else shot.get("query", "")
        options[shot["id"]] = [c for c in library.search(query, limit=100, min_duration=2,
            min_height=project.get("min_video_height", 720)) if not relevance(shot, c["source"])[0]]
    pending, options = library_match.quick_candidates(project, pending, options, stats)
    compact_sources = _shared_sources(pending, options)
    lengths = {s["id"]: piece_lengths(s["duration"]) for s in pending}
    targets = {s["id"] for s in pending}
    existing = intervals(project | {"shots": [s for s in project["shots"] if s["id"] not in targets]})
    reported = set()
    def diagnostic(sid, stage, status):
        if (sid, stage, status) not in reported:
            reported.add((sid, stage, status))
            stats["allocation_diagnostics"].append({"shot_id": sid, "stage": stage, "status": status,
                "search_limit": _SEARCH_LIMIT, "temporary_reservations": stage == "allocation"})
    feasible = []
    for shot in pending:
        # Independent preflight: another unreviewed shot cannot permanently
        # exclude this shot before its paid ranking. This does not reserve time.
        plan, status = _plan(options[shot["id"]], lengths[shot["id"]], existing, set(), check,
                             compact_sources=compact_sources)
        if plan is not None:
            feasible.append(shot)
        else:
            diagnostic(shot["id"], "before_rank", status)
    pending = feasible
    if not pending:
        return set(), stats
    def started():
        stats["model_requests"] += 1
    library_match.rank_windows(pending, options, check, diagnostics=stats["failures"], request_started=started)
    compact_sources = _shared_sources(pending, options)
    targets = {s["id"] for s in pending}
    used = defaultdict(list)
    for key, windows in intervals(project | {"shots": [s for s in project["shots"] if s["id"] not in targets]}).items():
        used[_source_key(key)].extend(windows)
    parts, plans, owned = defaultdict(list), defaultdict(list), defaultdict(list)
    verified, failed, attempted = defaultdict(dict), defaultdict(set), defaultdict(set)
    def release(sid, reservations=None):
        for reservation in list(owned[sid] if reservations is None else reservations):
            if reservation in owned[sid]:
                key, window = reservation
                used[key].remove(window)
                owned[sid].remove(reservation)
    def reject(row):
        sid = row["shot"]["id"]
        failed[sid].add(_window_id(row["reserved"]))
        release(sid, [row["reserved"], *(p["reserved"] for p in plans.pop(sid, []))])
    def review(rows):
        check()
        # Record only actual per-piece review attempts, not planned/unused
        # reservations or download errors. This is local to this invocation;
        # format/transport failures do not become semantic negative evidence.
        for row in rows:
            if "reserved" in row:
                attempted[row["shot"]["id"]].add(_window_id(row["reserved"]))
        try:
            return review_batch(project, rows, check)
        except ValueError as exc:
            return [], [(r, str(exc)) for r in rows], {}
    try:
        for turn in range(5):
            check()
            changed = False
            # Rank may remove/reorder candidates: solve again using only its
            # remaining options. Reserve every remaining piece atomically.
            for shot in pending:
                sid = shot["id"]
                if plans[sid] or len(parts[sid]) == len(lengths[sid]):
                    continue
                plan, status = _plan(options[sid], lengths[sid][len(parts[sid]):], used, failed[sid], check,
                                     compact_sources=compact_sources, attempted=attempted[sid], reusable=verified[sid])
                if plan is None and parts[sid]:
                    # A passed prefix must not strand source capacity. Keep its
                    # exact in-call review in memory if a later full plan uses it.
                    release(sid)
                    parts[sid].clear()
                    changed = True
                    plan, status = _plan(options[sid], lengths[sid], used, failed[sid], check,
                                         compact_sources=compact_sources, attempted=attempted[sid], reusable=verified[sid])
                if plan is None:
                    diagnostic(sid, "allocation", status)
                    continue
                plans[sid] = plan
                for row in plan:
                    key, window = row["reserved"]
                    used[key].append(window)
                    owned[sid].append(row["reserved"])
            ready = []
            for shot in pending:
                sid = shot["id"]
                while plans[sid]:
                    allocation = plans[sid].pop(0)
                    reservation = allocation["reserved"]
                    cached = verified[sid].get(_window_id(reservation))
                    if cached is not None:
                        parts[sid].append(cached | {"reserved": reservation})
                        changed = True
                        continue
                    clip = allocation["clip"]
                    duration = lengths[sid][len(parts[sid])]
                    try:
                        asset = library.materialize(project["id"], clip, check)
                        start = clip["offset"]+reservation[1][0]-clip["start"]
                        ready.append({"shot": shot | {"clip_start": start, "duration": duration}, "asset": asset,
                                      "reserved": reservation})
                    except (ValueError, OSError) as exc:
                        reject({"shot": shot, "reserved": reservation})
                        stats["failures"].append(str(exc))
                        changed = True
                    break
            if not ready:
                if changed:
                    continue
                break
            progress(10+turn*13, f"逐段复核短视频组合 {turn+1}/5")
            batches = [ready[i:i+4] for i in range(0, len(ready), 4)]
            for _, (rows, errors, info) in execution.completed_map(review, batches, workers):
                stats["model_requests"] += info.get("model_requests", 0)
                for row, error in errors:
                    reject(row)
                    stats["failures"].append(error)
                folder = paths.DATA/"cache"/"montage-reviews"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder/(store.uid()+".json")
                try:
                    target.write_text(json.dumps({"project_id": project["id"], "parts": [{"shot_id": r["shot"]["id"],
                        "asset_id": r["asset"]["id"], "clip_start": r["shot"]["clip_start"], "duration": r["shot"]["duration"],
                        "verification": r["verification"], "preview": r["preview"]} for r in rows]}, ensure_ascii=False, indent=2), "utf-8")
                    stats["review_summaries"].append(str(target))
                except OSError as exc:
                    stats["failures"].append("拼接审核记录保存失败："+str(exc))
                for row in rows:
                    result = row["verification"]
                    frames = (result.get("identity_audit") or {}).get("frame_observations") or (result.get("observations") or {}).get("frame_observations") or []
                    if (result.get("verdict") == "match" and result.get("scene_alignment") == "match"
                            and not result.get("selection_blocked", True) and not result.get("needs_source_check", True)
                            and not context_labels.relative_statistics(frames)):
                        sid = row["shot"]["id"]
                        parts[sid].append(row)
                        verified[sid][_window_id(row["reserved"])] = row
                    else:
                        reject(row)
        for shot in pending:
            if len(parts[shot["id"]]) != len(lengths[shot["id"]]):
                release(shot["id"])
        finished = []
        for shot in pending:
            if len(parts[shot["id"]]) != len(lengths[shot["id"]]):
                continue
            check()
            try:
                asset = assemble(project, shot, parts[shot["id"]], check)
                finished.append({"shot": shot | {"clip_start": 0, "fit": "contain"}, "asset": asset})
            except (ValueError, OSError) as exc:
                release(shot["id"])
                stats["failures"].append(str(exc))
        passed = set()
        batches = [finished[i:i+4] for i in range(0, len(finished), 4)]
        for _, (rows, errors, info) in execution.completed_map(review, batches, workers):
            stats["model_requests"] += info.get("model_requests", 0)
            stats["failures"].extend(message for _, message in errors)
            for row in rows:
                result, shot, asset = row["verification"], row["shot"], row["asset"]
                if (result.get("verdict") != "match" or result.get("scene_alignment") != "match"
                        or result.get("selection_blocked", True) or result.get("needs_source_check", True)):
                    continue
                check()
                committed = False
                def choose(p, shot=shot, asset=asset, result=result, row=row):
                    nonlocal committed
                    current = next(s for s in p["shots"] if s["id"] == shot["id"])
                    if (current.get("locked") or any(current.get(k) != shot.get(k)
                            for k in ("narration", "duration", "editorial_contract"))):
                        return
                    current.update(selected_asset=asset["id"], clip_start=0, fit="contain", approved=False,
                        verification=result, match_result=result, source_review_note="", source_review_url="",
                        search_note="短视频逐段核验并拼接，已重新审核完整时间窗；"+result["reason"],
                        match_preview=row["preview"] | {"asset_id": asset["id"], "clip_start": 0, "duration": shot["duration"]})
                    current.update(curation.reviewed_label_fields(result))
                    committed = True
                store.mutate(project["id"], choose, internal=True)
                if committed:
                    passed.add(shot["id"])
        stats["selected"] = len(passed)
        return passed, stats
    finally:
        # Only this invocation's reservations are removed, including on budget
        # exceptions/cancellation. Real selected-source occupancy is untouched.
        for sid in list(owned):
            release(sid)
