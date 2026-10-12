"""Bounded, user-triggered collection into the persistent clip library."""
from __future__ import annotations

import json
import time
from urllib.parse import urlparse

from . import config, execution, library, media, neutral_observer, paths, sources, stock_sources, store, video_sources


def project_id():
    with execution.keyed_lock(("library-project", str(paths.DATA))):
        with store.connection() as con:
            rows = con.execute("SELECT body FROM projects").fetchall()
        for row in rows:
            p = json.loads(row["body"])
            if p.get("system_library"):
                return p["id"]
        p = store.new_project("素材库采集", media_policy="video_only")
        store.mutate(p["id"], lambda p: p.update(system_library=True))
        return p["id"]


def describe(rows, check):
    if not config.read()["model_api_key"]:
        return {}
    # Collection and production use the same pixel-bound blind observations.
    # A separately collected window can then earn a fast-path review slot;
    # old untraceable captions do not receive the new method marker.
    records = [{"shot": {"id": str(n)}, "frame_paths": row["frames"]} for n, row in enumerate(rows)]
    observations, _ = neutral_observer.observe(records, check)
    return {n: {"description": observations[str(n)]["scene_description"],
                "visible_text": "\n".join(dict.fromkeys(text for frame in observations[str(n)]["frames"]
                                                       for text in frame["text"])),
                "replace_observations": True}
            for n in range(len(rows))}


def run(pid, options, check, progress):
    from .pooled_match import bounded_check, diverse_candidates, sample_frames
    began = time.monotonic()
    if options.get("action") == "index":
        from . import source_index
        return source_index.run(pid, options, check, progress)
    if options.get("action") == "rebuild":
        report = library.rebuild(check, progress)
        store.mutate(pid, lambda p: p.update(library_report=report), internal=True)
        return report
    candidates, errors = [], []
    for url in options.get("urls", []):
        url = library.source_url(url)
        provider = ("bilibili" if "bilibili.com/" in url else "tencent" if "v.qq.com/" in url
                    else "chinanews" if "chinanews.com" in url else "cctv" if "cctv.com/" in url else "web_video")
        provider = next((name for name, hosts in stock_sources.SOURCE_HOSTS.items()
                         if urlparse(url).hostname in hosts), provider)
        candidates.append(video_sources._candidate(provider, "导入的公开视频", url, ""))
    explicit_count = len(candidates)
    queries = options.get("queries", [])
    def search(q):
        check()
        return video_sources.search_videos(q)
    for query, (rows, failures) in execution.completed_map(search, queries, 3):
        tokens = sources.tokens(query)
        rows.sort(key=lambda c: len(tokens & sources.tokens(c["title"] + " " + c.get("description", ""))), reverse=True)
        candidates.extend(diverse_candidates(rows, 12))
        errors.extend(failures)
    candidates = list({c["source_url"]: c for c in candidates}.values())[:max(24, explicit_count)]
    library.remember_sources(candidates)
    progress(10, f"找到 {len(candidates)} 个来源，检查可下载画面")
    max_sources = options.get("max_sources", 6)
    seconds = options.get("clip_seconds", 12)
    windows = options.get("clips_per_source", 2)
    def inspect(c):
        try:
            result = video_sources.inspect_video_candidate(c, bounded_check(check, 25, "来源检查"))
            if result.get("download_status") != "available" or min(result.get("width", 0), result.get("height", 0)) < 720:
                return None, result.get("download_reason") or "源视频低于720p"
            return result, None
        except ValueError as exc:
            return None, str(exc)
    accepted = []
    # Inspect only a bounded shortlist; stop launching work after enough sources.
    for offset in range(0, len(candidates), 4):
        for c, (row, error) in execution.completed_map(inspect, candidates[offset:offset+4], 4):
            check()
            if row:
                accepted.append(row)
            else:
                errors.append(c["title"][:80] + "：" + error)
        if len(accepted) >= max_sources:
            break
    accepted = accepted[:max_sources]
    tasks = []
    for c in accepted:
        length = float(c["duration"])
        duration = min(seconds, length)
        if duration < 2:
            continue
        positions = [max(0, min(length-duration, length*f)) for f in (.12, .56)[:windows]]
        last = -duration
        for start in sorted(set(round(s, 3) for s in positions)):
            if start < last + duration - .1:
                continue
            tasks.append((c, start, duration))
            last = start
    def download(task):
        c, start, duration = task
        limited = bounded_check(check, 40, "采集片段")
        try:
            target = paths.project_dir(pid) / "assets" / (store.uid()+".mp4")
            metadata = video_sources.download_video_candidate(c, target, start, duration, limited)
            asset = media.inspect_asset(pid, target, c["title"])
            asset.update({k: c[k] for k in library.FIELDS if k in c})
            asset.update(source_clip_start=metadata["source_clip_start"], candidate_id=c["id"])
            frames, preview = sample_frames(pid, {"duration": asset["duration"], "clip_start": 0}, asset, limited)
            if not preview["motion_detected"]:
                raise ValueError("抽样未检测到动态画面")
            return {"asset": asset, "frames": frames}, None
        except ValueError as exc:
            return None, str(exc)
    ready = []
    for n, (_, (row, error)) in enumerate(execution.completed_map(download, tasks, 4)):
        check()
        if row:
            ready.append(row)
            store.mutate(pid, lambda p, a=row["asset"]: p["assets"].append(a), internal=True)
        else:
            errors.append(error)
        progress(25+(n+1)/max(1, len(tasks))*40, f"读取动态片段 {n+1}/{len(tasks)}")
    saved = []
    batches = [ready[i:i+4] for i in range(0, len(ready), 4)]
    def caption(rows):
        try:
            return describe(rows, check), None
        except ValueError as exc:
            return {}, str(exc)
    for n, (rows, (descriptions, error)) in enumerate(execution.completed_map(caption, batches, 3)):
        check()
        if error:
            errors.append(error)
        for i, row in enumerate(rows):
            saved.append(library.remember_window(pid, row["asset"], check=check, **descriptions.get(i, {})))
        progress(70+(n+1)/max(1, len(batches))*25, f"已建立 {len(saved)} 个片段索引")
    report = {"indexed": len(saved), "source_count": len(accepted), "clip_ids": saved,
              "errors": list(dict.fromkeys(errors))[:20], "elapsed_seconds": round(time.monotonic()-began, 3)}
    store.mutate(pid, lambda p: p.update(library_report=report), internal=True)
    if not saved:
        raise ValueError("本次没有取得可入库的动态视频；" + "；".join(report["errors"][:3]))
    return report
