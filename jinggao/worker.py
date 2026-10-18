from __future__ import annotations

import contextvars
from copy import deepcopy
import time
import math
import multiprocessing
import os
import re
import threading

from . import (captions, config, costs, curation, execution, express_plan, identity, instance, media, paths, planner, power,
               sources, store, throttle, video_sources)


def shot_by_id(project: dict, shot_id: str) -> dict:
    for shot in project["shots"]:
        if shot["id"] == shot_id:
            return shot
    raise KeyError("分镜不存在")


def ensure_narration_alignment(pid, check=lambda: None, progress=lambda *_: None):
    project = store.get_project(pid)
    if not project.get("full_audio"):
        return project
    progress(0, "正在用本地语音时间戳对齐镜头，复用已有字幕缓存")
    for attempt in range(8):
        try:
            proposal = media.prepare_narration_alignment(project, check, progress)
        except captions.ShotLengthOutOfRange as exc:
            # Fast voices (Baidu reads ~5 characters a second) can make a short line last under 1 s.
            # In express projects such a shot joins its neighbour; the narration audio is unchanged.
            if exc.long or not exc.short or not curation.express_project(project):
                raise
            project = store.mutate(pid, lambda p: express_plan.merge_short(p, exc.short), internal=True)
            continue
        # Express shots that the real voice makes shorter than about two seconds flash past: join them to a
        # neighbour, but only before footage is chosen (re-exports of finished projects keep their cut).
        flashing = [i for i, d in enumerate((proposal.get("timing") or {}).get("durations") or []) if d < express_plan.MIN_SECONDS]
        if (flashing and attempt < 7 and len(project["shots"]) > 1 and curation.express_project(project)
                and not any(s.get("selected_asset") for s in project["shots"])):
            project = store.mutate(pid, lambda p: express_plan.merge_short(p, flashing), internal=True)
            continue
        break
    else:
        raise ValueError("真实语音对应的镜头不足 1 秒，自动合并后仍未对齐，请调整分镜；已保留音频")
    check()
    return store.mutate(pid, lambda p: media.apply_narration_alignment(p, proposal), internal=True)


SOURCE_FIELDS = ("source_url", "license", "license_url", "author", "description", "provider",
                 "source_evidence", "publisher", "publisher_url", "published_at", "source_verified",
                 "discovery_evidence", "platform_provider")


def save_candidate(pid: str, sid: str, candidate: dict) -> None:
    def change(p):
        shot = shot_by_id(p, sid)
        for index, row in enumerate(shot["candidates"]):
            if row["id"] == candidate["id"]:
                shot["candidates"][index] = candidate
                break
    store.mutate(pid, change, internal=True)


def clip_starts(project: dict, shot: dict, candidate: dict) -> list[float]:
    """Try separate temporal windows; repeated sources must not repeat a chosen clip."""
    length = float(candidate.get("duration") or 0)
    duration = float(shot["duration"])
    assets = {a["id"]: a for a in project["assets"]}
    local = assets.get(candidate.get("asset_id"))
    base = float((local or {}).get("source_clip_start") or 0)
    used = []
    for other in project["shots"]:
        if other["id"] == shot["id"]:
            continue
        asset = assets.get(other.get("selected_asset"))
        if not asset or not (asset.get("candidate_id") == candidate["id"]
                             or asset["id"] == candidate.get("asset_id")
                             or (candidate.get("source_url") and asset.get("source_url") == candidate["source_url"])):
            continue
        offset = float(other.get("clip_start") or 0) + float(asset.get("source_clip_start") or 0) - base
        used.append((offset, offset+float(other["duration"])))
    preferred = [min(12.0, length*.1)]
    if length > 35:
        preferred.extend([length*.4, length*.7])
    preferred.extend(end+.4 for _, end in sorted(used))
    preferred.append(0.0)
    available = []
    for start in preferred:
        if length and start+duration > length+.12:
            continue
        if any(start < end and start+duration > begin for begin, end in used):
            continue
        if not any(abs(start-old) < .4 for old in available):
            available.append(round(start, 3))
    return available[:3]


class ClipReservations:
    """Reserve source intervals before concurrent previews start downloading."""
    def __init__(self, pid):
        self.pid = pid
        self.lock = threading.Lock()
        self.pending = {}

    def claim(self, shot, candidate):
        with self.lock:
            project = store.get_project(self.pid)
            local = next((a for a in project["assets"] if a["id"] == candidate.get("asset_id")), {})
            base = float(local.get("source_clip_start") or 0)
            for entries in self.pending.values():
                for fake_shot, fake_asset in entries:
                    project["shots"].append(fake_shot)
                    project["assets"].append(fake_asset)
            starts = clip_starts(project, shot, candidate)
            entries = self.pending.setdefault(shot["id"], [])
            for start in starts:
                token = store.uid()
                entries.append(({"id": token, "selected_asset": token, "duration": shot["duration"], "clip_start": 0},
                                {"id": token, "candidate_id": candidate["id"], "kind": "video",
                                 "source_url": candidate.get("source_url", ""), "source_clip_start": base+start,
                                 "duration": shot["duration"]}))
            return starts

    def release(self, shot_id):
        with self.lock:
            self.pending.pop(shot_id, None)


def materialize(pid: str, candidate: dict, check, *, clip_start: float = 0, duration: float = 12) -> dict:
    project = store.get_project(pid)
    existing = next((a for a in project["assets"] if a["id"] == candidate.get("asset_id")), None)
    if existing:
        return existing | {"selected_clip_start": clip_start}
    for asset in project["assets"]:
        if asset.get("candidate_id") != candidate["id"]:
            continue
        if asset["kind"] == "image":
            return asset
        offset = clip_start-float(asset.get("source_clip_start", 0))
        if 0 <= offset and offset+duration <= asset["duration"]+.12:
            return asset | {"selected_clip_start": offset}
    target = paths.project_dir(pid) / "assets" / f"{store.uid()}.media"
    downloaded = {}
    if candidate.get("kind") == "video":
        candidate = video_sources.inspect_video_candidate(candidate, check)
        candidate["min_video_height"] = project.get("min_video_height", 720)
        target = target.with_suffix(".mp4")
        downloaded = video_sources.download_video_candidate(candidate, target, clip_start, duration, check)
    else:
        sources.download(candidate["url"], target)
    check()
    asset = media.inspect_asset(pid, target, candidate["title"],
                **{k: candidate.get(k, "") for k in ("source_url", "license", "license_url", "author", "description", "provider")})
    asset.update({k: candidate[k] for k in SOURCE_FIELDS if k in candidate})
    asset.update({k: downloaded[k] for k in ("source_clip_start", "download_mode", "downloaded_bytes") if k in downloaded})
    asset["candidate_id"] = candidate["id"]
    store.mutate(pid, lambda p: p["assets"].append(asset), internal=True)
    return asset | {"selected_clip_start": 0}


def preview_candidate(pid: str, shot: dict, candidate: dict, check, *, clip_start: float | None = None) -> tuple[dict, dict]:
    """Download a bounded sample, record temporal frames, then ask the cloud reviewer."""
    project = store.get_project(pid)
    folder = paths.project_dir(pid) / "review" / store.uid()
    folder.mkdir(parents=True)
    target = folder / "preview.mp4"
    local = next((a for a in project["assets"] if a["id"] == candidate.get("asset_id")), None)
    c = candidate.copy()
    if local:
        c.update({k: local[k] for k in SOURCE_FIELDS if k in local})
        c["duration"] = local["duration"]
    else:
        c = video_sources.inspect_video_candidate(c, check)
        save_candidate(pid, shot["id"], c)
    length = float(c.get("duration") or 0)
    if clip_start is None:
        clip_start = min(12.0, length*.1) if length else 0.0
    duration = min(12.0, float(shot["duration"])+.3)
    if length:
        duration = min(duration, max(0.0, length-clip_start))
    if duration < .5:
        raise ValueError("所选预检起点后没有足够的画面")
    if local:
        media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-y", "-ss", str(clip_start),
                   "-i", str(paths.file_path(pid, local["path"])), "-t", str(duration),
                   "-vf", "scale=960:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", str(target)],
                  timeout=90, check=check)
        downloaded = {"source_clip_start": clip_start, "download_mode": "local_preview", "downloaded_bytes": 0}
    else:
        downloaded = video_sources.preview_video_candidate(c, target, clip_start, duration, check)
    temporary = media.inspect_asset(pid, target, c["title"], **{k: c[k] for k in SOURCE_FIELDS if k in c})
    temporary.update({k: c[k] for k in SOURCE_FIELDS if k in c})
    preview = {"path": temporary["path"], "duration": temporary["duration"],
               "width": temporary["width"], "height": temporary["height"],
               "source_clip_start": downloaded.get("source_clip_start", clip_start),
               "download_mode": downloaded.get("download_mode"), "downloaded_bytes": downloaded.get("downloaded_bytes", 0)}
    frames, comparisons = [], []
    from PIL import Image, ImageChops, ImageStat
    for index, fraction in enumerate((.08, .3, .5, .7, .92)):
        check()
        path = folder / f"frame-{index}.jpg"
        at = float(temporary["duration"])*fraction
        media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-y", "-ss", str(at),
                   "-i", str(target), "-frames:v", "1", "-vf", "scale=960:-2", str(path)], timeout=30, check=check)
        frames.append({"path": path.relative_to(paths.project_dir(pid)).as_posix(),
                       "at": round(preview["source_clip_start"]+at, 2), "kind": "video_frame"})
        with Image.open(path) as im:
            comparisons.append(im.convert("RGB").resize((96, 54)))
    changes = [sum(ImageStat.Stat(ImageChops.difference(a, b)).mean)/3 for a, b in zip(comparisons, comparisons[1:], strict=False)]
    preview["motion_detected"] = max(changes, default=0) > .5
    preview["frame_change"] = round(max(changes, default=0), 3)
    preview["frames"] = frames
    c.update(preview=preview, preview_paths=frames, preflight_status="sampled")
    save_candidate(pid, shot["id"], c)
    if config.read()["model_api_key"]:
        result = planner.verify(pid, shot | {"clip_start": 0, "duration": temporary["duration"]}, temporary, check=check)
        preview["verification"] = result
        c["preflight_status"] = "rejected" if result.get("selection_blocked") or result["verdict"] == "mismatch" else "reviewed"
    if not preview["motion_detected"]:
        c["preflight_status"] = "rejected"
        c["preflight_reason"] = "抽样画面未检测到变化，可能为静态封面或冻结画面"
    save_candidate(pid, shot["id"], c)
    return c, temporary


@costs.job
def process(job: dict, stop=None, *, progress_sink=None, finalize=True) -> None:
    pid, jid, options = job["project_id"], job["id"], job["payload"]
    project = store.get_project(pid)

    def check():
        if (stop is not None and stop.is_set()) or store.get_job(jid)["cancel"]:
            raise media.Cancelled()

    def progress(value, message):
        check()
        if progress_sink:
            progress_sink(value, message)
        else:
            store.job_update(jid, progress=max(0, min(99, value)), message=message)

    check()
    kind = job["kind"]
    if kind in ("match", "verify", "select", "preview") or (
            kind == "render" and all(s.get("selected_asset") for s in project["shots"])):
        project = ensure_narration_alignment(pid, check, lambda value, message: progress(value*4, message))
    if kind == "produce":
        if project.get("script_changed"):
            raise ValueError("原稿已经修改，请先重新规划分镜，再继续整片制作")
        timings, wall_timings = {}, {}
        def save_workflow(**extra):
            store.mutate(pid, lambda p: p.update(workflow_info={"job_id": jid,
                "stages": timings, "stage_wall_seconds": wall_timings,
                "state": "running", **extra}), internal=True)
        save_workflow()
        def stage(name, label, lo, hi, **extra):
            began = time.monotonic()
            wall_began = time.time()
            progress(lo, label)
            save_workflow(current_stage=name)
            try:
                process(job | {"kind": name, "payload": options | extra}, stop, finalize=False,
                        progress_sink=lambda value, message: progress(lo+(hi-lo)*value/100, label+" · "+message))
            finally:
                # Failed/cancelled work consumed time too. Keep its stage and
                # duration so a saved workflow cannot look like an idle hang.
                timings[name] = round(time.monotonic()-began, 3)
                wall_timings[name] = round(time.time()-wall_began, 3)
                save_workflow(current_stage=name)
        if not project["shots"] and not project.get("script", "").strip() and project.get("source_audio"):
            stage("transcribe", "0/4 识别音频文字", 0, 5)
            project = store.get_project(pid)
        if not project["shots"] and not project.get("script", "").strip() and (project.get("news") or {}).get("text"):
            stage("rewrite", "0/4 AI 洗稿", 0, 5)
            project = store.get_project(pid)
        early = None
        if (not project["shots"] and project.get("script", "").strip() and not project.get("full_audio")
                and not project.get("source_audio") and config.read().get("pipeline", "express") == "express"
                and config.read().get("tts_provider") in ("edge_tts", "qwen_tts")):
            # 千问配音 needs only the standard text, not the shots: put the script in its standard form, then
            # voice it while the shots are planned (the account allows three speech requests at a time).
            from . import script_norm
            progress(1, "1/4 规划分镜 · 整理文稿标点")
            standard, _ = script_norm.normalize(project["script"], check)
            def keep_standard(p):
                if standard and standard != p["script"]:
                    p["script_original"] = p.get("script_original") or p["script"]
                    p["script"] = standard
            project = store.mutate(pid, keep_standard, internal=True)
            early = {"target": paths.project_dir(pid) / "audio" / f"{store.uid()}.wav", "error": None,
                     "began": time.monotonic(), "wall": time.time()}
            def voice_early():
                try:
                    media.synthesize(standard, early["target"], check)
                except BaseException as exc:   # reported (or retried by the voice stage) after planning
                    early["error"] = exc
                finally:
                    early["seconds"] = round(time.monotonic() - early["began"], 3)
            early["thread"] = threading.Thread(target=contextvars.copy_context().run, args=(voice_early,),
                                               name="early-voice", daemon=True)
            early["thread"].start()
        if not project["shots"]:
            stage("plan", "1/4 规划分镜" + ("（同时生成配音）" if early else ""), 0 if not project.get("news") else 5, 12,
                  mode="model", replace=False)
        if early:
            progress(12, "2/4 生成完整配音 · 等待配音完成")
            early["thread"].join()
            timings["voice_full"], wall_timings["voice_full"] = early.get("seconds", 0), round(time.time() - early["wall"], 3)
            save_workflow(current_stage="voice_full")
            if isinstance(early["error"], (media.Cancelled, costs.BudgetExceeded)):
                raise early["error"]
            if early["error"] is None:
                duration = float(media.probe(early["target"])["format"]["duration"])
                def attach_voice(p):
                    if p.get("full_audio") or any(s.get("audio") for s in p["shots"]):
                        return
                    media.align_narration(p, {"path": early["target"].relative_to(paths.project_dir(pid)).as_posix(),
                                              "duration": duration, "text": p["script"], "title": "整段自动配音",
                                              "provider": config.read().get("tts_provider")})
                store.mutate(pid, attach_voice, internal=True)
            # Any other failure: the voice stage below tries again and reports it.
        project = store.get_project(pid)
        if project.get("source_audio") and not project.get("full_audio") and not any(s.get("audio") for s in project["shots"]):
            # 音频成片: the uploaded narration is the soundtrack; shots are aligned to it and nothing is voiced.
            def attach(p):
                source = p["source_audio"]
                media.align_narration(p, {"path": source["path"], "duration": source["duration"],
                                          "title": source.get("title") or "上传的音频", "text": p["script"]})
            project = store.mutate(pid, attach, internal=True)
        warmup = None
        pending_voice = not project.get("full_audio") and not all(s.get("audio") for s in project["shots"])
        to_match = [s for s in project["shots"] if not s.get("locked") and (not s.get("selected_asset") or not s.get("approved"))
                    and not (s.get("selected_asset") and curation.manual_choice(s))]
        if (config.read().get("pipeline", "express") == "express" and (project.get("plan_info") or {}).get("mode") == "express"
                and pending_voice and to_match and config.read().get("model_api_key")):
            # Footage search and preview sheets need no narration: fetch them while the voice is made and aligned.
            snapshot = deepcopy(project)
            def warm():
                try:
                    from . import express_match
                    wanted = {s["id"] for s in to_match}
                    express_match.warm(snapshot, [s for s in snapshot["shots"] if s["id"] in wanted], check)
                except Exception:
                    pass      # only a head start; the matching stage does the real work
            # Same cost scope (project/job budget) and cancel check as the job itself.
            warmup = threading.Thread(target=contextvars.copy_context().run, args=(warm,), name="express-warmup",
                                      daemon=True)
            warmup.start()
        if not project.get("full_audio") and not all(s.get("audio") for s in project["shots"]):
            if any(s.get("locked") and not s.get("audio") for s in project["shots"]):
                raise ValueError("锁定镜头缺少配音，请先解锁或导入该镜配音")
            # Resume partial per-shot narration without replacing paid or imported audio.
            if any(s.get("audio") for s in project["shots"]):
                stage("voice", "2/4 补全逐镜配音", 12, 22, replace=False)
            else:
                stage("voice_full", "2/4 生成完整配音", 12, 22, replace=False)
        project = store.get_project(pid)
        if project.get("full_audio"):
            began, wall_began = time.monotonic(), time.time()
            project = ensure_narration_alignment(pid, check, lambda value, message: progress(22+value*2, message))
            timings["align"] = round(time.monotonic()-began, 3)
            wall_timings["align"] = round(time.time()-wall_began, 3)
            save_workflow(current_stage="align")
        if warmup is not None:
            warmup.join(timeout=240)
        # Generating a new full audio track can change durations and invalidate
        # old reviews. Recheck retained assets before deciding the job needs help.
        current_assets = {a["id"]: a for a in project["assets"]}
        express = config.read().get("pipeline", "express") == "express"
        retained = [] if express else [s for s in project["shots"] if s.get("selected_asset") and not s.get("locked")
                    and (not s.get("approved") and (not (s.get("verification") or {}).get("scene_alignment")
                         or ((s.get("verification") or {}).get("identity_audit") or {}).get("version") != planner.FRAME_AUDIT_VERSION)
                         or curation.selection_error(project, s, current_assets.get(s["selected_asset"], {})))]
        from . import library_match
        retrieval_misses = {s["id"] for s in retained if library_match.retained_retrieval_miss(
            project, s, current_assets.get(s["selected_asset"], {}))}
        if retrieval_misses:
            def defer_unrelated(p):
                for shot in p["shots"]:
                    if shot["id"] in retrieval_misses:
                        curation.invalidate(shot)
                        shot["search_note"] = "原库内片段缺少当前检索的具体内容线索，继续查找其他素材"
            project = store.mutate(pid, defer_unrelated, internal=True)
            retained = [s for s in retained if s["id"] not in retrieval_misses]
        if retained:
            from . import pooled_match
            began = time.monotonic()
            wall_began = time.time()
            progress(22, f"复核已有素材（共 {len(retained)} 镜），复用现有配音")
            assets = {a["id"]: a for a in project["assets"]}
            rows = [{"shot": s, "asset": assets[s["selected_asset"]]} for s in retained if s["selected_asset"] in assets]
            batches = [rows[i:i+4] for i in range(0, len(rows), 4)]
            def review(items):
                try:
                    return pooled_match.review_batch(project, items, check)
                except ValueError as exc:
                    return [], [(r, str(exc)) for r in items], {}
            for n, (_, (reviewed, errors, _)) in enumerate(execution.completed_map(review, batches,
                    int(config.read().get("match_workers", "3")))):
                def save(p, reviewed=reviewed, errors=errors):
                    for row in reviewed:
                        current = shot_by_id(p, row["shot"]["id"])
                        current.update(verification=row["verification"], search_note=row["verification"]["reason"],
                                       selected_asset=row["asset"]["id"], clip_start=row["shot"].get("clip_start", 0),
                                       fit=row["shot"].get("fit", "cover"), approved=False,
                                       match_preview=row["preview"] | {"asset_id": row["asset"]["id"],
                                           "clip_start": row["shot"].get("clip_start", 0), "duration": row["shot"]["duration"]})
                        current.update(curation.reviewed_label_fields(row["verification"]))
                    for row, message in errors:
                        # A failed fresh window review supersedes an older
                        # automatic match. Otherwise approve_verified below
                        # could silently approve the stale result again.
                        # Keep the selected file and explicit source notes.
                        current = shot_by_id(p, row["shot"]["id"])
                        current.update(verification=None, approved=False, search_note=message)
                store.mutate(pid, save, internal=True)
                progress(22+(n+1)/max(1, len(batches))*8, "按配音时长复核已有素材")
            timings["verify"] = round(time.monotonic()-began, 3)
            wall_timings["verify"] = round(time.time()-wall_began, 3)
        project = store.mutate(pid, curation.approve_verified, internal=True)
        retry_shots = [s["id"] for s in project["shots"] if not s.get("locked")
                       and (not s.get("selected_asset") or not s.get("approved"))
                       and not (s.get("selected_asset") and curation.manual_choice(s))]
        if retry_shots:
            stage("match", "3/4 匹配实际画面", 30 if retained else 22, 75,
                  shot_ids=retry_shots, replace=True, shared_pool=True)
            project = store.mutate(pid, curation.approve_verified, internal=True)
        pending = [i+1 for i, s in enumerate(project["shots"]) if not s.get("selected_asset") or not s.get("approved")]
        report, note = curation.composition(project), ""
        if (curation.express_project(project) and not pending and not report["meets_target"]
                and report["video_ratio"] >= 0.5):
            # A batch video is better delivered with a few neutral cards than stopped for review: lower this
            # project's target to what was found and say so on its page.
            achieved = math.floor(report["video_ratio"] * 100) / 100
            project = store.mutate(pid, lambda p: p.update(min_video_ratio=achieved), internal=True)
            note = f"实拍视频占 {report['video_ratio']:.0%}，低于原定 {report['target_video_ratio']:.0%}，已照常导出"
        try:
            curation.validate_export(project)
        except ValueError as exc:
            reason = str(exc)
        else:
            reason = f"镜头 {'、'.join(map(str, pending))} 仍需补素材或核对依据" if pending else ""
        if reason:
            save_workflow(state="needs_review", reason=reason)
            store.job_update(jid, status="needs_review", progress=75, message="配音与本轮匹配已完成，补齐后可继续导出", error=reason)
            return
        stage("render", "4/4 合成画面与字幕", 75, 99)
        save_workflow(state="completed", **({"note": note} if note else {}))
    elif kind == "library":
        from . import library_collect
        report = library_collect.run(pid, options, check, progress)
        if finalize:
            message = (f"来源索引已更新 · 共 {report['total']:,} 条 · 本次新增 {report['added']:,} 条"
                       if report.get("action") == "index" else
                       f"索引已更新 · 本次处理 {report.get('indexed', report.get('processed', 0))} 个片段")
            if report.get("action") == "index" and not report.get("reached_target"):
                message += " · 本轮未达目标，可查看渠道状态后续建"
            store.job_update(jid, status="completed", progress=100,
                             message=message)
        return
    elif kind == "footage_index":
        from . import footage_index
        queries = [(name, q) for name in options.get("packs", []) for q in footage_index.PACKS.get(name, [])]
        queries += [("自定义", q) for q in options.get("keywords", [])]
        if not queries:
            raise ValueError("请选择题材包或填写关键词")
        report = footage_index.build(list(dict.fromkeys(queries)), check, progress, pages=int(options.get("pages", 1)),
                                     widen=int(options.get("widen", 8)))
        store.mutate(pid, lambda p: p.update(footage_index_report=report), internal=True)
        if finalize:
            store.job_update(jid, status="completed", progress=100,
                             message=f"素材库已更新 · 本次收录 {report['indexed']} 个视频、描述 {report['described_frames']} 帧"
                                     + ("（B站接口限流，部分关键词没搜完，稍后可再跑一次）" if report.get("blocked") else ""))
            return
    elif kind == "transcribe":
        from . import local_asr, script_norm
        import tempfile
        from pathlib import Path
        source = project.get("source_audio")
        if not source:
            raise ValueError("项目没有上传的音频")
        progress(5, "本地识别音频里的文字 · 不调用付费模型")
        with tempfile.TemporaryDirectory(prefix="transcribe-", dir=paths.project_dir(pid)) as directory:
            wav = Path(directory) / "speech.wav"
            media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(paths.file_path(pid, source["path"])),
                       "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=check)
            words = local_asr.transcribe(wav, check, lambda share, message: progress(5 + 55 * share, message))
        raw = "".join(str(w.get("text", "")) for w in words).strip()
        if len(raw) < 4:
            raise ValueError("没有从音频里识别出中文口播，请检查音频是否有人声")
        progress(65, "按语义补全标点，整理成文稿")
        script, _ = script_norm.normalize(raw, check)
        def save_transcript(p):
            if p["shots"] or p.get("script", "").strip():
                raise ValueError("文稿已存在，未覆盖")
            p["script"], p["script_original"] = script, raw
            if p.get("title", "").strip() in ("", "未命名项目", "音频成片"):
                first = next((part for part in re.split(r"[，,。！？!?；;\n]", script) if part.strip()), "音频成片")
                p["title"] = first.strip()[:24]
        store.mutate(pid, save_transcript, internal=True)
    elif kind == "rewrite":
        from . import rewrite
        news = project.get("news") or {}
        if not news.get("text"):
            raise ValueError("项目没有可改写的新闻原文")
        too_long = bool(news.get("minutes")) and len(news.get("koubo") or "") > news["minutes"] * rewrite.CHARS_PER_MINUTE * 1.15
        if news.get("koubo") and not too_long:
            # The news crawler already rewrote it (its own 口播 prompt, on Qwen): lay it out and fact-check it.
            progress(10, "使用爬虫生成的口播稿，对照原文核对事实")
            result = rewrite.from_crawler(news["koubo"], news["text"], title=news.get("koubo_title") or news.get("title", ""),
                                          project_id=pid, check=check)
        else:
            progress(10, "正在改写新闻并对照原文核对事实")
            result = rewrite.rewrite(news["text"], title=news.get("title", ""), minutes=news.get("minutes"),
                                     aspect=project.get("aspect", "16:9"), project_id=pid, check=check)
        def save(p):
            if p["shots"] or p.get("script", "").strip():
                raise ValueError("文稿已存在，未覆盖")
            p["script"] = result["script"]
            if result["title"] and p.get("title") in ("", news.get("title", ""), "新闻选题"):
                p["title"] = result["title"]
            p["news"] = news | {"rewrite": {k: result.get(k) for k in ("chars", "fixes", "check_numbers", "unmatched_fixes",
                                                                         "checked", "model")}}
        store.mutate(pid, save, internal=True)
    elif kind == "plan":
        if project["shots"] and not options.get("replace"):
            raise ValueError("已存在分镜，重新规划需要确认替换")
        progress(5, "正在分析文稿与选材条件")
        def plan_progress(done, total):
            batches = max(total-1, 1)
            if done >= total:
                message = "文稿规划与一致性复核已完成"
            elif done == total-1:
                message = "正在复核整份计划的内容与检索条件"
            else:
                message = f"分批规划文稿 {done}/{batches} · 保留原文并提取人物条件"
            progress(5+done/max(total, 1)*85, message)
        if config.read().get("pipeline", "express") == "express" and options.get("mode", "auto") != "rules":
            from . import express_plan
            progress(10, "按句切分文稿并生成检索规划")
            shots, info = express_plan.plan(project, check)
        else:
            shots, info = planner.plan(project, options.get("mode", "auto"), options.get("rewrite", False), check=check,
                                      progress=plan_progress)
        check()
        standard = info.pop("script", None)
        def update(p):
            if standard and standard != p.get("script"):
                # Narration and subtitles use the standard text; the original stays restorable.
                p["script_original"] = p.get("script_original") or p.get("script", "")
                p["script"] = standard
            p["shots"] = shots
            p["plan_info"] = info
            p["full_audio"] = None
            p["script_changed"] = False
        store.mutate(pid, update, internal=True)
    elif kind == "search":
        selected = [shot_by_id(project, options["shot_id"])] if options.get("shot_id") else project["shots"]
        if not selected:
            raise ValueError("请先创建分镜")
        failures = []
        for i, shot in enumerate(selected):
            if shot["locked"]:
                continue
            progress(i/len(selected)*95, f"查找镜头 {i+1}/{len(selected)} 的候选素材")
            try:
                candidates, note = sources.search_shot(project, shot)
                def save(p, shot=shot, candidates=candidates, note=note):
                    target = shot_by_id(p, shot["id"])
                    target["candidates"] = candidates
                    target["search_note"] = note
                store.mutate(pid, save, internal=True)
            except ValueError as exc:
                failures.append(f"镜头 {i+1}: {exc}")
                store.mutate(pid, lambda p, sid=shot["id"], error=str(exc):
                             shot_by_id(p, sid).update(search_note=error), internal=True)
            check()
        if failures:
            raise ValueError("部分检索未完成，成功结果已保存。" + "；".join(failures[:3]))
    elif kind == "match":
        if not config.read()["model_api_key"]:
            raise ValueError("自动配素材需要视觉模型，请先在设置中填写 API Key")
        selected = [shot_by_id(project, options["shot_id"])] if options.get("shot_id") else project["shots"]
        if options.get("shot_ids"):
            selected = [shot_by_id(project, sid) for sid in dict.fromkeys(options["shot_ids"])]
        selected = [s for s in selected if not s["locked"] and (not s["selected_asset"] or options.get("replace"))]
        if not selected:
            raise ValueError("没有需要自动配素材的镜头；可在单镜中重新匹配")
        express = config.read().get("pipeline", "express") == "express"
        if express or len(selected) >= 4 or options.get("shared_pool"):
            if express:
                from . import express_match, express_plan
                if not (project.get("plan_info") or {}).get("mode") == "express":
                    # Older plans: add express queries without re-splitting shots. The paid
                    # call runs outside the database transaction on a copy.
                    draft = deepcopy(project["shots"])
                    plan_info = express_plan.annotate(draft, project.get("script", ""),
                                                      project.get("requirements", ""), check)
                    fields = ("visual", "visual_origin", "query", "search_queries", "persons",
                              "evidence_mode", "event", "event_scope", "express")
                    targets = {s["id"] for s in selected}
                    def annotate(p, draft=draft, plan_info=plan_info, targets=targets):
                        rows = {s["id"]: s for s in draft}
                        for shot in p["shots"]:
                            row = rows.get(shot["id"])
                            if not row or shot["id"] not in targets or shot.get("locked") or shot.get("approved"):
                                continue
                            update = {k: row[k] for k in fields if k in row}
                            # User-entered people and visuals stay authoritative.
                            if shot.get("persons"):
                                names = {x["name"] for x in shot["persons"]}
                                update["persons"] = shot["persons"] + [x for x in row.get("persons", [])
                                                                       if x["name"] not in names]
                                update["evidence_mode"] = "person"
                            if shot.get("visual_origin") == "manual":
                                update.pop("visual", None)
                                update.pop("visual_origin", None)
                            if shot.get("event"):
                                update.pop("event", None)
                                update.pop("event_scope", None)
                            shot.update(update)
                        p["plan_info"] = plan_info
                    project = store.mutate(pid, annotate, internal=True)
                    selected = [shot_by_id(project, s["id"]) for s in selected]
                info = express_match.match(project, selected, check, progress)
            else:
                from . import pooled_match
                info = pooled_match.match(project, selected, check, progress,
                                          workers=int(options.get("match_workers") or config.read().get("match_workers", "3")))
            if info["unmatched"] and not info["selected"] and finalize:
                raise ValueError(f"共享素材池在 {info['elapsed_seconds']:.1f} 秒内通过 {info['selected']}/{len(selected)} 镜；"
                                 "其余镜头缺少合格素材或来源证据，已保留具体核验结果")
            gaps = f"；{len(info['unmatched'])} 镜缺少合格素材，查看对应镜头原因后补充" if info["unmatched"] else ""
            message = f"本轮检索完成：已配 {info['selected']}/{len(selected)} 镜，用时 {info['elapsed_seconds']:.1f} 秒"+gaps
            if finalize:
                store.job_update(jid, status="completed", progress=100, message=message)
            else:
                progress(100, message)
            return
        failures = []

        def prepare(shot):
            check()
            current = store.get_project(pid)
            try:
                candidates, note = sources.search_shot(current, shot)
                check()
                ranking = planner.rank_pool(current, shot, candidates, check)
                return shot, candidates, note, ranking, None
            except ValueError as exc:
                return shot, [], "", [], str(exc)

        workers = int(options.get("match_workers") or config.read().get("match_workers", "1"))
        reservations = ClipReservations(pid)
        completed, completion_lock = [0], threading.Lock()

        def match_one(item):
            i, original = item
            failure = None
            shot, candidates, note, ranking, error = prepare(original)
            check()
            progress(completed[0]/len(selected)*95, f"匹配镜头 {i+1}/{len(selected)}：检索、预览筛选与来源核验")
            try:
                if error:
                    raise ValueError(error)
                current_project = store.get_project(pid)
                store.mutate(pid, lambda p, sid=shot["id"], cs=candidates, note=note:
                             shot_by_id(p, sid).update(candidates=cs, search_note=note), internal=True)
                policy = current_project.get("media_policy", "mixed")
                candidate_map = {c["id"]: c for c in candidates}
                if policy != "mixed" or shot.get("preferred_media") == "video":
                    ranking.sort(key=lambda r: candidate_map[r["candidate_id"]].get("kind") != "video")
                picked, attempts = None, []
                for rank in ranking[:5]:
                    check()
                    c = next(c for c in candidates if c["id"] == rank["candidate_id"])
                    try:
                        clip_start = 0.0
                        if c.get("kind") == "video":
                            progress(completed[0]/len(selected)*95, f"镜头 {i+1}：先检查 {c['title'][:26]} 的小样")
                            inspected = video_sources.inspect_video_candidate(c, check) if not c.get("asset_id") else c
                            save_candidate(pid, shot["id"], inspected)
                            # A temporal sample cannot repair a different event
                            # explicitly stated by the retrieved source record.
                            _, source_text = identity.source_record(inspected)
                            if source_text and shot.get("event") and shot.get("event_scope", "exact") == "exact":
                                qualifiers = identity.event_checks(shot["event"], source_text, identity.persons(shot.get("persons", [])))
                                conflicts = [q["field"] for q in qualifiers if q["status"] == "conflict"]
                                if conflicts:
                                    raise ValueError("来源文字已明确冲突，跳过片段下载：" + "、".join(conflicts))
                            if inspected.get("download_status") in ("blocked", "unsupported"):
                                raise ValueError(inspected.get("download_reason") or "该来源只提供页面预览，未取得公开可下载流")
                            if inspected.get("height") and inspected.get("width") and min(inspected["height"], inspected["width"]) < current_project.get("min_video_height", 720):
                                raise ValueError(f"来源最高仅 {inspected['width']}×{inspected['height']}，低于方案要求；可手动预检，但不自动采用")
                            preview_ok = False
                            # A single opening card is not representative of a whole video.
                            starts = reservations.claim(shot, inspected)
                            if not starts:
                                raise ValueError("同一来源已用片段占满或素材短于口播，请更换来源或拆分镜头")
                            for start in starts[:3]:
                                c, _ = preview_candidate(pid, shot, inspected, check, clip_start=start)
                                preflight = c.get("preview", {}).get("verification") or {}
                                if (c.get("preflight_status") == "rejected" or preflight.get("selection_blocked")
                                        or preflight.get("verdict") == "mismatch"):
                                    attempts.append({"title": c["title"], "stage": "preview", "result": "rejected",
                                                     "source_url": c.get("source_url", ""), "clip_start": start,
                                                     "reason": c.get("preflight_reason") or preflight.get("identity_evidence", {}).get("reason") or preflight.get("reason", "小样未通过")})
                                    evidence = preflight.get("identity_evidence") or {}
                                    if evidence.get("event_status") in ("conflict", "insufficient"):
                                        # Different timestamps in this same file
                                        # cannot fill gaps in its source metadata.
                                        break
                                    continue
                                preview_ok = True
                                clip_start = c["preview"]["source_clip_start"]
                                break
                            if not preview_ok:
                                continue
                            progress(completed[0]/len(selected)*95, f"镜头 {i+1}：小样通过，下载所需视频片段")
                            asset = materialize(pid, c, check, clip_start=clip_start, duration=min(120, float(shot["duration"])+.3))
                        else:
                            if policy == "video_only":
                                continue
                            current = store.get_project(pid)
                            composition = curation.composition(current)
                            old_asset = next((a for a in current["assets"] if a["id"] == shot.get("selected_asset")), None)
                            replaced_image = shot["duration"] if old_asset and old_asset["kind"] == "image" else 0
                            if (policy == "video_preferred" and composition["image_seconds"]-replaced_image+shot["duration"]
                                    > composition["total_seconds"]*(1-current_project.get("min_video_ratio", .8))+.05):
                                attempts.append({"title": c["title"], "result": "image_budget",
                                                 "reason": "图片时长预算不足，保留视频缺口，不用图片凑满"})
                                continue
                            if rank.get("selection_blocked"):
                                attempts.append({"title": c["title"], "result": "evidence_missing",
                                                 "reason": rank.get("identity_evidence", {}).get("reason", "人物依据不足")})
                                continue
                            asset = materialize(pid, c, check)
                        if (asset["kind"] == "image" and
                                (max(asset["width"], asset["height"]) < 1024 or min(asset["width"], asset["height"]) < 576)):
                            attempts.append({"title": asset["title"], "result": "low_resolution",
                                             "reason": f"原图仅 {asset['width']}×{asset['height']}，未自动选用"})
                            continue
                        selected_start = asset.get("selected_clip_start", 0)
                        if asset["kind"] == "video":
                            if min(asset["width"], asset["height"]) < current_project.get("min_video_height", 720):
                                raise ValueError(f"实际视频仅 {asset['width']}×{asset['height']}，未达到制作方案清晰度")
                            if float(asset["duration"])-selected_start+.16 < float(shot["duration"]):
                                raise ValueError("视频片段短于当前口播，不用冻结末帧补足")
                        verdict = planner.verify(pid, shot | {"clip_start": selected_start}, asset, check=check)
                        if asset["kind"] == "video":
                            from . import frame_sampling
                            checked, start, _ = frame_sampling.verified_selection(
                                store.get_project(pid), shot | {"clip_start": selected_start}, asset, verdict)
                            asset = checked | {"selected_clip_start": start}
                        attempts.append({"title": asset["title"], "result": verdict["verdict"],
                                         "reason": "；".join(verdict.get("conflicts", [])) or verdict["reason"]})
                        if verdict["verdict"] == "mismatch" or verdict.get("selection_blocked"):
                            continue
                        if picked is None or verdict["verdict"] == "match":
                            picked = (asset, verdict)
                        if verdict["verdict"] == "match":
                            break
                    except ValueError as exc:
                        attempts.append({"title": c["title"], "result": "error", "reason": str(exc)})
                        continue
                store.mutate(pid, lambda p, sid=shot["id"], rows=attempts:
                             shot_by_id(p, sid).update(match_attempts=rows), internal=True)
                if not picked:
                    detail = "；".join(a["reason"][:130] for a in attempts[:3])
                    raise ValueError("未找到通过画面核验的候选，请调整关键词或导入素材" + ("。"+detail if detail else ""))
                asset, verdict = picked
                def choose(p):
                    current = shot_by_id(p, shot["id"])
                    if asset["kind"] == "image" and p.get("media_policy") == "video_preferred":
                        report = curation.composition(p)
                        old = next((a for a in p["assets"] if a["id"] == current.get("selected_asset")), None)
                        replaced = current["duration"] if old and old["kind"] == "image" else 0
                        if report["image_seconds"]-replaced+current["duration"] > report["total_seconds"]*(1-p.get("min_video_ratio", .8))+.05:
                            raise ValueError("其他镜头已用完图片预算，请补充视频")
                    current.update(selected_asset=asset["id"], approved=False, clip_start=asset.get("selected_clip_start", 0),
                                   source_review_note="", source_review_url="", verification=verdict,
                                   search_note="已自动选用并核验实际画面，请确认；"+verdict["reason"])
                    current.update(curation.reviewed_label_fields(verdict))
                store.mutate(pid, choose, internal=True)
            except ValueError as exc:
                failure = f"镜头 {i+1}: {exc}"
                store.mutate(pid, lambda p, sid=shot["id"], error=str(exc):
                             shot_by_id(p, sid).update(search_note=error), internal=True)
            finally:
                reservations.release(shot["id"])
            with completion_lock:
                completed[0] += 1
                progress(completed[0]/len(selected)*95, f"已处理 {completed[0]}/{len(selected)} 镜 · {workers} 路并发")
            return failure

        for failure in execution.ordered_map(match_one, enumerate(selected), workers):
            if failure:
                failures.append(failure)
        if failures:
            raise ValueError("部分镜头未匹配，已完成的结果已保存。" + "；".join(failures[:3]))
    elif kind in ("select", "preview"):
        shot = shot_by_id(project, options["shot_id"])
        if shot["locked"]:
            raise ValueError("请先解锁镜头")
        candidate = next((c for c in shot["candidates"] if c["id"] == options["candidate_id"]), None)
        if not candidate:
            raise ValueError("候选不存在，请重新搜索")
        if kind == "preview":
            if candidate.get("kind") != "video":
                raise ValueError("片段预检适用于视频候选")
            progress(10, "先读来源与低码率片段，抽取多个时间点检查")
            preview_candidate(pid, shot, candidate, check, clip_start=options.get("clip_start"))
            store.job_update(jid, status="completed", progress=100, message="小样预检完成，请查看画面与依据")
            return
        progress(10, "正在准备选定素材")
        clip_start = options.get("clip_start")
        if clip_start is None:
            clip_start = candidate.get("preview", {}).get("source_clip_start", 0)
        if candidate.get("kind") == "video" and not candidate.get("asset_id"):
            if not candidate.get("preview"):
                candidate, _ = preview_candidate(pid, shot, candidate, check, clip_start=clip_start)
            asset = materialize(pid, candidate, check, clip_start=clip_start, duration=min(120, float(shot["duration"])+.3))
        else:
            asset = materialize(pid, candidate, check)
        asset_id = asset["id"]
        def select(p):
            s = shot_by_id(p, shot["id"])
            s.update(selected_asset=asset_id, approved=False, verification=None,
                     clip_start=asset.get("selected_clip_start", 0), source_review_note="", source_review_url="")
            curation.manual_pick(p, s, asset)
        store.mutate(pid, select, internal=True)
    elif kind == "voice_full":
        if not project["shots"] or project.get("script_changed"):
            raise ValueError("请先为当前完整文稿创建分镜")
        if any(s["locked"] for s in project["shots"]):
            raise ValueError("整段配音需要重排镜头时长，请先解锁镜头")
        if (project.get("full_audio") or any(s.get("audio") for s in project["shots"])) and not options.get("replace"):
            raise ValueError("已有配音；使用整段配音需要确认替换并重新分配镜头时长")
        text = "\n".join(s["narration"] for s in project["shots"])
        if planner.normalized(text) == planner.normalized(project["script"]):
            text = project["script"]
        if config.read()["tts_provider"] in ("baidu_web", "baidu_http") and len(text) > 5000:
            raise ValueError("一镜整段配音单次最多 5000 字；请分成多个项目或使用逐镜配音")
        progress(5, "一次生成完整音轨；素材核验前将用本地语音时间戳对齐镜头")
        target = paths.project_dir(pid) / "audio" / f"{store.uid()}.wav"
        media.synthesize(text, target, check)
        duration = float(media.probe(target)["format"]["duration"])
        check()
        audio = {"path": target.relative_to(paths.project_dir(pid)).as_posix(), "duration": duration,
                 "text": text, "title": "整段自动配音", "provider": config.read()["tts_provider"]}
        store.mutate(pid, lambda p: media.align_narration(p, audio), internal=True)
    elif kind == "voice":
        if project.get("full_audio"):
            raise ValueError("本项目正在使用整段配音。请先移除整段配音后生成逐镜配音")
        selected = [shot_by_id(project, options["shot_id"])] if options.get("shot_id") else project["shots"]
        if not selected:
            raise ValueError("请先创建分镜")
        for i, shot in enumerate(selected):
            if shot["locked"]:
                continue
            if shot.get("audio") and not options.get("replace"):
                continue
            progress(i/len(selected)*95, f"生成配音 {i+1}/{len(selected)}")
            target = paths.project_dir(pid) / "audio" / f"{store.uid()}.wav"
            media.synthesize(shot["narration"], target, check)
            duration = float(media.probe(target)["format"]["duration"])
            if duration > 120:
                raise ValueError(f"镜头 {i+1} 配音超过 120 秒，请拆分镜头")
            check()
            audio = {"path": target.relative_to(paths.project_dir(pid)).as_posix(), "duration": duration,
                     "text": shot["narration"]}
            def save_voice(p, sid=shot["id"], audio=audio, duration=duration):
                current = shot_by_id(p, sid)
                retimed = abs(current["duration"] - duration) > .01
                current.update(audio=audio, duration=duration)
                if retimed and not curation.keep_express(p, current):
                    curation.invalidate(current)
            store.mutate(pid, save_voice, internal=True)
    elif kind == "verify":
        shot = shot_by_id(project, options["shot_id"])
        asset = next((a for a in project["assets"] if a["id"] == shot["selected_asset"]), None)
        if not asset:
            raise ValueError("请先选择一个素材")
        from . import frame_sampling, window_attempts
        def review_context(p, s):
            return deepcopy({"project": {k: p.get(k) for k in (
                "script", "requirements", "aspect", "media_policy", "min_video_height")},
                "shot": {k: s.get(k) for k in (*window_attempts.SHOT_FIELDS, "selected_asset", "source_unit_id")},
                "context": planner.narrative_context(p, s)})
        original_context = review_context(project, shot)
        progress(10, "正在抽帧并检查画面与分镜是否一致")
        result = planner.verify(pid, shot, asset, check=check)
        check()
        def save_verification(p):
            current = shot_by_id(p, shot["id"])
            if review_context(p, current) != original_context:
                raise ValueError("核验期间原文、要求或镜头已改变，未覆盖当前选择")
            if asset["kind"] == "video":
                original = next((a for a in p["assets"] if a["id"] == asset["id"]), None)
                if original is None:
                    raise ValueError("核验原素材已移除，未覆盖当前选择")
                reviewed_asset, start, preview = frame_sampling.verified_selection(p, current, original, result)
                # Privacy processing may have created a different, re-reviewed
                # file. Install exactly that file, never the unmasked original.
                current.update(selected_asset=reviewed_asset["id"], clip_start=start, match_preview=preview)
            express = curation.express_result(current) if curation.express_project(p) else {}
            manual = current.get("manual_asset")
            curation.invalidate(current)
            # Express projects: the AI check is advice on top of the pick; the pick (and a manual choice) stays.
            current.update(verification=result | ({"express": express} if express else {}))
            if manual and manual == current.get("selected_asset"):
                current["manual_asset"] = manual
            current.update(curation.reviewed_label_fields(result))
        store.mutate(pid, save_verification, internal=True)
    elif kind == "render":
        result = media.render(project, jid, options, progress, check)
        try:
            result["saved_to"] = media.deliver(project, result)
        except OSError:
            result["saved_to"] = ""      # the export stays in the project; only the copy failed
        store.mutate(pid, lambda p: p["exports"].insert(0, result), internal=True)
    else:
        raise ValueError("不支持的任务类型")
    check()
    if finalize:
        message = "已完成"
        if kind == "produce":
            ratio = curation.composition(store.get_project(pid))["video_ratio"]
            burned = (store.get_project(pid).get("exports") or [{}])[0].get("subtitles", True)
            message = f"成片已完成 · 实际视频 {ratio:.0%} · " + ("字幕已烧录" if burned else "画面不带字幕（附 SRT 字幕文件）")
        store.job_update(jid, status="completed", progress=100, message=message)


_current = {"job": None, "title": ""}


def run_once(stop=None, lane="main") -> bool:
    job = store.claim(lane)
    if not job:
        return False
    throttle.waited.clear()
    throttle.video_started(last=lane == "background")
    _current.update(job=job["id"], kind=job["kind"])
    if _current.get("name"):
        try:
            _write_diag(_current["name"])     # at once: a crash in the first seconds is still traced to this job
        except Exception:
            pass
    began, wall_began = time.monotonic(), time.time()
    power_info = {}
    try:
        with power.keep_awake() as power_info:
            process(job, stop)
    except media.Cancelled:
        store.job_update(job["id"], status="cancelled", message="已取消，已完成的结果保留")
    except (costs.BudgetExceeded, costs.PricingUnavailable) as exc:
        store.job_update(job["id"], status="failed", message="费用保护已停止模型请求，进度已保存", error=str(exc))
    except Exception as exc:
        store.job_update(job["id"], status="failed", message="任务未完成，可重试", error=str(exc)[:2200])
    finally:
        throttle.video_ended()
        _current.update(job=None)
        ended = time.time()
        run = {"job_id": job["id"], "started_at": wall_began, "finished_at": ended,
               "elapsed_wall_seconds": round(ended-wall_began, 3),
               "elapsed_monotonic_seconds": round(time.monotonic()-began, 3),
               "power_assertion": power_info}
        final_job = store.get_job(job["id"])
        def save_execution(p):
            p["execution_info"] = run
            if job["kind"] == "produce":
                workflow = p.get("workflow_info") or {}
                if workflow.get("job_id") != job["id"]:
                    workflow = {"job_id": job["id"], "stages": {}, "stage_wall_seconds": {}}
                workflow.update(state=final_job["status"])
                if final_job.get("error"):
                    workflow["reason"] = final_job["error"]
                elif final_job["status"] == "cancelled":
                    workflow["reason"] = final_job["message"]
                p["workflow_info"] = workflow
        store.mutate(job["project_id"], save_execution, internal=True)
    return True


_memory_cap = []


def _memory_videos() -> int:
    if not _memory_cap:
        _memory_cap.append(throttle.max_videos())
    return _memory_cap[0]


def _write_diag(name: str) -> None:
    import json
    target = paths.DATA / "diag" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.temporary(target)
    temporary.write_text(json.dumps({"job": _current.get("job"), "pid": os.getpid(), "updated": time.time(),
                                     "waited": {k: round(v, 1) for k, v in dict(throttle.waited).items()}}), "utf-8")
    paths.publish(temporary, target)


def _report_waits(name: str) -> None:
    """Every 10 s: which job this worker is on and what it has queued for so far (shared limits, B站 pauses),
    for the diagnostics text and for requeueing the job if this process dies."""
    _current["name"] = name
    while True:
        try:
            _write_diag(name)
        except Exception:
            pass
        time.sleep(10)


CRASH_RETRIES = 2


def requeue_after_crash(name: str, pid: int) -> str:
    """A worker process ended while making a video (a native crash leaves no Python error behind): mark the job
    interrupted and queue it again, at most CRASH_RETRIES times. Returns a line for the launcher window."""
    import json
    try:
        item = json.loads((paths.DATA / "diag" / f"{name}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return ""
    if item.get("pid") != pid or not item.get("job"):
        return ""
    try:
        job = store.get_job(item["job"])
    except KeyError:
        return ""
    if job["status"] != "running":
        return ""
    tries = int((job.get("payload") or {}).get("crash_retries", 0))
    if tries >= CRASH_RETRIES:
        store.job_update(job["id"], status="interrupted", message="制作进程连续意外退出，请稍后手动重试",
                         error="制作进程意外退出")
        return f"制作进程意外退出（第 {tries + 1} 次），这条视频已停止，可稍后重试"
    store.job_update(job["id"], status="interrupted", message="制作进程意外退出，已自动重新排队", error="制作进程意外退出")
    store.enqueue(job["project_id"], job["kind"], (job.get("payload") or {}) | {"crash_retries": tries + 1})
    return "制作进程意外退出，这条视频已自动重新排队"


def video_workers() -> int:
    """How many videos are made at the same time (setting video_workers)."""
    try:
        return max(1, min(throttle.MAX_VIDEOS, _memory_videos(), int(config.read().get("video_workers") or 1)))
    except (TypeError, ValueError):
        return 1


def loop(stop=None, lane="main", slot=0, shared=None, recover=True) -> None:
    """One worker. The main lane runs one process per video made at the same time (slot 0, 1, ...); a slot
    above the video_workers setting stays idle. The launcher recovers the queue before starting any slot."""
    throttle.install(shared, throttle.MAX_VIDEOS if lane == "background" else slot)
    parent = multiprocessing.parent_process()
    if parent is not None:
        # A worker process started by the launcher: its encodes must not make the computer sluggish.
        power.lower_priority()
        threading.Thread(target=_report_waits, args=(f"{lane}-{slot}",), daemon=True).start()
        def watch_parent():
            while parent.is_alive():
                time.sleep(.5)
            # A forcibly closed launcher cannot run its cleanup. Do not leave a worker
            # consuming new jobs with obsolete code or credentials after it exits.
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()
    ownership = None
    while ownership is None:
        if stop is not None and stop.is_set():
            return
        ownership = instance.acquire("worker-background.lock" if lane == "background" else
                                     "worker.lock" if slot == 0 else f"worker-{slot + 1}.lock")
        if ownership is None:
            time.sleep(.2)
    if lane == "background":
        # Index building is many small picture descriptions; in its own process it may use more parallel
        # model calls without taking any from video production.
        execution.MODEL_SLOTS = threading.BoundedSemaphore(10)
    try:
        store.init()
        if recover:
            store.recover(lane)
        while stop is None or not stop.is_set():
            if slot and slot >= video_workers():
                time.sleep(1)
                continue
            if not run_once(stop, lane):
                time.sleep(0.5)
    finally:
        ownership.close()
