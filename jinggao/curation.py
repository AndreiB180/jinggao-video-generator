"""Editorial checks based on source evidence and actual playable durations."""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse

from . import context_labels, privacy


def invalidate(shot: dict) -> None:
    """A changed time window or new review requires fresh editorial confirmation."""
    shot.update(approved=False, verification=None, source_review_note="", source_review_url="")


def preferred_fit(shot: dict, asset: dict, aspect: str) -> str:
    """Keep the inspected scene visible when auto-matching across orientations."""
    if shot.get("fit_origin") == "manual":
        return shot.get("fit", "cover")
    width, height = asset.get("width", 0), asset.get("height", 0)
    if not width or not height:
        return shot.get("fit", "cover")
    target = 9/16 if aspect == "9:16" else 16/9
    ratio = width / height / target
    if asset.get("kind") == "video":
        # Footage fills the frame (cropping a band) unless its shape is far off, e.g. a vertical video in a
        # landscape film: customers read blurred side or top bands as "not filled".
        return "cover" if .7 <= ratio <= 1.43 else "contain"
    # A picture near the film's shape fills it (a 4:3 photo loses a band top and bottom); a tall one sits on
    # a blurred copy of itself.
    return "cover" if .74 <= ratio <= 1.36 else "contain"


def needs_identity(shot: dict) -> bool:
    return (bool(shot.get("persons")) or shot.get("evidence_mode") in ("person", "event")
            or bool(shot.get("event") and shot.get("event_scope", "exact") == "exact"))


def has_manual_source(shot: dict) -> bool:
    url = urlparse(shot.get("source_review_url", ""))
    return (len(shot.get("source_review_note", "").strip()) >= 8
            and url.scheme in ("http", "https") and bool(url.hostname)
            and not url.username and not url.password)


def express_result(shot: dict) -> dict:
    """Express-pipeline evidence; ordinary shots need no per-frame audit chain."""
    result = shot.get("verification") or {}
    return result.get("express") if isinstance(result.get("express"), dict) else {}


def express_project(project: dict) -> bool:
    """Planned by the express pipeline: ordinary shots need no paid per-frame audit chain."""
    return (project.get("plan_info") or {}).get("mode") == "express"


# Workbench edits that change how the chosen footage is framed, timed or labelled, not what it is.
FRAMING_FIELDS = {"fit", "overlay_label", "clip_start", "duration", "visual", "subject", "location", "era",
                  "exclusions", "preferred_media", "query", "source_review_note", "source_review_url"}
IDENTITY_FIELDS = {"persons", "event", "evidence_mode", "event_scope"}


def express_edit(project: dict, shot: dict, before: dict, changed: set) -> bool:
    """Keep an express pick across a workbench edit instead of demanding a classic re-audit.
    A changed person/event target drops the identity result; any edit beyond framing needs a new confirm."""
    if not express_project(project) or not express_result(before):
        return False
    verification = deepcopy(before["verification"])
    if changed & IDENTITY_FIELDS:
        verification["express"]["identity"] = ""
        verification["identity_evidence"] = None
        # The editor re-sends the old source note; it described the previous person or event.
        if not changed & {"source_review_note", "source_review_url"}:
            shot.update(source_review_note="", source_review_url="")
    if changed - FRAMING_FIELDS:
        # New narration or a new target: the footage needs a new look (a re-match, or the user's confirm).
        verification["needs_source_check"] = True
    shot.update(verification=verification)
    shot["approved"] = bool(before.get("approved")) and not (changed - FRAMING_FIELDS) and identity_ready(shot)
    return True


def keep_express(project: dict, shot: dict, *, recheck: bool = False) -> bool:
    """A retimed, re-voiced or metadata-edited express pick keeps its pick (and a manual choice stays
    the user's) instead of falling back to the paid classic audit. It needs a new look (a re-match on
    produce, or the user's confirm) when asked to, or when the clip no longer covers the shot."""
    if not express_project(project) or not express_result(shot):
        return False
    asset = next((a for a in project.get("assets", []) if a["id"] == shot.get("selected_asset")), None)
    short = bool(asset and asset.get("kind") == "video" and float(shot.get("duration") or 0)
                 > float(asset.get("duration") or 0) - float(shot.get("clip_start") or 0) + .16)
    if recheck or short:
        shot["verification"]["needs_source_check"] = True
        shot["approved"] = False
    return True


def manual_choice(shot: dict) -> bool:
    """The user picked this footage by hand; automatic re-matching must not replace it (also after an
    AI check replaced its verification)."""
    return bool(shot.get("selected_asset")) and (shot.get("manual_asset") == shot.get("selected_asset")
                                                  or express_result(shot).get("lane") == "manual")


def manual_pick(project: dict, shot: dict, asset: dict) -> None:
    """The user's own pick in an express project counts as confirmed, except where a named
    person or event must be shown: that still needs the source note and link."""
    if not express_project(project) or not asset or asset.get("kind") not in ("video", "image"):
        return
    shot.update(verification={"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
                              "needs_source_check": False, "reason": "手动选择的画面",
                              "express": {"version": 0, "lane": "manual", "identity": "", "flags": []}},
                search_note="手动选择的画面", manual_asset=asset.get("id"))
    shot["approved"] = identity_ready(shot)


def identity_ready(shot: dict) -> bool:
    if not needs_identity(shot):
        return True
    express = express_result(shot)
    if express:
        # Supported: source text names the person and the pick shows a person
        # as subject. not_shown: people-free context footage, never someone else.
        evidence = (shot.get("verification") or {}).get("identity_evidence") or {}
        checked = evidence.get("status") == "supported" and not (shot.get("verification") or {}).get("selection_blocked", True)
        return express.get("identity") in ("supported", "not_shown") or has_manual_source(shot) or checked
    evidence = (shot.get("verification") or {}).get("identity_evidence") or {}
    return (evidence.get("status") == "supported"
            and not (shot.get("verification") or {}).get("selection_blocked", True)) or has_manual_source(shot)


def approve_verified(project: dict) -> int:
    """Approve only completed, conflict-free reviews; leave every gap visible."""
    count = 0
    assets = {a["id"]: a for a in project.get("assets", [])}
    for shot in project["shots"]:
        result = shot.get("verification") or {}
        if selection_error(project, shot, assets.get(shot.get("selected_asset"), {})):
            shot["approved"] = False
            continue
        if (not shot.get("locked") and shot.get("selected_asset") and result.get("verdict") == "match"
                and result.get("scene_alignment") == "match"
                and not result.get("selection_blocked", True) and not result.get("needs_source_check", True)
                and identity_ready(shot)):
            shot["approved"] = True
            count += 1
    return count


def selection_error(project, shot, asset, *, verify_files=False):
    from . import editorial_contract, editorial_review, frame_contract, frame_sampling, planner
    if express_result(shot):
        if not asset or asset.get("kind") not in ("video", "image"):
            return "所选素材已失效，请重新匹配"
        if verify_files:
            from . import paths
            try:
                if not paths.file_path(project["id"], asset["path"]).exists():
                    return "素材文件缺失，请重新匹配"
            except ValueError:
                return "素材文件路径无效，请重新匹配"
        return ""
    error = privacy.selection_error(shot, asset) or context_labels.selection_error(
        shot, asset, project_id=project.get("id") if verify_files else None)
    if error:
        return error
    result = shot.get("verification") or {}
    audit = result.get("identity_audit") or {}
    version = audit.get("version")
    if (type(version) is int and version < planner.FRAME_AUDIT_VERSION
            and editorial_contract.for_shot(shot, project) and not shot.get("locked")
            and not has_manual_source(shot)):
        return "原稿与实际活动的核验规则已更新，需重新核验已有素材；原配音和素材文件已保留"
    if asset.get("kind") == "video" and (result.get("visual_quality") or {}).get("representation") == "uncertain":
        return "片段含无法确认类型的帧，需检查空白或片尾转场后重选"
    frames = (result.get("identity_audit") or {}).get("frame_observations") or []
    if not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames):
        return "实际帧核验记录无效，需要重新核验"
    if not frames:
        return frame_sampling.selection_error(project, shot, asset)
    separate = {key for key in frame_contract.requirements(shot, frames) if key.startswith("frame_scope_")}
    checked = {row.get("key") for row in (result.get("identity_audit") or {}).get("primary_checks", [])}
    if not separate <= checked:
        return "片段含会议或演播等另一活动，尚未单独核验该转场帧"
    if editorial_review.alignment_policy(shot, project) == "related_background":
        # Complete caption semantics already passed the ordinary content audit.
        # Do not turn a different background reporting year into a new veto.
        return frame_sampling.selection_error(project, shot, asset)
    try:
        context = context_labels.review_context(shot, asset, frames)
    except ValueError:
        context = {}
    period = context.get("source_fields", {}).get("published_at", "")[:4] if context.get("source_bound") else ""
    labels = [{"frame_index": f["frame_index"], "quote": text} for f in frames for text in f.get("text", [])]
    return (planner.statistical_caption_conflict(shot, planner.narrative_context(project, shot), labels, report_year=period)
            or frame_sampling.selection_error(project, shot, asset))


def reviewed_label_fields(result):
    """Install exactly the disclosure that passed the ordinary content audit."""
    label = result.get("context_label")
    accepted = (result.get("verdict") == "match" and not result.get("selection_blocked", True)
            and not result.get("needs_source_check", True)
            and (result.get("identity_audit") or {}).get("status") == "supported")
    if accepted and label and result.get("reviewed_overlay_label") == label.get("text"):
        return {"overlay_label": label["text"], "overlay_label_origin": "source_context",
                **({"match_preview": result["review_preview"]} if result.get("review_preview") else {})}
    if (accepted and result.get("reset_source_label") is True
            and (result.get("identity_audit") or {}).get("reset_source_label") is True
            and result.get("reviewed_overlay_label") == "资料画面"):
        return {"overlay_label": "资料画面", "overlay_label_origin": "generated"}
    return {}


def image_in_use(project, asset, shot_id):
    """Avoid automatically repeating the same photo or exact source frame."""
    if asset.get("kind") != "image":
        return False
    def keys(item):
        result = {(name, item[name]) for name in ("id", "path", "origin_image_url") if item.get(name)}
        if item.get("asset_id"):
            result.add(("id", item["asset_id"]))
        origin = item.get("derived_from_video") or {}
        if origin.get("source_url") and isinstance(origin.get("source_at"), (int, float)):
            result.add(("source_frame", origin["source_url"], round(origin["source_at"], 2)))
        return result
    wanted = keys(asset)
    assets = {a["id"]: a for a in project.get("assets", [])}
    return any(wanted & keys(assets.get(s.get("selected_asset"), {}))
               for s in project["shots"] if s["id"] != shot_id)


def composition(project: dict) -> dict:
    assets = {a["id"]: a for a in project["assets"]}
    total = video = images = missing = frozen = 0.0
    illustration = unclassified = legacy = 0.0
    unresolved, short_shots, low_res, illustration_shots, unclassified_shots = [], [], [], [], []
    for index, shot in enumerate(project["shots"], 1):
        duration = float(shot.get("duration") or 0)
        if not project.get("full_audio") and shot.get("audio"):
            duration = float(shot["audio"].get("duration") or duration)
        total += duration
        asset = assets.get(shot.get("selected_asset"))
        if not asset:
            missing += duration
        elif asset["kind"] == "video":
            playable = max(0, float(asset.get("duration") or 0)-float(shot.get("clip_start") or 0))
            used = min(duration, playable)
            quality = (shot.get("verification") or {}).get("visual_quality")
            if quality is None:
                # Older projects retain their duration calculation, not a new
                # claim that their pixels have been classified as live action.
                video += used
                legacy += used
            elif isinstance(quality, dict) and quality.get("representation") in ("animation", "mixed"):
                illustration += used
                illustration_shots.append(index)
            elif isinstance(quality, dict) and quality.get("representation") in ("live_action", "screen_capture", "recorded_mix"):
                video += used
            else:
                unclassified += used
                unclassified_shots.append(index)
            frozen += max(0, duration-playable)
            if duration-playable > .16:
                short_shots.append(index)
            if min(asset.get("width", 0), asset.get("height", 0)) < project.get("min_video_height", 720):
                low_res.append(index)
        elif asset["kind"] == "image":
            images += duration
        else:
            missing += duration
        if not identity_ready(shot):
            unresolved.append(index)
    policy = project.get("media_policy", "mixed")
    target = 1.0 if policy == "video_only" else float(project.get("min_video_ratio", .8))
    ratio = video/total if total else 0
    return {"total_seconds": round(total, 3), "video_seconds": round(video, 3),
            "illustration_video_seconds": round(illustration, 3),
            "unclassified_video_seconds": round(unclassified, 3), "legacy_video_seconds": round(legacy, 3),
            "image_seconds": round(images, 3), "missing_seconds": round(missing, 3),
            "frozen_seconds": round(frozen, 3), "video_ratio": round(ratio, 4),
            "target_video_ratio": target, "meets_target": policy == "mixed" or ratio+.004 >= target,
            "identity_pending": unresolved, "short_video_shots": short_shots, "low_resolution_shots": low_res,
            "illustration_video_shots": illustration_shots, "unclassified_video_shots": unclassified_shots}


def validate_export(project: dict, *, image_motion: bool = True) -> dict:
    if project.get("script_changed"):
        raise ValueError("原稿已修改，当前分镜仍对应旧文稿。请重新规划分镜后再导出")
    report = composition(project)
    assets = {a["id"]: a for a in project["assets"]}
    for index, shot in enumerate(project["shots"], 1):
        error = selection_error(project, shot, assets.get(shot.get("selected_asset"), {}), verify_files=True)
        if error:
            raise ValueError(f"镜头 {index} 的资料说明需要重新核验：{error}")
    if report["identity_pending"]:
        raise ValueError("人物或事件依据尚未确认：镜头 " + "、".join(map(str, report["identity_pending"]))
                         + "。请核对字幕/计分牌与原始来源，或填写人工核对的来源链接和说明")
    import re
    if any(needs_identity(s) and s.get("event_scope") == "background"
           and not re.search(r"资料|示意", s.get("overlay_label", "")) for s in project["shots"]):
        raise ValueError("资料画面需要叠加“资料”或“示意”说明，不能冒充本次事件")
    if project.get("media_policy", "mixed") != "mixed":
        if not report["meets_target"]:
            raise ValueError(f"计入方案目标的视频仅占 {report['video_ratio']:.0%}，当前方案要求至少 "
                             f"{report['target_video_ratio']:.0%}。请补足实拍/录屏视频；"
                             "动画/混合示意与类型待核视频不计入目标，图片运镜不计作视频素材")
        if report["short_video_shots"]:
            raise ValueError("视频片段短于口播：镜头 " + "、".join(map(str, report["short_video_shots"]))
                             + "。请换用足够长的片段，不能靠末帧定格补足动态时长")
        if report["low_resolution_shots"]:
            raise ValueError("视频源清晰度未达方案要求：镜头 " + "、".join(map(str, report["low_resolution_shots"]))
                             + "。请替换素材，或在制作方案中明确调整最低清晰度")
        if report["image_seconds"] and not image_motion:
            raise ValueError("视频优先方案的少量图片需要开启运镜")
    return report
