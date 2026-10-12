"""Ground archival labels in the selected source and actual frame observations.

This module performs no search or model calls and never approves footage. The
existing text audit may propose an extractive label and must review that exact
label together with every original caption. Successful compilation proves the
references and binding, not that the scene, action, identity or story is suitable.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

from . import identity, neutral_observer


VERSION = "source_context_label_v1"
MAX_LABEL = 30
SOURCE_FIELDS = ("title", "description", "published_at", "publisher")
SCOPES = {"time", "place", "project"}

# Append to the existing text judge, never to the blind image observer. The
# caller continues to enforce all original scene/person/quality requirements.
PROTOCOL = """overlay_label是最终将烧录的后期说明，不能当作原片文字、身份或动作证据。
context_label_context中的captions列出全部去重原片字幕，不能仅引用“资料”、台标或有利的衣服文字而忽略其他字幕。
资料说明只能交代来源报道的时间、地点或项目范围，不能修复错误主体、动作、特定人物/建筑、广告剧情或无关故事；现有source_alignment、caption_context及所有原文条件仍须独立通过。
发布日期只能称报道时间；报道地点、联盟成立地点不证明当前镜头的拍摄地点。原片已有“资料”也不能证明它属于原稿所指现场。
如果无需新标签，context_label返回null。需要具体说明且证据足够时，可在本次响应提出context_label，但必须按以下固定格式先组成text，并审核这个最终text与全部原字幕组合后的含义。
source_quotes为1–2项{field:"title|description",quote:"对应source_fields字段中的完整连续短语"}；不改字、不补全专名、不借其他来源。短语以“·”连接，接“资料（来源”+source_fields.publisher+“）”。
report_period为none|year|month|day；非none时在前面加来源published_at的“YYYY年报道·”/“YYYY年M月报道·”/“YYYY年M月D日报道·”，不是拍摄日期。完整text最多30字，不能截断缺失的身份/地域范围来凑长度。
frame_evidence须逐字引用当前帧objects/action及text，证明所谈的实际主体与原字幕是什么，不用标题补出运行或人物。至少一条objects/action和一条text。
caption_checks对每个caption_id恰好一项：{caption_id,status,scope,reason}。status为compatible|scope_resolved|unknown|conflict|unrelated_story；scope仅time|place|project或空。只有确实只是范围悬空的文字才能scope_resolved。
无关人物关系、联盟/平台的独立发言、广告或其他故事不能改称project范围问题；需要更换画面/时段，不能靠“资料画面”免责。任何unknown/conflict/unrelated_story仍不得通过。具体故宫/某人/某动作的明确要求不因标签而改变。
context_label对象只含text、source_quotes、report_period、frame_evidence、caption_checks；不含批准字段。标签无法消除疑问时保留原审核的unknown/conflict，不补造标签。
"""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path, check):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check()
            digest.update(chunk)
    return digest.hexdigest()


def review_context(shot, asset, frames, *, frame_paths=(), check=lambda: None):
    """Return model-safe references plus an immutable-selection fingerprint.

    Pass the actual ``frame_paths`` from the existing audit. Without those files
    the payload can still explain the current label, but no automatic label can
    be compiled. This deliberately does not inherit a previous clip's label.
    """
    check()
    if (not isinstance(frames, list) or not frames
            or any(not isinstance(f, dict) or type(f.get("frame_index")) is not int
                   or not isinstance(f.get("text"), list)
                   or any(not isinstance(f.get(key), str) for key in ("objects", "action")) for f in frames)
            or [f["frame_index"] for f in frames] != list(range(len(frames)))):
        raise ValueError("资料说明需要完整且编号不重复的实际帧观察")
    record, _ = identity.source_record(asset)
    source_bound = bool(record and record.get("evidence_origin") == "publisher_page"
                        and asset.get("source_verified") is True)
    fields = {key: str(record.get(key) or "") for key in SOURCE_FIELDS} if source_bound else {}
    captions, by_text = [], {}
    for frame in frames:
        for text in frame["text"]:
            if not isinstance(text, str) or not text.strip():
                continue
            if text not in by_text:
                row = {"caption_id": f"c{len(captions)+1}", "text": text, "frame_indices": []}
                by_text[text] = row
                captions.append(row)
            if frame["frame_index"] not in by_text[text]["frame_indices"]:
                by_text[text]["frame_indices"].append(frame["frame_index"])
    frame_paths = list(frame_paths)
    if frame_paths and len(frame_paths) != len(frames):
        raise ValueError("资料说明的实际帧文件与观察数量不符")
    hashes = [_file_hash(path, check) for path in frame_paths]
    start = float(asset.get("source_clip_start") or 0) + float(shot.get("clip_start") or 0)
    duration = float(shot.get("duration") or 0)
    if not all(math.isfinite(n) for n in (start, duration)) or start < 0 or duration <= 0:
        raise ValueError("资料说明的实际时间窗无效")
    shot_facts = {key: shot.get(key) for key in (
        "id", "narration", "visual", "visual_origin", "location", "era", "exclusions", "persons",
        "event", "event_scope", "evidence_mode", "fit", "editorial_contract")}
    binding = {"shot_id": shot.get("id"), "asset_id": asset.get("id"), "asset_path": asset.get("path"),
               "source_url": asset.get("source_url", ""), "source_digest": _digest(record),
               "source_start": round(start, 6), "duration": round(duration, 6),
               "shot_digest": _digest(shot_facts), "observations_digest": _digest(frames),
               "frame_sha256": hashes}
    return {"version": VERSION, "overlay_label": str(shot.get("overlay_label") or ""),
            "source_url": asset.get("source_url", ""), "source_bound": source_bound,
            "source_fields": fields, "captions": captions, "binding": binding}


def _report_prefix(value, period):
    if period == "none":
        return ""
    if not isinstance(period, str) or period not in {"year", "month", "day"}:
        raise ValueError("资料说明的报道时间粒度无效")
    match = re.match(r"^((?:19|20)\d{2})(?:[-/年](\d{1,2})(?:[-/月](\d{1,2}))?)?(?=$|[T\s日])", value)
    needed = {"year": 1, "month": 2, "day": 3}[period]
    if not match or any(not match[i] for i in range(1, needed+1)):
        raise ValueError("来源未给出所需的报道日期，不能猜拍摄日期")
    year, month, day = (int(x) if x else 1 for x in match.groups())
    try:
        date(year, month, day)
    except ValueError:
        raise ValueError("来源报道日期无效") from None
    return f"{year}年" + (f"{month}月" if needed >= 2 else "") + (f"{day}日" if needed == 3 else "") + "报道·"


def compile_label(context, proposal, frames):
    """Validate an existing judge's proposal; never promote its content verdict.

    All captions must be accounted for. Source/quote grounding cannot establish
    semantic suitability, so the result remains ``requires_content_review``.
    The caller must commit the exact text and the normal successful audit
    atomically; a compiled proposal alone must never enter approve_verified.
    """
    if (not isinstance(proposal, dict) or set(proposal) != {
            "text", "source_quotes", "report_period", "frame_evidence", "caption_checks"}):
        raise ValueError("资料说明提案字段无效")
    binding = context.get("binding") or {}
    if (context.get("version") != VERSION or not context.get("source_bound")
            or not binding.get("shot_id") or not binding.get("asset_id")
            or len(binding.get("frame_sha256", [])) != len(frames)
            or not frames or binding.get("observations_digest") != _digest(frames)):
        raise ValueError("资料说明未绑定同一原页及实际帧/时间窗")
    fields = context["source_fields"]
    publisher = fields.get("publisher", "")
    quotes = proposal["source_quotes"]
    if (not isinstance(quotes, list) or not 1 <= len(quotes) <= 2 or not publisher
            or len(publisher) > 20 or re.search(r"[\x00-\x1f{}\\]", publisher)):
        raise ValueError("资料说明缺少可引用的来源短语或发布者")
    parts = []
    for row in quotes:
        if (not isinstance(row, dict) or set(row) != {"field", "quote"}
                or not isinstance(row["field"], str) or row["field"] not in {"title", "description"}
                or not isinstance(row["quote"], str) or not 2 <= len(row["quote"]) <= MAX_LABEL
                or row["quote"] != row["quote"].strip()
                or re.search(r"[\x00-\x1f{}\\]", row["quote"])
                or row["quote"] not in fields.get(row["field"], "") or row["quote"] in parts):
            raise ValueError("资料说明短语不是该来源字段的逐字原文")
        parts.append(row["quote"])
    prefix = _report_prefix(fields.get("published_at", ""), proposal["report_period"])
    text = prefix + "·".join(parts) + f"资料（来源{publisher}）"
    if len(text) > MAX_LABEL or proposal["text"] != text:
        raise ValueError("资料说明必须与已审核的完整标签一致且不超过30字，不能截断")
    evidence = proposal["frame_evidence"]
    if (not isinstance(evidence, list) or not evidence
            or any(not neutral_observer.grounded(row, frames) for row in evidence)
            or not any(row.get("field") in {"objects", "action"} for row in evidence)
            or not any(row.get("field") == "text" for row in evidence)):
        raise ValueError("资料说明缺少当前实际画面与原字幕的逐字依据")
    expected = {row["caption_id"] for row in context["captions"]}
    checks = proposal["caption_checks"]
    if (not isinstance(checks, list) or not expected or len(checks) != len(expected)
            or any(not isinstance(row, dict) or not isinstance(row.get("caption_id"), str) for row in checks)
            or {row["caption_id"] for row in checks} != expected):
        raise ValueError("资料说明未逐项处理全部原片字幕，不能只引用资料角标")
    for row in checks:
        status, scope = row.get("status"), row.get("scope", "")
        if (not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not isinstance(status, str) or not isinstance(scope, str)
                or status not in {"compatible", "scope_resolved"}
                or scope not in (SCOPES if status == "scope_resolved" else {""})):
            raise ValueError("资料说明不能修复未知、错误动作、错人或无关故事字幕")
        if scope == "time" and not prefix:
            raise ValueError("涉及历史范围的字幕尚未写明来源报道时间")
    return {"version": VERSION, "text": text, "source_url": context["source_url"],
            "binding": deepcopy(binding), "proposal": deepcopy(proposal),
            "requires_content_review": True}


def is_current(label, context):
    """Fail closed if the label, selected source, window or observed frames changed.

    ``context`` must be newly built for the current selection. This only checks
    provenance freshness and never substitutes for its successful content audit.
    """
    return bool(isinstance(label, dict) and label.get("version") == VERSION
                and context.get("version") == VERSION and context.get("source_bound")
                and label.get("text") == context.get("overlay_label")
                and label.get("binding") == context.get("binding")
                and bool((label.get("binding") or {}).get("frame_sha256")))


def relative_statistics(frames):
    """Relative statistical captions need a visible, source-bound time scope.

    This is a request for dated review, not a semantic match or a finding that
    the statistics are false. Split OCR lines are considered in the same frame.
    """
    for frame in frames:
        text = " ".join(frame.get("text") or [])
        if (re.search(r"当前|今年|本年|目前|截至|预计今年", text)
                and re.search(r"业务量|处理量|处理能力|吞吐量|揽收|投递|产量|销量|同比|总量|增长|收入|支出|利润|就业|人口", text)
                and re.search(r"[0-9零〇一二三四五六七八九十百千万亿]+", text)):
            return True
    return False


def _source_period(context):
    if not context.get("source_bound"):
        raise ValueError("相对时间的统计字幕需要已核实原页的报道日期")
    value = context.get("source_fields", {}).get("published_at", "")
    for period in ("day", "month", "year"):
        try:
            return period, _report_prefix(value, period)
        except ValueError:
            continue
    raise ValueError("原页没有可核实的报道日期，不能猜测统计字幕所说的当前/今年")


def _discloses_period(context, text):
    _, prefix = _source_period(context)
    pattern = r"((?:19|20)\d{2})年(?:(\d{1,2})月)?(?:(\d{1,2})日)?"
    source = re.match(pattern, prefix).groups()
    dates = re.findall(pattern, text)
    return bool(dates and re.search(r"报道|资料", text)
                and all(all(not part or part == actual for part, actual in zip(item, source, strict=True))
                        for item in dates))


def report_date_label(context, shot):
    """Prepare an exact date disclosure *before* the ordinary content audit.

    Unlike a proposed place/project label, this adds only a publisher's report
    date. It claims no filming date, matching action or approval. The subsequent
    judge sees and reviews the exact text that selection will commit.
    """
    from . import editorial_contract
    period, prefix = _source_period(context)
    existing = context.get("overlay_label", "")
    # An explicit correctly dated manual disclosure can be judged as written.
    if shot.get("overlay_label_origin") != "source_context" and _discloses_period(context, existing):
        return None
    if (shot.get("locked") or shot.get("overlay_label_origin") == "manual"
            or editorial_contract.for_shot(shot).get("onscreen_text")
            or existing not in {"", "资料", "资料画面"}
            and shot.get("overlay_label_origin") != "source_context"):
        raise ValueError("统计字幕需要注明来源报道时间；请保留原文叠字要求并核对资料说明")
    binding = context.get("binding") or {}
    if not binding.get("frame_sha256"):
        raise ValueError("报道时间说明尚未绑定实际抽帧")
    text = "资料画面 · " + prefix.rstrip("·")
    if len(text) > MAX_LABEL:
        raise ValueError("报道时间说明过长")
    return {"version": VERSION, "kind": "report_date", "text": text,
            "source_url": context["source_url"], "binding": deepcopy(binding),
            "report_period": period, "requires_content_review": True}


def selection_error(shot, asset, *, project_id=None):
    """Check saved disclosure against current source, edit and actual frames."""
    from . import editorial_review
    verification = shot.get("verification") or {}
    audit = verification.get("identity_audit") or {}
    frames = audit.get("frame_observations") or []
    label = verification.get("context_label")
    needs_period = editorial_review.alignment_policy(shot) != "related_background" and relative_statistics(frames)
    if shot.get("overlay_label_origin") == "source_context" and not label:
        return "自动资料说明未绑定当前素材，需要重新核验"
    if not label and not needs_period:
        return ""
    try:
        frame_paths = []
        if project_id is not None and label:
            from . import paths
            preview = verification.get("review_preview") or shot.get("match_preview") or {}
            if preview.get("asset_id") != asset.get("id"):
                raise ValueError("资料说明对应的预览素材已改变")
            frame_paths = [paths.file_path(project_id, row["path"]) for row in preview.get("frames", [])]
            if len(frame_paths) != len(frames):
                raise ValueError("资料说明缺少完整实际帧")
        context = review_context(shot, asset, frames, frame_paths=frame_paths)
        if label:
            if project_id is None:
                # In-memory checks cover source/edit/window drift. Export also
                # hashes the actual saved frame files above.
                context["binding"]["frame_sha256"] = (label.get("binding") or {}).get("frame_sha256", [])
            if (verification.get("reviewed_overlay_label") != shot.get("overlay_label")
                    or not is_current(label, context)):
                raise ValueError("资料说明、来源或时间窗已改变，需要重新核验")
        if needs_period:
            if not _discloses_period(context, str(shot.get("overlay_label") or "")):
                raise ValueError("原片含当前/今年等统计字幕，尚未明确来源报道时间")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return str(exc)
    return ""
