"""Validate ordinary B-roll judgments and explicit, stricter shot requirements."""
from __future__ import annotations

import re

from . import editorial_review, review_evidence


def requirements(shot, frames=()):
    source_requirements = editorial_review.requirements(shot)
    if source_requirements:
        origin = (shot.get("editorial_contract") or {}).get("full_script", "")
        activity = r"会议|会场|论坛|主席台|演播室|主持人|发布会|致辞"
        if editorial_review.alignment_policy(shot) == "strict_source" and not re.search(activity, origin):
            for frame in frames:
                # A short cut to a different activity must receive its own
                # grounded judgment, instead of disappearing in a majority vote.
                if re.search(activity, str(frame.get("objects", ""))+str(frame.get("action", ""))):
                    index = frame["frame_index"]
                    source_requirements[f"frame_scope_{index}"] = (
                        f"第{index}帧出现会议/演播等另一活动迹象。须单独判断这帧实际活动是否服务当前原稿；"
                        "不能用其他帧的劳动场景替它通过，也不能只因时间短忽略。只引用本帧证据。")
        return source_requirements
    plan = shot.get("visual_plan")
    if not isinstance(plan, dict) or plan.get("mode") != "single":
        return {}
    primary = plan.get("primary")
    if not isinstance(primary, dict):
        return {}
    return {key: primary[key].strip() for key in ("subject", "action", "scene")
            if isinstance(primary.get(key), str) and primary[key].strip()}


def validate(shot, judgments, frames):
    # New B-roll relations require the code-owned policy. Old strict judgments
    # remain readable, but an old conflict is not silently turned into approval.
    if isinstance(judgments, list):
        from . import review_schema
        for row in judgments:
            if not isinstance(row, dict) or row.get("key") != "source_alignment" or "source_relation" not in row:
                continue
            relation = row["source_relation"]
            required = {"required_activity", "observed_activity", "relationship"}
            if (not isinstance(relation, dict) or set(relation) not in (required, required | {"policy"})
                    or any(not isinstance(relation.get(key), str) or not relation[key].strip() for key in relation)
                    or review_schema.SOURCE_RELATIONS.get(relation["relationship"]) != row.get("status")):
                return "unknown", "原稿活动关系格式或结论不一致，不能自动修复为通过", []
            policy = relation.get("policy", "strict_source")
            if policy == "related_background":
                allowed = review_schema.BACKGROUND_SOURCE_RELATIONS
                if editorial_review.alignment_policy(shot) != policy:
                    return "unknown", "指定人物或取景要求不能按普通背景资料通过", []
            elif policy == "strict_source":
                allowed = review_schema.STRICT_SOURCE_RELATIONS
            else:
                return "unknown", "未知画面匹配政策", []
            if relation["relationship"] not in allowed:
                return "unknown", "画面关系不符合本次匹配政策", []
    expected = requirements(shot, frames)
    if not expected:
        return "supported", "", []
    if (not isinstance(judgments, list) or len(judgments) != len(expected)
            or any(not isinstance(row, dict) or not isinstance(row.get("key"), str) for row in judgments)
            or {row["key"] for row in judgments} != set(expected)):
        return "unknown", "画面审核缺少必需判断项，不能确认本段内容与字幕是否合适", []
    checked = []
    for row in judgments:
        if (row.get("status") not in ("supported", "unknown", "conflict")
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not isinstance(row.get("evidence"), list)):
            return "unknown", "选定画面的条件核验格式无效", []
        citations = row["evidence"]
        if (any(not review_evidence.condition_grounded(row["key"], citation, frames) for citation in citations)
                or row["status"] in ("supported", "conflict") and not citations):
            return "unknown", "选定画面的条件核验没有对应实际帧的逐字依据", []
        if row["key"].startswith("frame_scope_") and any(
                citation["frame_index"] != int(row["key"].removeprefix("frame_scope_")) for citation in citations):
            return "unknown", "转场核验必须引用该帧自身，不能借用其他画面", []
        checked.append({"key": row["key"], "required": expected[row["key"]],
                        "status": row["status"], "reason": row["reason"][:500], "evidence": citations,
                        **({"source_relation": row["source_relation"]} if "source_relation" in row else {})})
    # These are not model citations. They are exact local checks against a
    # validated source command and the text that will actually be rendered.
    checked.extend(editorial_review.postproduction_text_checks(shot))
    failed = next((row for row in checked if row["status"] == "conflict"), None)
    failed = failed or next((row for row in checked if row["status"] != "supported"), None)
    if failed:
        label = {"subject": "主体", "action": "动作", "scene": "场景",
                 "source_alignment": "原稿内容", "caption_context": "源字幕"}.get(failed["key"], "原文明确条件")
        return failed["status"], f"选定画面的{label}未通过：" + failed["reason"], checked
    return "supported", "", checked
