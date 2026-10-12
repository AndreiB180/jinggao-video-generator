"""Resolve shot-scoped references to unchanged blind frame observations.

The judge selects an observation; Python supplies its literal quotation. This
avoids turning a correctly selected action into a failed review because the
model abbreviated its wording. Unknown or cross-shot references still fail.
"""
from __future__ import annotations

import json


def output_limit(content, *, structured=False):
    """Reserve output for the actual review work, retaining the legacy ceiling.

    This is a generous generation limit, not a guaranteed token bound. All
    frames, source conditions and citation checks stay in the input. Invalid
    packets retain the original ceiling; a truncated response still fails the
    ordinary parser and is charged, without an automatic retry.
    """
    if not isinstance(content, list) or not content:
        return 12000
    maximum = 512
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            return 12000
        try:
            item = json.loads(part["text"])
        except (KeyError, TypeError, ValueError):
            return 12000
        if not isinstance(item, dict):
            return 12000
        checks = item.get("selected_visual_requirements")
        frames = item.get("frame_observations")
        people = item.get("requested_people")
        references = item.get("observation_references")
        if (not isinstance(checks, dict) or not checks
                or not isinstance(frames, list) or not frames
                or not isinstance(people, list)
                or not isinstance(references, list) or not references):
            return 12000
        # Named checks contain a short reason and exact observation refs.
        # Per-frame/name margins cover summary evidence and identity labels.
        maximum += 256 + 384 * len(checks) + 96 * len(frames) + 256 * len(people)
    return min(12000, max(2048 if structured else 4096, ((maximum + 255) // 256) * 256))


PROTOCOL = """\n本次引用格式使用 observation_references 中的 ref，覆盖前述手抄引用格式。
每个 ref 只属于当前 shot_id 的某一实际帧和字段，text_index 指向该帧 text 数组的一项。按观察事实选取引用，不改写观察，不借另一镜的 ref。
evidence 和 primary_checks[].evidence 均返回 [{"ref":"原样复制本镜ref"}]。
引用对象只写ref，不重复frame_index、field或quote；事实原文由程序按ref提取。
labels 仅为指定人物的姓名关系返回 [{"ref":"本镜text字段ref","relation":"on_subject|scoreboard_participant|headline|reference_only|other_subject"}]；没有指定人物或没有可用姓名文字时返回 []。
若提供field=text_absence的ref，它只表示本镜所有采样帧的text数组均为空（没有记录到可读文字），不是保证无水印。caption_context判断无源文字时应引用此ref，仍需核对实际overlay_label；不能空着evidence。此ref只可用于caption_context，不能证明主体、动作、身份或整段内容相符。
各项 status、reason、primary_checks 的逐项要求保持不变。引用只是事实定位，不代表该事实足以支持原稿；错误动作、无关字幕、其他人物仍须 conflict/unknown。
"""


def text_absence(frames):
    """Explicit absence in all completed observations, never missing OCR data."""
    if (not isinstance(frames, list) or not frames
            or any(not isinstance(frame, dict) or type(frame.get("frame_index")) is not int
                   or frame["frame_index"] != i or frame.get("text") != []
                   or any(not isinstance(frame.get(key), str) or not frame[key].strip()
                          for key in ("objects", "action")) for i, frame in enumerate(frames))):
        return None
    return {"field": "text_absence", "frame_indices": list(range(len(frames))), "quote": []}


def condition_grounded(key, citation, frames):
    from .neutral_observer import grounded
    if isinstance(citation, dict) and citation.get("field") == "text_absence":
        expected = text_absence(frames)
        return (key == "caption_context" and expected is not None
                and set(citation) == set(expected)
                and isinstance(citation.get("frame_indices"), list)
                and all(type(index) is int for index in citation["frame_indices"])
                and citation == expected)
    return grounded(citation, frames)


def references(shot_id, frames):
    public, lookup = [], {}
    for frame in frames:
        index = frame["frame_index"]
        for field in ("objects", "action", "text"):
            values = frame[field] if field == "text" else [frame[field]]
            for offset, quote in enumerate(values):
                if not quote.strip():
                    continue
                ref = f"{shot_id}:f{index}:{field}" + (f":{offset}" if field == "text" else "")
                location = {"ref": ref, "frame_index": index, "field": field}
                if field == "text":
                    location["text_index"] = offset
                public.append(location)
                lookup[ref] = {"frame_index": index, "field": field, "quote": quote}
    absent = text_absence(frames)
    if absent:
        ref = shot_id + ":text_absence"
        public.append({"ref": ref, **absent})
        lookup[ref] = absent
    return public, lookup


def resolve(row, lookup):
    """Retain invalid references for the normal grounding validator to reject.

    Redundant literal metadata must exactly match the referenced observation.
    Legacy literal citations remain subject to the normal grounding validator.
"""
    def citations(value, *, labels=False):
        if not isinstance(value, list):
            return value
        result = []
        for citation in value:
            if not isinstance(citation, dict) or "ref" not in citation:
                result.append(citation)
                continue
            keys = {"ref", "relation"} if labels else {"ref"}
            metadata = {"frame_index", "frame_indices", "field", "quote"}
            ref = citation.get("ref")
            target = lookup.get(ref) if isinstance(ref, str) else None
            if (not keys <= set(citation) or set(citation) - keys - metadata or target is None
                    or labels and target["field"] != "text"
                    or any(key not in target or type(citation[key]) is not type(target[key]) or citation[key] != target[key]
                           for key in metadata & citation.keys())):
                # Do not let a bad ref fall back to otherwise groundable
                # literal fields and silently change its evidence binding.
                result.append({"ref": ref})
                continue
            result.append(target | ({"relation": citation["relation"]} if labels else {}))
        return result

    checked = row | {"evidence": citations(row.get("evidence")),
                     "labels": citations(row.get("labels"), labels=True)}
    if isinstance(row.get("primary_checks"), list):
        checked["primary_checks"] = [
            item | {"evidence": citations(item.get("evidence"))} if isinstance(item, dict) else item
            for item in row["primary_checks"]]
    return checked
