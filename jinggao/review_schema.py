"""Fixed text-review keys and scoped references; all semantic gates stay local.

The source comparison is explicit before the final verdict. Input facts and
scope remain complete; declared topic-only similarity cannot approve footage.
"""
from copy import deepcopy
import json
from urllib.parse import urlparse

from .pixel_results import unique_object

MODELS = {"qwen3.7-flash", "qwen3.7-flash-2026-07-15"}
STATES = ["supported", "unknown", "conflict"]
RELATIONS = ["on_subject", "scoreboard_participant", "headline", "reference_only", "other_subject"]
STRICT_SOURCE_RELATIONS = {"direct_match": "supported", "listed_example": "supported", "topic_only": "conflict",
                          "different_activity": "conflict", "insufficient": "unknown"}
BACKGROUND_SOURCE_RELATIONS = {"same_activity": "supported", "related_background": "supported",
                              "unrelated": "conflict", "insufficient": "unknown"}
SOURCE_RELATIONS = STRICT_SOURCE_RELATIONS | BACKGROUND_SOURCE_RELATIONS
PROTOCOL = """\n输出遵循指定JSON Schema：reviews以短shot_id为键，primary_checks以selected_visual_requirements的全部key为键。不要再输出shot_id/key字段或数组。先写逐项比较，最后才写总判断。
source_alignment先分别填写required_activity（当前口播的题材及用户明确的画面要求）和observed_activity（实帧实际出现的主体及活动），各用一个短句，不强制摘抄原文分句。然后按代码给出的政策选择本镜Schema允许的relationship，不单独填写status：related_background政策允许same_activity（实际活动直接对应）或related_background（同主题、同类装备、相关工序或环境的自然背景），两者都通过本项；unrelated表示明显跨题材/不相关，insufficient表示看不清或证据不足。普通背景不要求精确型号年份、逐个动作或同一工序，但不免除全部requirements/must和人物身份条件。
strict_source政策只用direct_match/listed_example/topic_only/different_activity/insufficient：指定主体和要求直接相符才用direct_match，明确列举中的直接例子可用listed_example；仅宽泛题材相关用topic_only，实际另一指定主体/动作/阶段用different_activity。topic_only/different_activity仍不通过；具体本人、事件实录和手动要求不能借普通背景关系放宽。required_activity和observed_activity只是比较说明，不取代完整要求和实帧依据；reason/evidence仍须给出。旧的冲突关系不因新政策在程序中自动改为通过。
其他审核项填写status、reason、evidence。引用只来自本镜observation_references；text_absence只可用于caption_context，其余规则与输入全文保持不变。"""


def enabled(cfg, model):
    return urlparse(cfg["model_base_url"]).hostname == "dashscope.aliyuncs.com" and model in MODELS


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _refs(packet, record, *, field=None, absence=False):
    return [packet.forward[row["ref"]] for row in record.get("observation_references", [])
            if (field is None or row.get("field") == field)
            and (absence or row.get("field") != "text_absence")]


def relation_policy(record):
    """A background policy comes from validated planner context, not the model."""
    context = record.get("source_context")
    contract = context.get("contract") if isinstance(context, dict) else None
    if (record.get("visual_origin") == "generated" and not record.get("requested_people")
            and isinstance(contract, dict) and contract.get("use") == "illustration"
            and isinstance(contract.get("must"), list)
            and all(isinstance(row, dict) and row.get("kind") != "person" for row in contract["must"])
            and context.get("alignment_policy") == "related_background"):
        return "related_background"
    return "strict_source"


def relation_options(record):
    return BACKGROUND_SOURCE_RELATIONS if relation_policy(record) == "related_background" else STRICT_SOURCE_RELATIONS


def response_format(packet):
    definitions, reviews = {}, {}
    def evidence(refs):
        # Empty text sets legitimately have no name-label references.
        return {"type": "array", "items": _object({"ref": {"type": "string", **({"enum": refs} if refs else {})}}),
                **({"maxItems": 0} if not refs else {})}
    for record in packet.original:
        sid = packet.forward[record["shot_id"]]
        names = _refs(packet, record, field="text")
        definitions[sid + "_evidence"] = evidence(_refs(packet, record))
        definitions[sid + "_caption_evidence"] = evidence(_refs(packet, record, absence=True))
        citations = {"$ref": "#/$defs/" + sid + "_evidence"}
        def check(ref):
            return _object({"status": {"type": "string", "enum": STATES},
                            "reason": {"type": "string"}, "evidence": ref})
        definitions[sid + "_check"] = check(citations)
        definitions[sid + "_alignment"] = _object({
            "required_activity": {"type": "string"}, "observed_activity": {"type": "string"},
            "relationship": {"type": "string", "enum": list(relation_options(record))},
            "reason": {"type": "string"}, "evidence": citations})
        definitions[sid + "_caption_check"] = check({"$ref": "#/$defs/" + sid + "_caption_evidence"})
        labels = {"type": "array", "items": _object({
            "ref": {"type": "string", **({"enum": names} if names else {})},
            "relation": {"type": "string", "enum": RELATIONS}}), **({"maxItems": 0} if not names else {})}
        reviews[sid] = _object({"primary_checks": _object({key: {"$ref": "#/$defs/" + sid + (
                "_alignment" if key == "source_alignment" else
                "_caption_check" if key == "caption_context" else "_check")}
                for key in record.get("selected_visual_requirements", {})}),
            "labels": labels, "status": {"type": "string", "enum": STATES},
            "reason": {"type": "string"}, "evidence": citations})
    return {"type": "json_schema", "json_schema": {"name": "grounded_frame_reviews_v4", "strict": True,
            "schema": _object({"reviews": _object(reviews)}) | {"$defs": definitions}}}


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("strict_review_keys_missing_or_foreign")


def parse(packet, content):
    result = json.loads(content, object_pairs_hook=unique_object)
    _keys(result, {"reviews"})
    records = {packet.forward[r["shot_id"]]: r for r in packet.original}
    _keys(result["reviews"], records)
    rows = []
    for sid, record in records.items():
        row = result["reviews"][sid]
        _keys(row, {"status", "reason", "evidence", "labels", "primary_checks"})
        _keys(row["primary_checks"], record.get("selected_visual_requirements", {}))
        if not isinstance(row["labels"], list):
            raise ValueError("strict_review_labels_not_array")
        for label in row["labels"]:
            _keys(label, {"ref", "relation"})
            if label["ref"] not in _refs(packet, record, field="text") or label["relation"] not in RELATIONS:
                raise ValueError("strict_review_label_outside_record")
        for key, item in [(None, row), *row["primary_checks"].items()]:
            if key == "source_alignment":
                _keys(item, {"required_activity", "observed_activity", "relationship", "reason", "evidence"})
                if (item["relationship"] not in relation_options(record) or any(
                        not isinstance(item[field], str) or not item[field].strip()
                        for field in ("required_activity", "observed_activity"))):
                    raise ValueError("strict_review_invalid_activity_comparison")
                status = SOURCE_RELATIONS[item["relationship"]]
            else:
                status = item.get("status")
            if key is not None and key != "source_alignment":
                _keys(item, {"status", "reason", "evidence"})
            if (status not in STATES or not isinstance(item["reason"], str)
                    or not item["reason"].strip() or not isinstance(item["evidence"], list)):
                raise ValueError("strict_review_invalid_judgment")
            refs = _refs(packet, record, absence=key == "caption_context")
            for citation in item["evidence"]:
                _keys(citation, {"ref"})
                if citation["ref"] not in refs:
                    raise ValueError("strict_review_reference_outside_condition")
        comparison = row["primary_checks"].get("source_alignment")
        if comparison is not None:
            row["primary_checks"]["source_alignment"] = {
                "status": SOURCE_RELATIONS[comparison["relationship"]],
                "reason": comparison["reason"], "evidence": comparison["evidence"],
                "source_relation": {"policy": relation_policy(record), **{key: comparison[key] for key in
                                    ("required_activity", "observed_activity", "relationship")}}}
        # Missing evidence remains a failed individual judgment downstream;
        # the adapter never fills it or discards valid sibling judgments.
        rows.append(deepcopy(row) | {"shot_id": sid, "primary_checks": [
            {"key": key, **deepcopy(value)} for key, value in row["primary_checks"].items()]})
    return {"reviews": rows}
