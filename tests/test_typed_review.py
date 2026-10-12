"""Typed review is a closed trial, not evidence of model semantic accuracy."""
from copy import deepcopy
import hashlib
import json
import socket

import httpx
import pytest

from jinggao import (config, context_labels, costs, editorial_contract, editorial_review,
                     frame_contract, network, paths, planner, review_batches, review_evidence,
                     review_instructions, review_packet, review_schema, review_view)


CFG = config.DEFAULTS | {"model_api_key": "offline-typed-test-only", "model_name": "qwen3.7-flash",
    "review_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "_strict_text_review_trial": True}


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("typed review tests must not use a real transport")
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: CFG.copy())
    monkeypatch.setattr(network, "post", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def fixture():
    narrations = ["工作人员使用扳手维修水泵。", "公共设备维护也有许多具体劳动。"]
    project = {"script": "\n\n".join(narrations), "requirements": "不能用等待冒充正在维修。"}
    shots = [planner.new_shot(narration=value) | {"id": f"typed-shot-{i}", "source_unit_id": f"u{i}"}
             for i, value in enumerate(narrations)]
    contracts = {"contracts": [{"ref": f"s{i+1}", "use": "illustration", "refs": [],
        "facts": [["activity", f"s{i+1}", value]], "must": [], "queries": []}
        for i, value in enumerate(narrations)]}
    shots = editorial_contract.attach(project, shots, editorial_contract.compile_responses(project, shots, [contracts]))
    result = []
    for i, shot in enumerate(shots):
        frames = [{"frame_index": 0, "objects": "工作人员、扳手、水泵、墙上标牌",
                   "action": "手持扳手接触水泵，正在转动扳手", "text": ["维护间", "禁止吸烟"]},
                  {"frame_index": 1, "objects": "工作人员与水泵外壳", "action": "手持扳手继续转动",
                   "text": ["维护间"]}]
        context = editorial_review.source_context(shot, project)
        refs, _ = review_evidence.references(shot["id"], frames)
        record = {"shot_id": shot["id"], "narration": shot["narration"], "requirements": project["requirements"],
            "requested_people": [], "source_context": context, "source_contract": context["contract"],
            "selected_visual_requirements": frame_contract.requirements(shot, frames),
            "frame_observations": frames, "observation_references": refs, "scene_description": "水泵维护现场",
            "context_label_context": {"captions": context_labels.frame_captions(frames)},
            "evidence_binding": {"frame_sha256": [hashlib.sha256(f"{i}-{n}".encode()).hexdigest() for n in range(2)],
                                 "clip_start": 0, "duration": 4},
            "visual_quality": {"representation": "live_action", "quality_status": "not_detected"},
            "overlay_label": "设备维护", "source_evidence": {"title": "原来源，不是帧内字幕"}}
        result.append(record)
    return project, shots, result


def blocks(records):
    return [{"type": "text", "text": json.dumps(row, ensure_ascii=False)} for row in records]


def prepare(records, *, cfg=CFG, maximum=8704):
    return planner._prepare_chat_payload(cfg, "qwen3.7-flash", review_instructions.PROTOCOL, blocks(records),
        task="evidence_review", vision=False, max_tokens=maximum, reasoning=False, reasoning_effort=None)


def wire_answer(packet, *, status="unknown", relation="unclear"):
    result = {}
    for record in packet.original:
        sid = packet.forward[record["shot_id"]]
        captions = {f"caption_{c['caption_id']}": c for c in context_labels.frame_captions(record["frame_observations"])}
        checks = {}
        for key in record["selected_visual_requirements"]:
            if key in captions:
                caption = captions[key]
                index = caption["frame_indices"][0]
                offset = record["frame_observations"][index]["text"].index(caption["text"])
                ref = f"{sid}:f{index}:text:{offset}"
            else:
                index = int(key.removeprefix("frame_scope_")) if key.startswith("frame_scope_") else 0
                ref = f"{sid}:f{index}:action"
            checks[key] = {"status": status, "evidence": [{"ref": ref}],
                "reason": {"observed": "本夹具仅验证结构和引用", "relation": relation,
                           "why": "测试回答不是模型语义准确性的证据"}}
        result[sid] = {"source_use": record["narration"], "primary_checks": checks,
            "labels": [], "evidence": [{"ref": f"{sid}:f0:action"}], "status": status,
            "reason": "保留原始模型总解释"}
    return {"reviews": result}


def test_model_view_wraps_canonical_values_without_changing_source_pixels_or_ids():
    _, _, records = fixture()
    records[0]["selected_visual_requirements"]["future_check"] = {"source": "不能猜测", "nested": [1, 2]}
    records[0]["selected_visual_requirements"]["wrapper_shaped_condition"] = {
        "kind": "condition", "value": "原始条件恰巧也有 kind 和 value，仍须原样保留"}
    before = deepcopy(records)
    packet = prepare(records)["packet"]
    assert packet.original == before == records
    projected = packet.unpack()
    for original, view in zip(records, projected, strict=True):
        assert list(view["selected_visual_requirements"]) == list(original["selected_visual_requirements"])
        assert {key: value["value"] for key, value in view["selected_visual_requirements"].items()} == original["selected_visual_requirements"]
        for key in ("narration", "requirements", "requested_people", "frame_observations", "observation_references",
                    "scene_description", "visual_quality", "context_label_context", "source_evidence", "overlay_label"):
            assert view[key] == original[key]
        assert view["source_context"]["full_script"] == original["source_context"]["full_script"]
        assert view["source_context"]["requirements"] == original["source_context"]["requirements"]
    assert packet.view_version == "review_model_view_v2"
    assert all(row[review_view.VIEW_MARKER] == review_view.VERSION for row in projected)
    for projected_copy in (projected, json.loads(json.dumps(projected))):
        with pytest.raises(ValueError, match="重复投影"):
            review_view.build(projected_copy)
    assert packet.original == before == records


@pytest.mark.parametrize("marker", [None, "review_model_view_v1", "arbitrary source value"])
def test_reserved_view_marker_cannot_be_misread_as_canonical_input(marker):
    records = fixture()[2]
    records[0][review_view.VIEW_MARKER] = marker
    before = deepcopy(records)
    with pytest.raises(ValueError, match="保留字段"):
        review_view.build(records)
    assert records == before


@pytest.mark.parametrize("key,expected", [("source_alignment", "source"), ("caption_context", "captions"),
    ("frame_scope_0", "frame"), ("frame_scope_12", "frame"), ("caption_c1", "caption"),
    ("caption_c20", "caption"), ("must_1", "condition"), ("action", "condition"),
    ("caption_context_future", "condition"), ("caption_c0", "condition")])
def test_check_kind_is_determined_by_key_not_caption_or_topic(key, expected):
    assert review_view.check_kind(key) == expected


def test_v2_schema_requires_source_use_once_and_every_original_obligation():
    _, _, records = fixture()
    prepared = prepare(records)
    spec = prepared["payload"]["response_format"]["json_schema"]
    assert spec["name"] == "frame_reviews_v2" and spec["strict"]
    checks = spec["schema"]["$defs"]["check"]["properties"]
    assert checks["reason"]["required"] == ["observed", "relation", "why"]
    assert checks["reason"]["properties"]["relation"]["enum"] == list(review_schema.USE_RELATIONS)
    for original, row in zip(records, spec["schema"]["properties"]["reviews"]["properties"].values(), strict=True):
        assert list(row["properties"])[0] == "source_use"
        assert row["required"].count("source_use") == 1
        assert row["properties"]["primary_checks"]["required"] == list(original["selected_visual_requirements"])
        assert row["additionalProperties"] is False
    assert prepared["payload"]["max_tokens"] == 8704


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_normal_production_config_never_enables_the_typed_trial(flag):
    _, _, records = fixture()
    cfg = {key: value for key, value in CFG.items() if key != "_strict_text_review_trial"}
    if flag is not None:
        cfg["_strict_text_review_trial"] = flag
    result = prepare(records, cfg=cfg)
    assert result["payload"]["response_format"] == {"type": "json_object"}
    assert result["packet"].view_version is None
    assert result["packet"].unpack() == records
    assert "_strict_text_review_trial" not in config.DEFAULTS


@pytest.mark.parametrize("relation,expected", [("same_topic_only", "unknown"), ("unrelated", "conflict"),
    ("contradictory", "conflict"), ("unclear", "unknown")])
@pytest.mark.parametrize("model_status", ["supported", "unknown", "conflict"])
def test_negative_relationship_only_vetoes_supported_and_preserves_raw_model_result(relation, expected, model_status):
    packet = prepare(fixture()[2])["packet"]
    raw = wire_answer(packet, status="supported", relation="direct")
    raw["reviews"]["r0"]["primary_checks"]["caption_c1"].update(status=model_status,
        reason={"observed": "原片标识的含义", "relation": relation, "why": "与原句关系须单独说明"})
    before = deepcopy(raw)
    normalized = review_schema.normalize(packet, raw)["reviews"][0]
    checked = next(c for c in normalized["primary_checks"] if c["key"] == "caption_c1")
    assert checked["status"] == (expected if model_status == "supported" else model_status)
    assert normalized["status"] == checked["status"]
    assert normalized["typed_response"] == before["reviews"]["r0"] and raw == before
    assert all(value in checked["reason"] for value in ("原片标识的含义", relation, "与原句关系须单独说明"))
    assert bool([v for v in normalized["adapter_vetoes"] if v["key"] == "caption_c1"]) == (model_status == "supported")
    assert normalized["review_schema"] == "frame_reviews_v2"


@pytest.mark.parametrize("relation", ["direct", "compatible_context"])
@pytest.mark.parametrize("status", ["supported", "unknown", "conflict"])
def test_positive_relationship_does_not_upgrade_model_status(relation, status):
    packet = prepare(fixture()[2])["packet"]
    raw = wire_answer(packet, status=status, relation=relation)
    normalized = review_schema.normalize(packet, raw)
    assert all(row["status"] == status and not row["adapter_vetoes"] for row in normalized["reviews"])
    assert all(check["status"] == status for row in normalized["reviews"] for check in row["primary_checks"])


def test_total_unknown_is_not_upgraded_even_if_every_check_claims_direct_support():
    packet = prepare(fixture()[2])["packet"]
    raw = wire_answer(packet, status="supported", relation="direct")
    raw["reviews"]["r0"]["status"] = "unknown"
    assert review_schema.normalize(packet, raw)["reviews"][0]["status"] == "unknown"


@pytest.mark.parametrize("corruption", ["missing_source_use", "bad_source_use", "empty_source_use", "missing_check",
    "foreign_check", "missing_shot", "foreign_shot", "legacy_reason", "missing_observed", "empty_observed",
    "bad_why", "bad_relation", "nontext_relation", "extra_comparison", "nontext_status", "no_evidence",
    "cross_shot", "invented_ref", "nontext_label"])
def test_malformed_or_unbound_typed_output_fails_without_partial_repair(corruption):
    packet = prepare(fixture()[2])["packet"]
    raw = wire_answer(packet, status="supported", relation="direct")
    row = raw["reviews"]["r0"]
    check = row["primary_checks"]["caption_c1"]
    if corruption == "missing_source_use":
        row.pop("source_use")
    elif corruption == "bad_source_use":
        row["source_use"] = {}
    elif corruption == "empty_source_use":
        row["source_use"] = " "
    elif corruption == "missing_check":
        row["primary_checks"].pop("caption_c1")
    elif corruption == "foreign_check":
        row["primary_checks"]["invented"] = deepcopy(check)
    elif corruption == "missing_shot":
        raw["reviews"].pop("r1")
    elif corruption == "foreign_shot":
        raw["reviews"]["wrong"] = raw["reviews"].pop("r1")
    elif corruption == "legacy_reason":
        check["reason"] = "原来可见"
    elif corruption == "missing_observed":
        check["reason"].pop("observed")
    elif corruption == "empty_observed":
        check["reason"]["observed"] = " "
    elif corruption == "bad_why":
        check["reason"]["why"] = []
    elif corruption == "bad_relation":
        check["reason"]["relation"] = "automatically_approved"
    elif corruption == "nontext_relation":
        check["reason"]["relation"] = []
    elif corruption == "extra_comparison":
        check["reason"]["new_fact"] = "没有观察的东西"
    elif corruption == "nontext_status":
        check["status"] = []
    elif corruption == "no_evidence":
        check["evidence"] = []
    elif corruption == "cross_shot":
        check["evidence"] = [{"ref": "r1:f0:text:0"}]
    elif corruption == "invented_ref":
        check["evidence"] = [{"ref": "r0:f8:text:0"}]
    else:
        row["labels"] = [{"ref": "r0:f0:objects", "relation": "on_subject"}]
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="strict_review_"):
        review_schema.normalize(packet, raw)
    assert raw == before


def test_duplicate_comparison_json_key_is_not_last_wins():
    packet = prepare(fixture()[2])["packet"]
    raw = json.dumps(wire_answer(packet))
    raw = raw.replace('"relation": "unclear"', '"relation": "direct", "relation": "unclear"', 1)
    with pytest.raises(ValueError, match="duplicate_key"):
        review_schema.parse(packet, raw)


@pytest.mark.parametrize("bad_check,bad_ref", [("frame_scope_1", "r0:f0:action"),
    ("caption_c1", "r0:f0:text:1"), ("caption_c1", "r0:f0:objects")])
def test_typed_direct_does_not_bypass_existing_same_frame_and_complete_caption_gates(bad_check, bad_ref):
    _, shots, records = fixture()
    packet = prepare(records)["packet"]
    raw = wire_answer(packet, status="supported", relation="direct")
    raw["reviews"]["r0"]["primary_checks"][bad_check]["evidence"] = [{"ref": bad_ref}]
    normalized = packet.restore(review_schema.normalize(packet, raw))["reviews"][0]
    _, lookup = review_evidence.references(shots[0]["id"], records[0]["frame_observations"])
    resolved = review_evidence.resolve(normalized, lookup)
    status, _, _ = frame_contract.validate(shots[0], resolved["primary_checks"], records[0]["frame_observations"])
    assert status == "unknown"
    assert resolved["typed_response"]["primary_checks"][bad_check]["status"] == "supported"


def test_all_explanations_are_retained_without_becoming_frame_facts():
    _, _, records = fixture()
    before = deepcopy(records)
    packet = prepare(records)["packet"]
    raw = wire_answer(packet)
    reason = raw["reviews"]["r0"]["primary_checks"]["source_alignment"]["reason"]
    reason["observed"] = "这是一段需要保留的模型解释。" * 60
    reason["why"] = "模型解释不改写真实观察。" * 60
    normalized = review_schema.normalize(packet, raw)["reviews"][0]
    check = next(c for c in normalized["primary_checks"] if c["key"] == "source_alignment")
    assert reason["observed"] in check["reason"] and reason["why"] in check["reason"]
    assert normalized["typed_response"] == raw["reviews"]["r0"]
    assert packet.original == records == before


def test_v1_wire_is_never_accepted_as_v2_and_old_nonview_adapter_remains_available():
    records = fixture()[2]
    legacy = review_packet.Packet(blocks(records))
    raw = wire_answer(legacy)
    for row in raw["reviews"].values():
        row.pop("source_use")
        for check in row["primary_checks"].values():
            check["reason"] = "历史v1解释"
    assert review_schema.response_format(legacy)["json_schema"]["name"] == "frame_reviews_v1"
    assert review_schema.normalize(legacy, raw)["reviews"][0]["primary_checks"][0]["reason"] == "历史v1解释"
    with pytest.raises(ValueError, match="strict_review_"):
        review_schema.normalize(prepare(records)["packet"], raw)


def test_typed_cache_binds_full_original_schema_protocol_model_and_endpoint():
    records = fixture()[2]
    original = prepare(records)
    changed = deepcopy(records)
    changed[0]["evidence_binding"]["frame_sha256"][0] = "f" * 64
    altered = prepare(changed)
    assert original["payload"] == altered["payload"] and original["cache"].path != altered["cache"].path
    packet, payload = original["packet"], original["payload"]
    for updated in [payload | {"model": "different-model"}, payload | {"response_format": {"type": "json_object"}},
                    payload | {"messages": [*payload["messages"], {"role": "system", "content": "另一完整协议"}]}]:
        assert review_packet.Cache(packet, updated, CFG["model_base_url"]).path != original["cache"].path
    assert review_packet.Cache(packet, payload, "https://other.invalid").path != original["cache"].path
    assert packet.view_version in {"review_model_view_v2"}


@pytest.mark.parametrize("invalid", [False, True])
def test_mock_transport_charges_once_and_veto_or_failure_cannot_trigger_paid_retry(monkeypatch, invalid):
    records = fixture()[2]
    prepared = prepare(records)
    raw = wire_answer(prepared["packet"], status="supported", relation="direct")
    raw["reviews"]["r0"]["primary_checks"]["caption_c1"]["reason"]["relation"] = "same_topic_only"
    if invalid:
        raw["reviews"]["r0"].pop("source_use")
    sent = []
    def respond(url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}})
    monkeypatch.setattr(network, "post", respond)
    with costs.scope("typed-mock-only", "job"):
        result, info = planner.chat(review_instructions.PROTOCOL, blocks(records),
                                    task="evidence_review", max_tokens=8704)
        assert len(sent) == info["request_count"] == 1
        assert result["reviews"][0]["status"] == "unknown"
        if not invalid:
            repeated, cached = planner.chat(review_instructions.PROTOCOL, blocks(records),
                                            task="evidence_review", max_tokens=8704)
            assert cached["cache_hit"] and cached["request_count"] == 0 and repeated == result
    assert costs.summary("typed-mock-only")["project"]["requests"] == 1
    if invalid:
        assert not list((paths.DATA / "cache/text-judgments").glob("*.json"))
    else:
        row = result["reviews"][0]
        assert row["typed_response"]["status"] == "supported" and row["adapter_vetoes"]


def test_schema_and_shared_protocol_are_counted_without_changing_output_limits():
    records = fixture()[2]
    result = prepare(records, maximum=3072)
    payload = result["payload"]
    schema_bytes = len(json.dumps(payload["response_format"], ensure_ascii=False, separators=(",", ":")).encode())
    assert costs._input_bound(CFG, payload) - costs._input_bound(CFG, payload | {"response_format": {"type": "json_object"}}) == schema_bytes
    assert payload["messages"][0]["content"].count(review_instructions.PROTOCOL) == 1
    assert payload["max_tokens"] == 3072
    assert review_batches.output_limit(CFG, "qwen3.7-flash", records, maximum=3072) == 3072
    assert "caption问" in review_instructions.PROTOCOL and "frame问" in review_instructions.PROTOCOL
