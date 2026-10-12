"""Complete wire responses remain evidence-bound and fully budgeted."""
from copy import deepcopy
import json

import httpx
import pytest

from jinggao import config, costs, network, paths, planner, review_packet, review_schema, review_view
from test_review_packet import answer, packet, records, strict_answer

CFG = config.DEFAULTS | {"model_api_key": "test-only", "model_name": "qwen3.7-flash",
    "review_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "_strict_text_review_trial": True}


def prepare(rows, *, cfg=CFG, model="qwen3.7-flash", task="evidence_review", vision=False):
    return planner._prepare_chat_payload(cfg, model, "完整 JSON 审核规则",
        [{"type": "text", "text": json.dumps(row, ensure_ascii=False)} for row in rows],
        task=task, vision=vision, max_tokens=2048, reasoning=True, reasoning_effort=None)


def test_schema_requires_exact_shots_and_checks_and_adapter_retains_citations():
    value = packet()
    spec = review_schema.response_format(value)
    assert spec["type"] == "json_schema" and spec["json_schema"]["strict"] is True
    schema = spec["json_schema"]["schema"]
    reviews = schema["properties"]["reviews"]
    assert reviews["required"] == ["r0", "r1"]
    for row in reviews["properties"].values():
        assert row["additionalProperties"] is False
        checks = row["properties"]["primary_checks"]
        assert checks["required"] == ["action"] and checks["additionalProperties"] is False
        assert row["properties"]["evidence"] == {"$ref": "#/$defs/evidence"}
    raw = strict_answer(answer())
    saved = deepcopy(raw)
    restored = review_schema.normalize(value, raw)
    assert raw == saved and restored == answer() and value.valid_response(restored)


@pytest.mark.parametrize("corruption", ["missing_shot", "foreign_shot", "legacy_array", "missing_check",
    "extra_check", "extra_field", "nontext_status", "empty_reason", "missing_evidence", "cross_shot",
    "made_up_ref", "literal_quote", "invalid_label", "label_from_objects", "extra_root"])
def test_incomplete_or_foreign_structure_is_rejected_without_repair(corruption):
    raw = strict_answer(answer())
    row = raw["reviews"]["r0"]
    if corruption == "missing_shot":
        raw["reviews"].pop("r1")
    elif corruption == "foreign_shot":
        raw["reviews"]["r999"] = raw["reviews"].pop("r1")
    elif corruption == "legacy_array":
        raw = answer()
    elif corruption == "missing_check":
        row["primary_checks"] = {}
    elif corruption == "extra_check":
        row["primary_checks"]["unrequested"] = deepcopy(row["primary_checks"]["action"])
    elif corruption == "extra_field":
        row["primary_checks"]["action"]["approved"] = True
    elif corruption == "nontext_status":
        row["status"] = []
    elif corruption == "empty_reason":
        row["reason"] = " "
    elif corruption == "missing_evidence":
        row["evidence"] = []
    elif corruption == "cross_shot":
        row["evidence"][0]["ref"] = "r1:f0:text:1"
    elif corruption == "made_up_ref":
        row["evidence"][0]["ref"] = "r0:f8:action"
    elif corruption == "literal_quote":
        row["evidence"][0]["quote"] = "未经观察的猜测"
    elif corruption == "invalid_label":
        row["labels"] = [{"ref": "r0:f0:text:0", "relation": "looks_like"}]
    elif corruption == "label_from_objects":
        row["labels"] = [{"ref": "r0:f0:objects", "relation": "on_subject"}]
    else:
        raw["approved"] = True
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="strict_review_"):
        review_schema.normalize(packet(), raw)
    assert raw == before


def test_duplicate_json_check_cannot_silently_replace_a_conflicting_check():
    content = json.dumps(strict_answer(answer()))
    content = content.replace('"action": {', '"action": {}, "action": {', 1)
    with pytest.raises(ValueError, match="duplicate_key"):
        review_schema.parse(packet(), content)


@pytest.mark.parametrize("changes", [{"model": "qwen3.6-flash"}, {"vision": True}, {"task": "planning"},
    {"cfg": CFG | {"_strict_text_review_trial": False}},
    {"cfg": {key: value for key, value in CFG.items() if key != "_strict_text_review_trial"}},
    {"cfg": CFG | {"model_base_url": "https://models.example/v1"}}])
def test_only_known_provider_pure_text_review_opts_in(changes):
    assert prepare(records())["payload"]["response_format"]["type"] == "json_schema"
    assert prepare(records(), **changes)["payload"]["response_format"] == {"type": "json_object"}


def test_unpromoted_trial_cannot_be_enabled_in_saved_application_settings():
    with pytest.raises(ValueError, match="未知设置字段"):
        config.update({"_strict_text_review_trial": "true"})


def test_hiding_program_binding_does_not_hide_it_from_cache_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    rows = records()
    first = prepare(rows)
    changed = deepcopy(rows)
    changed[0]["evidence_binding"]["frame_sha256"] = ["0"*64]
    changed[0]["evidence_binding"]["clip_start"] += 1
    second = prepare(changed)
    assert first["payload"] == second["payload"]  # Hashes aren't model evidence.
    assert first["packet"].original == rows
    assert first["cache"].path != second["cache"].path
    first["cache"].save(answer())
    assert first["cache"].read() == answer() and second["cache"].read() is None


def test_short_shared_observations_keep_negation_and_source_words_losslessly():
    rows = records()
    observation = "人群正在等待，没有操作"
    assert len(observation.encode()) < 48
    for row in rows:
        row["frame_observations"][0]["action"] = observation
        row["scene_description"] = observation
        row["notes"] = [observation, observation]
    blocks = [{"type": "text", "text": json.dumps(row, ensure_ascii=False)} for row in rows]
    projected = review_packet.Packet(blocks, model_view=True)
    assert observation in projected.body["texts"].values()
    assert projected.unpack() == review_view.build(rows)
    assert projected.original == rows
    assert projected.stats["packed_bytes"] < len(review_packet.encoded(review_view.build(rows)).encode())


def test_schema_is_included_in_price_tier_and_reservation():
    payload = prepare(records())["payload"]
    original = deepcopy(payload)
    plain = payload | {"response_format": {"type": "json_object"}}
    schema_bytes = len(json.dumps(payload["response_format"], ensure_ascii=False, separators=(",", ":")).encode())
    assert costs._input_bound(CFG, payload) - costs._input_bound(CFG, plain) == schema_bytes
    assert payload == original
    # Adding a schema must not let an oversized request pass as a lower tier.
    plain["messages"] = [{"role": "user", "content": "x"*(32000-388)}]
    assert costs._input_bound(CFG, plain) == 32000
    assert costs._input_bound(CFG, plain | {"response_format": payload["response_format"]}) > 32000


def test_provider_schema_violation_is_charged_once_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: CFG.copy())
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        raw = strict_answer(answer(), typed=True)
        raw["reviews"]["r0"]["primary_checks"] = {}
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}})
    monkeypatch.setattr(network, "post", post)
    content = [{"type": "text", "text": json.dumps(row)} for row in records()]
    with costs.scope("strict-incomplete", "job"):
        result, info = planner.chat("审核 JSON", content, task="evidence_review", max_tokens=2048)
    assert len(calls) == info["request_count"] == 1
    assert all(row["status"] == "unknown" for row in result["reviews"])
    assert not info["batches"][0]["valid_ids"]
    assert costs.summary("strict-incomplete")["project"]["estimated_cny"] > 0
    assert not list((tmp_path / "cache" / "text-judgments").glob("*.json"))
    diagnostics = list((tmp_path / "diagnostics" / "model-responses").glob("*.json"))
    assert len(diagnostics) == 1 and "strict_review_keys" in diagnostics[0].read_text()
