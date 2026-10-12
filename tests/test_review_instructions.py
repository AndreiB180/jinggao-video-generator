"""Offline transport checks; these tests cannot prove semantic review accuracy."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jinggao import config, network, paths, planner, review_evidence, review_instructions, review_packet


def _record():
    # Synthetic transport data, unrelated to the held-out visual judgments.
    frames = [{"frame_index": 0, "objects": "工人手持扳手接触水泵外壳。",
               "action": "转动扳手。", "text": ["维护中"]}]
    refs, _ = review_evidence.references("instruction-test", frames)
    return {"shot_id": "instruction-test", "narration": "工人修理水泵。",
            "requirements": "不得把准备工具说成正在修理。",
            "selected_visual_requirements": {"action": "修理水泵", "caption_c1": "维护中"},
            "frame_observations": frames, "observation_references": refs,
            "evidence_binding": {"frame_sha256": [hashlib.sha256(b"synthetic transport fixture").hexdigest()],
                                 "clip_start": 0, "duration": 2}}


def _prepare(record, protocol=review_instructions.PROTOCOL):
    # This is payload preparation only; no settings file or remote client is read.
    cfg = config.DEFAULTS | {"model_name": "qwen3.7-flash",
        "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "_strict_text_review_trial": True}
    return planner._prepare_chat_payload(cfg, "qwen3.7-flash", protocol,
        [{"type": "text", "text": json.dumps(record, ensure_ascii=False)}],
        task="evidence_review", vision=False, max_tokens=2048, reasoning=True, reasoning_effort=None)


def test_module_is_only_a_literal_protocol_not_a_request_or_approval_path():
    module = ast.parse(Path(review_instructions.__file__).read_text("utf-8"))
    assert len(module.body) == 2
    assignment = module.body[1]
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1 and assignment.targets[0].id == "PROTOCOL"
    assert isinstance(assignment.value, ast.Constant) and isinstance(assignment.value.value, str)
    # Allow typed questions while bounding instruction growth. Actual complete
    # requests still pass the production price-tier and reservation checks.
    assert 4096 <= len(review_instructions.PROTOCOL.encode("utf-8")) <= 6000


def test_unified_protocol_uses_strict_wire_without_replacing_input_obligations(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(network, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    record = _record()
    original = deepcopy(record)
    prepared = _prepare(record)
    system = prepared["payload"]["messages"][0]["content"]
    assert system.startswith(review_instructions.PROTOCOL)
    assert system.count(review_instructions.PROTOCOL) == 1
    assert prepared["packet"].original == [original] and record == original
    response = prepared["payload"]["response_format"]
    assert response["type"] == "json_schema" and response["json_schema"]["strict"] is True
    row = response["json_schema"]["schema"]["properties"]["reviews"]["properties"]["r0"]
    assert list(row["properties"])[:2] == ["source_use", "primary_checks"]
    assert row["properties"]["primary_checks"]["required"] == list(record["selected_visual_requirements"])
    assert prepared["payload"]["max_tokens"] == 2048


def test_protocol_change_cannot_reuse_a_previous_structurally_valid_judgment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    first = _prepare(_record())
    changed = _prepare(_record(), review_instructions.PROTOCOL + "\n本轮协议变动。")
    # A complete unknown response tests cache transport without asserting that
    # invented pixels or a synthetic source have passed semantic review.
    checks = [{"key": key, "status": "unknown", "reason": "仅作离线传输测试。",
               "evidence": [{"ref": "r0:f0:text:0"}] if key == "caption_c1" else []}
              for key in _record()["selected_visual_requirements"]]
    raw = {"reviews": [{"shot_id": "r0", "primary_checks": checks, "status": "unknown",
                        "reason": "仅作离线传输测试。", "labels": [], "evidence": []}]}
    assert first["packet"].valid_response(raw)
    first["cache"].save(raw)
    assert first["cache"].read() == raw
    assert changed["cache"].path != first["cache"].path
    assert changed["cache"].read() is None
    assert isinstance(first["cache"], review_packet.Cache)
