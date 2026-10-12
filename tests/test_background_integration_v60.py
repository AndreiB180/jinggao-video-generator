"""Exercise the user's ordinary B-roll policy across transport and local gates."""
from copy import deepcopy
import json

import pytest

from jinggao import config, costs, frame_contract, planner, review_evidence, review_schema
from jinggao import segment_discovery as discovery
from tests.test_editorial_background_v60 import answer, make_shot, packet_for
from tests.test_library_rank_context_v60 import bound_shots


def resolved(shot, relation="related_background"):
    packet = packet_for(shot)
    frames = packet.original[0]["frame_observations"]
    parsed = review_schema.parse(packet, json.dumps(answer(packet, relation), ensure_ascii=False))
    restored = packet.restore(parsed)["reviews"][0]
    _, lookup = review_evidence.references(shot["id"], frames)
    return review_evidence.resolve(restored, lookup)["primary_checks"], frames


def test_equipment_background_passes_local_gate_with_grounded_frame_evidence():
    shot = make_shot("这种坦克在2010年服役，正在高速行驶。")
    checks, frames = resolved(shot)
    # Actual observations are a stationary tank of another model. This is a
    # controlled judge response, proving plumbing rather than real model quality.
    assert frames[0]["action"] == "坦克静置在训练场"
    assert frame_contract.validate(shot, checks, frames)[0] == "supported"


@pytest.mark.parametrize("mutation", ["named-person", "manual", "locked", "unbound", "missing-policy"])
def test_background_verdict_cannot_override_person_or_explicit_shot_policy(mutation):
    shot = make_shot()
    checks, frames = resolved(shot)
    if mutation == "named-person":
        shot["persons"] = [{"name": "张三", "aliases": []}]
    elif mutation == "manual":
        shot["visual_origin"] = "manual"
    elif mutation == "locked":
        shot["locked"] = True
    elif mutation == "unbound":
        shot["narration"] += "新的内容。"
    else:
        checks[0]["source_relation"].pop("policy")
    assert frame_contract.validate(shot, checks, frames)[0] == "unknown"


def test_old_conflict_remains_conflict_and_explicit_condition_remains_required():
    shot = make_shot(requirements="必须完整展示装卸步骤。", must=[["sequence", "r", "必须完整展示装卸步骤"]])
    checks, frames = resolved(shot)
    omitted = [row for row in checks if row["key"] != "must_1"]
    assert frame_contract.validate(shot, omitted, frames)[0] == "unknown"
    checks[0].update(status="conflict", source_relation={"required_activity": "旧严格动作",
        "observed_activity": "旧记录不相符", "relationship": "different_activity"})
    assert frame_contract.validate(shot, checks, frames)[0] == "conflict"


def test_background_has_no_extra_per_frame_process_gate_but_person_still_does():
    shot = make_shot("讲讲军事装备。")
    frames = [{"frame_index": 0, "objects": "军事论坛会场", "action": "人员坐着", "text": []}]
    assert set(frame_contract.requirements(shot, frames)) == {"source_alignment", "caption_context"}
    shot["persons"] = [{"name": "张三"}]
    assert "frame_scope_0" in frame_contract.requirements(shot, frames)


def test_online_ranking_receives_shared_full_context_and_computed_policy():
    first, shots = bound_shots(["自行车适合日常出行。", "它也可以停在路边。"])
    other, second = bound_shots(["坦克装备的日常检查。"], prefix="second")
    shots += second
    for shot in shots:
        shot["duration"] = 8
    shots[1]["persons"] = [{"name": "张三"}]
    before = deepcopy(shots)
    content, snapshot, _ = discovery._rank_payload(shots, [], lambda: None)
    request = json.loads(content[0]["text"])
    assert request["source_contexts"] == snapshot["source_contexts"] == {
        "SRC1": {"full_script": first["script"], "requirements": ""},
        "SRC2": {"full_script": other["script"], "requirements": ""}}
    assert [row["source_context_ref"] for row in request["shots"]] == ["SRC1", "SRC1", "SRC2"]
    assert [row["alignment_policy"] for row in request["shots"]] == [
        "related_background", "strict_source", "related_background"]
    assert content[0]["text"].count('"full_script"') == 2 and shots == before


def test_shared_budget_bound_prices_the_same_full_context_as_ranking(monkeypatch):
    from jinggao import sample_catalog
    _, shots = bound_shots(["自行车适合日常出行。", "它也可以停在路边。"], "使用横屏。")
    for shot in shots:
        shot["duration"] = 8
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS)
    monkeypatch.setattr(sample_catalog, "prepared_facts", lambda *a, **kw: ({}, {}))
    captured = []
    def prepare(cfg, model, protocol, text, **kwargs):
        captured.append(json.loads(text))
        return {"payload": {"content": text}}
    monkeypatch.setattr(planner, "_prepare_chat_payload", prepare)
    monkeypatch.setattr(costs, "_input_bound", lambda cfg, payload: len(payload["content"].encode()))
    bound = discovery.shared_rank_input_bound([], {}, lambda: None)
    assert bound(shots) > 0
    content, _, _ = discovery._rank_payload(shots, [], lambda: None)
    assert captured == [json.loads(content[0]["text"])]
