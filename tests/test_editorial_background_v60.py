"""User-selected B-roll latitude never changes explicit person requirements.

The replies below are controlled schema fixtures, not new model judgments or
claims that a generated video has passed a paid end-to-end acceptance run.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from jinggao import editorial_contract, editorial_review, planner
from jinggao import review_evidence, review_packet, review_schema


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("editorial-background regression attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def cycling():
    return json.loads((Path(__file__).parent / "fixtures/editorial-background-cycling-v60.json").read_text("utf-8"))["shots"]


def make_shot(text="这种坦克在2010年服役，正在高速行驶。", *, use="illustration", requirements="", must=(), facts=()):
    project = {"script": text, "requirements": requirements}
    shot = planner.new_shot(text, visual_origin="generated")
    response = {"contracts": [{"ref": "s1", "use": use, "refs": [],
        "facts": list(facts) or ([["event", "s1", text]] if use == "event_record" else []),
        "must": list(must), "queries": []}]}
    bundle = editorial_contract.compile_responses(project, [shot], [response])
    return editorial_contract.attach(project, [shot], bundle)[0]


def packet_for(shot, frames=None):
    frames = frames or [{"frame_index": 0, "objects": "另一型号的坦克", "action": "坦克静置在训练场", "text": []}]
    refs, _ = review_evidence.references(shot["id"], frames)
    record = {"shot_id": shot["id"], "narration": shot["narration"], "visual_origin": shot["visual_origin"],
        "requested_people": shot.get("persons", []), "source_context": editorial_review.source_context(shot),
        "selected_visual_requirements": editorial_review.requirements(shot),
        "frame_observations": frames, "observation_references": refs}
    return review_packet.Packet([{"type": "text", "text": json.dumps(record, ensure_ascii=False)}])


def answer(packet, relation, *, required="当前军事装备题材", observed="同类坦克的背景资料"):
    record = packet.original[0]
    actual = next(row for row in record["observation_references"] if row["field"] == "action")
    ref = packet.forward[actual["ref"]]
    checks = {key: {"status": "supported", "reason": "受控测试判断", "evidence": [{"ref": ref}]}
              for key in record["selected_visual_requirements"]}
    checks["source_alignment"] = {"required_activity": required, "observed_activity": observed,
        "relationship": relation, "reason": "根据此记录的编辑政策比较实帧。", "evidence": [{"ref": ref}]}
    return {"reviews": {"r0": {"status": "supported", "reason": "乐观总判断不得覆盖逐项失败",
        "labels": [], "evidence": [{"ref": ref}], "primary_checks": checks}}}


def parse(packet, body):
    return review_schema.parse(packet, json.dumps(body, ensure_ascii=False))["reviews"][0]


def alignment(row):
    return next(item for item in row["primary_checks"] if item["key"] == "source_alignment")


def test_ordinary_equipment_and_date_facts_do_not_select_strict_policy():
    text = "这种坦克在2010年服役，正在高速行驶。"
    shot = make_shot(text, facts=[["object", "s1", "这种坦克"], ["event_date", "s1", "2010年"]])
    original = deepcopy(shot)
    assert editorial_review.alignment_policy(shot) == "related_background"
    context = editorial_review.source_context(shot)
    assert context["alignment_policy"] == "related_background"
    assert context["full_script"] == text and context["contract"]["facts"]
    assert shot == original


def test_author_attribution_is_not_an_instruction_to_show_the_named_author():
    shot = make_shot("张三认为，这类装备仍然重要。", facts=[["person", "s1", "张三"]])
    assert editorial_review.alignment_policy(shot) == "related_background"
    packet = packet_for(shot)
    assert review_schema.relation_policy(packet.original[0]) == "related_background"
    assert alignment(parse(packet, answer(packet, "related_background")))["status"] == "supported"


@pytest.mark.parametrize("change", ["requested-person", "explicit-person", "manual", "locked", "event", "invalid-binding"])
def test_specific_people_manual_requirements_and_event_record_retain_strict_policy(change):
    text = "张三检查车辆。"
    shot = make_shot(text, use="event_record" if change == "event" else "illustration",
                     requirements="必须展示张三本人。" if change == "explicit-person" else "",
                     must=[["person", "r", "必须展示张三本人"]] if change == "explicit-person" else [])
    if change == "requested-person":
        shot["persons"] = [{"name": "张三", "aliases": []}]
    elif change == "manual":
        shot["visual_origin"] = "manual"
    elif change == "locked":
        shot["locked"] = True
    elif change == "invalid-binding":
        shot["narration"] += "修改原稿。"
    assert editorial_review.alignment_policy(shot) == "strict_source"
    packet = packet_for(shot)
    packet.original[0]["selected_visual_requirements"]["source_alignment"] = shot["narration"]
    schema = review_schema.response_format(packet)["json_schema"]["schema"]
    allowed = schema["$defs"]["r0_alignment"]["properties"]["relationship"]["enum"]
    assert set(allowed) == set(review_schema.STRICT_SOURCE_RELATIONS)
    for relation in ("related_background", "same_activity"):
        with pytest.raises(ValueError, match="strict_review_invalid_activity_comparison"):
            parse(packet, answer(packet, relation))


def test_project_source_change_does_not_reuse_a_wider_policy():
    shot = make_shot()
    assert editorial_review.alignment_policy(shot, {"script": shot["narration"], "requirements": ""}) == "related_background"
    assert editorial_review.alignment_policy(shot, {"script": shot["narration"], "requirements": "必须展示张三"}) == "strict_source"
    assert editorial_review.source_context(shot, {"script": "改过的原稿", "requirements": ""}) == {}


@pytest.mark.parametrize("relation,expected", [("same_activity", "supported"), ("related_background", "supported"),
                                              ("unrelated", "conflict"), ("insufficient", "unknown")])
def test_ordinary_schema_exposes_the_users_background_standard(relation, expected):
    packet = packet_for(make_shot())
    schema = review_schema.response_format(packet)["json_schema"]["schema"]
    assert set(schema["$defs"]["r0_alignment"]["properties"]["relationship"]["enum"]) == set(review_schema.BACKGROUND_SOURCE_RELATIONS)
    result = alignment(parse(packet, answer(packet, relation)))
    assert result["status"] == expected
    assert result["source_relation"] == {"policy": "related_background", "required_activity": "当前军事装备题材",
                                        "observed_activity": "同类坦克的背景资料", "relationship": relation}


@pytest.mark.parametrize("number", ["3", "28"])
def test_real_cycling_records_keep_full_source_and_all_old_pixels_under_background_policy(cycling, number):
    sample = cycling[number]
    original = deepcopy(sample)
    packet = packet_for(sample["shot"], sample["frames"])
    assert review_schema.relation_policy(packet.original[0]) == "related_background"
    result = alignment(parse(packet, answer(packet, "related_background", required="日常骑行",
                                           observed="实际观察到的自行车整备或骑行画面")))
    assert result["status"] == "supported"
    assert packet.original[0]["source_context"]["full_script"] == sample["shot"]["editorial_contract"]["full_script"]
    assert packet.original[0]["frame_observations"] == sample["frames"]
    assert sample == original


def test_legacy_mismatch_is_not_upgraded_by_changing_relation_mappings(cycling):
    old = cycling["3"]["old_primary_checks"][0]
    assert old["source_relation"]["relationship"] == "different_activity"
    assert review_schema.SOURCE_RELATIONS[old["source_relation"]["relationship"]] == "conflict"
    packet = packet_for(cycling["3"]["shot"], cycling["3"]["frames"])
    packet.original[0]["source_context"].pop("alignment_policy")
    assert review_schema.relation_policy(packet.original[0]) == "strict_source"
    result = alignment(parse(packet, answer(packet, "different_activity")))
    assert result["status"] == "conflict" and result["source_relation"]["policy"] == "strict_source"


@pytest.mark.parametrize("kind,requirement", [("object", "必须展示指定A型坦克"),
                                            ("sequence", "必须完整展示先停车再拿手机的顺序")])
def test_background_relation_preserves_all_explicit_requirements_and_their_conflicts(kind, requirement):
    shot = make_shot(requirements=requirement, must=[[kind, "r", requirement]])
    packet = packet_for(shot)
    assert packet.original[0]["source_context"]["requirements"] == requirement
    assert "must_1" in packet.original[0]["selected_visual_requirements"]
    body = answer(packet, "related_background")
    body["reviews"]["r0"]["primary_checks"]["must_1"]["status"] = "conflict"
    rows = parse(packet, body)["primary_checks"]
    assert next(row for row in rows if row["key"] == "must_1")["status"] == "conflict"


@pytest.mark.parametrize("damage", ["missing-condition", "foreign-frame", "no-person-evidence", "topic-as-person"])
def test_specific_person_cannot_use_background_permission_or_invented_evidence(damage):
    shot = make_shot("张三检查车辆。")
    shot["persons"] = [{"name": "张三", "aliases": []}]
    packet = packet_for(shot)
    body = answer(packet, "direct_match")
    row = body["reviews"]["r0"]
    if damage == "missing-condition":
        row["primary_checks"].pop("caption_context")
    elif damage == "foreign-frame":
        row["primary_checks"]["source_alignment"]["evidence"] = [{"ref": "r9:f0:action"}]
    elif damage == "no-person-evidence":
        row["labels"] = [{"ref": "r0:f0:action", "relation": "on_subject"}]
    else:
        row["primary_checks"]["source_alignment"]["relationship"] = "related_background"
    with pytest.raises(ValueError):
        parse(packet, body)


def test_unrelated_story_caption_is_not_overridden_by_a_related_background_summary():
    packet = packet_for(make_shot())
    body = answer(packet, "related_background")
    body["reviews"]["r0"]["primary_checks"]["caption_context"].update(status="conflict", reason="无关剧情广告")
    rows = parse(packet, body)["primary_checks"]
    assert next(row for row in rows if row["key"] == "caption_context")["status"] == "conflict"


def test_missing_or_forged_policy_cannot_relax_named_person_or_event_record():
    shot = make_shot("张三检查车辆。")
    packet = packet_for(shot)
    record = packet.original[0]
    record["requested_people"] = [{"name": "张三"}]
    assert record["source_context"]["alignment_policy"] == "related_background"
    assert review_schema.relation_policy(record) == "strict_source"
    record["requested_people"] = []
    record["source_context"]["contract"]["use"] = "event_record"
    assert review_schema.relation_policy(record) == "strict_source"
