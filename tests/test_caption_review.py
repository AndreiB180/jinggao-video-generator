"""Every observed caption is reviewed; page metadata is never visible text."""
from copy import deepcopy
import json

import pytest

from jinggao import (config, context_labels, frame_contract, neutral_observer, planner,
                     review_evidence, review_instructions, review_packet)


FRAMES = [
    {"frame_index": 0, "objects": "车辆加油口与加油枪", "action": "加油枪插入加油口",
     "text": ["记录者", "第一厢油", "123元"]},
    {"frame_index": 1, "objects": "骑车的人与白色叠加字幕", "action": "人在车旁",
     "text": ["记录者", "后续路上11公里熄火3次！"]},
]


def complete_checks(shot, frames):
    captions = {f"caption_{row['caption_id']}": row for row in context_labels.frame_captions(frames)}
    checks = []
    for key in frame_contract.requirements(shot, frames):
        if key in captions:
            caption = captions[key]
            evidence = [{"frame_index": caption["frame_indices"][0], "field": "text", "quote": caption["text"]}]
        else:
            index = int(key.removeprefix("frame_scope_")) if key.startswith("frame_scope_") else 0
            evidence = [{"frame_index": index, "field": "action", "quote": frames[index]["action"]}]
        checks.append({"key": key, "status": "supported", "reason": "本项独立语义结论", "evidence": evidence})
    return checks


@pytest.mark.parametrize("shot", [{}, {"visual_plan": {"mode": "single", "primary": {"action": "加油"}}}])
def test_complete_caption_ids_are_required_even_without_a_source_contract(shot):
    requirements = frame_contract.requirements(shot, FRAMES)
    assert {key: text for key, text in requirements.items() if key.startswith("caption_")} == {
        "caption_c1": "记录者", "caption_c2": "第一厢油", "caption_c3": "123元", "caption_c4": "后续路上11公里熄火3次！"}
    checks = complete_checks(shot, FRAMES)
    status, _, verified = frame_contract.validate(shot, checks, FRAMES)
    assert status == "supported" and len(verified) == len(requirements)
    assert "approved" not in verified[-1]
    # The complete evidence set alone does not claim that the model's semantic
    # result is true. A negative caption judgment must still veto the total.
    checks[-1].update(status="conflict", reason="具体故障叙事与本句用途不相容")
    status, reason, _ = frame_contract.validate(shot, checks, FRAMES)
    assert status == "conflict" and "c4" in reason


@pytest.mark.parametrize("change", ["missing", "duplicate", "logo", "partial", "wrong-frame", "objects-only",
                                    "other-caption", "metadata", "no-evidence", "unknown", "caption-N"])
def test_omitted_or_substituted_caption_cannot_pass(change):
    checks = complete_checks({}, FRAMES)
    last = checks[-1]
    if change == "missing":
        checks.pop()
    elif change == "duplicate":
        checks[-1] = deepcopy(checks[0])
    elif change == "logo":
        last["evidence"] = [{"frame_index": 1, "field": "text", "quote": "记录者"}]
    elif change == "partial":
        last["evidence"][0]["quote"] = "后续路上11公里"
    elif change == "wrong-frame":
        last["evidence"][0]["frame_index"] = 0
    elif change == "objects-only":
        last["evidence"] = [{"frame_index": 1, "field": "objects", "quote": FRAMES[1]["objects"]}]
    elif change == "other-caption":
        last["evidence"].append({"frame_index": 1, "field": "text", "quote": "记录者"})
    elif change == "metadata":
        last["evidence"] = [{"frame_index": 1, "field": "description", "quote": "旅途愉快"}]
    elif change == "no-evidence":
        last["evidence"] = []
    elif change == "unknown":
        last["status"] = "unknown"
    else:
        last["key"] = "caption_cN"
    assert frame_contract.validate({}, checks, FRAMES)[0] == "unknown"


def test_caption_can_use_its_observed_carrier_but_not_another_frame():
    checks = complete_checks({}, FRAMES)
    checks[-1]["evidence"].append({"frame_index": 1, "field": "objects", "quote": FRAMES[1]["objects"]})
    assert frame_contract.validate({}, checks, FRAMES)[0] == "supported"
    checks[-1]["evidence"].append({"frame_index": 0, "field": "action", "quote": FRAMES[0]["action"]})
    assert frame_contract.validate({}, checks, FRAMES)[0] == "unknown"


def test_caption_rows_preserve_every_literal_and_occurrence_without_rewriting():
    frames = deepcopy(FRAMES)
    frames[1]["text"] += ["123元", "123 元", "123元 ", "", " "]
    before = deepcopy(frames)
    captions = context_labels.frame_captions(frames)
    assert captions[0] == {"caption_id": "c1", "text": "记录者", "frame_indices": [0, 1]}
    assert captions[2] == {"caption_id": "c3", "text": "123元", "frame_indices": [0, 1]}
    assert [row["text"] for row in captions[-2:]] == ["123 元", "123元 "]
    assert frames == before


def test_metadata_does_not_create_caption_requirements_or_change_ids():
    frames = [{"frame_index": 0, "objects": "工人与包裹", "action": "搬运包裹", "text": ["中新视频"]}]
    shot = {"id": "s", "narration": "认真对待劳动。", "duration": 5}
    url = "https://www.chinanews.com.cn/shipin/example.shtml"
    asset = {"source_url": url, "source_verified": True, "source_evidence": {
        "url": url, "readable": True, "evidence_origin": "publisher_page", "title": "全自动化智能分拣",
        "description": "无人矩阵日均处理150万件", "published_at": "2020-10-30", "publisher": "中国新闻网"}}
    context = context_labels.review_context(shot, asset, frames)
    assert context["source_fields"]["description"] == "无人矩阵日均处理150万件"
    assert context["source_fields_kind"] == "publisher_page_metadata"
    assert context["captions_kind"] == "observed_frame_text"
    assert context["captions"] == [{"caption_id": "c1", "text": "中新视频", "frame_indices": [0]}]
    assert frame_contract.requirements(shot, frames) == {"caption_c1": "中新视频"}
    # Publisher metadata stays available for its own source checks; nothing
    # removes it or adds it to the actual observed caption inventory.
    assert "2020-10-30" in json.dumps(context, ensure_ascii=False)


@pytest.mark.usefixtures("isolated_pixel_quality")
@pytest.mark.parametrize("cached, omit", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("strict", [False, True])
def test_real_audit_uses_each_caption_ref_and_revalidates_cached_results(monkeypatch, cached, omit, strict):
    cfg = config.DEFAULTS | {"model_api_key": "test-only", "review_model": "qwen3.7-flash",
        "_strict_text_review_trial": strict,
        "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1" if strict else "https://models.test/v1"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    shot = planner.new_shot(narration="使用体验需要结合实际。")
    frames = deepcopy(FRAMES)
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "加油资料与行驶情况文字", "frames": frames}}, 0))
    sent = []

    def judge(system, content, **options):
        data = json.loads(content[0]["text"])
        assert len(data["context_label_context"]["captions"]) == 4
        if strict:
            assert system == review_instructions.PROTOCOL
        else:
            assert context_labels.CAPTION_PROTOCOL in system
        sent.append(data)
        checks = complete_checks(shot, frames)
        for check in checks:
            check["evidence"] = [{"ref": f"{shot['id']}:f{citation['frame_index']}:text:"
                f"{frames[citation['frame_index']]['text'].index(citation['quote'])}"} for citation in check["evidence"]]
        if omit:
            checks.pop()
        return {"reviews": [{"shot_id": shot["id"], "status": "supported", "reason": "模型整体结论",
            "labels": [], "evidence": [{"ref": f"{shot['id']}:f0:action"}], "primary_checks": checks}]}, {
                "cache_hit": cached, "request_count": 0 if cached else 1}

    monkeypatch.setattr(planner, "chat", judge)
    audit = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"}, "frame_paths": []}],
                                          lambda: None)[shot["id"]]
    assert len(sent) == 1 and audit["request_count"] == (0 if cached else 1)
    assert audit["status"] == ("unknown" if omit else "supported")
    assert "approved" not in audit and not shot["approved"]
    if not omit:
        assert audit["primary_checks"][-1]["evidence"][0]["quote"] == FRAMES[-1]["text"][-1]


def test_packet_requires_all_caption_ids_and_resolves_only_same_shot_refs():
    public, lookup = review_evidence.references("s", FRAMES)
    record = {"shot_id": "s", "frame_observations": FRAMES, "observation_references": public,
              "selected_visual_requirements": frame_contract.requirements({}, FRAMES)}
    packet = review_packet.Packet([{"type": "text", "text": json.dumps(record)}])
    assert packet.unpack() == [record]
    checks = complete_checks({}, FRAMES)
    for check in checks:
        literal = check["evidence"][0]
        check["evidence"] = [{"ref": next(ref for ref, value in lookup.items() if value == literal).replace("s:", "r0:", 1)}]
    response = {"reviews": [{"shot_id": "r0", "status": "supported", "reason": "独立判断",
        "labels": [], "evidence": [{"ref": "r0:f0:action"}], "primary_checks": checks}]}
    assert packet.valid_response(response)
    response["reviews"][0]["primary_checks"].pop()
    assert not packet.valid_response(response)
