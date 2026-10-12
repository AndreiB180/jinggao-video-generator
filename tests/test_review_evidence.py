"""Observation refs preserve exact evidence without weakening attribution."""
from copy import deepcopy
import json

import pytest

from jinggao import neutral_observer, planner, review_evidence


FRAMES = [{"frame_index": 0, "objects": "传送带旁的人与包裹",
           "action": "手持设备靠近包裹上的面单", "text": ["快件处理员", "国家职业技能标准"]}]


def test_refs_are_shot_scoped_and_never_mutate_original_observations():
    frames = deepcopy(FRAMES)
    public, lookup = review_evidence.references("shot-a", frames)
    other, _ = review_evidence.references("shot-b", frames)
    assert {r["ref"] for r in public}.isdisjoint(r["ref"] for r in other)
    ref = next(r["ref"] for r in public if r["field"] == "action")
    row = {"evidence": [{"ref": ref}], "labels": [], "primary_checks": [
        {"key": "action", "status": "supported", "evidence": [{"ref": ref}]}]}
    raw = deepcopy(row)
    result = review_evidence.resolve(row, lookup)
    assert result["evidence"] == [{"frame_index": 0, "field": "action", "quote": frames[0]["action"]}]
    assert result["primary_checks"][0]["evidence"] == result["evidence"]
    assert all(neutral_observer.grounded(r, frames) for r in result["evidence"])
    assert row == raw and frames == FRAMES


@pytest.mark.parametrize("bad", [
    {"ref": "shot-b:f0:action"}, {"ref": "shot-a:f9:action"}, {"ref": []},
    {"ref": "shot-a:f0:action", "quote": "正在支付"},
    {"frame_index": 0, "field": "action", "quote": "正在支付"},
])
def test_invalid_and_altered_refs_are_not_repaired(bad):
    _, lookup = review_evidence.references("shot-a", FRAMES)
    result = review_evidence.resolve({"evidence": [bad], "labels": []}, lookup)
    assert not neutral_observer.grounded(result["evidence"][0], FRAMES)


def test_objects_cannot_become_person_name_label():
    _, lookup = review_evidence.references("shot-a", FRAMES)
    row = review_evidence.resolve({"labels": [{"ref": "shot-a:f0:objects", "relation": "on_subject"}],
                                  "evidence": []}, lookup)
    assert not neutral_observer.grounded(row["labels"][0], FRAMES, label=True)


def test_redundant_provider_fields_require_exact_agreement_with_reference():
    _, lookup = review_evidence.references("shot-a", FRAMES)
    literal = {"frame_index": 0, "field": "action", "quote": FRAMES[0]["action"]}
    name = {"frame_index": 0, "field": "text", "quote": FRAMES[0]["text"][0], "relation": "headline"}
    row = {"labels": [{"ref": "shot-a:f0:text:0", **name}],
           "evidence": [{"ref": "shot-a:f0:action", **literal}],
           "primary_checks": [{"evidence": [{"ref": "shot-a:f0:action", **literal}]}]}
    original = deepcopy(row)
    result = review_evidence.resolve(row, lookup)
    assert result["evidence"] == result["primary_checks"][0]["evidence"] == [literal]
    assert result["labels"] == [name] and row == original


@pytest.mark.parametrize("change", [
    {"ref": "shot-b:f0:action"}, {"ref": "shot-a:f0:objects"},
    {"frame_index": 1}, {"frame_index": False}, {"quote": "手持设备"},
    {"field": "objects"}, {"unexpected": "ignored"},
])
def test_invalid_ref_metadata_cannot_fall_back_to_literal_grounding(change):
    _, lookup = review_evidence.references("shot-a", FRAMES)
    citation = {"ref": "shot-a:f0:action", "frame_index": 0,
                "field": "action", "quote": FRAMES[0]["action"]} | change
    result = review_evidence.resolve({"evidence": [citation], "labels": []}, lookup)
    assert not neutral_observer.grounded(result["evidence"][0], FRAMES)


@pytest.mark.usefixtures("isolated_pixel_quality")
@pytest.mark.parametrize("status", ["supported", "conflict", "unknown"])
def test_real_audit_resolves_refs_but_preserves_the_semantic_verdict(monkeypatch, status):
    shot = planner.new_shot(narration="处理包裹仍需要人。", query="快递")
    shot["visual_plan"] = {"mode": "single", "primary": {"action": "人处理包裹"}}
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "手持设备靠近面单", "frames": FRAMES}}, 0))
    raw = []
    def judge(system, content, **kwargs):
        payload = json.loads(content[0]["text"])
        assert payload["frame_observations"] == FRAMES
        assert "observation_references" in system
        ref = next(r["ref"] for r in payload["observation_references"] if r["field"] == "action")
        row = {"shot_id": shot["id"], "status": status, "reason": "按原稿判断实际动作", "labels": [],
               "evidence": [{"ref": ref}], "primary_checks": [
                   {"key": "action", "status": status, "reason": "实际动作", "evidence": [{"ref": ref}]}]}
        row["primary_checks"].extend({"key": f"caption_c{i+1}", "status": status,
            "reason": "逐条原片文字", "evidence": [{"ref": f"{shot['id']}:f0:text:{i}"}]}
            for i in range(len(FRAMES[0]["text"])))
        raw.append(deepcopy(row))
        return {"reviews": [row]}, {}
    monkeypatch.setattr(planner, "chat", judge)
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"},
                                            "frame_paths": []}], lambda: None)[shot["id"]]
    assert result["status"] == status
    assert result["evidence"][0]["quote"] == FRAMES[0]["action"]
    assert result["raw_review"] == raw[0]
