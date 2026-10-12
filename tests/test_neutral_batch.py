"""The generic fast route keeps requirements and accounting without a duplicate vision call."""
import json

import pytest
from PIL import Image

from jinggao import planner, pooled_match


@pytest.fixture
def rows(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (128, 72), "white").save(frame)
    monkeypatch.setattr(pooled_match, "sample_frames", lambda *args: ([frame] * 5, {"motion_detected": True}))
    return [{"shot": {"id": sid, "persons": [], "duration": 5, "clip_start": 0, "narration": "包裹运输",
                      "visual_plan": {"mode": "single", "primary": {
                          "subject": "包裹", "action": "随输送带移动", "scene": "分拨中心"}}},
             "asset": {"id": sid, "title": sid, "kind": "video", "duration": 5,
                       "width": 1280, "height": 720, "source_url": "https://www.iqiyi.com/v_test.html"}}
            for sid in ("good", "bad")]


def audit(records, check):
    assert all(r["requirements"] == "必须使用真实工厂作业画面" for r in records)
    assert all(len(r["frame_paths"]) == 5 for r in records)
    return {r["shot"]["id"]: {
        "status": "supported" if r["shot"]["id"] == "good" else "conflict",
        "reason": "实际画面结论", "request_count": 2, "scene_description": "实际输送带上的包裹",
        "frame_observations": []} for r in records}


def test_generic_batch_keeps_requirements_and_rejections_with_two_calls(rows, monkeypatch):
    monkeypatch.setattr(planner, "audit_selected_frames", audit)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("generic route must not upload frames twice"))
    reviewed, errors, info = pooled_match.review_batch(
        {"id": "test", "requirements": "必须使用真实工厂作业画面"}, rows, lambda: None)
    assert not errors and info["model_requests"] == 2
    assert [r["verification"]["verdict"] for r in reviewed] == ["match", "mismatch"]
    assert reviewed[1]["verification"]["selection_blocked"]
    assert reviewed[0]["verification"]["observations"]["observation_method"] == "blind_actual_frames_v1"


@pytest.mark.parametrize("identity_fields", [
    {"persons": [{"name": "刘翔", "aliases": []}]},
    {"persons": [], "event": "某场具体比赛", "event_scope": "exact"},
])
def test_person_or_event_still_uses_source_evidence_route(rows, monkeypatch, identity_fields):
    rows[1]["shot"].update(identity_fields)
    monkeypatch.setattr(planner, "audit_selected_frames", audit)
    def chat(system, content, **kwargs):
        assert kwargs["vision"] is True
        ids = [json.loads(p["text"])["shot_id"] for p in content
               if p["type"] == "text" and "shot_id" in p["text"]]
        assert set(ids) == {"bad"}
        return {"reviews": [{"shot_id": "bad", "verdict": "mismatch", "reason": "人物或事件不符",
                             "conflicts": ["明确不符"], "needs_source_check": False,
                             "scene_alignment": "mismatch", "scene_reason": "不符"}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    reviewed, errors, info = pooled_match.review_batch(
        {"id": "test", "requirements": "必须使用真实工厂作业画面"}, rows, lambda: None)
    assert len(reviewed) == 2 and not errors and info["model_requests"] == 3
    assert reviewed[1]["verification"]["selection_blocked"]


def test_generic_account_failure_propagates(rows, monkeypatch):
    def unavailable(*args):
        raise planner.ModelUnavailable("模型余额不足")
    monkeypatch.setattr(planner, "audit_selected_frames", unavailable)
    with pytest.raises(planner.ModelUnavailable):
        pooled_match.review_batch({"id": "test"}, rows, lambda: None)
