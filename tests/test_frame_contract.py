"""Specific visual requirements cannot disappear inside a broad topic match."""
from copy import deepcopy

import pytest
from PIL import Image

from jinggao import frame_contract, neutral_observer, planner

pytestmark = pytest.mark.usefixtures("isolated_pixel_quality", "separate_pixel_observer")


@pytest.fixture
def record(tmp_path):
    frame = tmp_path / "mock-observed-frame.jpg"
    Image.new("RGB", (1280, 720)).save(frame)
    return {"shot": {"id": "purchase", "persons": [], "narration": "买东西的时候，手指只动了几下。",
                     "visual_plan": {"mode": "single", "primary": {
                         "subject": "手持手机", "action": "完成购买下单", "scene": "购物页面"}}},
            "asset": {"kind": "video", "title": "手机操作", "source_url": "https://www.bilibili.com/video/BV1234567890"},
            "frame_paths": [str(frame)]}


@pytest.fixture
def observation():
    # Reproduces the independently rejected shopping-vs-query failure from
    # round 5, without accepting the original model's broad topic verdict.
    return {"scene_description": "手机显示快递物流进度。", "frames": [
        {"frame_index": 0, "objects": "手机显示快递查询页面", "action": "手指点击查快递", "text": ["查快递"]}]}


def citation(field="action", quote="手指点击查快递"):
    return {"frame_index": 0, "field": field, "quote": quote}


def checks():
    # v60 has no per-caption caption_cN keys for legacy visual plans.
    return [{"key": key, "status": "supported", "reason": "已核验", "evidence": [citation()]}
            for key in ("subject", "action", "scene")]


def test_short_meeting_cut_requires_its_own_evidence_not_a_later_work_frame(monkeypatch):
    monkeypatch.setattr(frame_contract.editorial_review, "requirements", lambda shot: {"source_alignment": "说明实际物流劳动"})
    monkeypatch.setattr(frame_contract.editorial_review, "postproduction_text_checks", lambda shot: [])
    shot = {"editorial_contract": {"full_script": "屏幕上的几行物流进度，装不下包裹走过的路。"}}
    frames = [{"frame_index": 0, "objects": "会议室内人们坐在长桌前", "action": "与会者坐着听讲话", "text": []},
              {"frame_index": 1, "objects": "仓库内工人搬运纸箱", "action": "工人拿起包裹", "text": []}]
    work = {"frame_index": 1, "field": "action", "quote": frames[1]["action"]}
    ordinary = {"key": "source_alignment", "status": "supported", "reason": "后面有实际劳动", "evidence": [work]}
    # v60: only the meeting/studio-like frame gets its own check (not every frame).
    assert set(frame_contract.requirements(shot, frames)) == {"source_alignment", "frame_scope_0"}
    assert frame_contract.validate(shot, [ordinary], frames)[0] == "unknown"
    borrowed = {"key": "frame_scope_0", "status": "supported", "reason": "借用后面劳动帧", "evidence": [work]}
    assert frame_contract.validate(shot, [ordinary, borrowed], frames)[0] == "unknown"
    actual = borrowed | {"status": "conflict", "reason": "会议不是物流作业", "evidence": [
        {"frame_index": 0, "field": "objects", "quote": frames[0]["objects"]}]}
    assert frame_contract.validate(shot, [ordinary, actual], frames)[0] == "conflict"
    shot["editorial_contract"]["full_script"] = "介绍本次行业会议与物流作业。"
    # v60 policy: when the script itself covers meetings, a meeting cut needs no separate check.
    assert set(frame_contract.requirements(shot, frames)) == {"source_alignment"}


def test_broad_summary_cannot_overrule_failed_primary_action(record, observation, monkeypatch):
    detailed = checks()
    detailed[1].update(status="conflict", reason="实际是查询运输进度，不是购买下单")
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check: ({"purchase": observation}, 0))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"reviews": [{
        "shot_id": "purchase", "status": "supported", "reason": "手机主题相同",
        "labels": [], "evidence": [citation()], "primary_checks": detailed}]}, {}))
    result = planner.audit_selected_frames([record], lambda: None)["purchase"]
    assert result["status"] == "conflict"
    assert "动作" in result["reason"] and result["primary_checks"][1]["status"] == "conflict"


@pytest.mark.parametrize("change", ["omitted", "duplicate", "foreign-frame", "invented-quote", "no-evidence"])
def test_incomplete_or_fabricated_condition_proof_cannot_pass(record, observation, change):
    rows = checks()
    if change == "omitted":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "foreign-frame":
        rows[1]["evidence"][0]["frame_index"] = 5
    elif change == "invented-quote":
        rows[1]["evidence"][0]["quote"] = "订单支付成功"
    elif change == "no-evidence":
        rows[1]["evidence"] = []
    status, _, _ = frame_contract.validate(record["shot"], rows, observation["frames"])
    assert status == "unknown"


def test_matching_primary_retains_grounded_evidence(record, observation):
    record["shot"]["visual_plan"]["primary"].update(action="点击查快递", scene="快递查询页面")
    status, reason, result = frame_contract.validate(record["shot"], checks(), observation["frames"])
    assert status == "supported" and not reason
    assert result[1]["required"] == "点击查快递"


def test_optional_scene_is_not_invented_and_legacy_plan_is_not_rewritten(record, observation):
    primary = record["shot"]["visual_plan"]["primary"]
    primary["scene"] = ""
    assert set(frame_contract.requirements(record["shot"])) == {"subject", "action"}
    assert frame_contract.validate(record["shot"], [row for row in checks() if row["key"] != "scene"],
                                   observation["frames"])[0] == "supported"
    original = deepcopy(record["shot"])
    assert frame_contract.requirements(record["shot"] | {"visual_plan": None}) == {}
    assert record["shot"] == original
