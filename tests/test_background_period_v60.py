"""Ordinary B-roll reaches caption review without a mandatory publication date."""
from copy import deepcopy
import json
import socket

import pytest

from jinggao import config, context_labels, curation, editorial_review, frame_sampling
from jinggao import neutral_observer, planner, privacy, visual_quality
from tests.test_editorial_background_v60 import make_shot


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("background-period regression attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {"pixel_observer_mode": "separate"})


def sample(text="日常物流运输带来便利。", *, caption="目前日处理量100万件", strict=False):
    shot = make_shot(text)
    shot.update(duration=8, clip_start=0, overlay_label="资料画面")
    if strict:
        shot["locked"] = True
    frames = [{"frame_index": 0, "objects": "物流仓库和车辆", "action": "车辆在仓库场地中行驶", "text": [caption]}]
    asset = {"id": "test-asset", "kind": "video", "title": "物流作业", "path": "test-only.mp4"}
    return shot, asset, frames


def controlled_audit(monkeypatch, shot, asset, frames, *, caption_status="supported"):
    calls = []
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "物流运输场景", "frames": frames}}, 0))
    monkeypatch.setattr(visual_quality, "observe", lambda rows, check: ({shot["id"]: {
        "representation": "live_action", "quality_status": "not_detected", "frames": []}}, {"model_requests": 0}))
    def judge(system, content, **options):
        record = json.loads(content[0]["text"])
        calls.append((system, record))
        action = [{"frame_index": 0, "field": "action", "quote": frames[0]["action"]}]
        text = [{"frame_index": 0, "field": "text", "quote": frames[0]["text"][0]}]
        checks = [{"key": key, "status": caption_status if key == "caption_context" else "supported",
                   "reason": "受控测试已读完整字幕", "evidence": text if key == "caption_context" else action}
                  for key in record["selected_visual_requirements"]]
        return {"reviews": [{"shot_id": shot["id"], "status": "supported", "reason": "受控背景配图判断",
                              "labels": [], "evidence": action, "primary_checks": checks}]}, {"request_count": 0}
    monkeypatch.setattr(planner, "chat", judge)
    result = planner.audit_selected_frames([{"shot": shot, "asset": asset, "frame_paths": []}], lambda: None)[shot["id"]]
    return result, calls


@pytest.mark.parametrize("strict", [False, True])
def test_missing_reporting_date_blocks_only_strict_path_before_text_review(monkeypatch, strict):
    shot, asset, frames = sample(strict=strict)
    result, calls = controlled_audit(monkeypatch, shot, asset, frames)
    if strict:
        assert result["status"] == "unknown" and calls == []
        assert "报道日期" in result["reason"]
    else:
        assert result["status"] == "supported" and len(calls) == 1
        assert calls[0][1]["context_label_context"]["captions"][0]["text"] == frames[0]["text"][0]
        assert calls[0][1]["source_context"]["alignment_policy"] == "related_background"
        assert "context_label" not in result


@pytest.mark.parametrize("strict", [False, True])
def test_different_background_statistical_year_is_not_a_post_review_veto(monkeypatch, strict):
    shot, asset, frames = sample("2024年物流业务量持续增长。", caption="2022年物流业务量增长", strict=strict)
    result, calls = controlled_audit(monkeypatch, shot, asset, frames)
    assert len(calls) == 1
    assert result["status"] == ("unknown" if strict else "supported")
    if strict:
        assert "统计年度" in result["reason"]


def test_ordinary_background_still_rejects_an_actual_caption_conflict(monkeypatch):
    shot, asset, frames = sample()
    result, calls = controlled_audit(monkeypatch, shot, asset, frames, caption_status="conflict")
    assert len(calls) == 1 and result["status"] == "conflict"


@pytest.mark.parametrize("strict", [False, True])
def test_saved_selection_does_not_reintroduce_date_requirement(strict):
    shot, asset, frames = sample(strict=strict)
    shot["verification"] = {"identity_audit": {"frame_observations": frames}}
    error = context_labels.selection_error(shot, asset)
    assert bool(error) is strict
    if strict:
        assert "报道日期" in error


def test_existing_automatic_label_integrity_is_still_required():
    shot, asset, frames = sample()
    shot.update(overlay_label_origin="source_context", verification={"identity_audit": {"frame_observations": frames}})
    assert "未绑定" in context_labels.selection_error(shot, asset)
    shot["verification"].update(context_label={"version": 1}, reviewed_overlay_label="改过的标签")
    assert context_labels.selection_error(shot, asset)


@pytest.mark.parametrize("strict", [False, True])
def test_curation_skips_only_the_ordinary_period_veto_and_keeps_file_checks(monkeypatch, strict):
    shot, asset, frames = sample("2024年物流业务量持续增长。", caption="2022年物流业务量增长", strict=strict)
    shot["verification"] = {"identity_audit": {"frame_observations": frames, "primary_checks": [
        {"key": key} for key in ("source_alignment", "caption_context")]}}
    project = {"script": shot["narration"], "requirements": "", "shots": [shot], "assets": [asset]}
    calls = []
    monkeypatch.setattr(privacy, "selection_error", lambda *a, **k: "")
    monkeypatch.setattr(frame_sampling, "selection_error", lambda *a, **k: calls.append("files") or "file-check")
    monkeypatch.setattr(planner, "statistical_caption_conflict", lambda *a, **k: calls.append("period") or "period-veto")
    before = deepcopy(shot)
    assert editorial_review.alignment_policy(shot, project) == ("strict_source" if strict else "related_background")
    error = curation.selection_error(project, shot, asset)
    assert error == ("period-veto" if strict else "file-check")
    assert calls == (["period"] if strict else ["files"])
    assert shot == before
