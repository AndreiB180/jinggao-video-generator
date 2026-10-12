"""Opt-in integration preserves the production review gates; no external calls."""
from copy import deepcopy

import pytest

from jinggao import config, costs, media, neutral_observer, paths, pixel_observer, planner, source_planner, visual_quality, window_attempts


@pytest.fixture
def configured(monkeypatch, tmp_path):
    cfg = config.DEFAULTS | {"model_api_key": "test-only", "source_plan_mode": "unified",
                             "pixel_observer_mode": "joint"}
    monkeypatch.setattr(config, "read", lambda: dict(cfg))
    monkeypatch.setattr(paths, "DATA", tmp_path)
    for module, name in ((neutral_observer, "observe"), (visual_quality, "observe"),
                         (planner, "_plan_model_batch")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("No paid legacy fallback"))
    monkeypatch.setattr(planner, "chat", lambda *a, **k: pytest.fail("Unexpected model request"))
    return cfg


@pytest.mark.parametrize("mode", ["auto", "model"])
def test_unified_plan_uses_original_project_and_shared_cancellation_progress(configured, monkeypatch, mode):
    project = {"script": "  包裹沿输送带移动。\n", "requirements": "保持原文"}
    seen, stages = [], []
    def check():
        seen.append("check")
    def plan(p, *, chat, check, progress):
        assert p is project and chat is planner.chat
        check()
        progress(1, 1)
        return [{"narration": p["script"]}], {"model_requests": 1, "cached_batches": 0, "method": "source_planner_v1"}
    monkeypatch.setattr(source_planner, "plan", plan)
    rows, info = planner.plan(project, mode, False, check=check, progress=lambda n, t: stages.append((n, t)))
    assert rows[0]["narration"] == project["script"] and stages == [(1, 1)]
    assert len(seen) == 2 and info["planning_model_requests"] == 1


def test_rule_mode_does_not_call_unified_model(configured, monkeypatch):
    monkeypatch.setattr(source_planner, "plan", lambda *a, **k: pytest.fail("Rules must stay local"))
    rows, info = planner.plan({"script": "包裹沿输送带移动。"}, "rules", False)
    assert info["method"] == "rules" and rows[0]["narration"] == "包裹沿输送带移动。"


def test_unified_failure_propagates_without_replanning(configured, monkeypatch):
    failure = ValueError("source quote mismatch")
    failure.request_count = failure.model_requests = 1
    def fail(*a, **k):
        raise failure
    monkeypatch.setattr(source_planner, "plan", fail)
    with pytest.raises(ValueError) as caught:
        planner.plan({"script": "包裹沿输送带移动。"}, "model", False)
    assert caught.value is failure and failure.request_count == 1


@pytest.mark.parametrize("rewrite,setting", [(True, "unified"), (False, "legacy")])
def test_rewrite_or_legacy_setting_retains_existing_plan(configured, monkeypatch, rewrite, setting):
    configured["source_plan_mode"] = setting
    monkeypatch.setattr(source_planner, "plan", lambda *a, **k: pytest.fail("Wrong source route"))
    def sentinel(*a, **k):
        raise RuntimeError("existing plan selected")
    monkeypatch.setattr(planner, "_plan_model_batch", sentinel)
    with pytest.raises(RuntimeError, match="existing plan selected"):
        planner.plan({"script": "包裹沿输送带移动。"}, "model", rewrite)


@pytest.fixture
def joint(configured, monkeypatch):
    records = [{"shot": {"id": "s", "narration": "包裹沿输送带移动。", "persons": []},
                "asset": {"kind": "video"}, "frame_paths": []}]
    facts = {"s": {"scene_description": "输送带上的纸箱", "frames": [
        {"frame_index": 0, "objects": "输送带和纸箱", "action": "纸箱移动", "text": []}]}}
    quality = {"s": {"representation": "live_action", "quality_status": "not_detected",
                      "quality_reason": "未发现遮挡", "frames": []}}
    stats = {"model_requests": 1, "request_count": 1, "all_cache_hit": False, "task": pixel_observer.TASK}
    response = {"reviews": [{"shot_id": "s", "status": "supported", "reason": "纸箱在传送",
                            "labels": [], "evidence": [{"frame_index": 0, "field": "action", "quote": "纸箱移动"}]}]}
    observed, judged = [], []
    def observe(rows, check):
        check()
        observed.append(deepcopy(rows))
        return facts, quality, stats
    def chat(*args, **kwargs):
        judged.append((args, kwargs))
        return response, {"request_count": 1}
    monkeypatch.setattr(pixel_observer, "observe", observe)
    monkeypatch.setattr(planner, "chat", chat)
    return records, facts, quality, stats, response, observed, judged


def test_joint_observer_still_requires_grounded_text_review_and_counts_once(joint):
    records, _, _, _, _, observed, judged = joint
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["status"] == "supported" and row["request_count"] == 2
    assert len(observed) == len(judged) == 1
    assert row["observation_info"]["task"] == pixel_observer.TASK
    assert not judged[0][1]["vision"] and judged[0][1]["task"] == "evidence_review"
    saved = planner.apply_identity_audit({"verdict": "match"}, row)
    assert saved["identity_audit"]["observation_info"] == row["observation_info"]


@pytest.mark.parametrize("state,expected", [("reject", "conflict"), ("uncertain", "unknown")])
def test_joint_quality_gate_blocks_before_text(joint, state, expected):
    records, _, quality, _, _, _, judged = joint
    quality["s"].update(quality_status=state, quality_reason="广告覆盖主体")
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["status"] == expected and row["request_count"] == 1 and not judged


def test_joint_animation_still_needs_label(joint):
    records, _, quality, _, _, _, judged = joint
    quality["s"]["representation"] = "animation"
    assert planner.audit_selected_frames(records, lambda: None)["s"]["status"] == "unknown"
    assert not judged


def test_joint_text_cannot_invent_pixel_evidence(joint):
    records, _, _, _, response, _, _ = joint
    response["reviews"][0]["evidence"][0]["quote"] = "完成扫码付款"
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["status"] == "unknown" and "逐字" in row["reason"]


def test_joint_named_person_still_needs_attribution(joint):
    records, _, _, _, _, _, _ = joint
    records[0]["shot"]["persons"] = [{"name": "刘翔", "aliases": []}]
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["status"] == "unknown" and "姓名" in row["reason"]


def test_joint_cache_hit_is_not_a_new_charge(joint):
    records, _, _, stats, _, _, _ = joint
    stats.update(model_requests=0, request_count=0, all_cache_hit=True)
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["request_count"] == 1 and row["observation_info"]["all_cache_hit"]


def test_joint_failure_never_falls_back_or_calls_text(joint, monkeypatch):
    records, _, _, _, _, _, judged = joint
    def fail(*a):
        raise pixel_observer.PixelObservationError("response incomplete", request_count=1)
    monkeypatch.setattr(pixel_observer, "observe", fail)
    row = planner.audit_selected_frames(records, lambda: None)["s"]
    assert row["status"] == "unknown" and row["request_count"] == 1 and not judged


@pytest.mark.parametrize("kind", [planner.ModelUnavailable, costs.BudgetExceeded, costs.PricingUnavailable, media.Cancelled])
def test_joint_fatal_stays_fatal(joint, monkeypatch, kind):
    records, _, _, _, _, _, judged = joint
    failure = kind("stop")
    failure.request_count = 1
    def fail(*a):
        raise failure
    monkeypatch.setattr(pixel_observer, "observe", fail)
    with pytest.raises(kind) as caught:
        planner.audit_selected_frames(records, lambda: None)
    assert caught.value is failure and not judged


def test_rejection_cache_binds_observer_mode_and_new_protocol_files(configured):
    joint = window_attempts.policy()
    configured["pixel_observer_mode"] = "separate"
    separate = window_attempts.policy()
    assert joint != separate
    assert {"pixel_observer", "pixel_packet", "source_planner"} <= set(joint["files"])


@pytest.mark.parametrize("key", ["source_plan_mode", "pixel_observer_mode"])
def test_invalid_pipeline_setting_rejected_before_write(configured, key):
    with pytest.raises(ValueError):
        config.update({key: "invalid"})
