import socket
from threading import Barrier

import pytest

from jinggao import config, neutral_observer, planner, visual_quality


@pytest.fixture(autouse=True)
def offline_review_config(monkeypatch, request):
    mode = getattr(request, "param", "strict")
    model = "qwen3.7-flash" if mode == "strict" else "deepseek-flash"
    cfg = config.DEFAULTS | {"model_api_key": "test-only-no-network", "model_name": model,
        "review_model": model, "pixel_observer_mode": "separate", "_strict_text_review_trial": mode == "strict",
        "model_base_url": ("https://dashscope.aliyuncs.com/compatible-mode/v1"
                           if mode == "strict" else "https://api.deepseek.com")}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    def forbidden(*args, **kwargs):
        pytest.fail("quality audit integration test attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    return mode


@pytest.fixture
def setup(monkeypatch):
    records = [{"shot": {"id": "s", "narration": "包裹沿输送带移动。", "persons": []},
                "asset": {"kind": "video"}, "frame_paths": []}]
    observed = {"s": {"scene_description": "输送带上的纸箱", "frames": [
        {"frame_index": 0, "objects": "输送带和纸箱", "action": "纸箱移动", "text": []}]}}
    qualities = {"s": {"representation": "live_action", "quality_status": "not_detected",
                        "quality_reason": "没有重复促销遮挡", "frame_sha256": ["snapshot"], "frames": []}}
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: (observed, 1))
    monkeypatch.setattr(visual_quality, "observe", lambda rows, check: (qualities, {"model_requests": 1}))
    calls = []
    def chat(*args, **kwargs):
        calls.append(args)
        return {"reviews": [{"shot_id": "s", "status": "supported", "reason": "纸箱在传送",
            "labels": [], "evidence": [{"frame_index": 0, "field": "action", "quote": "纸箱移动"}]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    return records, observed, qualities, calls


def test_pixel_obstruction_blocks_even_when_neutral_ocr_omitted_it(setup):
    records, _, qualities, calls = setup
    qualities["s"].update(quality_status="reject", quality_reason="点关注贴纸盖住主要操作界面")
    audit = planner.audit_selected_frames(records, lambda: None)["s"]
    assert audit["status"] == "conflict" and "点关注" in audit["reason"]
    assert audit["request_count"] == 2 and not calls
    result = planner.apply_identity_audit({"verdict": "match"}, audit)
    assert result["selection_blocked"] and result["visual_quality"] == qualities["s"]


@pytest.mark.parametrize("offline_review_config", ["strict", "legacy"], indirect=True)
def test_animation_needs_explicit_label_and_then_the_same_content_judge(setup, offline_review_config):
    records, _, qualities, calls = setup
    qualities["s"]["representation"] = "animation"
    assert planner.audit_selected_frames(records, lambda: None)["s"]["status"] == "unknown"
    assert not calls
    records[0]["shot"]["overlay_label"] = "动画示意"
    audit = planner.audit_selected_frames(records, lambda: None)["s"]
    assert audit["status"] == "supported" and len(calls) == 1
    assert audit["visual_quality"]["representation"] == "animation"
    # v60 uses one wording for both transports (planner evidence-review protocol).
    assert "动画/建模即使标明示意，也不能替代原稿要求的实际操作、实录人物或事件" in calls[0][0]


def test_quality_timeout_is_counted_and_does_not_become_a_pass(setup, monkeypatch):
    records, _, _, calls = setup
    def timeout(*a):
        raise visual_quality.QualityObservationError("timed out", request_count=1)
    monkeypatch.setattr(visual_quality, "observe", timeout)
    audit = planner.audit_selected_frames(records, lambda: None)["s"]
    assert audit["status"] == "unknown" and audit["request_count"] == 2 and not calls
    assert audit["visual_quality"]["quality_status"] == "uncertain"


def test_quality_account_failure_stops_before_text_audit(setup, monkeypatch):
    records, _, _, calls = setup
    def unavailable(*a):
        raise planner.ModelUnavailable("account unavailable")
    monkeypatch.setattr(visual_quality, "observe", unavailable)
    with pytest.raises(planner.ModelUnavailable):
        planner.audit_selected_frames(records, lambda: None)
    assert not calls


def test_two_blind_observers_run_together_and_total_requests_include_both(setup, monkeypatch):
    records, observations, qualities, calls = setup
    barrier = Barrier(2)
    def observe(rows, check):
        barrier.wait(timeout=2)
        return observations, 1
    def quality(rows, check):
        barrier.wait(timeout=2)
        return qualities, {"model_requests": 1}
    monkeypatch.setattr(neutral_observer, "observe", observe)
    monkeypatch.setattr(visual_quality, "observe", quality)
    audit = planner.audit_selected_frames(records, lambda: None)["s"]
    assert audit["status"] == "supported" and audit["request_count"] == 3 and len(calls) == 1
