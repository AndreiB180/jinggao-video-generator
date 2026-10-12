"""Independent visual quality stays pixel-bound, bounded and non-editorial."""
import asyncio
from copy import deepcopy
import json
import time

import pytest
from PIL import Image

from jinggao import media, planner, visual_quality as quality


def frame(index=0, *, representation="live_action", overlay=None):
    return {"frame_index": index, "representation": representation,
            "type_evidence": "手与实体手机接触，现实背景纹理" if representation == "live_action" else "白背景下规则几何建模材质",
            "overlays": [] if overlay is None else [overlay]}


def sticker(kind="engagement_cta", impact="material"):
    return {"kind": kind, "text": "点关注", "bbox": [.4, .4, .65, .65],
            "impact": impact, "occludes": "中央手机聊天单号" if impact == "material" else ""}


def test_promotion_needs_spatial_obstruction_not_a_word_or_news_lower_third():
    observed = {"frames": [frame(i, overlay=sticker()) for i in range(5)]}
    result = quality.assess(observed)
    assert result["quality_status"] == "reject" and result["representation"] == "live_action"
    assert len(result["obstruction_evidence"]) == 5
    assert not result["content_approved"] and not result["authenticity_assessed"]
    for overlay in [sticker("caption"), sticker("interface"), sticker(impact="minor")]:
        control = quality.assess({"frames": [frame(i, overlay=overlay) for i in range(5)]})
        assert control["quality_status"] == "not_detected"
    single = quality.assess({"frames": [frame(0, overlay=sticker()), frame(1)]})
    assert single["quality_status"] == "uncertain"


def test_animation_mixed_and_uncertain_are_types_not_automatic_quality_failures():
    for kind in ("animation", "mixed", "uncertain", "screen_capture"):
        result = quality.assess({"frames": [frame(i, representation=kind) for i in range(5)]})
        assert result["representation"] == kind and result["quality_status"] == "not_detected"
        assert "approved_seconds" not in result and not result["content_approved"]
    mixed = quality.assess({"frames": [frame(0), frame(1, representation="animation")]})
    assert mixed["representation"] == "mixed"


def test_recorded_phone_interface_and_real_scene_are_not_animation_mix():
    recorded = [frame(0, representation="screen_capture"), frame(1)]
    result = quality.assess({"frames": recorded})
    assert result["representation"] == "recorded_mix"
    assert result["representation_counts"] == {"live_action": 1, "screen_capture": 1}
    assert not result["content_approved"]
    for kind in ("animation", "mixed", "uncertain"):
        result = quality.assess({"frames": [*recorded, frame(2, representation=kind)]})
        assert result["representation"] == ("uncertain" if kind == "uncertain" else "mixed")


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_frame", "invalid_box", "missing_basis"])
def test_incomplete_or_ungrounded_spatial_observations_are_not_cached_as_good(change):
    group = {"id": "g0", "frames": [frame(0, overlay=sticker()), frame(1)]}
    other = {"id": "g1", "frames": [frame(0), frame(1)]}
    if change == "duplicate_id":
        other["id"] = "g0"
    elif change == "duplicate_frame":
        group["frames"][1]["frame_index"] = 0
    elif change == "invalid_box":
        group["frames"][0]["overlays"][0]["bbox"] = [.7, 0, .1, 1]
    else:
        group["frames"][0]["overlays"][0]["occludes"] = ""
    with pytest.raises(ValueError):
        quality.parse_groups(json.dumps({"groups": [group, other]}), {"g0": 2, "g1": 2})


@pytest.fixture
def environment(tmp_path, monkeypatch):
    cfg = {"vision_model": "vision-test", "model_base_url": "https://model.test", "model_api_key": "test-only-secret"}
    cfg.update(cost_input_per_million="2", cost_cached_per_million="0.04", cost_output_per_million="8", cost_image_tokens="1024")
    monkeypatch.setattr(quality.config, "read", lambda: deepcopy(cfg))
    monkeypatch.setattr(quality.paths, "DATA", tmp_path)
    records = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (96, 96), (30+i*30, 0, 0)).save(path)
        records.append({"shot": {"id": str(i), "narration": "不要发送原稿", "visual": "不要发送期望画面"},
                        "asset": {"title": "不要发送来源标题"}, "verification": {"reason": "不要发送旧审核"},
                        "frame_paths": [path]})
    return cfg, records


def test_two_request_cap_cache_dedup_and_pixel_changes(environment, monkeypatch, tmp_path):
    cfg, records = environment
    calls = []
    def request(content, cfg, check, deadline):
        calls.append(content)
        groups = {}
        for row in content:
            if row["type"] == "text":
                obj = json.loads(row["text"])
                assert set(obj) == {"group_id", "frame_index"}
                groups.setdefault(obj["group_id"], []).append(frame(obj["frame_index"]))
        assert len(groups) <= 2
        return json.dumps({"groups": [{"id": k, "frames": v} for k, v in groups.items()]}), {"model": cfg["vision_model"]}
    monkeypatch.setattr(quality, "_request", request)
    rows, info = quality.observe(records, lambda: None, diagnostics_dir=tmp_path / "diagnostics")
    assert info["model_requests"] == 2 and len(rows) == 4
    sent = json.dumps(calls, ensure_ascii=False)
    assert "不要发送" not in sent and cfg["model_api_key"] not in sent
    for path in (tmp_path / "diagnostics").glob("*.json"):
        assert cfg["model_api_key"] not in path.read_text()
    same = [records[0], records[0] | {"shot": {"id": "other", "narration": "不同原稿"}}]
    cached, stats = quality.observe(same, lambda: None)
    assert stats["model_requests"] == 0 and stats["unique_groups"] == 1
    assert cached["other"]["cached"] and cached["0"]["frame_sha256"] == cached["other"]["frame_sha256"]
    Image.new("RGB", (96, 96), "white").save(records[0]["frame_paths"][0])
    assert quality.observe(records[:1], lambda: None)[1]["model_requests"] == 1
    with pytest.raises(ValueError, match="最多4组"):
        quality.observe(records + [records[0]], lambda: None)


@pytest.mark.parametrize("cancel", [False, True])
def test_real_deadline_and_cancellation_cancel_inflight_http_and_release_only_owned_slot(environment, monkeypatch, cancel):
    cfg, _ = environment
    state = {"started": False, "cancelled": False, "closed": False, "acquired": False, "released": False}
    class Slot:
        def acquire(self, timeout):
            state["acquired"] = True
            return True
        def release(self):
            assert state["acquired"]
            state["released"] = True
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            state["closed"] = True
        async def post(self, *args, **kwargs):
            state["started"] = True
            try:
                await asyncio.sleep(10)
            finally:
                state["cancelled"] = True
    def check():
        if cancel and state["started"]:
            raise media.Cancelled()
    monkeypatch.setattr(quality.execution, "MODEL_SLOTS", Slot())
    monkeypatch.setattr(quality.httpx, "AsyncClient", Client)
    began = time.monotonic()
    with pytest.raises(media.Cancelled if cancel else quality.QualityObservationError) as failure:
        quality._request([], cfg, check, began + (.4 if cancel else .04))
    assert time.monotonic()-began < .7 and all(state.values())
    assert failure.value.request_count == failure.value.model_requests == 1


def test_account_failure_stops_before_second_batch_and_partial_json_stays_uncertain(environment, monkeypatch):
    _, records = environment
    attempts = []
    def unavailable(*args):
        attempts.append(1)
        failure = planner.ModelUnavailable("HTTP 402")
        failure.request_count = failure.model_requests = 1
        raise failure
    monkeypatch.setattr(quality, "_request", unavailable)
    with pytest.raises(planner.ModelUnavailable) as failure:
        quality.observe(records, lambda: None)
    assert len(attempts) == 1 and failure.value.request_count == failure.value.model_requests == 1
    monkeypatch.setattr(quality, "_request", lambda *args: ('{"groups":', {"finish_reason": "length"}))
    rows, info = quality.observe(records[:1], lambda: None)
    assert info["model_requests"] == 1 and rows["0"]["quality_status"] == "uncertain"
    assert rows["0"]["representation"] == "uncertain" and not rows["0"]["content_approved"]


def test_extra_valid_overlays_are_retained_and_later_obstructions_still_reject():
    frames = [frame(i) | {"overlays": [sticker("watermark", "minor"),
                sticker("caption", "minor"), sticker("interface", "minor"), sticker()]} for i in range(2)]
    parsed = quality.parse_groups(json.dumps({"groups": [{"id": "g0", "frames": frames}]}), {"g0": 2})
    assert all(len(row["overlays"]) == 4 for row in parsed["g0"]["frames"])
    assert quality.assess(parsed["g0"])["quality_status"] == "reject"
    frames[0]["overlays"][3]["bbox"] = [.8, 0, .2, 1]
    with pytest.raises(ValueError, match="有效位置"):
        quality.parse_groups(json.dumps({"groups": [{"id": "g0", "frames": frames}]}), {"g0": 2})
