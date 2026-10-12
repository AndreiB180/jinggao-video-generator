"""Blind image facts stay separate from editorial requirements and judgments."""
import copy
import json

import httpx
from PIL import Image
import pytest

from jinggao import neutral_observer as observer
from jinggao import media, planner


def group(ident="g0", count=1):
    return {"id": ident, "scene_description": "桌面摆放一把刷子。", "frames": [
        {"frame_index": i, "objects": "桌面和一把刷子", "action": "刷子平放", "text": ["唇刷"]}
        for i in range(count)]}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    cfg = {"vision_model": "vision-test", "model_name": "text-test", "model_api_key": "test-only-secret",
           "model_base_url": "https://models.test"}
    cfg.update(cost_input_per_million="2", cost_cached_per_million="0.04", cost_output_per_million="8", cost_image_tokens="1024")
    monkeypatch.setattr(observer.config, "read", lambda: cfg.copy())
    monkeypatch.setattr(observer.paths, "DATA", tmp_path)
    monkeypatch.setattr(observer.network, "post", lambda *a, **kw: pytest.fail("unmocked network request"))
    image = tmp_path / "actual.jpg"
    Image.new("RGB", (1280, 720), "white").save(image)
    record = {"shot": {"id": "private-shot", "narration": "不可泄露的原稿", "visual": "不可泄露的构图"},
              "asset": {"title": "不可泄露的源标题", "source_url": "https://example.test/private-source"},
              "verification": {"reason": "不可泄露的首审"}, "frame_paths": [image]}
    return cfg, record


def responder(calls):
    def request(content, cfg, check):
        calls.append(content)
        counts = {}
        manifest = json.loads(content[0]["text"])["observation_groups"]
        for part in content[1:]:
            if part["type"] == "text":
                value = json.loads(part["text"])
                assert set(value) == {"group_id", "frame_index"}
                counts[value["group_id"]] = counts.get(value["group_id"], 0) + 1
        assert manifest == [{"group_id": ident, "frame_indices": list(range(count))}
                            for ident, count in counts.items()]
        return json.dumps({"groups": [group(k, n) for k, n in counts.items()]}, ensure_ascii=False), {}
    return request


def test_complete_adjacent_groups_are_accepted_without_synthesizing_observations():
    a, b = group("A", 2), group("B")
    a["group"] = a.pop("id")
    a["changes"] = a.pop("scene_description")
    for frame in a["frames"]:
        frame["i"] = frame.pop("frame_index")
    raw = json.dumps(a, ensure_ascii=False) + "\n" + json.dumps(b, ensure_ascii=False)
    result = observer.parse_groups(raw, {"A": 2, "B": 1})
    assert result["A"]["frames"] == group("A", 2)["frames"]
    assert result["B"]["scene_description"] == b["scene_description"]


@pytest.mark.parametrize("mutate", [
    lambda groups: groups.append(group("g0")),
    lambda groups: groups[0].update(id="unknown"),
    lambda groups: groups[0].update(group="contradictory"),
    lambda groups: groups[0]["frames"].append(copy.deepcopy(groups[0]["frames"][0])),
    lambda groups: groups[0]["frames"][0].update(frame_index=True),
    lambda groups: groups[0]["frames"][0].update(frame_index=1),
    lambda groups: groups[0]["frames"][0].update(i=1),
    lambda groups: groups[0]["frames"][0].update(text="唇刷"),
    lambda groups: groups[0]["frames"][0].pop("action"),
])
def test_invalid_ids_frames_and_facts_are_rejected(mutate):
    groups = [group()]
    mutate(groups)
    with pytest.raises(ValueError):
        observer.parse_groups(json.dumps({"groups": groups}), {"g0": 1})


def test_equal_count_does_not_hide_duplicate_ids_or_frame_indices():
    with pytest.raises(ValueError, match="重复"):
        observer.parse_groups(json.dumps({"groups": [group("A"), group("A")]}), {"A": 1, "B": 1})
    duplicated = group(count=2)
    duplicated["frames"][1]["frame_index"] = 0
    with pytest.raises(ValueError, match="重复"):
        observer.parse_groups(json.dumps({"groups": [duplicated]}), {"g0": 2})


@pytest.mark.parametrize("tail", ['\n{"id":', " extra words"])
def test_partial_or_trailing_json_is_never_repaired(tail):
    with pytest.raises(ValueError, match="不完整"):
        observer.parse_groups(json.dumps(group()) + tail, {"g0": 1})


def test_blind_request_and_cache_never_include_editorial_metadata(environment, monkeypatch, tmp_path):
    cfg, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    observations, count = observer.observe([record], lambda: None)
    assert count == 1 and observations["private-shot"]["frames"] == group()["frames"]
    prompt = json.dumps(calls, ensure_ascii=False)
    assert "不可泄露" not in prompt and "private-shot" not in prompt and "private-source" not in prompt
    assert len([b for b in calls[0] if b["type"] == "image_url"]) == 1
    files = list((tmp_path / "cache" / "neutral-observer").glob("*.json"))
    cached = files[0].read_text()
    assert "不可泄露" not in cached and cfg["model_api_key"] not in cached
    changed = record | {"shot": {"id": "other-shot", "narration": "完全不同的稿件"}}
    second, second_count = observer.observe([changed], lambda: None)
    assert second_count == 0 and second["other-shot"] == observations["private-shot"] and len(calls) == 1


def test_identical_actual_frames_coalesce_even_for_distinct_shots(environment, monkeypatch):
    _, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    other = record | {"shot": {"id": "other"}}
    result, count = observer.observe([record, other], lambda: None)
    assert count == 1 and len([b for b in calls[0] if b["type"] == "image_url"]) == 1
    assert result["private-shot"] == result["other"]


@pytest.mark.parametrize("change", ["pixels", "model", "protocol", "endpoint"])
def test_cache_is_bound_to_actual_pixels_model_protocol_and_endpoint(environment, monkeypatch, change):
    cfg, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    observer.observe([record], lambda: None)
    if change == "pixels":
        Image.new("RGB", (1280, 720), "black").save(record["frame_paths"][0])
    elif change == "model":
        cfg["vision_model"] = "new-vision-model"
    elif change == "endpoint":
        cfg["model_base_url"] = "https://different-model.test"
    else:
        monkeypatch.setattr(observer, "PROTOCOL", observer.PROTOCOL + "协议新增规则")
    _, count = observer.observe([record], lambda: None)
    assert count == 1 and len(calls) == 2


def test_reordered_frame_sequence_needs_new_observation(environment, monkeypatch, tmp_path):
    _, record = environment
    second = tmp_path / "second.jpg"
    Image.new("RGB", (1280, 720), "black").save(second)
    record["frame_paths"].append(second)
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    observer.observe([record], lambda: None)
    record["frame_paths"].reverse()
    _, count = observer.observe([record], lambda: None)
    assert count == 1 and len(calls) == 2


def test_more_than_four_distinct_groups_are_bounded_and_counted(environment, monkeypatch, tmp_path):
    _, record = environment
    records, calls = [], []
    for i in range(5):
        frame = tmp_path / f"{i}.jpg"
        Image.new("RGB", (1280, 720), (i * 30, 0, 0)).save(frame)
        records.append(record | {"shot": {"id": str(i)}, "frame_paths": [frame]})
    monkeypatch.setattr(observer, "_request", responder(calls))
    result, count = observer.observe(records, lambda: None)
    assert len(result) == 5 and count == 2
    assert [sum(b["type"] == "image_url" for b in call) for call in calls] == [4, 1]


def test_failed_observer_response_counts_the_request_and_stores_no_cache(environment, monkeypatch, tmp_path):
    cfg, record = environment
    monkeypatch.setattr(observer, "_request", lambda *args: ('{"groups":', {}))
    with pytest.raises(observer.ObservationError) as error:
        observer.observe([record], lambda: None)
    assert error.value.request_count == 1
    assert list(tmp_path.glob("cache/**/*.json")) == []
    evidence = next(tmp_path.glob("diagnostics/neutral-observer/*.json")).read_text()
    assert json.loads(evidence)["raw_content"] == '{"groups":'
    assert cfg["model_api_key"] not in evidence and "data:image" not in evidence


def test_truncation_is_rejected_even_if_provider_returns_complete_json(environment, monkeypatch, tmp_path):
    _, record = environment
    monkeypatch.setattr(observer, "_request", lambda *args: (
        json.dumps({"groups": [group()]}), {"finish_reason": "length"}))
    with pytest.raises(observer.ObservationError, match="截断"):
        observer.observe([record], lambda: None)
    assert list(tmp_path.glob("cache/**/*.json")) == []


def test_corrupt_cached_facts_cannot_be_reused(environment, monkeypatch, tmp_path):
    _, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    observer.observe([record], lambda: None)
    cache = next(tmp_path.glob("cache/**/*.json"))
    body = json.loads(cache.read_text())
    body["observation"]["frames"][0]["frame_index"] = 8
    cache.write_text(json.dumps(body))
    _, count = observer.observe([record], lambda: None)
    assert count == 1 and len(calls) == 2


@pytest.mark.parametrize("corruption", ["list", "expired", "future"])
def test_invalid_cache_container_or_timestamp_is_ignored(environment, monkeypatch, tmp_path, corruption):
    _, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    observer.observe([record], lambda: None)
    cache = next(tmp_path.glob("cache/**/*.json"))
    body = json.loads(cache.read_text())
    if corruption == "list":
        body = []
    else:
        body["created_at"] = 0 if corruption == "expired" else 10**20
    cache.write_text(json.dumps(body))
    assert observer.observe([record], lambda: None)[1] == 1
    assert len(calls) == 2


def test_cache_write_failure_does_not_discard_paid_facts(environment, monkeypatch):
    _, record = environment
    calls = []
    monkeypatch.setattr(observer, "_request", responder(calls))
    def cannot_write(*args):
        raise OSError("disk full")
    monkeypatch.setattr(observer, "_cache_write", cannot_write)
    result, count = observer.observe([record], lambda: None)
    assert count == 1 and result["private-shot"]["frames"] == group()["frames"]


def test_cancellation_while_waiting_for_model_slot_never_sends_or_releases_unowned_slot(environment, monkeypatch):
    _, record = environment
    class BusySlot:
        waiting = False
        def acquire(self, *, timeout):
            self.waiting = True
            return False
        def release(self):
            pytest.fail("cannot release a slot that was never acquired")
    slot = BusySlot()
    monkeypatch.setattr(observer.execution, "MODEL_SLOTS", slot)
    def check():
        if slot.waiting:
            raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        observer.observe([record], check)


def test_cancellation_after_response_releases_slot_and_does_not_cache(environment, monkeypatch, tmp_path):
    _, record = environment
    sent, held = [], []
    class Slot:
        def acquire(self, *, timeout):
            held.append(True)
            return True
        def release(self):
            held.pop()
    monkeypatch.setattr(observer.execution, "MODEL_SLOTS", Slot())
    def post(url, **kwargs):
        sent.append(True)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [
            {"message": {"content": json.dumps({"groups": [group()]})}}]})
    def check():
        if sent:
            raise media.Cancelled()
    monkeypatch.setattr(observer.network, "post", post)
    with pytest.raises(media.Cancelled):
        observer.observe([record], check)
    assert held == [] and len(sent) == 1 and list(tmp_path.glob("cache/**/*.json")) == []


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_account_failures_stop_observation_instead_of_caching_unknown(environment, monkeypatch, status):
    cfg, record = environment
    request = httpx.Request("POST", "https://models.test/chat/completions")
    monkeypatch.setattr(observer.network, "post", lambda *a, **kw: httpx.Response(status, request=request))
    with pytest.raises(planner.ModelUnavailable, match=f"HTTP {status}") as error:
        observer.observe([record], lambda: None)
    assert cfg["model_api_key"] not in str(error.value)


def test_network_payload_uses_only_configured_vision_model_and_neutral_protocol(environment, monkeypatch):
    cfg, record = environment
    def post(url, **kwargs):
        data = kwargs["json"]
        assert data["model"] == "vision-test" and data["max_tokens"] == 4000
        assert data["messages"][0]["content"] == observer.PROTOCOL
        assert "不可泄露" not in json.dumps(data, ensure_ascii=False)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [
            {"message": {"content": json.dumps({"groups": [group()]})}}], "usage": {"total_tokens": 12}})
    monkeypatch.setattr(observer.network, "post", post)
    assert observer.observe([record], lambda: None)[1] == 1
    assert cfg["model_name"] == "text-test"


@pytest.mark.parametrize("reference,label,expected", [
    ({"frame_index": 0, "field": "objects", "quote": "桌面"}, False, True),
    ({"frame_index": 0, "field": "action", "quote": "刷子平放"}, False, True),
    ({"frame_index": 0, "field": "text", "quote": "唇刷"}, False, True),
    ({"frame_index": 0, "quote": "唇刷"}, True, True),
    ({"frame_index": 1, "field": "text", "quote": "唇刷"}, False, False),
    ({"frame_index": 0, "quote": "桌面"}, True, False),
    ({"frame_index": 0, "field": "action", "quote": "打开包装"}, False, False),
    ({"frame_index": 0, "field": "action", "quote": "刷子 平放"}, False, False),
    ({"frame_index": 0, "field": "text", "quote": ""}, False, False),
    ({"frame_index": True, "quote": "唇刷"}, True, False),
])
def test_grounding_requires_exact_field_and_frame(reference, label, expected):
    assert observer.grounded(reference, group()["frames"], label=label) is expected


def test_actual_audit_reuses_cached_observation_but_rejudges_new_script(environment, monkeypatch, isolated_pixel_quality):
    _, record = environment
    calls, text_calls = [], []
    monkeypatch.setattr(observer, "_request", responder(calls))
    def chat(system, content, **kwargs):
        assert kwargs["vision"] is False and all(part["type"] == "text" for part in content)
        data = json.loads(content[0]["text"])
        text_calls.append(data)
        assert data["frame_observations"] == group()["frames"]
        return {"reviews": [{"shot_id": "private-shot", "status": "supported", "reason": "桌面有刷子", "labels": [],
            "evidence": [{"frame_index": 0, "field": "objects", "quote": "桌面和一把刷子"}]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    first = planner.audit_selected_frames([record], lambda: None)["private-shot"]
    record["shot"]["narration"] = "另一份原稿"
    second = planner.audit_selected_frames([record], lambda: None)["private-shot"]
    assert first["request_count"] == 2 and second["request_count"] == 1
    assert len(calls) == 1 and len(text_calls) == 2
    assert text_calls[0]["narration"] != text_calls[1]["narration"]
    assert first["scene_description"] == second["scene_description"] == group()["scene_description"]
