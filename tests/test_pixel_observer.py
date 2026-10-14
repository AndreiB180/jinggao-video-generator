"""No paid services: joint prototype schema, binding and one-transport ledger."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
from threading import Event
import time

import httpx
from PIL import Image
import pytest

from jinggao import costs, media, pixel_observer as observer, planner


def joint_group(ident="g0", count=1):
    return {"id": ident, "scene_description": "工作人员手持纸箱走动。", "frames": [
        {"frame_index": i, "objects": "工作人员、纸箱、车厢", "action": "双手托着纸箱走向车厢",
         "text": ["新闻"], "representation": "live_action", "type_evidence": "现实纹理与手和纸箱接触",
         "overlays": [{"kind": "caption", "text": "新闻", "bbox": [0, .8, 1, 1],
                       "impact": "minor", "occludes": ""}]} for i in range(count)]}


def response_for(payload):
    manifest = json.loads(payload["messages"][1]["content"][0]["text"])["observation_groups"]
    return json.dumps(observer.pixel_packet.pack({"groups": [
        joint_group(row["group_id"], len(row["frame_indices"])) for row in manifest]}), ensure_ascii=False)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    cfg = {"vision_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
           "model_api_key": "test-only-secret", "cost_image_tokens": "0",
           "model_project_budget_cny": "2", "model_daily_budget_cny": "3",
           "model_job_budget_cny": "2", "model_provider_budget_cny": "5"}
    monkeypatch.setattr(observer.config, "read", lambda: deepcopy(cfg))
    monkeypatch.setattr(observer.paths, "DATA", tmp_path)
    monkeypatch.setattr(observer.httpx, "AsyncClient", lambda **kw: pytest.fail("unmocked network"))
    monkeypatch.setattr(observer.neutral_observer, "observe", lambda *a, **kw: pytest.fail("no double fallback"))
    monkeypatch.setattr(observer.visual_quality, "observe", lambda *a, **kw: pytest.fail("no double fallback"))
    records = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (1280, 720), (30 + i * 30, 0, 0)).save(path)
        records.append({"shot": {"id": f"private-{i}", "narration": "不要发送原稿",
                                "persons": [{"name": "不要发送人物"}]},
                        "asset": {"title": "不要发送来源", "source_url": "https://private-source.test"},
                        "verification": {"reason": "不要发送旧审核"}, "frame_paths": [path]})
    return cfg, records


def fake_request(calls):
    def request(payload, cfg, check, deadline):
        check()
        calls.append(deepcopy(payload))
        return response_for(payload), {"model": cfg["vision_model"], "response_model": cfg["vision_model"],
                                       "finish_reason": "stop", "request_count": 1}
    return request


def test_lossless_projections_preserve_all_fields_and_never_approve():
    groups = [joint_group("A", 2), joint_group("B", 1)]
    overlay = {"kind": "engagement_cta", "text": "关注", "bbox": [.3, .3, .8, .8],
               "impact": "material", "occludes": "手与纸箱接触处"}
    for frame in groups[0]["frames"]:
        frame["overlays"].extend([deepcopy(overlay), deepcopy(overlay)])
    facts, quality = observer.parse_groups(json.dumps({"groups": groups}), {"A": 2, "B": 1})
    for group in groups:
        ident = group["id"]
        assert facts[ident]["scene_description"] == group["scene_description"]
        assert facts[ident]["frames"] == [
            {key: frame[key] for key in observer.FACT_FIELDS} for frame in group["frames"]]
        assert quality[ident]["frames"] == [
            {key: frame[key] for key in observer.QUALITY_FIELDS} for frame in group["frames"]]
    assessed = observer.visual_quality.assess(quality["A"])
    assert assessed["quality_status"] == "reject" and not assessed["content_approved"]


@pytest.mark.parametrize("change", [
    lambda g: g[0]["frames"][0].pop("action"),
    lambda g: g[0]["frames"][0].update(text="不是数组"),
    lambda g: g[0]["frames"][0].update(objects=""),
    lambda g: g[0]["frames"][0].update(frame_index=True),
    lambda g: g[0]["frames"][0].update(frame_index=1),
    lambda g: g[0]["frames"][0].update(representation="approved"),
    lambda g: g[0]["frames"][0].pop("type_evidence"),
    lambda g: g[0]["frames"][0]["overlays"][0].update(bbox=[.8, 0, .2, 1]),
    lambda g: g[0]["frames"][0]["overlays"][0].update(impact="material", occludes=""),
    lambda g: g[0]["frames"][0].update(content_approved=True),
    lambda g: g[0].update(id="other"),
    lambda g: g.append(deepcopy(g[0])),
    lambda g: g[0]["frames"].append(deepcopy(g[0]["frames"][0])),
    lambda g: g[0].pop("scene_description"),
])
def test_malformed_facts_or_quality_cannot_be_projected(change):
    groups = [joint_group()]
    change(groups)
    with pytest.raises(ValueError):
        observer.parse_groups(json.dumps({"groups": groups}), {"g0": 1})


@pytest.mark.parametrize("raw", ['{"groups":[],"groups":[]}', '{"groups":[', '{"groups":[]} more',
                                  '{"groups":[],"content_approved":true}'])
def test_duplicate_json_fields_and_incomplete_or_extra_json_are_rejected(raw):
    with pytest.raises(ValueError):
        observer.parse_groups(raw, {"g0": 1})


def test_one_request_per_pair_is_blind_and_joint_cache_survives_new_batchmates(environment, monkeypatch, tmp_path):
    cfg, records = environment
    calls = []
    monkeypatch.setattr(observer, "_request", fake_request(calls))
    facts, quality, stats = observer.observe(records, lambda: None)
    # v60: DashScope qwen-flash sends up to four groups in one request (pixel_observer._supports_four).
    assert stats["model_requests"] == stats["request_count"] == len(calls) == 1
    assert stats["unique_groups"] == 4 and not stats["all_cache_hit"]
    assert all(r["observation_status"] == "complete" and not r["content_approved"] for r in facts.values())
    assert all(r["quality_status"] == "not_detected" and not r["content_approved"] for r in quality.values())
    for payload in calls:
        text = json.dumps(payload, ensure_ascii=False)
        assert "不要发送" not in text and "private-" not in text and cfg["model_api_key"] not in text
        manifest = json.loads(payload["messages"][1]["content"][0]["text"])
        assert set(manifest) == {"observation_groups"} and len(manifest["observation_groups"]) == 4
    same = [records[3] | {"shot": {"id": "new", "narration": "new script"}}, records[1]]
    replay, replay_quality, hit = observer.observe(same, lambda: None)
    assert hit["model_requests"] == 0 and hit["cache_hits"] == 2 and hit["all_cache_hit"]
    assert replay["new"]["frames"] == facts["private-3"]["frames"]
    assert replay_quality["new"]["frames"] == quality["private-3"]["frames"]
    for file in (tmp_path / "cache" / "pixel-observer").glob("*.json"):
        saved = file.read_text()
        assert cfg["model_api_key"] not in saved and "data:image" not in saved and "不要发送" not in saved


@pytest.mark.parametrize("change", ["pixels", "sequence", "model", "endpoint", "protocol", "output_limit", "transport"])
def test_cache_binding_includes_every_effective_input(environment, monkeypatch, change):
    cfg, records = environment
    record = deepcopy(records[0])
    record["frame_paths"] += records[1]["frame_paths"]
    calls = []
    monkeypatch.setattr(observer, "_request", fake_request(calls))
    observer.observe([record], lambda: None)
    if change == "pixels":
        Image.new("RGB", (1280, 720), "white").save(record["frame_paths"][0])
    elif change == "sequence":
        record["frame_paths"].reverse()
    elif change == "model":
        cfg["vision_model"] = "qwen3.7-flash-2026-07-15"
    elif change == "endpoint":
        cfg["model_base_url"] += "/other"
    elif change == "protocol":
        monkeypatch.setattr(observer, "PROTOCOL", observer.PROTOCOL + "\n新增规则")
    elif change == "output_limit":
        monkeypatch.setattr(observer, "MAX_TOKENS", observer.MAX_TOKENS + 1)
    else:
        old = observer.model_protocol.apply
        def apply(payload, cfg):
            old(payload, cfg)
            payload["temperature"] = .1
        monkeypatch.setattr(observer.model_protocol, "apply", apply)
    assert observer.observe([record], lambda: None)[2]["model_requests"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["malformed", "missing_group", "truncated", "wrong_model", "unknown_finish"])
def test_failed_joint_response_is_unknown_uncached_and_not_retried(environment, monkeypatch, tmp_path, kind):
    _, records = environment
    calls = []
    def request(payload, cfg, check, deadline):
        calls.append(1)
        raw, info = response_for(payload), {"finish_reason": "stop"}
        if kind == "malformed":
            raw = '{"groups":'
        elif kind == "missing_group":
            raw = json.dumps(observer.pixel_packet.pack({"groups": [joint_group()]}))
        elif kind == "truncated":
            info["finish_reason"] = "length"
        elif kind == "wrong_model":
            info["response_model"] = "different-model"
        else:
            info["finish_reason"] = None
        return raw, info
    monkeypatch.setattr(observer, "_request", request)
    facts, quality, stats = observer.observe(records[:2], lambda: None)
    assert len(calls) == stats["model_requests"] == 1 and not stats["all_cache_hit"]
    failed = list(facts.values())
    if kind == "missing_group":
        # v60 retains a strictly parsed, complete group from a partial reply (no retry, no fill-in);
        # only the omitted group stays unknown and uncached.
        assert [row["observation_status"] for row in failed] == ["complete", "unknown"]
        assert len(list((tmp_path / "cache" / "pixel-observer").glob("*.json"))) == 1
        failed, quality = failed[1:], {"x": list(quality.values())[1]}
    else:
        assert not list((tmp_path / "cache" / "pixel-observer").glob("*.json"))
    assert all(row["observation_status"] == "unknown" and not row["frames"] for row in failed)
    assert all(row["quality_status"] == "uncertain" and not row["content_approved"] for row in quality.values())
    assert len(stats["diagnostic_paths"]) == 1


def test_changed_image_during_response_cannot_be_returned_as_current_facts(environment, monkeypatch):
    _, records = environment
    def request(payload, cfg, check, deadline):
        Image.new("RGB", (1280, 720), "white").save(records[0]["frame_paths"][0])
        return response_for(payload), {"finish_reason": "stop"}
    monkeypatch.setattr(observer, "_request", request)
    facts, quality, stats = observer.observe(records[:1], lambda: None)
    assert facts["private-0"]["observation_status"] == "unknown"
    assert quality["private-0"]["quality_status"] == "uncertain" and stats["model_requests"] == 1


@pytest.mark.parametrize("bad", ["corrupt", "expired"])
def test_invalid_or_expired_joint_cache_is_not_trusted(environment, monkeypatch, tmp_path, bad):
    _, records = environment
    calls = []
    monkeypatch.setattr(observer, "_request", fake_request(calls))
    observer.observe(records[:1], lambda: None)
    path = next((tmp_path / "cache" / "pixel-observer").glob("*.json"))
    body = json.loads(path.read_text())
    if bad == "corrupt":
        body["observation"]["frames"][0].pop("overlays")
    else:
        body["created_at"] = time.time() - observer.TTL - 1
    path.write_text(json.dumps(body))
    assert observer.observe(records[:1], lambda: None)[2]["model_requests"] == 1


def test_equal_pixels_coalesce_and_concurrent_waiters_do_not_duplicate_requests(environment, monkeypatch):
    _, records = environment
    calls, entered, release = [], Event(), Event()
    def request(payload, cfg, check, deadline):
        calls.append(1)
        entered.set()
        assert release.wait(2)
        return response_for(payload), {"finish_reason": "stop"}
    monkeypatch.setattr(observer, "_request", request)
    records = [records[0], records[0] | {"shot": {"id": "other"}}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(observer.observe, records, lambda: None) for _ in range(4)]
        assert entered.wait(1)
        time.sleep(.05)
        release.set()
        results = [future.result() for future in futures]
    assert len(calls) == 1 and sum(r[2]["model_requests"] for r in results) == 1
    assert all(r[2]["unique_groups"] == 1 for r in results)
    assert sum(r[2]["all_cache_hit"] for r in results) == 3


def test_reservation_transport_and_real_ledger_share_one_final_payload(environment, monkeypatch, tmp_path):
    cfg, records = environment
    captured, timeouts, reserve = [], [], costs.request
    def charge(config, payload, category, **kwargs):
        captured.append(payload)
        assert category == "pixel_observation_joint"
        return reserve(config, payload, category, **kwargs)
    monkeypatch.setattr(observer.costs, "request", charge)
    class Client:
        def __init__(self, **kwargs):
            timeouts.append(kwargs["timeout"])
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, *, json, headers, **kwargs):
            assert json is captured[0]
            assert json["enable_thinking"] is False and json["temperature"] == 0
            assert json["max_tokens"] == 4000
            pictures = [b for b in json["messages"][1]["content"] if b["type"] == "image_url"]
            assert len(pictures) == 2 and all(b["max_pixels"] == 942080 for b in pictures)
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "model": cfg["vision_model"], "choices": [{"message": {"content": response_for(json)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})
    # v60 sends through network.model_async_client (route probe + trace hooks), not httpx directly.
    monkeypatch.setattr(observer.network, "model_async_client", lambda url, **options: Client(**options))
    with costs.scope("joint-test", "job", "offline-test"):
        facts, quality, stats = observer.observe(records[:2], lambda: None)
        cached = observer.observe(records[:2], lambda: None)
    assert len(captured) == stats["model_requests"] == 1 and cached[2]["model_requests"] == 0
    # The default 45-second total deadline must reach the transport; no hidden
    # 25-second cap should discard a still-pending, potentially billed response.
    assert len(timeouts) == 1 and 25 < timeouts[0] <= 45
    assert stats["model_info"][0]["input_bound"] == costs._input_bound(cfg, captured[0])
    with sqlite3.connect(tmp_path / "model-costs.sqlite3") as con:
        rows = con.execute("SELECT category,task,state,amount,input_tokens,output_tokens FROM charges").fetchall()
    assert rows == [("pixel_observation_joint", "pixel_observation_joint", "settled", 360, 1000, 200)]
    assert all(not r["content_approved"] for r in [*facts.values(), *quality.values()])


def test_budget_rejection_is_zero_sent_requests_and_no_fallback(environment, monkeypatch):
    cfg, records = environment
    cfg["model_project_budget_cny"] = "0.000001"
    with costs.scope("joint-budget-test", "job"):
        with pytest.raises(costs.BudgetExceeded) as failure:
            observer.observe(records, lambda: None)
    assert failure.value.request_count == failure.value.model_requests == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_cancel_http_and_release_owned_slot(environment, monkeypatch, cancel):
    cfg, records = environment
    state = {"started": False, "cancelled": False, "closed": False, "released": False}
    class Slot:
        def acquire(self, timeout):
            return True
        def release(self):
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
    monkeypatch.setattr(observer.execution, "MODEL_SLOTS", Slot())
    monkeypatch.setattr(observer.httpx, "AsyncClient", Client)
    prepared = [observer.neutral_observer._frame(records[0]["frame_paths"][0])]
    payload = observer._payload({"g0": prepared}, cfg)
    began = time.monotonic()
    with pytest.raises(media.Cancelled if cancel else observer.PixelObservationError) as failure:
        observer._request(payload, cfg, check, began + (.5 if cancel else .05))
    assert time.monotonic() - began < .8 and all(state.values())
    assert failure.value.request_count == failure.value.model_requests == 1


def test_account_error_counts_sent_request_and_stops_remaining_batches(environment, monkeypatch):
    _, records = environment
    calls = []
    def request(*args):
        calls.append(1)
        error = planner.ModelUnavailable("HTTP 402")
        error.request_count = error.model_requests = 1
        raise error
    monkeypatch.setattr(observer, "_request", request)
    with pytest.raises(planner.ModelUnavailable) as failure:
        observer.observe(records, lambda: None)
    assert len(calls) == failure.value.request_count == 1


@pytest.mark.parametrize("after_first", [False, True])
def test_cancellation_before_transport_or_between_batches_counts_only_sent_calls(environment, monkeypatch, after_first):
    _, records = environment
    calls = []
    monkeypatch.setattr(observer, "_request", fake_request(calls))
    def check():
        if not after_first or calls:
            raise media.Cancelled()
    with pytest.raises(media.Cancelled) as failure:
        observer.observe(records, check)
    assert len(calls) == failure.value.request_count == failure.value.model_requests == int(after_first)


def test_parse_rejects_repeated_nested_id_before_projection_can_hide_it():
    raw = json.dumps({"groups": [joint_group()]}).replace('"id": "g0"', '"id": "g1", "id": "g0"')
    with pytest.raises(ValueError, match="无歧义"):
        observer.parse_groups(raw, {"g0": 1})
