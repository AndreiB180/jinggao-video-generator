"""Concurrent pixel checks reuse complete facts, never approvals or bad facts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from threading import Event, local
from types import SimpleNamespace
import time

import httpx
from PIL import Image
import pytest

from jinggao import config, costs, execution, media, neutral_observer, paths, planner, visual_quality


@pytest.fixture(params=[neutral_observer, visual_quality], ids=["facts", "quality"])
def observer(request):
    return request.param


@pytest.fixture
def environment(tmp_path, monkeypatch):
    cfg = config.DEFAULTS | {
        "vision_model": "test-vision", "model_api_key": "test-only-key",
        "model_base_url": "https://model.test", "cost_input_per_million": "2",
        "cost_cached_per_million": "0.04", "cost_output_per_million": "8",
        "cost_image_tokens": "1024", "model_job_budget_cny": "10",
        "model_project_budget_cny": "10", "model_daily_budget_cny": "10",
        "model_provider_budget_cny": "10",
    }
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(neutral_observer.network, "post", lambda *a, **kw: pytest.fail("unmocked paid call"))
    records = []
    for i in range(3):
        frames = []
        for j in range(i+1):
            path = tmp_path / f"{i}-{j}.png"
            Image.new("RGB", (64, 48), (40+i*50, 20+j*40, 30)).save(path)
            frames.append(path)
        records.append({"shot": {"id": str(i), "narration": "never send this script"},
                        "asset": {"title": "never send this title"}, "frame_paths": frames})
    return cfg, records


def response_for(observer, content, *, partial=False):
    indices = {}
    for item in content:
        if item["type"] != "text":
            continue
        item = json.loads(item["text"])
        if "group_id" in item:
            indices.setdefault(item["group_id"], []).append(item["frame_index"])
    if observer is neutral_observer:
        assert json.loads(content[0]["text"])["observation_groups"] == [
            {"group_id": key, "frame_indices": values} for key, values in indices.items()]
    groups = []
    for key, values in indices.items():
        assert values == list(range(len(values)))
        if observer is neutral_observer:
            groups.append({"id": key, "scene_description": "桌面物体", "frames": [
                {"frame_index": i, "objects": "桌面上的物体", "action": "物体平放", "text": []}
                for i in values]})
        else:
            groups.append({"id": key, "frames": [{"frame_index": i, "representation": "live_action",
                "type_evidence": "实体纹理和自然光影", "overlays": []} for i in values]})
    return json.dumps({"groups": groups[:1] if partial else groups}), {}


def count_calls(observer, outcome):
    kind, value = outcome
    if kind == "error":
        return getattr(value, "request_count", 0)
    return value[1] if observer is neutral_observer else value[1]["model_requests"]


@contextmanager
def inflight(observer, records, monkeypatch, *, partial=False, real_timeout=False, request_error=None):
    """Keep the leader in transport until a follower registered its key cohort."""
    started, release, waiting = Event(), Event(), Event()
    role, calls, order = local(), [], {}
    original_lock = execution.keyed_lock

    @contextmanager
    def tracked_lock(key, *, check=None):
        name = getattr(role, "name", "main")
        order.setdefault(name, []).append(key[1])
        if name == "follower":
            # _group_flights registers all of this caller's keys before this
            # first acquisition, so response errors are shared deterministically.
            waiting.set()
        with original_lock(key, check=check):
            yield

    monkeypatch.setattr(execution, "keyed_lock", tracked_lock)

    def request(content, *args):
        calls.append(content)
        if len(calls) == 1:
            started.set()
            assert release.wait(4), "test transport was not released"
        if real_timeout:
            raise httpx.ReadTimeout("offline simulated timeout")
        if request_error:
            failure = request_error()
            failure.request_count = failure.model_requests = 1
            raise failure
        return response_for(observer, content, partial=partial and len(calls) == 1)

    if real_timeout:
        def post(*args, **kwargs):
            return request(kwargs["json"]["messages"][1]["content"])
        if observer is neutral_observer:
            monkeypatch.setattr(neutral_observer.network, "post", post)
        else:
            class Client:
                def __init__(self, **kwargs):
                    pass
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                async def post(self, *args, **kwargs):
                    return post(*args, **kwargs)
            monkeypatch.setattr(visual_quality.httpx, "AsyncClient", Client)
    else:
        monkeypatch.setattr(observer, "_request", request)

    def invoke(name, rows, check=lambda: None, **kwargs):
        role.name = name
        try:
            with costs.scope("singleflight", "job", "pixel_test"):
                return "result", observer.observe(rows, check, **kwargs)
        except Exception as exc:
            return "error", exc

    pool = ThreadPoolExecutor(max_workers=3)
    leader = pool.submit(invoke, "leader", records)
    try:
        assert started.wait(2), "leader never reached the mocked transport"
        yield SimpleNamespace(pool=pool, leader=leader, invoke=invoke, calls=calls,
                              release=release, waiting=waiting, order=order)
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


def test_reverse_order_batches_wait_reread_and_observe_every_frame(observer, environment, monkeypatch):
    _, records = environment
    with inflight(observer, records[:2], monkeypatch) as run:
        follower = run.pool.submit(run.invoke, "follower", list(reversed(records[:2])))
        assert run.waiting.wait(1)
        assert len(run.calls) == 1
        run.release.set()
        first, second = run.leader.result(timeout=2), follower.result(timeout=2)
    assert first[0] == second[0] == "result"
    assert [count_calls(observer, row) for row in (first, second)] == [1, 0]
    for outcome in (first, second):
        facts = outcome[1][0]
        assert {sid: len(value["frames"]) for sid, value in facts.items()} == {"0": 1, "1": 2}
        if observer is visual_quality:
            assert all(not value["content_approved"] for value in facts.values())
    assert all(keys == sorted(keys) for keys in run.order.values())
    assert not neutral_observer._flights


def test_overlapping_batch_only_sends_the_new_group(observer, environment, monkeypatch):
    _, records = environment
    with inflight(observer, records[:2], monkeypatch) as run:
        follower = run.pool.submit(run.invoke, "follower", records[1:])
        assert run.waiting.wait(1)
        run.release.set()
        assert run.leader.result(timeout=2)[0] == "result"
        second = follower.result(timeout=2)
    assert second[0] == "result" and count_calls(observer, second) == 1
    assert len(run.calls) == 2
    assert sum(item["type"] == "image_url" for item in run.calls[1]) == 3
    assert len(second[1][0]["1"]["frames"]) == 2 and len(second[1][0]["2"]["frames"]) == 3


def test_incomplete_group_failure_is_shared_and_retry_sends_only_missing_pixels(observer, environment, monkeypatch, tmp_path):
    _, records = environment
    with inflight(observer, records[:2], monkeypatch, partial=True) as run:
        follower = run.pool.submit(run.invoke, "follower", list(reversed(records[:2])))
        assert run.waiting.wait(1)
        run.release.set()
        results = [run.leader.result(timeout=2), follower.result(timeout=2)]
        assert [count_calls(observer, row) for row in results] == [1, 0]
        assert len(run.calls) == 1 and len(list(tmp_path.glob("cache/**/*.json"))) == 1
        if observer is neutral_observer:
            assert all(row[0] == "error" and isinstance(row[1], neutral_observer.ObservationError) for row in results)
        else:
            for row in results:
                assert row[1][0]["0"]["quality_status"] == "not_detected"
                assert row[1][0]["1"]["quality_status"] == "uncertain"
                assert not any(value["content_approved"] for value in row[1][0].values())
        retry = run.invoke("retry", records[:2])
        assert retry[0] == "result" and count_calls(observer, retry) == 1
        assert len(run.calls) == 2
        assert sum(item["type"] == "image_url" for item in run.calls[1]) == 2


def test_cache_write_failure_still_shares_paid_result_without_shared_mutation(observer, environment, monkeypatch, tmp_path):
    _, records = environment
    def disk_full(*args):
        raise OSError("offline simulated full disk")
    monkeypatch.setattr(observer, "_cache_write" if observer is neutral_observer else "_write_json", disk_full)
    with inflight(observer, records[:1], monkeypatch) as run:
        follower = run.pool.submit(run.invoke, "follower", records[:1])
        assert run.waiting.wait(1)
        run.release.set()
        first, second = run.leader.result(timeout=2), follower.result(timeout=2)
    assert first[0] == second[0] == "result" and len(run.calls) == 1
    assert count_calls(observer, second) == 0 and not list(tmp_path.glob("cache/**/*.json"))
    field = "objects" if observer is neutral_observer else "representation"
    before = second[1][0]["0"]["frames"][0][field]
    first[1][0]["0"]["frames"][0][field] = "caller mutation"
    assert second[1][0]["0"]["frames"][0][field] == before


@pytest.mark.parametrize("change", ["pixels", "frame_order", "model", "protocol", "endpoint"])
def test_changed_evidence_or_protocol_can_run_while_old_key_is_inflight(observer, environment, monkeypatch, change):
    cfg, records = environment
    record = records[1]
    with inflight(observer, [record], monkeypatch) as run:
        new = record | {"shot": {"id": "changed"}}
        if change == "pixels":
            Image.new("RGB", (64, 48), "white").save(record["frame_paths"][0])
        elif change == "frame_order":
            new["frame_paths"] = list(reversed(record["frame_paths"]))
        elif change == "model":
            cfg["vision_model"] = "different-vision"
        elif change == "endpoint":
            cfg["model_base_url"] = "https://other-model.test"
        else:
            monkeypatch.setattr(observer, "PROTOCOL", observer.PROTOCOL + "\nNew independent protocol.")
        changed = run.pool.submit(run.invoke, "changed", [new])
        result = changed.result(timeout=1)
        assert result[0] == "result" and count_calls(observer, result) == 1
        assert not run.leader.done() and len(run.calls) == 2
        run.release.set()
        assert run.leader.result(timeout=2)[0] == "result"


def test_waiter_can_cancel_without_cancelling_owner_or_spending(observer, environment, monkeypatch):
    _, records = environment
    cancel = Event()
    def check():
        if cancel.is_set():
            raise media.Cancelled()
    with inflight(observer, records[:1], monkeypatch) as run:
        follower = run.pool.submit(run.invoke, "follower", records[:1], check)
        assert run.waiting.wait(1)
        cancel.set()
        result = follower.result(timeout=1)
        assert result[0] == "error" and isinstance(result[1], media.Cancelled)
        assert count_calls(observer, result) == 0 and len(run.calls) == 1
        assert not run.leader.done()
        run.release.set()
        assert run.leader.result(timeout=2)[0] == "result"
    assert not neutral_observer._flights


def test_quality_deadline_includes_waiting_for_same_frames(environment, monkeypatch):
    _, records = environment
    with inflight(visual_quality, records[:1], monkeypatch) as run:
        began = time.monotonic()
        follower = run.pool.submit(run.invoke, "follower", records[:1], max_seconds=1)
        assert run.waiting.wait(1)
        result = follower.result(timeout=2)
        assert time.monotonic()-began < 1.5 and not run.leader.done()
        assert result[0] == "result" and result[1][1]["model_requests"] == 0
        assert result[1][0]["0"]["quality_status"] == "uncertain"
        assert "超时" in result[1][0]["0"]["quality_reason"]
        assert not result[1][0]["0"]["content_approved"] and len(run.calls) == 1


def test_transport_timeout_keeps_one_unknown_budget_reservation(observer, environment, monkeypatch):
    _, records = environment
    with inflight(observer, records[:1], monkeypatch, real_timeout=True) as run:
        follower = run.pool.submit(run.invoke, "follower", records[:1])
        assert run.waiting.wait(1)
        run.release.set()
        results = [run.leader.result(timeout=2), follower.result(timeout=2)]
    assert len(run.calls) == 1 and [count_calls(observer, row) for row in results] == [1, 0]
    report = costs.summary("singleflight")["project"]
    assert report["reserved_cny"] > 0 and report["unresolved_requests"] == 1
    assert not neutral_observer._flights


@pytest.mark.parametrize("account_failure", [False, True])
def test_shared_stopped_request_never_retries_or_confuses_follower_cancellation(
        observer, environment, monkeypatch, account_failure):
    _, records = environment
    error = (lambda: planner.ModelUnavailable("offline account unavailable")) if account_failure else media.Cancelled
    with inflight(observer, records[:1], monkeypatch, request_error=error) as run:
        follower = run.pool.submit(run.invoke, "follower", records[:1])
        assert run.waiting.wait(1)
        run.release.set()
        first, second = run.leader.result(timeout=2), follower.result(timeout=2)
    assert first[0] == "error" and len(run.calls) == 1
    assert [count_calls(observer, row) for row in (first, second)] == [1, 0]
    if account_failure:
        assert second[0] == "error" and isinstance(second[1], planner.ModelUnavailable)
    elif observer is neutral_observer:
        assert second[0] == "error" and isinstance(second[1], neutral_observer.ObservationError)
    else:
        assert second[0] == "result" and second[1][0]["0"]["quality_status"] == "uncertain"


def test_facts_and_quality_protocols_do_not_share_one_response(environment, monkeypatch):
    _, records = environment
    with inflight(neutral_observer, records[:1], monkeypatch) as run:
        quality_calls = []
        def quality_request(content, *args):
            quality_calls.append(content)
            return response_for(visual_quality, content)
        monkeypatch.setattr(visual_quality, "_request", quality_request)
        result, stats = visual_quality.observe(records[:1], lambda: None)
        assert stats["model_requests"] == 1 and len(quality_calls) == 1
        assert "scene_description" not in result["0"] and not result["0"]["content_approved"]
        assert not run.leader.done()
        run.release.set()
        assert run.leader.result(timeout=2)[0] == "result"


def test_key_lock_checks_after_acquisition_and_cleans_up_cancelled_owner():
    count = 0
    def check():
        nonlocal count
        count += 1
        if count == 2:
            raise media.Cancelled()
    with pytest.raises(media.Cancelled), execution.keyed_lock("cancel-after-acquire", check=check):
        pytest.fail("cancelled owner must not enter body")
    with execution.keyed_lock("cancel-after-acquire"):
        assert count == 2
