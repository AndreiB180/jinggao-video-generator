"""Exercise real packet/budget/cache boundaries with mocked transport only."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from threading import Event, Lock
import time

import httpx
import pytest

from jinggao import config, costs, execution, media, network, paths, planner, review_batches, review_packet
from test_review_packet import records, strict_answer

# Quarantined: written against a pre-v60 review transport (review_batches.output_limit,
# review_packet.SCHEMA_PROTOCOL, Packet(model_view=...), "frame_reviews_v2", fixed 3072/8192 output
# sizing). None of these symbols exist in jinggao/ history. v60 sizes output with
# review_evidence.output_limit and enforces review_schema "grounded_frame_reviews_v4". Rewrite or delete.
PRE_V60_TRANSPORT = pytest.mark.skip(reason="pre-v60 review transport API; see comment above")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"model_api_key": "test-only", "model_name": "qwen3.7-flash",
        "review_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_project_budget_cny": "10", "model_job_budget_cny": "10", "model_daily_budget_cny": "10",
        "_strict_text_review_trial": True}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    monkeypatch.setattr(network, "post", lambda *a, **k: pytest.fail("unexpected model send"))
    return cfg


def large_records():
    rows = records()
    for i, row in enumerate(rows):
        row["full_source_evidence"] = ("甲" if i == 0 else "乙")*6000
        row["event"] = {"year": "2024", "person": "李明", "round": "决赛"}
        row["must"] = [{"quote": "不能使用2019年的画面", "use": "action", "kind": "event"}]
        row["input_record"] = row["shot_id"]
    return rows


def content(rows):
    return [{"type": "text", "text": json.dumps(row, ensure_ascii=False)} for row in rows]


def prepare(cfg, rows, system="完整审核规则", maximum=12000):
    return planner._prepare_chat_payload(cfg, "qwen3.7-flash", system, content(rows),
        task="evidence_review", vision=False, max_tokens=maximum, reasoning=True, reasoning_effort="low")


def answer(payload):
    body = json.loads(payload["messages"][1]["content"])
    return {"reviews": [{"shot_id": row["shot_id"], "status": "conflict", "reason": "年份不符", "labels": [],
            "evidence": [{"ref": row["shot_id"]+":f0:text:1"}], "primary_checks": [
                {"key": key, "status": "unknown", "reason": "不能证明本项", "evidence": []}
                for key in row["selected_visual_requirements"]]}
            for row in body["records"]]}


def response(url, result, *, finish="stop", payload=None):
    if payload and payload["response_format"]["type"] == "json_schema":
        result = strict_answer(result, typed=payload["response_format"]["json_schema"]["name"] == "frame_reviews_v2")
    return httpx.Response(200, request=httpx.Request("POST", url), json={
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(result)}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}})


def invoke(rows, system="完整审核规则", *, maximum=12000):
    return planner.chat(system, content(rows), task="evidence_review", max_tokens=maximum,
                        reasoning=True, reasoning_effort="low")


@PRE_V60_TRANSPORT
def test_output_allowance_tracks_full_obligations_not_remaining_budget(environment):
    row = {"selected_visual_requirements": {f"check_{i}": "完整条件" for i in range(7)},
           "requested_people": []}
    single = review_batches.output_limit(environment, "qwen3.7-flash", [row])
    pair = review_batches.output_limit(environment, "qwen3.7-flash", [row, row])
    assert single == 3072 and pair == 5120
    environment["model_project_budget_cny"] = "0"
    assert review_batches.output_limit(environment, "qwen3.7-flash", [row, row]) == pair
    identified = deepcopy(row)
    identified["requested_people"] = ["甲", "乙"]
    assert review_batches.output_limit(environment, "qwen3.7-flash", [identified]) > single
    assert review_batches.output_limit(environment, "qwen3.7-flash", [row]*40) == 12000
    assert review_batches.output_limit(environment, "deepseek-flash", [row]) == 12000
    assert review_batches.output_limit(environment, "qwen3.7-flash", [row], maximum=1024) == 1024
    assert review_batches.output_limit(environment, "deepseek-flash", [row], maximum=16000) == 16000


@PRE_V60_TRANSPORT
@pytest.mark.parametrize("bound,expected_calls", [(32000, 1), (32001, 2)])
def test_exact_final_payload_price_boundary_includes_system_packet_and_protocol(environment, monkeypatch, bound, expected_calls):
    rows = records()
    base = prepare(environment, rows, system="")
    system = "规"*((bound-costs._input_bound(environment, base["payload"]))//3)
    remaining = bound-costs._input_bound(environment, prepare(environment, rows, system)["payload"])
    system += "x"*remaining
    assert costs._input_bound(environment, prepare(environment, rows, system)["payload"]) == bound
    sent = []
    def post(url, **kwargs):
        payload = kwargs["json"]
        sent.append(payload)
        assert payload["messages"][0]["content"] == system+review_packet.SCHEMA_PROTOCOL
        assert payload["max_tokens"] == (12000 if expected_calls == 1 else 3072)
        measured = costs._input_bound(environment, payload)
        assert measured <= 32000
        assert costs._rates(environment, payload["model"], measured)[0] == Decimal("0.2")
        return response(url, answer(payload), payload=payload)
    monkeypatch.setattr(network, "post", post)
    with costs.scope("price-boundary", "job"):
        result, info = invoke(rows, system)
    assert len(sent) == expected_calls == info["request_count"] == info["batch_count"]
    assert info["usage"]["prompt_tokens"] == 200*expected_calls
    assert [r["shot_id"] for r in result["reviews"]] == [r["shot_id"] for r in rows]


@PRE_V60_TRANSPORT
def test_split_is_lossless_scoped_and_reuses_only_exact_complete_batch_caches(environment, monkeypatch):
    rows = large_records()
    original = deepcopy(rows)
    expected = {row["shot_id"]: row for row in rows}
    sent = []
    def post(url, **kwargs):
        payload = kwargs["json"]
        sent.append(payload)
        body = json.loads(payload["messages"][1]["content"])
        assert len(body["records"]) == 1
        row = body["records"][0]
        wanted = expected[row["input_record"]]
        assert body == review_packet.Packet(content([wanted]), model_view=True).body
        assert payload["max_tokens"] == 3072
        assert costs._input_bound(environment, payload) <= 32000
        return response(url, answer(payload), payload=payload)
    monkeypatch.setattr(network, "post", post)
    with costs.scope("split-cache", "job"):
        first, info = invoke(rows)
        second, reused = invoke(rows)
        assert first == second and rows == original
        assert info["request_count"] == 2 and info["cache_hits"] == 0 and not info["cache_hit"]
        assert reused["request_count"] == 0 and reused["cache_hits"] == 2 and reused["cache_hit"]
        rows[0]["evidence_binding"]["frame_sha256"] = ["f"*64]
        _, mixed = invoke(rows)
    assert len(sent) == 3
    assert mixed["request_count"] == 1 and mixed["cache_hits"] == 1 and not mixed["cache_hit"]
    assert costs.summary("split-cache")["project"]["requests"] == 3
    assert info["batches"][0]["shot_ids"] == [rows[0]["shot_id"]]
    assert info["batches"][1]["shot_ids"] == [rows[1]["shot_id"]]


def test_original_oversized_complete_cache_is_checked_before_splitting(environment, monkeypatch):
    rows = large_records()
    batch = prepare(environment, rows)
    assert costs._input_bound(environment, batch["payload"]) > 32000
    batch["cache"].save(answer(batch["payload"]))
    environment["model_project_budget_cny"] = "0"
    monkeypatch.setattr(review_batches, "partition", lambda *a: pytest.fail("keep complete original cache"))
    result, info = invoke(rows)
    assert len(result["reviews"]) == 2
    assert info["request_count"] == 0 and info["cache_hits"] == 1 and info["cache_hit"]


@PRE_V60_TRANSPORT
@pytest.mark.parametrize("bad", ["missing", "duplicate", "foreign", "other_batch"])
def test_bad_batch_ids_become_unknown_without_borrowing_another_batch(environment, monkeypatch, bad):
    rows = large_records()
    calls = []
    def post(url, **kwargs):
        payload = kwargs["json"]
        raw = answer(payload)
        if not calls:
            if bad == "missing":
                raw["reviews"] = []
            elif bad == "duplicate":
                raw["reviews"].append(deepcopy(raw["reviews"][0]))
            else:
                raw["reviews"][0]["shot_id"] = "r999" if bad == "foreign" else rows[1]["shot_id"]
        calls.append(payload)
        return response(url, raw, payload=payload if len(calls) > 1 else None)
    monkeypatch.setattr(network, "post", post)
    with costs.scope("bad-id", "job"):
        result, info = invoke(rows)
    assert info["request_count"] == 2 and not info["batches"][0]["valid_ids"]
    assert [r["shot_id"] for r in result["reviews"]] == [r["shot_id"] for r in rows]
    assert result["reviews"][0]["status"] == "unknown"
    assert result["reviews"][1]["status"] == "conflict"


@PRE_V60_TRANSPORT
def test_truncated_batch_is_unknown_and_accounted_while_complete_batch_is_cached(environment, monkeypatch):
    rows, calls = large_records(), []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return response(url, answer(kwargs["json"]), finish="length" if len(calls) == 1 else "stop",
                        payload=kwargs["json"])
    monkeypatch.setattr(network, "post", post)
    with costs.scope("truncated", "job"):
        result, info = invoke(rows)
        again, retried = invoke(rows)
    assert result["reviews"][0]["status"] == "unknown" and result["reviews"][1]["status"] == "conflict"
    assert info["request_count"] == 2 and info["usage"]["prompt_tokens"] == 400
    assert "error" in info["batches"][0]
    assert len(calls) == 3 and retried["request_count"] == 1 and retried["cache_hits"] == 1
    assert all(row["status"] == "conflict" for row in again["reviews"])
    assert all(payload["max_tokens"] == 3072 for payload in calls)
    ledger = costs.summary("truncated")["project"]
    assert ledger["requests"] == 3 and ledger["unresolved_requests"] == 0


def test_single_record_above_tier_keeps_full_payload_and_honestly_hits_guard(environment):
    rows = large_records()[:1]
    rows[0]["full_source_evidence"] *= 2
    batch = prepare(environment, rows)
    assert costs._input_bound(environment, batch["payload"]) > 32000
    assert review_batches.partition(content(rows), None, batch, environment, 32000) == [batch]
    environment["model_project_budget_cny"] = ".02"
    with costs.scope("single-large", "job"), pytest.raises(costs.BudgetExceeded) as caught:
        invoke(rows)
    assert caught.value.request_count == 0
    assert caught.value.model_info["request_count"] == 0
    assert costs.summary("single-large")["project"]["requests"] == 0


@PRE_V60_TRANSPORT
def test_fatal_later_batch_counts_only_sent_requests_and_preserves_first_cache(environment, monkeypatch):
    rows = large_records()
    rows[1]["full_source_evidence"] *= 2
    environment["model_project_budget_cny"] = ".02"
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return response(url, answer(kwargs["json"]), payload=kwargs["json"])
    monkeypatch.setattr(network, "post", post)
    with costs.scope("partial-fatal", "job"), pytest.raises(costs.BudgetExceeded) as caught:
        invoke(rows)
    assert len(calls) == caught.value.request_count == caught.value.model_info["request_count"] == 1
    assert [info["request_count"] for info in caught.value.model_info["batches"]] == [1, 0]
    environment["model_project_budget_cny"] = "0"
    # The completed sub-batch's smaller output cap is part of its actual
    # request identity; asking for that exact request remains free.
    _, cached = invoke(rows[:1], maximum=3072)
    assert cached["cache_hit"] and cached["request_count"] == 0 and len(calls) == 1


@PRE_V60_TRANSPORT
@pytest.mark.parametrize("maximum", [None, 12000, 4096, 1024])
def test_each_split_batch_sizes_all_checks_and_people_with_its_own_bound_cache(environment, monkeypatch, maximum):
    rows = large_records()
    rows[0]["selected_visual_requirements"] = {f"caption_c{i+1}": f"全部文字条件{i}" for i in range(8)}
    rows[1]["selected_visual_requirements"] = {f"caption_c{i+1}": f"全部文字条件{i}" for i in range(22)}
    rows[0]["requested_people"] = []
    rows[1]["requested_people"] = [{"name": "李明", "aliases": []}, {"name": "王芳", "aliases": []}]
    before = deepcopy(rows)
    ceiling = 14000 if maximum is None else maximum
    expected_maxima = [min(ceiling, needed) for needed in (3584, 8192)]
    sent = []

    def post(url, **kwargs):
        payload = kwargs["json"]
        index = len(sent)
        assert payload["max_tokens"] == expected_maxima[index]
        assert payload["messages"][0]["content"] == "完整审核规则"+review_packet.SCHEMA_PROTOCOL
        assert json.loads(payload["messages"][1]["content"]) == review_packet.Packet(content([rows[index]]), model_view=True).body
        sent.append(deepcopy(payload))
        return response(url, answer(payload), payload=payload)

    monkeypatch.setattr(network, "post", post)
    with costs.scope("split-output", "job"):
        first, info = invoke(rows, maximum=maximum)
        again, reused = invoke(rows, maximum=maximum)
    assert rows == before and first == again
    assert info["request_count"] == 2 and len(sent) == 2
    assert reused["request_count"] == 0 and reused["cache_hit"] and reused["cache_hits"] == 2
    endpoint = environment["model_base_url"]+"/chat/completions"
    with costs._db() as con:
        charges = con.execute("SELECT request_hash,reserved FROM charges WHERE project_id=? ORDER BY created",
                              ("split-output",)).fetchall()
    assert len(charges) == 2
    for index, (payload, charge) in enumerate(zip(sent, charges, strict=True)):
        assert charge["request_hash"] == costs.request_hash(endpoint, payload)
        bound = costs._input_bound(environment, payload)
        output = payload["max_tokens"] + (payload.get("thinking_budget", 0) if payload.get("enable_thinking") else 0)
        assert charge["reserved"] == costs._amount(costs._rates(environment, payload["model"], bound), bound, output)
        actual = prepare(environment, [rows[index]], maximum=expected_maxima[index])
        assert actual["payload"] == payload and actual["cache"].read() is not None
        previous = prepare(environment, [rows[index]], maximum=ceiling)
        if ceiling != expected_maxima[index]:
            assert previous["cache"].path != actual["cache"].path and not previous["cache"].path.exists()


@pytest.mark.parametrize("provider,model", [("https://dashscope.aliyuncs.com/compatible-mode/v1", "unlisted-review"),
                                            ("https://models.test/v1", "qwen3.7-flash")])
def test_unknown_output_policy_keeps_original_limit_and_record_group(environment, monkeypatch, provider, model):
    environment.update(model_base_url=provider, model_name=model, review_model=model,
                       cost_input_per_million="1", cost_cached_per_million="1", cost_output_per_million="1")
    rows, sent = large_records(), []
    def post(url, **kwargs):
        payload = kwargs["json"]
        assert payload["max_tokens"] == 16000
        assert json.loads(payload["messages"][1]["content"]) == review_packet.Packet(content(rows)).body
        sent.append(payload)
        return response(url, answer(payload), payload=payload)
    monkeypatch.setattr(network, "post", post)
    with costs.scope("unknown-policy", "job"):
        _, info = invoke(rows, maximum=16000)
    assert len(sent) == info["request_count"] == info["batch_count"] == 1


def test_parallel_review_groups_send_one_text_request_at_a_time(environment, monkeypatch):
    first, second = large_records(), large_records()
    for row in second:
        old = row["shot_id"]
        row["shot_id"] = row["input_record"] = "other-"+old
        for ref in row["observation_references"]:
            ref["ref"] = ref["ref"].replace(old, row["shot_id"])
        row["evidence_binding"]["frame_sha256"] = [hashlib.sha256(row["shot_id"].encode()).hexdigest()]
    lock, active, maximum, calls = Lock(), 0, 0, []
    def post(url, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(kwargs["json"])
        time.sleep(.02)
        with lock:
            active -= 1
        return response(url, answer(kwargs["json"]), payload=kwargs["json"])
    monkeypatch.setattr(network, "post", post)
    with costs.scope("serial", "job"), execution.ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(invoke, [first, second]))
    assert len(calls) == 4 and maximum == 1
    assert sum(info["request_count"] for _, info in replies) == 4


def test_waiting_for_review_serial_lock_remains_cancellable(environment):
    entered, release, waiting, cancel = Event(), Event(), Event(), Event()
    key = ("text-review-serial", environment["model_base_url"], "qwen3.7-flash")
    def owner():
        with execution.keyed_lock(key):
            entered.set()
            assert release.wait(timeout=3)
    def waiter():
        def check():
            waiting.set()
            if cancel.is_set():
                raise media.Cancelled()
        token = costs._cancel_check.set(check)
        try:
            with pytest.raises(media.Cancelled):
                invoke(large_records())
        finally:
            costs._cancel_check.reset(token)
    with execution.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner)
        assert entered.wait(timeout=2)
        second = pool.submit(waiter)
        assert waiting.wait(timeout=2)
        cancel.set()
        try:
            second.result(timeout=1)
        finally:
            release.set()
        first.result(timeout=1)


@pytest.mark.usefixtures("isolated_pixel_quality", "separate_pixel_observer")
@pytest.mark.parametrize("calls,hits", [(2, False), (1, False), (0, True)])
def test_audit_uses_actual_batch_request_count_instead_of_one_or_zero(monkeypatch, calls, hits):
    from jinggao import neutral_observer
    shot = planner.new_shot("工人搬运纸箱。")
    frames = [{"frame_index": 0, "objects": "工人和纸箱", "action": "工人搬运纸箱", "text": []}]
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "工人搬运纸箱", "frames": frames}}, 3))
    info = {"request_count": calls, "batch_count": 2, "cache_hit": hits, "cache_hits": 2-calls}
    monkeypatch.setattr(planner, "chat", lambda *a, **k: ({"reviews": [{"shot_id": shot["id"],
        "status": "unknown", "reason": "尚未核对", "labels": [], "evidence": [], "primary_checks": []}]}, info))
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"}, "frame_paths": []}],
                                          lambda: None)[shot["id"]]
    assert result["request_count"] == 3+calls
    assert result["model_info"]["request_count"] == calls
    assert result["status"] == "unknown"
