"""No paid calls: exercise the guard at the real transport boundaries."""
import json
from threading import Barrier, Event
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from jinggao import config, costs, execution, network, neutral_observer, paths, planner, store, visual_quality, worker
from jinggao.api import create_app


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"model_api_key": "test-only-private-key"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    monkeypatch.setattr(network, "post", lambda *a, **kw: pytest.fail("unexpected model request"))
    return cfg


def payload(maximum=1000):
    return {"model": "deepseek-flash", "messages": [{"role": "user", "content": "hello"}], "max_tokens": maximum}


def response(*, code=200, usage=None, text='{"ok":true}'):
    body = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(code, request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"), json=body)


def charged(cfg, pid, *, jid="job", usage=None, code=200):
    with costs.scope(pid, jid, "match"), costs.request(cfg, payload(), "text") as charge:
        charge.mark_sent()
        charge.response(response(code=code, usage=usage))


def test_parallel_requests_reserve_atomically_and_retry_or_key_change_cannot_reset(environment):
    cfg = environment
    cfg["model_project_budget_cny"] = "0.02"
    barrier, release = Barrier(12), Event()
    started, rejected = [], []
    def call(i):
        with costs.scope("project", f"job-{i}", "match"):
            barrier.wait(timeout=5)
            try:
                with costs.request(cfg, payload(), "vision") as charge:
                    charge.mark_sent()
                    started.append(i)
                    release.wait(timeout=5)
                    charge.response(response(usage={"prompt_tokens": 100, "completion_tokens": 1000}))
            except costs.BudgetExceeded:
                rejected.append(i)
    with execution.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(call, i) for i in range(12)]
        try:
            deadline = time.monotonic()+4
            while len(started)<2 and time.monotonic()<deadline:
                time.sleep(.01)
            assert len(started) == 2
            assert costs.summary("project")["project"]["total_cny"] <= .02
        finally:
            release.set()
        for future in futures:
            future.result()
    assert len(rejected) == 10
    cfg["model_api_key"] = "another-test-only-key"
    with pytest.raises(costs.BudgetExceeded, match="重试不会清零"):
        charged(cfg, "project", jid="retry")
    report = costs.summary("project")
    assert report["project"]["requests"] == 2
    assert report["project"]["estimated_cny"] == pytest.approx(.0164)


def test_job_cap_stops_before_network_without_erasing_project_spend(environment):
    cfg = environment
    cfg.update(model_job_budget_cny="0.01", model_project_budget_cny="0.02")
    charged(cfg, "limited", jid="first", usage={"prompt_tokens": 100, "completion_tokens": 1000})
    with pytest.raises(costs.BudgetExceeded, match="本次任务"):
        charged(cfg, "limited", jid="first")
    charged(cfg, "limited", jid="second", usage={"prompt_tokens": 100, "completion_tokens": 1000})
    with pytest.raises(costs.BudgetExceeded, match="项目累计"):
        charged(cfg, "limited", jid="third")
    assert costs.summary("limited")["project"]["requests"] == 2


def test_waits_for_inflight_reservation_to_settle_before_declaring_budget_exceeded(environment):
    cfg = environment
    cfg["model_project_budget_cny"] = "0.01"
    entered, release, second_sent = Event(), Event(), Event()
    def first():
        with costs.scope("project", "first"), costs.request(cfg, payload(), "text") as charge:
            charge.mark_sent()
            entered.set()
            release.wait(timeout=3)
            charge.response(response(usage={"prompt_tokens": 10, "completion_tokens": 5}))
    def second():
        with costs.scope("project", "second"), costs.request(cfg, payload(), "text") as charge:
            charge.mark_sent()
            second_sent.set()
            charge.response(response(usage={"prompt_tokens": 10, "completion_tokens": 5}))
    with execution.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first)
        assert entered.wait(timeout=2)
        b = pool.submit(second)
        try:
            assert not second_sent.wait(timeout=.1) and not b.done()
        finally:
            release.set()
        a.result(timeout=2)
        b.result(timeout=2)
    assert second_sent.is_set()
    assert costs.summary("project")["project"]["estimated_cny"] == .00012


def test_budget_wait_remains_cancellable_without_a_second_charge(environment):
    cfg = environment
    cfg["model_project_budget_cny"] = "0.01"
    from jinggao.media import Cancelled
    cancel = Event()
    def check():
        if cancel.is_set():
            raise Cancelled
    with costs.scope("project", "first"), costs.request(cfg, payload(), "text") as charge:
        charge.mark_sent()
        def waiting():
            with costs.scope("project", "second"), costs.request(cfg, payload(), "text", check=check):
                pytest.fail("a waiting request cannot be sent")
        with execution.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiting)
            cancel.set()
            with pytest.raises(Cancelled):
                future.result(timeout=2)
    assert costs.summary("project")["project"]["requests"] == 1


def test_daily_cap_covers_other_projects_and_keeps_unknown_usage_after_midnight(environment, monkeypatch):
    cfg = environment
    cfg["model_daily_budget_cny"] = "0.01"
    monkeypatch.setattr(costs, "_day", lambda: "2026-10-09")
    charged(cfg, "one")  # 200 without usage: never interpreted as zero cost.
    before = costs.summary("one")["daily"]
    assert before["reserved_cny"] > 0 and before["unresolved_requests"] == 1
    monkeypatch.setattr(costs, "_day", lambda: "2026-10-10")
    with pytest.raises(costs.BudgetExceeded, match="今日"):
        charged(cfg, "two")
    assert costs.summary()["daily"]["reserved_cny"] == before["reserved_cny"]


def test_settlement_counts_cached_tokens_and_reasoning_once_and_saves_no_content(environment, tmp_path):
    cfg = environment
    usage = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 800, "completion_tokens": 100,
             "completion_tokens_details": {"reasoning_tokens": 90}}
    charged(cfg, "one", usage=usage)
    totals = costs.summary("one")["project"]
    assert totals["estimated_cny"] == pytest.approx(.001232)
    assert totals["reserved_cny"] == 0
    with costs._db() as con:
        rows = [dict(row) for row in con.execute("SELECT * FROM charges")]
    assert rows[0]["input_tokens"] == 1000 and rows[0]["output_tokens"] == 100
    encoded = json.dumps(rows)
    assert "test-only-private-key" not in encoded and "hello" not in encoded and "choices" not in encoded


@pytest.mark.parametrize("code,holds", [(401, False), (402, False), (429, False), (500, True), (503, True)])
def test_unbilled_rejections_release_reservation_server_failures_remain_unknown(environment, code, holds):
    charged(environment, "one", code=code)
    totals = costs.summary("one")["project"]
    assert bool(totals["reserved_cny"]) == holds
    assert totals["estimated_cny"] == 0


def test_timeout_retains_reservation_and_unsent_cancel_releases_it(environment):
    cfg = environment
    with costs.scope("one"):
        with pytest.raises(TimeoutError), costs.request(cfg, payload(), "text") as charge:
            charge.mark_sent()
            raise TimeoutError
    with costs.scope("two"):
        with pytest.raises(TimeoutError), costs.request(cfg, payload(), "text"):
            raise TimeoutError
    assert costs.summary("one")["project"]["reserved_cny"] > 0
    assert costs.summary("two")["project"]["total_cny"] == 0


def test_image_reservation_uses_image_bound_instead_of_base64_length(environment):
    one = payload()
    one["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"+"X"*1_000_000}}]
    with costs.scope("one"), costs.request(environment, one, "vision") as charge:
        charge.mark_sent()
    two = payload()
    two["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,small"}}]
    with costs.scope("two"), costs.request(environment, two, "vision") as charge:
        charge.mark_sent()
    assert costs.summary("one")["project"]["reserved_cny"] == costs.summary("two")["project"]["reserved_cny"]


def test_unknown_provider_fails_before_sending_and_custom_pricing_is_explicit(environment):
    cfg = environment
    cfg["model_base_url"] = "https://compatible.example/v1"
    with pytest.raises(costs.PricingUnavailable):
        charged(cfg, "one")
    cfg.update(cost_input_per_million="3", cost_cached_per_million="1", cost_output_per_million="9")
    charged(cfg, "one", usage={"prompt_tokens": 100, "completion_tokens": 100})
    assert costs.summary("one")["project"]["estimated_cny"] == .0012
    images = payload()
    images["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "test"}}]
    with pytest.raises(costs.PricingUnavailable, match="每图"):
        with costs.request(cfg, images, "vision"):
            pytest.fail("unknown image pricing must not send")


@pytest.mark.parametrize("transport", ["text", "vision", "neutral", "quality"])
def test_all_model_transports_charge_even_invalid_output_and_block_before_send(environment, monkeypatch, transport):
    cfg, sent = environment, []
    def post(*a, **kw):
        sent.append(1)
        return response(usage={"prompt_tokens": 100, "completion_tokens": 20}, text="invalid json")
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, *a, **kw):
            return post(*a, **kw)
    monkeypatch.setattr(network, "post", post)
    monkeypatch.setattr(visual_quality.httpx, "AsyncClient", Client)
    def invoke():
        if transport in {"text", "vision"}:
            return planner.chat("test", "test", vision=transport=="vision", max_tokens=1000)
        if transport == "neutral":
            return neutral_observer._request([], cfg, lambda: None)
        return visual_quality._request([], cfg, lambda: None, time.monotonic()+1)
    with costs.scope("one", "job1", "verify"):
        if transport in {"text", "vision"}:
            with pytest.raises(planner.ModelResponseError):
                invoke()
        else:
            invoke()
    assert len(sent) == 1
    report = costs.summary("one")
    assert report["project"]["estimated_cny"] == .00036
    assert report["categories"][0]["stage"] == "verify"
    cfg["model_project_budget_cny"] = "0.00"
    with costs.scope("one", "job2", "verify"), pytest.raises(costs.BudgetExceeded):
        invoke()
    assert len(sent) == 1


def test_nested_thread_pools_keep_project_job_and_stage(environment):
    cfg = environment
    def inner(_):
        with costs.request(cfg, payload(), "text") as charge:
            charge.mark_sent()
            charge.response(response(usage={"prompt_tokens": 10, "completion_tokens": 5}))
    def outer(_):
        list(execution.completed_map(inner, range(2), workers=2))
    with costs.scope("project", "job", "match"):
        list(execution.ordered_map(outer, range(2), workers=2))
    with costs._db() as con:
        records = list(con.execute("SELECT project_id,job_id,stage FROM charges"))
    assert len(records) == 4
    assert {tuple(row) for row in records} == {("project", "job", "match")}
    assert costs.summary("unassigned")["project"]["requests"] == 0


def test_api_shows_ledger_but_never_the_key_and_worker_preserves_checkpoint(environment, monkeypatch):
    cfg = environment
    cfg["model_project_budget_cny"] = "0.00"
    client = TestClient(create_app())
    p = store.new_project("cost-test", "test", "16:9")
    store.mutate(p["id"], lambda p: p.update(cost_test_checkpoint="saved"))
    task = store.enqueue(p["id"], "verify", {})
    @costs.job
    def process(job, stop=None):
        planner.chat("test", "test")
    monkeypatch.setattr(worker, "process", process)
    assert worker.run_once()
    finished = store.get_job(task["id"])
    assert finished["status"] == "failed" and "费用保护" in finished["message"]
    result = client.get('/api/projects/'+p['id']).json()
    assert result["cost_test_checkpoint"] == "saved"
    assert result["model_costs"]["project"]["requests"] == 0
    assert client.get('/api/costs').status_code == 200
    assert "test-only-private-key" not in json.dumps(result)


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "1.005", "1001"])
def test_invalid_budget_settings_are_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with pytest.raises(ValueError, match="预算"):
        config.update({"model_project_budget_cny": value})


def test_valid_settings_persist_and_zero_pauses_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    saved = config.update({"model_project_budget_cny": "0", "model_daily_budget_cny": "3.00"})
    assert saved["model_project_budget_cny"] == "0"
    with costs.scope("test"), pytest.raises(costs.BudgetExceeded):
        with costs.request(config.read(), payload(), "text"):
            pytest.fail("zero budget must pause requests")
