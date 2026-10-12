"""Remaining budget affects batching, never approval or the atomic ledger cap."""

from copy import deepcopy
import json
import sqlite3

import httpx
import pytest

from jinggao import costs, pixel_observer as observer
from test_pixel_observer import environment as _pixel_environment, response_for

environment = _pixel_environment


def prepared(records):
    return {
        f"g{i}": [observer.neutral_observer._frame(path) for path in r["frame_paths"]]
        for i, r in enumerate(records)
    }


def reserve(cfg, groups):
    payload = observer._payload(groups, cfg, max_tokens=observer._request_limit(len(groups)))
    bound = costs._input_bound(cfg, payload)
    return costs._amount(costs._rates(cfg, cfg["vision_model"], bound), bound, payload["max_tokens"])


def install_transport(monkeypatch, cfg, captures, *, truncated=False):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json, headers, **kwargs):
            captures.append(deepcopy(json))
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "model": cfg["vision_model"],
                    "choices": [
                        {
                            "message": {"content": response_for(json)},
                            "finish_reason": "length" if truncated else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
                },
            )

    monkeypatch.setattr(observer.httpx, "AsyncClient", Client)


def test_near_cap_splits_before_reservation_then_reuses_complete_caches(environment, monkeypatch):
    cfg, records = environment
    records = records[:2]
    for row in records:
        row["frame_paths"] *= 5
    groups = prepared(records)
    single = reserve(cfg, {"g0": groups["g0"]})
    double = reserve(cfg, groups)
    assert single + 1000 < double
    cfg["model_project_budget_cny"] = str((single + 1000) / 1_000_000)
    captures = []
    install_transport(monkeypatch, cfg, captures)
    with costs.scope("low-headroom", "job"):
        facts, quality, stats = observer.observe(records, lambda: None)
        again = observer.observe(records, lambda: None)
        assert costs.summary("low-headroom")["project"]["total_cny"] == pytest.approx(0.00072)
    assert stats["model_requests"] == len(captures) == 2
    assert [p["max_tokens"] for p in captures] == [2000, 2000]
    assert all(
        len(json.loads(p["messages"][1]["content"][0]["text"])["observation_groups"]) == 1 for p in captures
    )
    assert len(facts) == len(quality) == 2 and again[2]["model_requests"] == 0
    assert all(not value["content_approved"] for value in facts.values())


def test_existing_4000_token_cache_is_read_without_rebilling(environment, monkeypatch):
    cfg, records = environment
    captures = []
    install_transport(monkeypatch, cfg, captures)
    with costs.scope("old-response", "job"):
        first = observer.observe(records[:2], lambda: None)
        second = observer.observe(records[:1], lambda: None)
    assert first[2]["model_requests"] == 1 and captures[0]["max_tokens"] == 4000
    assert second[2]["model_requests"] == 0 and len(captures) == 1


def test_single_group_truncation_is_billed_and_never_cached_as_complete(environment, monkeypatch):
    cfg, records = environment
    captures = []
    install_transport(monkeypatch, cfg, captures, truncated=True)
    with costs.scope("truncated-single", "job"):
        result = observer.observe(records[:1], lambda: None)
    assert len(captures) == 1 and captures[0]["max_tokens"] == 2000
    assert result[2]["model_requests"] == 1 and not result[2]["all_cache_hit"]
    assert costs.summary("truncated-single")["project"]["total_cny"] > 0
    assert all(value.get("error") for value in result[0].values())


def test_headroom_includes_unknown_reservations_and_scope(environment, tmp_path):
    cfg, _ = environment
    cfg["model_project_budget_cny"] = "0.01"
    payload = {
        "model": cfg["vision_model"],
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1000,
    }
    with costs.scope("kept-spend", "job"):
        with costs.request(cfg, payload, "test") as charge:
            charge.mark_sent()
        with sqlite3.connect(tmp_path / "model-costs.sqlite3") as con:
            amount = con.execute("select amount from charges where project_id='kept-spend'").fetchone()[0]
        assert costs.project_headroom() == 10000 - amount
    with costs.scope("another-project", "job"):
        assert costs.project_headroom() == 10000
