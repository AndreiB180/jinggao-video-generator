"""Provider integration and lifetime budget checks; no network or paid requests."""
from decimal import Decimal
import json
import sqlite3
import time

import httpx
import pytest

from jinggao import config, costs, network, neutral_observer, paths, planner, visual_quality


@pytest.fixture
def aliyun(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                            "model_name": "qwen3.7-flash", "vision_model": "qwen3.7-flash",
                            "model_api_key": "private-test-only", "model_provider_budget_cny": "8.90"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    monkeypatch.setattr(network, "post", lambda *a, **kw: pytest.fail("unexpected request"))
    return cfg


def payload(content="hello", maximum=1000):
    return {"model": "qwen3.7-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": maximum, "enable_thinking": False}


def response(prompt=1000, completion=100, cached=0):
    return httpx.Response(200, request=httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
                          json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
                                "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                          "prompt_tokens_details": {"cached_tokens": cached}}})


@pytest.mark.parametrize("transport", ["text", "vision", "neutral", "quality"])
def test_every_qwen_transport_explicitly_disables_thinking_and_accounts(aliyun, monkeypatch, transport):
    sent = []
    def post(*a, **kw):
        sent.append(kw["json"])
        return response()
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
    with costs.scope("check"):
        if transport in {"text", "vision"}:
            planner.chat("JSON", "input", vision=transport=="vision", reasoning=True, reasoning_effort="low")
        elif transport == "neutral":
            neutral_observer._request([], aliyun, lambda: None)
        else:
            visual_quality._request([], aliyun, lambda: None, time.monotonic()+5)
    assert len(sent) == 1
    assert sent[0]["enable_thinking"] is False
    assert sent[0]["temperature"] == 0
    assert "thinking" not in sent[0] and "reasoning_effort" not in sent[0]
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert costs.summary()["provider"]["estimated_cny"] == .00028


def test_optional_qwen_thinking_reserves_both_output_components(aliyun):
    value = payload(maximum=1000) | {"enable_thinking": True, "thinking_budget": 2048}
    with costs.scope("judge"), costs.request(aliyun, value, "text") as charge:
        rates = costs._rates(aliyun, value["model"], costs._input_bound(aliyun, value))
        expected = costs._amount(rates, costs._input_bound(aliyun, value), 1000+2048)/costs.MICROS
        assert costs.summary()["provider"]["reserved_cny"] == expected
        charge.mark_sent()
        charge.response(response(completion=700))
    assert costs.summary()["provider"]["estimated_cny"] == .00076
    with costs.scope("unbounded"), pytest.raises(costs.PricingUnavailable, match="独立输出上限"):
        with costs.request(aliyun, payload() | {"enable_thinking": True}, "text"):
            pytest.fail("unbounded reasoning must not be sent")


@pytest.mark.parametrize("tokens,expected", [(32_000, (".2", ".04", ".8")),
                                              (32_001, (".6", ".12", "2.4")),
                                              (256_000, (".6", ".12", "2.4")),
                                              (256_001, ("1.2", ".24", "4.8"))])
def test_qwen_tiers_apply_to_whole_request(aliyun, tokens, expected):
    assert costs._rates(aliyun, "qwen3.7-flash", tokens) == tuple(Decimal(v) for v in expected)


@pytest.mark.parametrize("tokens,expected", [(256_000, ("2", ".4", "8")),
                                              (256_001, ("6", "1.2", "24"))])
def test_plus_text_judge_uses_its_own_official_rates(aliyun, tokens, expected):
    assert costs._rates(aliyun, "qwen3.7-plus", tokens) == tuple(Decimal(v) for v in expected)


def test_conservative_input_reservation_settles_at_actual_tier_and_cached_rate(aliyun):
    with costs.scope("check"), costs.request(aliyun, payload("a"*40_000), "text") as charge:
        charge.mark_sent()
        reserved = costs.summary()["provider"]["reserved_cny"]
        assert reserved > .026
        charge.response(response(prompt=1000, completion=100, cached=800))
    report = costs.summary()["provider"]
    assert report["reserved_cny"] == 0
    assert report["estimated_cny"] == .000152


def test_qwen_image_bound_covers_default_custom_and_high_resolution(aliyun):
    def bound(**options):
        return costs._input_bound(aliyun, payload([{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}, **options}]))
    assert bound() >= 2560
    assert bound(max_pixels=16_777_216)-bound() == 16_384-2560
    assert bound(vl_high_resolution_images=True) == bound(max_pixels=16_777_216)
    with pytest.raises(costs.PricingUnavailable):
        bound(max_pixels=0)


def test_provider_cap_survives_project_day_and_key_changes_without_counting_known_other_provider(aliyun, monkeypatch):
    aliyun["model_provider_budget_cny"] = ".01"
    with costs.scope("first"), costs.request(aliyun, payload(maximum=8000), "text") as charge:
        charge.mark_sent()
        charge.response(response(prompt=100, completion=8000))
    monkeypatch.setattr(costs, "_day", lambda: "2099-01-01")
    aliyun["model_api_key"] = "rotated-private-test-key"
    with costs.scope("second"), pytest.raises(costs.BudgetExceeded, match="当前服务累计"):
        with costs.request(aliyun, payload(maximum=8000), "text"):
            pytest.fail("cap must apply before sending")
    ds = aliyun | {"model_base_url": "https://api.deepseek.com"}
    with costs.scope("third"), costs.request(ds, payload() | {"model": "deepseek-flash"}, "text") as charge:
        charge.mark_sent()
        charge.response(response())
    assert costs.summary()["provider"]["requests"] == 1
    assert costs.summary()["provider"]["estimated_cny"] == .00642


def test_unknown_usage_remains_in_provider_budget_after_restart(aliyun, monkeypatch):
    aliyun["model_provider_budget_cny"] = ".01"
    with costs.scope("first"), costs.request(aliyun, payload(maximum=8000), "text") as charge:
        charge.mark_sent()  # missing response retains reservation
    monkeypatch.setattr(costs, "_day", lambda: "2099-01-01")
    with costs.scope("second"), pytest.raises(costs.BudgetExceeded, match="当前服务累计"):
        with costs.request(aliyun, payload(maximum=8000), "text"):
            pytest.fail("unknown usage is not free")


def test_legacy_ledger_migration_retains_unattributed_charges_conservatively(aliyun, tmp_path):
    with sqlite3.connect(tmp_path/"model-costs.sqlite3") as con:
        con.execute("CREATE TABLE charges (id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT, stage TEXT, category TEXT, "
                    "model TEXT, created REAL, day TEXT, state TEXT, reserved INTEGER, amount INTEGER, "
                    "input_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER, rates TEXT, http_status INTEGER)")
        con.execute("INSERT INTO charges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("legacy", "old", "old", "match", "text", "unknown", 1, "2000-01-01", "settled", 0,
                     10_000, 1, 0, 1, json.dumps(["1", "1", "1"]), 200))
    aliyun["model_provider_budget_cny"] = ".01"
    assert costs.summary()["provider"]["total_cny"] == .01
    with costs.scope("second"), pytest.raises(costs.BudgetExceeded, match="当前服务累计"):
        with costs.request(aliyun, payload(), "text"):
            pytest.fail("migration must not erase costs")


def test_provider_budget_validation_and_public_key_redaction(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    for limit in ("NaN", "-1", "8.901"):
        with pytest.raises(ValueError, match="预算"):
            config.update({"model_provider_budget_cny": limit})
    public = config.update({"model_provider_budget_cny": "8.90", "model_api_key": "private-test-only"})
    assert public["model_provider_budget_cny"] == "8.90"
    assert "private-test-only" not in json.dumps(public)
