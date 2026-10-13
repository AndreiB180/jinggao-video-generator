"""Speech accounting tests use local fake transports and temporary ledgers only."""
from datetime import date, timedelta
import json
import sqlite3

import httpx
import pytest

from jinggao import baidu_http, config, costs, media, paths, tts_costs


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    cfg = config.DEFAULTS | {"tts_provider": "compatible", "tts_base_url": "https://speech.test/v1",
                             "tts_api_key": "private-test-key", "tts_model": "test-speech"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "2.0"}})
    return cfg


def pricing(cfg, **updates):
    endpoint, model, voice = tts_costs.identity(cfg)
    tariff = {"provider": cfg["tts_provider"], "base_url": endpoint, "model": model, "voice": voice,
              "unit": "million_characters", "cny_per_unit": "100", "billing": "metered",
              "valid_until": (date.today()+timedelta(days=10)).isoformat(),
              "evidence": "explicit synthetic test tariff, not a real provider price"} | updates
    cfg["tts_pricing_json"] = json.dumps(tariff)
    return tariff


def transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(media.httpx, "Client",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def records():
    with costs._db() as con:
        return [dict(row) for row in con.execute("SELECT * FROM charges ORDER BY created")]


def test_task_and_request_hash_preserve_identity_without_content_or_key(environment):
    cfg = environment
    payload = {"model": "deepseek-flash", "max_tokens": 100,
               "messages": [{"role": "user", "content": "sensitive original script"}]}
    with costs.scope("project", "job", "match"):
        for key in ["first-secret", "rotated-secret"]:
            with costs.request(cfg | {"model_api_key": key}, payload, "text", task="local_rank") as charge:
                charge.mark_sent()
                charge.response(httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5}}))
    rows = records()
    assert {r["task"] for r in rows} == {"local_rank"}
    assert rows[0]["request_hash"] == rows[1]["request_hash"]
    assert len(rows[0]["request_hash"]) == 64
    serialized = json.dumps(rows)
    assert all(value not in serialized for value in ["first-secret", "rotated-secret", "sensitive original script"])
    assert costs.request_hash("https://a/v1", payload) != costs.request_hash("https://a/v2", payload)
    assert costs.request_hash("https://a/v1", payload) != costs.request_hash("https://a/v1", payload | {"max_tokens": 101})
    assert costs.summary("project")["tasks"][0]["task"] == "local_rank"


def test_old_ledger_migration_preserves_cost_and_does_not_invent_request_hash(environment):
    paths.prepare()
    with sqlite3.connect(paths.DATA / "model-costs.sqlite3") as con:
        con.executescript("""CREATE TABLE charges (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, job_id TEXT NOT NULL,
            stage TEXT NOT NULL, category TEXT NOT NULL, model TEXT NOT NULL,
            created REAL NOT NULL, day TEXT NOT NULL, state TEXT NOT NULL,
            reserved INTEGER NOT NULL, amount INTEGER NOT NULL, input_tokens INTEGER,
            cached_tokens INTEGER, output_tokens INTEGER, rates TEXT NOT NULL, http_status INTEGER);
            INSERT INTO charges VALUES ('old','project','job','match','text','model',1,'2026-01-01',
                                        'unknown',10000,10000,NULL,NULL,NULL,'[]',NULL);""")
    row = records()[0]
    assert row["amount"] == row["reserved"] == 10000
    assert row["state"] == "unknown" and row["request_hash"] == "" and row["provider"] == ""
    report = costs.summary("project")
    assert report["project"]["reserved_cny"] == .01
    assert report["tasks"][0]["task"] == "text"


def test_compatible_missing_or_mismatched_price_stops_before_transport(environment, monkeypatch, tmp_path):
    monkeypatch.setattr(media.httpx, "Client", lambda **kwargs: pytest.fail("must not open a client"))
    with pytest.raises(costs.PricingUnavailable, match="尚无已核实费率"):
        media.synthesize("正文", tmp_path / "audio.wav")
    pricing(environment, voice="other")
    with pytest.raises(costs.PricingUnavailable, match="不符"):
        media.synthesize("正文", tmp_path / "audio.wav")
    pricing(environment, valid_until="2000-01-01")
    with pytest.raises(costs.PricingUnavailable, match="过期"):
        media.synthesize("正文", tmp_path / "audio.wav")
    assert records() == []


def test_character_metered_speech_is_counted_and_uses_shared_project_budget(environment, monkeypatch, tmp_path):
    cfg = environment
    pricing(cfg)
    calls = []
    def post(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b"valid mock wav")
    transport(monkeypatch, post)
    cfg.update(model_project_budget_cny="0.01", model_job_budget_cny="0.01")
    with costs.scope("project", "tts-job", "narrate"):
        media.synthesize("中文😀", tmp_path / "audio.wav")
    row = records()[0]
    assert row["state"] == "settled" and row["amount"] == row["reserved"] == 300
    assert row["usage_units"] == 3 and row["unit"] == "million_characters"
    assert row["category"] == "tts" and row["task"] == "tts_compatible"
    assert row["provider"] == "speech.test" and row["pricing_hash"]
    assert json.loads(row["rates"]) == ["100"]
    assert "中文😀" not in json.dumps(records(), ensure_ascii=False)
    assert calls == [{"model": "test-speech", "voice": "alloy", "input": "中文😀", "response_format": "wav"}]
    # Existing model and speech costs share the same project budget across jobs.
    with costs.scope("project", "model-job", "match"):
        with costs.request(cfg, {"model": "deepseek-flash", "messages": [], "max_tokens": 1000}, "text") as charge:
            charge.mark_sent()
            charge.response(httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 1000}}))
    with costs.scope("project", "retry", "narrate"), pytest.raises(costs.BudgetExceeded):
        media.synthesize("字" * 100, tmp_path / "next.wav")
    assert len(calls) == 1
    assert costs.summary("project")["project"]["total_cny"] == pytest.approx(.0085)


@pytest.mark.parametrize(("code", "expected"), [(200, "settled"), (400, "rejected"), (401, "rejected"),
                                                 (429, "rejected"), (500, "unknown")])
def test_speech_http_outcomes_retain_ambiguous_reserves(environment, monkeypatch, tmp_path, code, expected):
    pricing(environment, unit="request", cny_per_unit="0.01")
    transport(monkeypatch, lambda request: httpx.Response(code, content=b"mock"))
    with costs.scope("project", "job", "narrate"):
        if code == 200:
            media.synthesize("正文", tmp_path / "audio.wav")
        else:
            with pytest.raises(ValueError, match="配音接口调用失败"):
                media.synthesize("正文", tmp_path / "audio.wav")
    row = records()[0]
    assert row["state"] == expected
    assert row["amount"] == (0 if expected == "rejected" else 10000)


def test_timeout_survives_day_change_and_key_rotation(environment, monkeypatch, tmp_path):
    cfg = environment
    pricing(cfg, unit="request", cny_per_unit="0.01", valid_until="2099-12-31")
    cfg.update(model_project_budget_cny="0.01", model_job_budget_cny="0.01")
    def timeout(request):
        raise httpx.ReadTimeout("test", request=request)
    transport(monkeypatch, timeout)
    with costs.scope("project", "first", "narrate"), pytest.raises(ValueError):
        media.synthesize("正文", tmp_path / "audio.wav")
    monkeypatch.setattr(costs, "_day", lambda: "2099-01-01")
    cfg["tts_api_key"] = "rotated"
    with costs.scope("project", "second", "narrate"), pytest.raises(costs.BudgetExceeded):
        media.synthesize("正文", tmp_path / "audio.wav")
    assert len(records()) == 1 and records()[0]["state"] == "unknown"
    assert costs.summary("project")["daily"]["reserved_cny"] == .01


def test_invalid_audio_keeps_old_track_and_successful_provider_charge(environment, monkeypatch, tmp_path):
    pricing(environment)
    target = tmp_path / "audio.wav"
    target.write_bytes(b"old valid")
    transport(monkeypatch, lambda request: httpx.Response(200, content=b"corrupt"))
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "0"}})
    with pytest.raises(ValueError, match="没有有效时长"):
        media.synthesize("正文", target)
    assert target.read_bytes() == b"old valid"
    assert records()[0]["state"] == "settled" and records()[0]["amount"] == 200
    assert not list(tmp_path.glob(".speech-*"))


def test_baidu_unpriced_is_not_silently_zero_or_disabled(environment, monkeypatch, tmp_path):
    cfg = environment
    cfg["tts_provider"] = "baidu_http"
    def generate(text, target, check, *, before_submit):
        before_submit()
        target.write_bytes(b"mock wav")
    monkeypatch.setattr(baidu_http, "generate", generate)
    with costs.scope("project", "job", "narrate"):
        media.synthesize("正文", tmp_path / "audio.wav")
    row = records()[0]
    assert row["state"] == row["pricing_status"] == "unpriced"
    assert json.loads(row["rates"]) == []
    report = costs.summary("project")
    assert report["project"]["unpriced_requests"] == 1
    assert not report["cost_complete"]
    assert report["complete_total_cny"] is None
    assert report["project"]["upper_bound_cny"] is None
    assert report["tasks"][0]["total_cny"] is None


def test_baidu_failure_before_auth_is_not_a_submitted_unpriced_request(environment, monkeypatch, tmp_path):
    environment["tts_provider"] = "baidu_http"
    def generate(text, target, check, *, before_submit):
        raise ValueError("登录已过期")
    monkeypatch.setattr(baidu_http, "generate", generate)
    with pytest.raises(ValueError, match="登录已过期"):
        media.synthesize("正文", tmp_path / "audio.wav")
    assert records()[0]["state"] == "not_sent"
    assert costs.summary()["daily"]["unpriced_requests"] == 0


def test_baidu_explicit_subscription_allocation_is_guarded(environment, monkeypatch, tmp_path):
    cfg = environment
    cfg["tts_provider"] = "baidu_http"
    pricing(cfg, unit="request", cny_per_unit="0.06", billing="included")
    calls = []
    def generate(text, target, check, *, before_submit):
        before_submit()
        calls.append(text)
        target.write_bytes(b"mock")
    monkeypatch.setattr(baidu_http, "generate", generate)
    with costs.scope("project", "first", "narrate"):
        media.synthesize("正文", tmp_path / "audio.wav")
    with costs.scope("project", "second", "narrate"), pytest.raises(costs.BudgetExceeded):
        media.synthesize("第二次", tmp_path / "next.wav")
    assert calls == ["正文"]
    assert records()[0]["cost_basis"] == "configured_included"
    assert costs.summary("project")["project"]["total_cny"] == .06


def test_local_voice_has_explicit_offline_basis_and_no_cloud_tariff(environment, monkeypatch, tmp_path):
    cfg = environment
    cfg["tts_provider"] = "system"
    monkeypatch.setattr(media.sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(media, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(media, "_say_voice", lambda: "Tingting")   # newer macOS names it "Ting-Ting"
    monkeypatch.setattr(paths, "binary", lambda name: name)
    monkeypatch.setattr(media.httpx, "Client", lambda **kwargs: pytest.fail("no cloud client"))
    cfg["model_project_budget_cny"] = "0.00"
    with costs.scope("project", "job", "narrate"):
        media.synthesize("正文", tmp_path / "audio.wav")
    assert calls[0][:3] == ["/usr/bin/say", "-v", "Tingting"]
    assert records()[0]["cost_basis"] == "offline_system"
    assert costs.summary("project")["project"]["total_cny"] == 0


@pytest.mark.parametrize("updates", [{"cny_per_unit": "NaN"}, {"cny_per_unit": "0"},
                                    {"unit": "token"}, {"billing": "assume_free"},
                                    {"valid_until": "tomorrow"}, {"evidence": ""}])
def test_invalid_tariffs_are_rejected(environment, updates):
    tariff = pricing(environment, **updates)
    with pytest.raises(ValueError, match="配音计费配置无效"):
        tts_costs.validate_pricing(json.dumps(tariff))


def test_config_accepts_explicit_tariff_without_changing_budgets(environment):
    pricing(environment)
    budget = environment["model_project_budget_cny"]
    result = config.update({"tts_pricing_json": environment["tts_pricing_json"]})
    assert result["model_project_budget_cny"] == budget
    assert result["tts_pricing_json"] == environment["tts_pricing_json"]
