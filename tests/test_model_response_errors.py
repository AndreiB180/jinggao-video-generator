"""Failed paid responses remain diagnosable without exposing request data."""
import json
import os

import httpx
import pytest

from jinggao import config, network, paths, planner


def setup_response(monkeypatch, tmp_path, content, *, finish_reason="stop", extras=None,
                   base_url="https://compatible.example/v1"):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {
        "model_base_url": base_url, "model_api_key": "secret-local-key",
        "model_name": "local-test-model", "vision_model": "local-test-model",
        "cost_input_per_million": "2", "cost_cached_per_million": "0.04", "cost_output_per_million": "8", "cost_image_tokens": "1024"})
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        body = {"choices": [{"finish_reason": finish_reason, "message": {
            "content": content, "reasoning_content": "private-model-reasoning"}}],
            "usage": {"total_tokens": 123, "completion_tokens": 100}, **(extras or {})}
        return httpx.Response(200, request=httpx.Request("POST", url), json=body)

    monkeypatch.setattr(network, "post", post)
    return calls


@pytest.mark.parametrize("content,finish,match", [
    ("", "stop", "返回空内容"),
    (None, "stop", "返回空内容"),
    ('{"rows":[', "length", "输出长度上限"),
    ('{"ok":true}', "length", "输出长度上限"),
    ("not-json", "stop", "有效 JSON"),
    ([], "stop", "有效 JSON"),
])
def test_failed_response_keeps_finish_reason_usage_without_request_secrets(
        monkeypatch, tmp_path, content, finish, match):
    calls = setup_response(monkeypatch, tmp_path, content, finish_reason=finish,
                           extras={"headers": {"Authorization": "secret-provider-header"}})
    with pytest.raises(planner.ModelResponseError, match=match) as caught:
        planner.chat("private-system-prompt", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,private-pixels"}}],
                     reasoning=True, max_tokens=321)
    assert len(calls) == 1  # No hidden paid retries.
    error = caught.value
    diagnostic = next((tmp_path / "diagnostics" / "model-responses").glob("*.json"))
    assert error.diagnostic_path == str(diagnostic)
    assert error.model_info["usage"]["total_tokens"] == 123
    text = diagnostic.read_text("utf-8")
    record = json.loads(text)
    assert record["finish_reason"] == finish
    assert record["request_options"] == {"max_tokens": 321, "reasoning": True, "reasoning_effort": None}
    assert record["reasoning_characters"] == len("private-model-reasoning")
    assert record["content"] == (content if isinstance(content, str) else None)
    if os.name == "posix":
        assert diagnostic.stat().st_mode & 0o777 == 0o600
    for secret in ("secret-local-key", "secret-provider-header", "private-model-reasoning",
                   "private-system-prompt", "private-pixels"):
        assert secret not in text and secret not in str(error)


def test_valid_response_does_not_create_diagnostic(monkeypatch, tmp_path):
    calls = setup_response(monkeypatch, tmp_path, '{"ok":true}')
    result, info = planner.chat("private-system", "private-user", max_tokens=1000)
    assert result == {"ok": True} and info["usage"]["total_tokens"] == 123
    assert len(calls) == 1 and not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize('content,finish', [
    ('{"frames":[],"frames":[]}', 'stop'),
    ('{"frames":[]}', None), ('{"frames":[]}', 'content_filter'),
])
def test_pixel_response_requires_complete_unambiguous_json(monkeypatch, tmp_path, content, finish):
    calls = setup_response(monkeypatch, tmp_path, content, finish_reason=finish)
    with pytest.raises(planner.ModelResponseError) as caught:
        planner.chat('pixel protocol', 'pixels', vision=True, task='pixel_observation', max_tokens=1000)
    assert len(calls) == caught.value.request_count == 1
    record = json.loads(next((tmp_path / 'diagnostics/model-responses').glob('*.json')).read_text())
    assert record['content'] == content and record['finish_reason'] == finish


def test_truncation_diagnostic_records_the_effort_actually_sent(monkeypatch, tmp_path):
    calls = setup_response(monkeypatch, tmp_path, '{"reviews":[', finish_reason="length",
                           base_url="https://api.deepseek.com")
    with pytest.raises(planner.ModelResponseError) as caught:
        planner.chat("private-system", "private-user", reasoning=True, reasoning_effort="low", max_tokens=12000)
    record = json.loads(next((tmp_path / "diagnostics" / "model-responses").glob("*.json")).read_text())
    assert len(calls) == 1
    assert calls[0]["json"]["reasoning_effort"] == record["request_options"]["reasoning_effort"] == "low"
    assert record["request_options"]["max_tokens"] == 12000
    assert "private-model-reasoning" not in json.dumps(record) and "secret-local-key" not in str(caught.value)


def test_incompatible_choices_are_recorded_without_crashing_error_handler(monkeypatch, tmp_path):
    calls = setup_response(monkeypatch, tmp_path, None, extras={"choices": None})
    with pytest.raises(planner.ModelResponseError, match="返回空内容"):
        planner.chat("protocol", "data", max_tokens=1000)
    assert len(calls) == 1
    assert len(list((tmp_path / "diagnostics" / "model-responses").glob("*.json"))) == 1
