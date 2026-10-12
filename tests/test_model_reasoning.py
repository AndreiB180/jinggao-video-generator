"""Only the final text judgment opts in to supported provider reasoning."""
import httpx
import pytest

from jinggao import config, network, paths, planner


@pytest.mark.parametrize("host,reasoning,expected", [
    ("https://api.deepseek.com", True, {"type": "enabled"}),
    ("https://api.deepseek.com", False, {"type": "disabled"}),
    ("https://compatible.example/v1", True, None),
])
def test_reasoning_is_opt_in_and_not_sent_to_compatibility_endpoints(monkeypatch, tmp_path, host, reasoning, expected):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"model_base_url": host, "model_api_key": "local-test-key",
        "model_name": "deepseek-flash", "vision_model": "deepseek-flash", "cost_input_per_million": "2",
        "cost_cached_per_million": "0.04", "cost_output_per_million": "8"})
    def post(url, *, json, **kwargs):
        assert json.get("thinking") == expected
        assert json["model"] == "deepseek-flash"
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {}})
    monkeypatch.setattr(network, "post", post)
    assert planner.chat("protocol", "data", reasoning=reasoning, max_tokens=1000)[0] == {"ok": True}


@pytest.mark.parametrize("host,thinking,expected", [
    ("https://api.deepseek.com", True, "low"),
    ("https://api.deepseek.com", False, None),
    ("https://compatible.example/v1", True, None),
])
def test_planning_effort_override_only_goes_to_deepseek_when_thinking(monkeypatch, tmp_path, host, thinking, expected):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"model_base_url": host, "model_api_key": "local-test-key",
        "model_name": "deepseek-flash", "vision_model": "deepseek-flash", "cost_input_per_million": "2",
        "cost_cached_per_million": "0.04", "cost_output_per_million": "8"})
    def post(url, *, json, **kwargs):
        assert json.get("reasoning_effort") == expected
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {}})
    monkeypatch.setattr(network, "post", post)
    assert planner.chat("protocol", "data", reasoning=thinking, reasoning_effort="low", max_tokens=1000)[0] == {"ok": True}
