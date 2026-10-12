from fastapi.testclient import TestClient
import pytest

from jinggao import config, paths, stock_sources
from jinggao.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    with TestClient(create_app()) as value:
        yield value


def test_check_uses_entered_key_without_saving_or_disclosing_it(client, monkeypatch):
    def search(provider, query, *, settings):
        assert (provider, query, settings["pixabay_api_key"]) == ("pixabay", "nature", "temporary-test-key")
        return {"status": "configured", "reason": "查询成功", "checked": True, "cache_hit": False,
                "observed_at": 123, "items": [{"id": "one"}], "api_key": settings["pixabay_api_key"]}
    monkeypatch.setattr(stock_sources, "search", search)
    result = client.post("/api/settings/stock/check", json={"provider": "pixabay", "api_key": " temporary-test-key "})
    assert result.status_code == 200
    assert result.json()["result_count"] == 1 and "temporary-test-key" not in result.text
    assert not config.read()["pixabay_api_key"] and not (paths.DATA / "settings.json").exists()


def test_blank_check_uses_saved_key_and_distinguishes_cached_success(client, monkeypatch):
    config.update({"pexels_api_key": "saved-test-key"})
    def search(provider, query, *, settings):
        assert provider == "pexels" and settings["pexels_api_key"] == "saved-test-key"
        return {"status": "configured", "reason": "上次成功查询的缓存", "checked": True,
                "cache_hit": True, "observed_at": 123, "items": []}
    monkeypatch.setattr(stock_sources, "search", search)
    result = client.post("/api/settings/stock/check", json={"provider": "pexels"}).json()
    assert result["cache_hit"] is True and result["result_count"] == 0
    assert config.read()["pexels_api_key"] == "saved-test-key"


@pytest.mark.parametrize("values", [
    {"provider": "custom", "api_key": "test"},
    {"provider": "coverr", "api_key": "test"},
    {"provider": "pixabay", "url": "https://example.org"},
    {"provider": "pixabay", "api_key": 123},
])
def test_invalid_check_cannot_select_an_arbitrary_destination(client, monkeypatch, values):
    monkeypatch.setattr(stock_sources, "search", lambda *a, **k: pytest.fail("No request for invalid configuration"))
    assert client.post("/api/settings/stock/check", json=values).status_code == 400


def test_missing_key_is_not_reported_as_connected(client, monkeypatch):
    # The real registry/search fast path must decline before its HTTP adapter.
    monkeypatch.setattr(stock_sources, "_request", lambda *a, **k: pytest.fail("Missing key must make no request"))
    result = client.post("/api/settings/stock/check", json={"provider": "pixabay"}).json()
    assert result["status"] == "needs_key" and result["checked"] is False
    assert result["result_count"] == 0
