from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from jinggao import config, library, media, paths, source_index, stock_sources as stock, video_sources


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    settings = {"pexels_api_key": "pexels-test-secret", "pixabay_api_key": "pixabay-test-secret"}
    monkeypatch.setattr(config, "read", lambda: dict(settings))
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: pytest.fail("unexpected network request"))
    with stock._detail_guard:
        stock._detail_memory.clear()
    with video_sources._inspection_guard:
        video_sources._inspection_cache.clear()
    return settings


def example(provider, ident="123"):
    source = (f"https://www.pexels.com/video/forest-{ident}/" if provider == "pexels"
              else f"https://pixabay.com/videos/id-{ident}/")
    row = video_sources._candidate(provider, "Forest path", source, "", video_id=ident)
    stream = {"width": 1280, "height": 720}
    if provider == "pexels":
        body = {"id": int(ident), "url": source, "duration": 12, "width": 1280, "height": 720,
                "video_files": [stream | {"link": f"https://videos.pexels.com/videos/{ident}/video.mp4",
                                            "file_type": "video/mp4"}]}
    else:
        body = {"hits": [{"id": int(ident), "pageURL": source, "tags": "forest, path", "duration": 12,
                          "type": "film", "videos": {"medium": stream | {
                              "url": f"https://cdn.pixabay.com/video/{ident}.mp4"}}}]}
    return row, body


def clock(monkeypatch):
    value = [1_800_000_000.0]
    timer = SimpleNamespace(time=lambda: value[0])
    monkeypatch.setattr(stock, "time", timer)
    monkeypatch.setattr(video_sources, "time", timer)
    return value


@pytest.mark.parametrize("provider", ["pexels", "pixabay"])
def test_force_inspection_and_restart_reuse_detail_for_full_24_hours(monkeypatch, tmp_path, provider):
    now = clock(monkeypatch)
    row, body = example(provider)
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append((a, kw)) or copy.deepcopy(body))
    library.remember_sources([row])

    first = video_sources.inspect_video_candidate(row, force=True)
    assert first["download_status"] == "available" and not first["stock_detail_cache_hit"]
    assert first["stock_detail_expires_at"] == now[0] + 86400
    assert not first["quality_verified"] and not first.get("approved") and not first.get("verification")
    now[0] += 601
    after_inspection_ttl = video_sources.inspect_video_candidate(first)
    assert after_inspection_ttl["stock_detail_cache_hit"] and len(calls) == 1
    assert video_sources.inspect_video_candidate(first, force=True)["stock_detail_cache_hit"]

    # A fresh process can reuse the sanitized durable response, not just memory.
    stock._detail_memory.clear()
    video_sources._inspection_cache.clear()
    now[0] = first["stock_detail_expires_at"] - 1
    assert video_sources.inspect_video_candidate(first, force=True)["stock_detail_cache_hit"]
    assert len(calls) == 1
    now[0] += 1
    refreshed = video_sources.inspect_video_candidate(first)
    assert not refreshed["stock_detail_cache_hit"] and len(calls) == 2
    assert refreshed["stock_detail_expires_at"] == now[0] + 86400
    assert not refreshed["quality_verified"]

    if provider == "pixabay":
        assert parse_qs(urlparse(calls[0][0][0]).query)["id"] == ["123"]
    else:
        assert calls[0][0][0] == "https://api.pexels.com/v1/videos/videos/123"
    persisted = "\n".join(f.read_text() for f in tmp_path.rglob("*.json"))
    assert "test-secret" not in persisted and "Authorization" not in persisted
    assert '"headers"' not in persisted and '"api_key"' not in persisted
    with library.connection() as con:
        saved = json.loads(con.execute("SELECT body FROM sources").fetchone()[0])
    assert "media_streams" not in saved and "stock_detail_expires_at" not in saved


def test_key_change_cannot_reuse_short_inspection_cache(monkeypatch, isolated, tmp_path):
    row, body = example("pixabay")
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append(1) or body)
    first = video_sources.inspect_video_candidate(row)
    isolated["pixabay_api_key"] = "changed-test-secret"
    second = video_sources.inspect_video_candidate(first)
    assert second["download_status"] == "available" and len(calls) == 2
    assert not second["stock_detail_cache_hit"]
    assert len(list((tmp_path / "cache" / "stock-details").glob("*.json"))) == 2
    isolated["pixabay_api_key"] = ""
    blocked = video_sources.inspect_video_candidate(second)
    assert blocked["download_status"] == "blocked" and not blocked["media_streams"] and len(calls) == 2
    assert "test-secret" not in json.dumps([first, second, blocked])


def test_source_and_identifier_bindings_do_not_accept_different_response(monkeypatch):
    row, body = example("pixabay")
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append(1) or body)
    video_sources._stock_video(row, lambda: None)
    different_source = row | {"source_url": "https://pixabay.com/videos/different-123/"}
    different_id = row | {"video_id": "999"}
    for other in (different_source, different_id):
        with pytest.raises(ValueError, match="不一致"):
            video_sources._stock_video(other, lambda: None)
    assert len(calls) == 3
    assert video_sources._stock_video(row, lambda: None)["stock_detail_cache_hit"]


def test_legacy_checked_stream_refreshes_when_inspection_expires(monkeypatch):
    now = clock(monkeypatch)
    row, body = example("pixabay")
    row.update(metadata_checked=True, inspection_at=now[0] - 601,
               media_streams=[{"url": "https://cdn.pixabay.com/video/stale.mp4", "width": 1280, "height": 720}])
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append(1) or body)
    refreshed = video_sources.inspect_video_candidate(row)
    assert len(calls) == 1 and refreshed["media_streams"][0]["url"].endswith("123.mp4")
    assert "stock_detail_expires_at" in refreshed


def test_signed_renditions_are_volatile_and_restart_does_not_repeat_api(monkeypatch, tmp_path):
    row, body = example("pexels")
    url = "https://videos.pexels.com/videos/123/video.mp4?token=temporary-signature&expires=999999"
    body["video_files"][0]["link"] = url
    body["headers"] = {"Authorization": "never-cache-raw-body"}
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append(1) or body)
    first = video_sources._stock_video(row, lambda: None)
    assert first["media_streams"][0]["url"] == url
    first["media_streams"][0]["url"] = "https://malformed.example/mutated"
    assert video_sources._stock_video(row, lambda: None)["media_streams"][0]["url"] == url
    text = next((tmp_path / "cache" / "stock-details").glob("*.json")).read_text()
    assert "temporary-signature" not in text and "never-cache-raw-body" not in text and '"headers"' not in text
    stock._detail_memory.clear()
    restored = video_sources.inspect_video_candidate(row, force=True)
    assert restored["stock_detail_cache_hit"] and len(calls) == 1
    assert not restored["media_streams"] and restored["download_status"] == "unsupported"
    assert "未重复请求" in restored["download_reason"]


@pytest.mark.parametrize("http_error", [False, True])
def test_auth_request_failures_never_reach_index_or_ui_with_secret(monkeypatch, tmp_path, http_error):
    row, _ = example("pixabay")
    library.remember_sources([row])
    calls = []
    def request(*a, **kw):
        calls.append(1)
        url = "https://pixabay.com/api/videos/?key=pixabay-test-secret&id=123"
        if http_error:
            httpx.Response(403, request=httpx.Request("GET", url)).raise_for_status()
        raise ValueError("failed GET " + url)
    monkeypatch.setattr(video_sources, "_json", request)
    result = video_sources.inspect_video_candidate(row, force=True)
    assert len(calls) == 1 and result["download_status"] == "blocked" and not result["media_streams"]
    assert "pixabay-test-secret" not in json.dumps(result) and "https://" not in result["download_reason"]
    with library.connection() as con:
        reason = con.execute("SELECT reason FROM source_catalog").fetchone()[0]
    assert reason == result["download_reason"] and "?key=" not in reason
    assert "pixabay-test-secret" not in json.dumps(source_index.search("forest", provider="pixabay"))
    assert not list((tmp_path / "cache" / "stock-details").glob("*.json"))


def test_unexpected_credential_echo_is_not_saved_or_used(monkeypatch, tmp_path):
    row, body = example("pexels")
    body["title"] = "Forest pexels-test-secret"
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: body)
    result = video_sources.inspect_video_candidate(row)
    assert result["download_status"] == "blocked" and not result["media_streams"]
    assert "pexels-test-secret" not in json.dumps(result)
    assert not list((tmp_path / "cache" / "stock-details").glob("*.json"))


def test_official_detail_same_id_concurrency_shares_one_response_and_cancels_waiter(monkeypatch):
    row, body = example("pixabay")
    entered, release = threading.Event(), threading.Event()
    calls = []
    def request(*a, **kw):
        calls.append(1)
        entered.set()
        assert release.wait(2)
        return body
    monkeypatch.setattr(video_sources, "_json", request)
    checks = []
    def cancel_waiter():
        checks.append(1)
        if len(checks) >= 2:
            raise media.Cancelled()
    with ThreadPoolExecutor(max_workers=5) as pool:
        first = pool.submit(video_sources._stock_video, row, lambda: None)
        assert entered.wait(1)
        cancelled = pool.submit(video_sources._stock_video, row, cancel_waiter)
        others = [pool.submit(video_sources._stock_video, row, lambda: None) for _ in range(3)]
        try:
            with pytest.raises(media.Cancelled):
                cancelled.result(timeout=1)
        finally:
            release.set()
        rows = [first.result()] + [f.result() for f in others]
    assert len(calls) == 1 and sum(r["stock_detail_cache_hit"] for r in rows) == 3
    assert all(r["media_streams"] for r in rows)


def test_cancelled_cache_read_does_not_send_or_invalidate_success(monkeypatch):
    row, body = example("pexels")
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **kw: calls.append(1) or body)
    video_sources._stock_video(row, lambda: None)
    def cancel():
        raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        video_sources._stock_video(row, cancel)
    assert video_sources._stock_video(row, lambda: None)["stock_detail_cache_hit"] and len(calls) == 1
