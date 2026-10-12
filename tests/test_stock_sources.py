from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from jinggao import media, paths, stock_sources as stock, video_sources


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(stock.config, "read", lambda: {})
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: pytest.fail("unexpected network request"))


def payload(provider):
    if provider == "pexels":
        return {"page": 1, "videos": [{"id": 1, "url": "https://www.pexels.com/video/forest-walk-1/",
            "image": "https://images.pexels.com/videos/1/free.jpg", "duration": 12,
            "width": 1920, "height": 1080, "user": {"name": "Example", "url": "https://www.pexels.com/@example"},
            "video_files": [{"link": "https://player.vimeo.com/private.mp4?token=do-not-store"}]}]}
    if provider == "pixabay":
        return {"hits": [{"id": 2, "type": "film", "tags": "forest, walking",
            "pageURL": "https://pixabay.com/videos/id-2/", "duration": 13, "user": "Example",
            "videos": {"medium": {"url": "https://cdn.pixabay.com/video/2.mp4", "width": 1280,
                "height": 720, "thumbnail": "https://cdn.pixabay.com/video/2.jpg"}}}]}
    if provider == "nasa":
        return {"collection": {"items": [{"href": "https://images-assets.nasa.gov/video/earth/collection.json",
            "data": [{"nasa_id": "earth", "title": "Earth from space", "media_type": "video",
                "center": "JSC", "description": "Our planet.", "date_created": "2024-01-01T00:00:00Z"}],
            "links": [{"rel": "preview", "href": "https://images-assets.nasa.gov/video/earth/thumb.jpg"}]}]}}
    if provider == "commons":
        return {"batchcomplete": "", "query": {"pages": {"3": {"pageid": 3, "index": 1,
            "title": "File:Forest.webm", "imageinfo": [{"mime": "video/webm", "duration": 14,
                "url": "https://upload.wikimedia.org/wikipedia/commons/a/aa/Forest.webm",
                "thumburl": "https://upload.wikimedia.org/wikipedia/commons/a/aa/Forest.jpg",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Forest.webm",
                "width": 1920, "height": 1080, "extmetadata": {
                    "Artist": {"value": "<a>Example Author</a>"},
                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                    "ImageDescription": {"value": "<p>A forest.</p>"}}}]}}}}
    return {"hits": [{"id": "sample-id", "title": "Walking in a forest", "duration": 11.625,
        "description": "A forest path", "thumbnail": "https://storage.coverr.co/t/image-id",
        "max_width": 2048, "max_height": 1152, "published_at": "2020-11-15T09:27:18.578Z",
        "urls": {"mp4": "https://storage.coverr.co/videos/asset?token=bound-secret"}}]}


def credentials(provider, key="example-test-key"):
    return {stock.PROVIDERS[provider]["key_field"]: key} if stock.PROVIDERS[provider]["key_field"] else {}


def test_registry_configuration_does_not_claim_live_verification():
    records = stock.registry()
    assert set(records) == {"pexels", "pixabay", "coverr", "nasa", "commons"}
    assert [records[p]["status"] for p in ("pexels", "pixabay")] == ["needs_key", "needs_key"]
    assert records["coverr"]["status"] == "restricted"
    assert records["coverr"]["configuration_status"] == "needs_key"
    for provider in ("nasa", "commons"):
        assert records[provider]["status"] == "configured"
        assert not records[provider]["key_required"]
    for row in records.values():
        assert not row["checked"] and row["observed_at"] is None
        assert row["website"].startswith("https://") and row["docs_url"]
        assert row["license_url"] and row["public_index"] and row["reason"]


def test_registry_never_exposes_secrets_or_unblocks_coverr():
    settings = {"pexels_api_key": "one-secret", "pixabay_api_key": "two-secret", "coverr_api_key": "three-secret"}
    records = stock.registry(settings)
    assert records["pexels"]["status"] == "configured"
    assert records["coverr"]["configuration_status"] == "configured"
    assert records["coverr"]["status"] == "restricted"
    encoded = json.dumps(records)
    assert all(key not in encoded for key in settings.values())


@pytest.mark.parametrize("provider", ["pexels", "pixabay", "coverr"])
def test_unconfigured_or_restricted_source_never_sends(provider, tmp_path):
    result = stock.search(provider, "forest")
    assert result["status"] in ("needs_key", "restricted")
    assert result["items"] == [] and not result["checked"]
    with pytest.raises(stock.SourceUnavailable) as caught:
        stock.search_items(provider, "forest")
    assert caught.value.provider == provider
    assert caught.value.status == result["status"]
    assert not list(tmp_path.rglob("*.json"))


def test_coverr_key_alone_does_not_allow_service_integration():
    result = stock.search("coverr", "forest", settings=credentials("coverr"))
    assert result["status"] == "restricted" and not result["checked"]
    assert result["configuration_status"] == "configured"


@pytest.mark.parametrize("provider", list(stock.PROVIDERS))
def test_official_metadata_schema_keeps_video_provenance_without_approving(provider):
    rows = stock.parse(provider, payload(provider))
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == provider and row["kind"] == "video"
    assert row["title"] and row["source_url"] and row["video_id"]
    assert row["license"] and row["license_url"]
    assert row["thumbnail_url"] == row["metadata_preview_url"]
    assert row["status"] == "unverified" and row["download_status"] == "unchecked"
    assert row["metadata_only"] and row["source_verified"]
    assert row["source_evidence"]["url"] == row["source_url"]
    assert row["source_evidence"]["evidence_origin"] == "platform_metadata"
    assert not any(k in row for k in ("approved", "verification", "preview", "media_streams"))
    assert "do-not-store" not in json.dumps(row) and "bound-secret" not in json.dumps(row)


def test_nasa_unknown_duration_stays_unknown_and_original_collection_is_preserved():
    row = stock.parse("nasa", payload("nasa"))[0]
    assert row["duration"] == 0
    assert row["collection_url"] == "https://images-assets.nasa.gov/video/earth/collection.json"
    assert row["publisher"] == "NASA / JSC"


@pytest.mark.parametrize("url", ["https://images-assets.nasa.gov/video/earth/collection.json?any=query",
    "https://images-assets.nasa.gov/video/earth/private.json", "https://evil.invalid/video/earth/collection.json"])
def test_nasa_collection_only_keeps_official_manifest_links(url):
    body = payload("nasa")
    body["collection"]["items"][0]["href"] = url
    assert stock.parse("nasa", body)[0]["collection_url"] == ""


def test_commons_per_item_license_and_author_are_preserved():
    row = stock.parse("commons", payload("commons"))[0]
    assert row["license"] == "CC BY-SA 4.0" and row["author"] == "Example Author"
    assert row["license_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert row["description"] == "A forest." and row["duration"] == 14


def test_coverr_does_not_invent_a_public_permalink_or_persist_signed_urls():
    row = stock.parse("coverr", payload("coverr"))[0]
    assert row["source_url"] == "https://api.coverr.co/videos/sample-id"
    assert row["source_access"] == "api_key"
    assert "urls" not in row and "token=" not in json.dumps(row)


@pytest.mark.parametrize("provider", ["pexels", "pixabay", "nasa", "commons"])
def test_real_page_parameter_cache_reuse_and_credentials_binding(monkeypatch, tmp_path, provider):
    calls = []
    def request(url, **options):
        calls.append((url, options))
        return payload(provider)
    monkeypatch.setattr(video_sources, "_json", request)
    settings = credentials(provider)
    first = stock.search(provider, "forest", 2, settings=settings)
    assert first["status"] == "configured" and first["checked"] and not first["cache_hit"]
    assert len(first["items"]) == 1
    again = stock.search(provider, "forest", 2, settings=settings)
    assert again["cache_hit"] and again["observed_at"] == first["observed_at"]
    assert len(calls) == 1
    params = parse_qs(urlparse(calls[0][0]).query)
    if provider == "commons":
        assert params["gsroffset"] == ["20"] and params["gsrsearch"] == ["(forest) filetype:video"]
    else:
        assert params["page"] == ["2"]
    if provider == "pexels":
        assert urlparse(calls[0][0]).path == "/v1/videos/search"
        assert calls[0][1]["headers"] == {"Authorization": "example-test-key"}
    if provider == "pixabay":
        assert urlparse(calls[0][0]).path == "/api/videos/"
    stock.search(provider, "forest", 3, settings=settings)
    stock.search(provider, "ocean", 2, settings=settings)
    assert len(calls) == 3
    if settings:
        stock.search(provider, "forest", 2, settings=credentials(provider, "another-test-key"))
        assert len(calls) == 4
    for file in tmp_path.rglob("*.json"):
        assert "example-test-key" not in file.read_text() and "another-test-key" not in file.read_text()


def test_successful_empty_page_is_cached_but_an_error_is_not(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        return {"collection": {"items": []}}
    monkeypatch.setattr(video_sources, "_json", request)
    assert stock.search_items("nasa", "forest") == []
    assert stock.search("nasa", "forest")["cache_hit"] and len(calls) == 1
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: {"not_a_collection": []})
    result = stock.search("nasa", "rocket")
    assert result["status"] == "unavailable" and result["checked"]
    assert not stock.search("nasa", "rocket")["cache_hit"]


@pytest.mark.parametrize("status,expected", [(401, "restricted"), (403, "restricted"),
    (429, "restricted"), (503, "unavailable")])
def test_http_failure_is_safe_distinct_and_not_automatically_retried(monkeypatch, status, expected):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        response = httpx.Response(status, request=httpx.Request("GET", "https://pixabay.com/api/videos/?key=secret-value"))
        response.raise_for_status()
    monkeypatch.setattr(video_sources, "_json", request)
    result = stock.search("pixabay", "forest", settings=credentials("pixabay", "secret-value"))
    assert result["status"] == expected and result["items"] == [] and len(calls) == 1
    assert "secret-value" not in json.dumps(result) and "https://" not in result["reason"]


def test_challenge_is_restricted_instead_of_a_successful_empty_index(monkeypatch):
    def request(*args, **kwargs):
        raise ValueError("来源没有返回公开数据，可能需要验证或页面已变化")
    monkeypatch.setattr(video_sources, "_json", request)
    result = stock.search("commons", "forest")
    assert result["status"] == "restricted" and result["items"] == []


def test_callback_cancellation_propagates_without_poisoning_cache(monkeypatch):
    def cancel():
        raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        stock.search("nasa", "forest", check=cancel)
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: payload("nasa"))
    assert stock.search("nasa", "forest")["items"]


def test_same_page_concurrent_searches_share_successful_request(monkeypatch):
    calls = []
    def request(*args, **kwargs):
        calls.append(1)
        time.sleep(.05)
        return payload("nasa")
    monkeypatch.setattr(video_sources, "_json", request)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda _: stock.search("nasa", "forest"), range(4)))
    assert len(calls) == 1
    assert sum(row["cache_hit"] for row in result) == 3
    assert all(row["items"] for row in result)


def test_waiting_request_can_cancel_independently(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def request(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return payload("nasa")
    monkeypatch.setattr(video_sources, "_json", request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(stock.search, "nasa", "forest")
        assert started.wait(1)
        checks = []
        def check():
            checks.append(1)
            if len(checks) >= 3:
                raise media.Cancelled()
        waiting = pool.submit(stock.search, "nasa", "forest", check=check)
        try:
            with pytest.raises(media.Cancelled):
                waiting.result(timeout=1)
        finally:
            release.set()
        assert first.result(timeout=1)["items"]


@pytest.mark.parametrize("provider", ["nasa", "commons", "pixabay"])
def test_non_video_results_do_not_become_video_candidates(provider):
    body = payload(provider)
    if provider == "nasa":
        body["collection"]["items"][0]["data"][0]["media_type"] = "image"
    elif provider == "commons":
        body["query"]["pages"]["3"]["imageinfo"][0]["mime"] = "image/jpeg"
    else:
        body["hits"][0]["type"] = "photo"
    assert stock.parse(provider, body) == []


@pytest.mark.parametrize("source", ["https://user:pass@www.pexels.com/video/x/",
    "https://www.pexels.com.evil.invalid/video/x/", "https://127.0.0.1/video/x/",
    "https://www.pexels.com/video/x/?token=secret", "file:///private"])
def test_invalid_or_credentialed_source_urls_are_not_indexed(source):
    body = payload("pexels")
    body["videos"][0]["url"] = source
    assert stock.parse("pexels", body) == []


@pytest.mark.parametrize("change", ["expired", "binding", "protocol"])
def test_cache_binding_ttl_and_protocol_are_enforced(monkeypatch, tmp_path, change):
    calls = []
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: calls.append(1) or payload("nasa"))
    stock.search("nasa", "forest")
    file = next(tmp_path.rglob("*.json"))
    saved = json.loads(file.read_text())
    if change == "expired":
        saved["observed_at"] -= stock.CACHE_TTL + 1
    elif change == "binding":
        saved["binding"] = "another-input"
    else:
        monkeypatch.setattr(stock, "VERSION", "changed-protocol")
    file.write_text(json.dumps(saved))
    assert not stock.search("nasa", "forest")["cache_hit"] and len(calls) == 2


@pytest.mark.parametrize("field,value", [("approved", True), ("preview", "https://example.invalid/image"),
    ("media_streams", [{"url": "https://example.invalid/stream?token=secret"}])])
def test_metadata_cache_cannot_restore_approvals_preflight_or_signed_streams(monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: payload("nasa"))
    stock.search("nasa", "forest")
    file = next(tmp_path.rglob("*.json"))
    saved = json.loads(file.read_text())
    saved["items"][0][field] = value
    file.write_text(json.dumps(saved))
    result = stock.search("nasa", "forest")
    assert not result["cache_hit"] and field not in result["items"][0]


def test_bad_input_does_not_silently_trim_or_send():
    for provider, query, page in (("unknown", "forest", 1), ("nasa", "", 1),
        ("nasa", "forest", True), ("nasa", "forest", 0), ("pixabay", "a" * 101, 1)):
        with pytest.raises(ValueError):
            stock.search(provider, query, page)


@pytest.mark.parametrize("provider", ["pexels", "pixabay"])
def test_on_demand_policy_keeps_explicit_keyed_search_working(monkeypatch, provider):
    calls = []
    settings = credentials(provider, "on-demand-test-key")
    access = stock.registry(settings)[provider]
    assert access["index_mode"] == "on_demand" and not access["bulk_index_allowed"]
    assert "按需索引" in access["reason"] and "通用素材库补全不扫描" in access["reason"]
    assert access["status"] == "configured" and not access["checked"] and not access["live_checked"]
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: calls.append(1) or payload(provider))

    first = stock.search(provider, "forest", settings=settings)
    again = stock.search(provider, "forest", settings=settings)

    assert first["checked"] and first["live_checked"] and not first["cache_hit"]
    assert again["checked"] and not again["live_checked"] and again["cache_hit"]
    assert len(calls) == 1 and first["items"] == again["items"]
    assert "on-demand-test-key" not in json.dumps([access, first, again])


@pytest.mark.parametrize("provider", ["pexels", "pixabay"])
def test_on_demand_search_reuses_the_full_24_hour_cache(monkeypatch, tmp_path, provider):
    now, calls = [1_800_000_000.0], []
    monkeypatch.setattr(stock.time, "time", lambda: now[0])
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: calls.append(1) or payload(provider))
    settings = credentials(provider, "full-day-test-key")
    first = stock.search(provider, "forest", settings=settings)
    now[0] += 86400 - .001
    cached = stock.search(provider, "forest", settings=settings)
    assert cached["cache_hit"] and not cached["live_checked"] and len(calls) == 1
    assert cached["observed_at"] == first["observed_at"]
    now[0] += .002
    refreshed = stock.search(provider, "forest", settings=settings)
    assert refreshed["live_checked"] and not refreshed["cache_hit"] and len(calls) == 2
    for file in tmp_path.rglob("*.json"):
        assert "full-day-test-key" not in file.read_text()


def test_missing_key_does_not_claim_a_live_check():
    result = stock.search("pixabay", "forest")
    assert result["status"] == "needs_key" and not result["checked"] and not result["live_checked"]
