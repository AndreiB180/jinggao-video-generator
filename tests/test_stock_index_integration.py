import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jinggao import config, domestic, library, library_collect, models, paths, planner, source_index, stock_queries, stock_sources, store, video_sources
from jinggao.api import create_app


@pytest.fixture
def empty(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()


def nasa(ident="launch1", title="Rocket launch from the launch pad"):
    return video_sources._candidate("nasa", title, "https://images.nasa.gov/details/" + ident,
        "https://images-assets.nasa.gov/video/" + ident + "/thumb.jpg", video_id=ident,
        collection_url="https://images-assets.nasa.gov/video/" + ident + "/collection.json",
        publisher="NASA", license="NASA Media Usage Guidelines")


def test_stock_hosts_are_exact_and_arbitrary_shared_cdns_stay_closed():
    for host in ("api.pexels.com", "static-videos.pexels.com", "cdn.pixabay.com", "images-assets.nasa.gov", "upload.wikimedia.org"):
        assert domestic.allowed("https://" + host + "/item")
        assert not domestic.allowed("https://evil." + host + "/item")
        assert not domestic.allowed("https://" + host + ".evil.example/item")
    assert not domestic.allowed("https://player.vimeo.com/external/video.mp4")


def test_stock_index_roundtrip_recall_keeps_manifest_and_no_approval(empty):
    source = nasa() | {"approved": True, "verification": {"verdict": "match"}, "media_streams": ["private"]}
    assert library.remember_sources([source], topic="航天国防", query="rocket launch") == 1
    rows = source_index.recall("火箭发射现场")
    assert len(rows) == 1
    assert rows[0]["collection_url"] == source["collection_url"]
    assert rows[0]["download_status"] == "unchecked" and rows[0]["index_state"] == "indexed"
    assert not {"approved", "verification", "media_streams"} & rows[0].keys()
    assert not source_index.recall("火箭发射失败现场")
    assert not source_index.recall("马龙火箭发射")
    assert source_index.search("火箭发射", provider="nasa")["total"] == 1
    assert not source_index.recall("火箭发射", channel_names={"央视"})


@pytest.mark.parametrize("manifest", ["https://images-assets.nasa.gov/video/x/collection.json?token=secret", "https://evil.example/collection.json"])
def test_index_does_not_persist_private_or_unknown_manifests(empty, manifest):
    _, row = library.source_record(nasa() | {"collection_url": manifest})
    assert "collection_url" not in row


def test_access_reports_real_counts_and_last_query_separately(empty):
    library.remember_sources([nasa()])
    rows = {r["provider"]: r for r in source_index.overview()["source_access"]}
    assert rows["nasa"]["indexed_sources"] == 1
    assert rows["nasa"]["status"] == "configured" and rows["nasa"]["last_query_status"] == "unchecked"
    assert rows["pexels"]["status"] == "needs_key" and rows["coverr"]["status"] == "restricted"
    with library.connection() as con:
        con.execute("INSERT INTO index_pages VALUES (?,?,?,?,?,?,?,?)", ("nasa", "rocket", 1, "failed", 0, 0, time.time(), "HTTP 403"))
    row = next(r for r in source_index.source_access() if r["provider"] == "nasa")
    assert row["status"] == "unavailable" and row["configuration_status"] == "configured"
    assert row["indexed_sources"] == 1 and "403" in row["reason"]


def test_index_build_skips_missing_keys_and_uses_real_stock_pagination(empty, monkeypatch):
    calls = []
    def adapter(query, page):
        calls.append((query, page))
        return [nasa(str(page))]
    monkeypatch.setattr(source_index, "routes", lambda: {"nasa": adapter})
    monkeypatch.setattr(stock_queries, "queries", lambda topic, provider: ("rocket launch",) if provider == "nasa" else ())
    monkeypatch.setattr(source_index, "REQUEST_INTERVAL", 0)
    def forbidden(*a, **kw):
        pytest.fail("Indexing must not download or call a model")
    monkeypatch.setattr(planner, "chat", forbidden)
    monkeypatch.setattr(video_sources, "download_video_candidate", forbidden)
    report = source_index.run(library_collect.project_id(), {"index_target": 100, "index_topics": ["航天国防"],
        "index_entities": False, "index_providers": ["nasa", "pexels", "pixabay", "coverr"]}, lambda: None, lambda *a: None)
    assert calls == [("rocket launch", 1), ("rocket launch", 2), ("rocket launch", 3)]
    assert report["added"] == 3 and report["model_requests"] == report["downloads"] == 0
    assert set(report["provider_stops"]) == {"pexels", "pixabay", "coverr"}
    assert not report["reached_target"]


def test_stock_sources_are_reachable_through_existing_api(empty):
    library.remember_sources([nasa()])
    client = TestClient(create_app())
    response = client.get("/api/library/sources", params={"provider": "nasa", "query": "火箭发射"})
    assert response.status_code == 200 and response.json()["total"] == 1
    assert client.get("/api/library").json()["coverage"]["source_access"]
    options = models.LibraryBuild(action="index", index_providers=list(source_index.PROVIDERS))
    assert len(options.index_providers) == 13


def test_portable_seed_keeps_license_and_public_manifest(empty, tmp_path, monkeypatch):
    source = nasa() | {"license_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/", "author": "NASA"}
    library.remember_sources([source], topic="航天国防", query="rocket launch")
    seed = tmp_path / "portable.jsonl.gz"
    assert source_index.export_seed(seed) == 1
    monkeypatch.setattr(paths, "DATA", tmp_path / "clean-install")
    assert source_index.install_seed(seed) == 1
    row = source_index.search("火箭发射")["items"][0]
    assert row["license_url"] == source["license_url"] and row["author"] == "NASA"
    assert row["collection_url"] == source["collection_url"] and row["index_state"] == "indexed"
    assert not {"approved", "verification", "media_streams"} & row.keys()


def test_generic_search_uses_stock_mapping_but_never_a_missing_optional_key(empty, monkeypatch):
    calls = []
    def search(provider, query, *a, **kw):
        calls.append((provider, query))
        return [nasa()]
    monkeypatch.setattr(stock_sources, "search_items", search)
    rows, failures = video_sources.search_videos("火箭发射现场", use_index=False, channel_names={"NASA", "Pexels"})
    assert calls == [("nasa", "rocket launch")] and not failures and len(rows) == 1
    video_sources.search_videos("火箭发射现场", include_stock=False, use_index=False, channel_names={"NASA", "Pexels"})
    assert len(calls) == 1


@pytest.mark.parametrize("provider", ["pexels", "pixabay"])
@pytest.mark.parametrize("configured", [False, True])
def test_topic_bulk_build_never_prewarms_demand_only_providers(empty, monkeypatch, provider, configured):
    if configured:
        config.update({stock_sources.PROVIDERS[provider]["key_field"]: "bulk-test-key"})
    def forbidden(*a, **kw):
        pytest.fail("Demand-only providers must not receive topic-prewarming calls")
    monkeypatch.setattr(source_index, "routes", lambda: {provider: forbidden})
    monkeypatch.setattr(stock_sources, "_request", forbidden)
    monkeypatch.setattr(planner, "chat", forbidden)
    monkeypatch.setattr(video_sources, "download_video_candidate", forbidden)

    report = source_index.run(library_collect.project_id(), {"index_target": 100,
        "index_entities": True, "index_providers": [provider]}, lambda: None, lambda *a: None)

    assert report["pages"] == report["added"] == report["downloads"] == report["model_requests"] == 0
    assert report["on_demand_providers"] == [provider]
    assert "按需索引" in report["provider_stops"][provider]
    assert "bulk-test-key" not in json.dumps(report)
    with library.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM index_pages WHERE provider=?", (provider,)).fetchone()[0] == 0


@pytest.mark.parametrize("provider", ["pexels", "pixabay"])
def test_explicit_stock_query_accumulates_index_and_reuses_cache(empty, monkeypatch, provider):
    config.update({stock_sources.PROVIDERS[provider]["key_field"]: "explicit-test-key"})
    calls = []
    source = ("https://www.pexels.com/video/forest-123/" if provider == "pexels"
              else "https://pixabay.com/videos/id-123/")
    body = ({"videos": [{"id": 123, "url": source, "title": "forest", "duration": 12,
                         "width": 1280, "height": 720}]} if provider == "pexels" else
            {"hits": [{"id": 123, "pageURL": source, "tags": "forest", "type": "film", "duration": 12,
                       "videos": {"medium": {"url": "https://cdn.pixabay.com/video/123.mp4",
                                              "width": 1280, "height": 720}}}]})
    monkeypatch.setattr(video_sources, "_json", lambda *a, **k: calls.append(1) or body)
    def forbidden(*a, **kw):
        pytest.fail("Metadata search must not download or call a model")
    monkeypatch.setattr(planner, "chat", forbidden)
    monkeypatch.setattr(video_sources, "download_video_candidate", forbidden)

    first, failures = video_sources.search_videos("森林", use_index=False,
        channel_names={source_index.PROVIDERS[provider]})
    again, again_failures = video_sources.search_videos("森林", use_index=False,
        channel_names={source_index.PROVIDERS[provider]})

    assert not failures and not again_failures and len(calls) == 1
    assert len(first) == len(again) == 1 and first[0]["source_url"] == source
    assert source_index.search("森林", provider=provider)["total"] == 1
    assert library.stats()["sources"] == 1
    assert not {"approved", "verification", "media_streams"} & first[0].keys()
    assert "explicit-test-key" not in json.dumps([first, again, source_index.source_access()])


def test_access_keeps_demand_policy_and_sanitizes_historical_auth_errors(empty):
    config.update({"pixabay_api_key": "private-old-key"})
    with library.connection() as con:
        con.execute("INSERT INTO index_pages VALUES (?,?,?,?,?,?,?,?)", ("pixabay", "forest", 1,
            "failed", 0, 0, time.time(), "HTTP 403 https://pixabay.com/api/videos/?key=private-old-key"))

    row = next(r for r in source_index.source_access() if r["provider"] == "pixabay")

    assert row["status"] == "unavailable" and row["configuration_status"] == "configured"
    assert row["index_mode"] == "on_demand" and not row["bulk_index_allowed"]
    assert "按需索引" in row["reason"] and "403" in row["reason"]
    assert "private-old-key" not in json.dumps(row) and "?key=" not in row["reason"]


def test_pexels_indexed_source_refreshes_by_id_and_does_not_store_streams(empty, monkeypatch):
    config.update({"pexels_api_key": "test-only-key"})
    source = "https://www.pexels.com/video/rocket-launch-123/"
    item = {"id": 123, "url": source, "duration": 12, "video_files": [
        {"file_type": "video/mp4", "link": "https://videos.pexels.com/videos/123/video.mp4", "width": 1280, "height": 720}],
        "user": {"name": "Creator"}}
    def read(url, **kwargs):
        assert url == "https://api.pexels.com/v1/videos/videos/123"
        assert kwargs["headers"] == {"Authorization": "test-only-key"}
        return item
    monkeypatch.setattr(video_sources, "_json", read)
    row = video_sources._candidate("pexels", "rocket launch", source, "", video_id="123")
    library.remember_sources([row])
    result = video_sources.inspect_video_candidate(row, force=True)
    assert result["download_status"] == "available" and len(result["media_streams"]) == 1
    with library.connection() as con:
        saved = json.loads(con.execute("SELECT body FROM sources").fetchone()[0])
    assert "media_streams" not in saved and not result["quality_verified"]
    item["id"] = 999
    # A successful detail response must be reused for 24h, even under force.
    assert video_sources._stock_video(row, lambda: None)["video_id"] == "123"
    monkeypatch.setattr(stock_sources, "time", SimpleNamespace(time=lambda: result["stock_detail_expires_at"]))
    with pytest.raises(ValueError, match="不一致"):
        video_sources._stock_video(row, lambda: None)
