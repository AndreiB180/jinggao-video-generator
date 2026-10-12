from jinggao import config, planner, sources


def test_multi_query_retains_diverse_results_and_partial_success(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "commons"})

    def search(project, query):
        if query == "failed":
            raise ValueError("network unavailable")
        return [{"id": "shared"}, {"id": query+"-unique"}], "ok"

    monkeypatch.setattr(sources, "search", search)
    candidates, note = sources.search_shot({}, {"query": "中文", "search_queries": ["english", "failed"]})
    assert {c["id"] for c in candidates} == {"shared", "中文-unique", "english-unique"}
    assert "3 组" in note and "1 组检索未完成" in note


def test_ranking_can_select_from_later_search_batches(monkeypatch):
    batches = []

    def compare(project, shot, candidates, check):
        batches.append([c["id"] for c in candidates])
        return [{"candidate_id": candidates[-1]["id"], "verdict": "match", "reason": "visible content"}]

    monkeypatch.setattr(planner, "rank_candidates", compare)
    candidates = [{"id": str(i)} for i in range(40)]
    ranking = planner.rank_pool({}, {}, candidates, lambda: None)
    assert ranking[0]["candidate_id"] == "29"
    assert sorted(sum(batches[:3], []), key=int) == [str(i) for i in range(30)]
    assert batches[3] == ["9", "19", "29"]


def test_used_exact_event_source_remains_in_preview_pool_with_many_fresh_results(monkeypatch):
    seen = []
    reused = {"id": "same-event", "asset_id": "used-asset", "kind": "video",
              "title": "2026 WTT中国大满贯 梁靖崑 MS R32"}
    project = {"shots": [{"id": "earlier-shot", "selected_asset": "used-asset"}], "assets": []}
    shot = {"id": "current", "event": "2026 WTT中国大满贯 R32", "persons": [{"name": "梁靖崑", "aliases": []}]}

    def compare(project, shot, candidates, check):
        seen.extend(candidate["id"] for candidate in candidates)
        selected = next((candidate for candidate in candidates if candidate["id"] == "same-event"), candidates[0])
        return [{"candidate_id": selected["id"], "verdict": "match", "reason": "same event, another interval"}]

    monkeypatch.setattr(planner, "rank_candidates", compare)
    candidates = [{"id": f"fresh-{i}", "kind": "video", "title": "其他比赛"} for i in range(40)] + [reused]
    ranking = planner.rank_pool(project, shot, candidates, lambda: None)
    assert "same-event" in seen and ranking[0]["candidate_id"] == "same-event"


def test_commons_retries_transient_failure_and_reuses_cache(monkeypatch, tmp_path):
    import httpx
    from jinggao import paths
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(503 if len(calls) == 1 else 200, json={"query": {"pages": {}}},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(sources.httpx, "get", get)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    assert sources.commons("test-only") == []
    assert sources.commons("test-only") == []
    assert len(calls) == 2


def test_web_image_results_keep_original_source_and_reject_non_web_urls():
    import html
    import json
    valid = {"murl": "https://example.com/photo.jpg", "purl": "https://example.com/article",
             "turl": "https://example.com/preview.jpg", "t": "<b>轨道检修</b>"}
    rows = [valid, valid, valid | {"murl": "file:///secret"}]
    markup = "".join('<a m="'+html.escape(json.dumps(row), quote=True)+'">x</a>' for row in rows)
    results = sources.parse_bing_images(markup)
    assert len(results) == 1
    assert results[0]["source_url"] == valid["purl"]
    assert results[0]["title"] == "轨道检修" and "待核实" in results[0]["license"]


def test_mixed_search_keeps_other_channel_when_one_is_unavailable(monkeypatch):
    from jinggao import publisher_videos, video_sources
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    def offline(query):
        raise ValueError("unavailable")
    monkeypatch.setattr(publisher_videos, "chinanews_images", offline)
    monkeypatch.setattr(video_sources, "search_videos", lambda query: ([{"id": "web", "kind": "video",
        "source_url": "https://v.qq.com/x/page/abcdefghijk.html"}], []))
    results, note = sources.search({"assets": []}, "轨道")
    assert results[0]["id"] == "web" and "国内图片检索未完成" in note


def test_download_honors_server_cooldown_across_attempts(monkeypatch, tmp_path):
    import httpx
    import pytest
    from jinggao import paths
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(sources, "public_url", lambda url: None)
    client = httpx.Client
    calls = []
    def reply(request):
        calls.append(request.url)
        return httpx.Response(429, headers={"Retry-After": "600"})
    monkeypatch.setattr(sources.network, "client", lambda **kw: client(transport=httpx.MockTransport(reply), **kw))
    for name in ("one", "two"):
        with pytest.raises(ValueError, match="10 分钟"):
            sources.download("https://media.example/"+name, tmp_path / (name+".jpg"))
    assert len(calls) == 1
