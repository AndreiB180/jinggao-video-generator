"""Optional English suggestions reuse grouping, with no translation requests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import socket
import time

import pytest

from jinggao import (config, library, library_match, model_routing, paths, planner, pooled_match,
                     retrieval_cache, source_index, stock_queries, stock_sources, store, video_sources)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline stock hint test attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def example():
    query = "金属零件表面抛光"
    shot = {"id": "s1", "narration": "工人抛光金属零件。", "query": query, "search_queries": [],
            "persons": [], "event": "", "event_scope": "background", "duration": 8}
    project = {"id": "hint-project", "script": shot["narration"], "requirements": "", "shots": [shot]}
    return project, [shot]


def bound_group(project, shots):
    group = {"label": "机械加工", "indices": [0], "queries": ["机械加工 工作记录", "金属零件表面抛光"]}
    context = stock_queries.hint_context(project, shots)
    group["stock_hints"] = [stock_queries.bind_hint(q, en, context) for q, en in zip(group["queries"],
        ["metal machining work", "metal component polishing"], strict=True)]
    return group


def test_targets_cover_each_first_query_before_alternatives_and_never_change_originals():
    inputs = [[f"第{i}项首词", f"第{i}项备词"] for i in range(30)]
    inputs[29][0] = inputs[0][0]
    before = deepcopy(inputs)
    targets = stock_queries.hint_targets(inputs)
    assert len(targets) == 32 and len({r["query"] for r in targets}) == 32
    assert [r["query"] for r in targets[:29]] == [row[0] for row in inputs[:29]]
    assert [r["query"] for r in targets[29:]] == [row[1] for row in inputs[:3]]
    assert targets[0]["shot_indices"] == [0, 29] and inputs == before


@pytest.mark.parametrize("rows", [None, {}, [["q1", "metal machining"], ["q1", "different action"]],
                                  [["q2", "metal machining"]], [["q1"]], [["q1", {}]], [{"id": "q1"}]])
def test_missing_or_ambiguous_optional_ids_do_not_create_hints(rows):
    assert stock_queries.response_hints(rows, [{"id": "q1", "query": "金属加工"}]) == {}


@pytest.mark.parametrize("english", ["tea", "rocket launch", "2023 C920 rocket launch", "2024 C919 rocket launch",
                                     "a" * 101, {}, "<script>rocket launch</script>"])
def test_binding_rejects_generic_or_malformed_hints_and_lost_year_model_literals(english):
    assert stock_queries.bind_hint("2023 C919首飞", english, "a" * 64) is None


def test_explicit_person_event_and_numeric_constraints_remain_bound_without_aliases():
    project, shots = example()
    shots[0].update(persons=[{"name": "张三", "aliases": []}], event_persons=[{"name": "李四", "aliases": []}],
                    event="2023 R32 事件", event_scope="exact")
    before = deepcopy(shots)
    required = stock_queries.hint_literals("人物赛事", shots)
    assert required == ["张三", "李四", "2023", "R32"]
    context = stock_queries.hint_context(project, shots)
    assert stock_queries.bind_hint("人物赛事", "2023 R32 sports match", context, literals=required) is None
    hint = stock_queries.bind_hint("人物赛事", "张三 李四 2023 R32 sports match", context, literals=required)
    assert hint and hint["preserved_literals"] == required
    assert shots == before
    assert stock_queries.bind_hint("人物赛事", "sports match", context, literals=[{}]) is None


def test_model_hint_matching_retains_negation_and_all_english_qualifiers():
    hint = stock_queries.bind_hint("不能使用破损零件", "not damaged metal components", "a" * 64)
    assert hint
    assert not stock_queries.hint_match(hint["query"], hint, "damaged metal components", context="a" * 64)
    assert stock_queries.hint_match(hint["query"], hint, "not damaged metal components", context="a" * 64)


@pytest.fixture
def grouping(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.read()
    monkeypatch.setattr(config, "read", lambda: cfg)
    project, shots = example()
    calls = []
    response = {"groups": [{"label": "机械加工", "indices": [0], "queries": ["机械加工 工作记录", "备用加工资料"],
                             "stock_queries": ["metal machining work", "machine working footage"]}],
                "stock_hints": [["q1", "metal component polishing"]]}
    def chat(system, content, **options):
        calls.append((system, json.loads(content), options))
        time.sleep(.01)
        return deepcopy(response), {"model": model_routing.model(cfg, "retrieval_grouping"), "request_count": 1}
    monkeypatch.setattr(planner, "chat", chat)
    return project, shots, calls, response, cfg


def test_grouping_reuses_one_call_and_complete_cache_across_parallel_resume(grouping):
    project, shots, calls, _, _ = grouping
    before = deepcopy((project, shots))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: pooled_match.retrieval_groups(project, shots), range(6)))
    assert len(calls) == 1 and sum(info.get("cache_hit", False) for _, info in results) == 5
    groups = results[0][0]
    assert len(groups[0]["stock_hints"]) == 3
    hint = pooled_match.stock_search_hint(project, groups, shots, "金属零件表面抛光")
    assert hint["english"] == "metal component polishing"
    assert calls[0][1]["full_script"] == project["script"]
    assert calls[0][1]["stock_hint_targets"] == [["q1", "金属零件表面抛光", []]]
    assert calls[0][2]["max_tokens"] == 4500
    assert all(info["stock_hint_extra_model_requests"] == 0 for _, info in results)
    assert (project, shots) == before


@pytest.mark.parametrize("malformed", [False, True])
def test_legacy_or_bad_optional_hints_cache_good_group_without_paid_repair(grouping, malformed):
    project, shots, calls, response, _ = grouping
    response.pop("stock_hints")
    response["groups"][0].pop("stock_queries")
    if malformed:
        response.update(stock_hints=[["q1", "a" * 101]])
        response["groups"][0]["stock_queries"] = [{}, "too", "many"]
    first, info = pooled_match.retrieval_groups(project, shots)
    second, later = pooled_match.retrieval_groups(project, shots)
    assert first == second and first[0]["stock_hints"] == []
    assert len(calls) == 1 and later["cache_hit"] and info["stock_hints_available"] == 0


def test_previous_group_protocol_does_not_replan_old_selected_shots(grouping):
    project, shots, calls, _, _ = grouping
    shots[0].update(selected_asset="retained", approved=True, locked=True)
    original = deepcopy(shots)
    cache = retrieval_cache.Groups(project, "old grouping protocol", {"script": project["script"]},
                                   {"task": "retrieval_grouping", "max_tokens": 4500})
    cache.save({"groups": [{"label": "旧组", "indices": [0], "queries": ["旧查询"]}]})
    groups, _ = pooled_match.retrieval_groups(project, shots)
    again, info = pooled_match.retrieval_groups(project, shots)
    assert groups == again and len(calls) == 1 and info["cache_hit"]
    assert shots == original and groups[0]["stock_hints"]


@pytest.mark.parametrize("change", ["script", "requirements", "narration", "persons", "event", "query"])
def test_stale_context_hints_are_not_used_for_search_or_ranking(change):
    project, shots = example()
    group = bound_group(project, shots)
    if change in ("script", "requirements"):
        project[change] += "新限定"
    elif change == "persons":
        shots[0]["persons"] = [{"name": "指定人物", "aliases": []}]
    else:
        shots[0][change] = "另一条原始限定"
    assert pooled_match.stock_search_hint(project, [group], shots, "金属零件表面抛光") is None


@pytest.mark.parametrize("rows", [None, {}, "metal component polishing"])
def test_malformed_optional_group_hints_fall_back_without_breaking_search(rows):
    project, shots = example()
    group = bound_group(project, shots)
    group["stock_hints"] = rows
    assert pooled_match.stock_search_hint(project, [group], shots, shots[0]["query"]) is None


def test_primary_hint_keeps_metadata_opportunity_out_of_evidence_and_approvals():
    project, shots = example()
    group = bound_group(project, shots)
    rows = [{"id": str(i), "kind": "video", "provider": "cctv", "title": "机械零件加工", "status": "unverified"}
            for i in range(30)]
    rows.append({"id": "stock", "kind": "video", "provider": "pixabay", "title": "metal component polishing",
                 "source_evidence": {"title": "original metadata"}, "status": "unverified"})
    before = deepcopy((project, shots, group, rows))
    pool, audit = pooled_match.primary_candidate_pool(project, group, shots, rows, 24)
    assert pool[0]["id"] == "stock" and audit[0]["reason"] == "primary_query_terms"
    assert audit[0]["model_hint_matches"] == [{"query": "金属零件表面抛光", "english": "metal component polishing",
                                               "origin": "model_retrieval_hint"}]
    assert audit[0]["metadata_only"] and not pool[0].get("approved")
    assert (project, shots, group, rows) == before


def test_group_hint_cannot_substitute_for_an_untranslated_primary():
    project, shots = example()
    group = bound_group(project, shots)
    group["stock_hints"] = group["stock_hints"][:1]
    _, audit = pooled_match.primary_candidate_pool(project, group, shots,
        [{"id": "stock", "kind": "video", "title": "metal machining work"}], 1)
    assert audit[0]["reason"] == "group_metadata_fallback"


@pytest.fixture
def search_routes(monkeypatch):
    from jinggao import baidu_videos, platform_videos, portal_videos, publisher_videos
    cfg = {"pixabay_api_key": "unit-test-placeholder"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    monkeypatch.setattr(stock_sources, "registry", lambda *a: {
        "pixabay": {"status": "configured"}, "pexels": {"status": "needs_key"}})
    calls, index_calls = [], []
    def domestic(query):
        calls.append(("cctv", query))
        return []
    for module, name in ((video_sources, "cctv_search"), (platform_videos, "tencent_search"),
                         (platform_videos, "bilibili_search"), (publisher_videos, "chinanews_search"),
                         (portal_videos, "iqiyi_search"), (portal_videos, "sohu_search"),
                         (portal_videos, "youku_search"), (baidu_videos, "baidu_video_search")):
        monkeypatch.setattr(module, name, domestic)
    evidence = {"title": "metal component polishing", "evidence_origin": "platform_metadata"}
    def stock(provider, query, **kwargs):
        calls.append((provider, query))
        return [{"id": "stock", "provider": provider, "kind": "video", "source_url": "https://pixabay.com/videos/id-123/",
                 "title": evidence["title"], "source_evidence": deepcopy(evidence)}]
    def recall(query, **kwargs):
        index_calls.append((query, kwargs))
        return []
    monkeypatch.setattr(stock_sources, "search_items", stock)
    monkeypatch.setattr(source_index, "recall", recall)
    monkeypatch.setattr(library, "remember_sources", lambda *a, **kw: None)
    monkeypatch.setattr(video_sources, "_recent_unavailable", lambda rows, *a: (rows, {}))
    return calls, index_calls, evidence


def test_only_stock_routes_and_stock_index_use_bound_english_with_truthful_diagnostics(search_routes):
    calls, index_calls, evidence = search_routes
    project, shots = example()
    group = bound_group(project, shots)
    query = shots[0]["query"]
    hint = pooled_match.stock_search_hint(project, [group], shots, query)
    diagnostic = {}
    rows, errors = video_sources.search_videos(query, stock_query_hint=hint,
        channel_names={"央视", "Pixabay"}, diagnostics=diagnostic)
    assert not errors and set(calls) == {("cctv", query), ("pixabay", "metal component polishing")}
    assert index_calls == [(query, {"channel_names": {"央视", "Pixabay"}}),
                           ("metal component polishing", {"channel_names": {"Pixabay"}})]
    assert rows[0]["source_evidence"] == evidence and not rows[0].get("approved")
    provider = next(p for p in diagnostic["providers"] if p["provider"] == "Pixabay")
    assert provider["requested_query"] == query and provider["effective_query"] == hint["english"]
    assert provider["query_reason"] == "model_stock_hint"


def test_wrong_query_hint_and_disabled_stock_never_redirect_other_sources(search_routes):
    calls, index_calls, _ = search_routes
    hint = stock_queries.bind_hint("另一原始查询", "different source footage", "a" * 64)
    video_sources.search_videos("原始查询", stock_query_hint=hint, channel_names={"Pixabay"})
    assert calls == [("pixabay", "原始查询")] and len(index_calls) == 1
    calls.clear()
    video_sources.search_videos("另一原始查询", include_stock=False, stock_query_hint=hint,
        channel_names={"央视", "Pixabay"})
    assert calls == [("cctv", "另一原始查询")]


def test_initial_and_existing_rescue_queries_reuse_hints_without_more_query_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("加工", "工人抛光金属零件。", media_policy="video_only")
    original = planner.new_shot(project["script"], query="金属零件表面抛光")
    store.mutate(project["id"], lambda p: p.update(shots=[original]))
    project = store.get_project(project["id"])
    group = bound_group(project, project["shots"])
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {"cache_hit": True}))
    monkeypatch.setattr(planner, "rank_candidates", lambda *a: [])
    monkeypatch.setattr(pooled_match, "discover_windows", lambda *a: ({}, {}, {
        "model_requests": 0, "sampled_bytes": 0, "failures": []}))
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [], []
    monkeypatch.setattr(video_sources, "search_videos", search)
    stats = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    assert [q for q, _ in calls] == group["queries"]
    assert all(kw["stock_query_hint"] and kw["include_stock"] for _, kw in calls)
    assert stats["search_queries"] == 2 and stats["downloads"] == 0 and stats["model_requests"] == 0
    assert stats["query_budget"] == {"video": 16, "rescue": 8, "image": 4}
