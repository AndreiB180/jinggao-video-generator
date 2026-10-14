"""Provider feedback earns bounded search opportunities, never visual approval."""
from copy import deepcopy

import pytest

from jinggao import editorial_contract, provider_recall


def case(lines=("茶叶需要揉捻。", "茶叶完成包装。"), hints=None):
    project = {"script": "\n".join(lines), "requirements": ""}
    shots = [{"id": f"shot-{i}", "narration": line, "duration": 8,
              "visual_origin": "generated", "locked": False}
             for i, line in enumerate(lines)]
    hints = hints or [[line] for line in lines]
    rows = [{"ref": f"s{i+1}", "use": "illustration", "refs": [], "facts": [], "must": [],
             "queries": queries} for i, queries in enumerate(hints)]
    bundle = editorial_contract.compile_responses(project, shots, [{"contracts": rows}])
    shots = editorial_contract.attach(project, shots, bundle)
    project["shots"] = shots
    group = {"indices": list(range(len(shots))), "queries": ["茶叶制作完整流程纪录片"]}
    return project, shots, group


def feedback(query, **changes):
    return {"query": query, "providers": [{"provider": "中新网", "status": "ok", "candidates": 0,
            "requested_query": query, "effective_query": query, **changes}]}


def planned(project, shots, group, *, diagnostics=None, broad=None, remaining=None, limit=8, found=None):
    query = group["queries"][0]
    return provider_recall.plan(project, [group], shots, shots if remaining is None else remaining,
        {query: []} if found is None else found,
        [feedback(query)] if diagnostics is None else diagnostics, broad or {}, limit=limit)


def test_successful_empty_provider_can_retry_when_other_channels_return_rows():
    project, shots, group = case()
    query = group["queries"][0]
    source = {"title": "其他平台返回的茶叶素材", "id": "other-provider"}
    before = deepcopy((project, shots, group))
    selected, details = planned(project, shots, group, found={query: [source]})
    assert "茶叶 揉捻" in selected and "茶叶 包装" in selected
    assert all(groups[0] is group for groups in selected.values())
    assert details["茶叶 揉捻"]["shot_ids"] == ["shot-0"]
    assert details["茶叶 包装"]["source_bindings"] == {"shot-1": shots[1]["editorial_contract"]["checksum"]}
    assert all(row["provider"] == "中新网" and row["metadata_only"] for row in details.values())
    assert (project, shots, group) == before


@pytest.mark.parametrize("changes", [
    {"status": "error"}, {"status": "index_only"}, {"candidates": 1}, {"candidates": False},
    {"candidates": "0"}, {"candidates": None}, {"provider": "腾讯视频"},
    {"requested_query": "其他查询"}, {"effective_query": "已经被替换的短词"},
])
def test_only_exact_successful_empty_response_triggers_retry(changes):
    project, shots, group = case()
    assert planned(project, shots, group, diagnostics=[feedback(group["queries"][0], **changes)]) == ({}, {})


def test_missing_provider_feedback_and_foreign_group_are_not_empty_response():
    project, shots, group = case()
    assert planned(project, shots, group, diagnostics=[]) == ({}, {})
    assert planned(project, shots, group, diagnostics=[feedback("另一个组的长查询")]) == ({}, {})


def test_literal_local_words_and_existing_hints_are_both_required():
    project, shots, group = case(("茶叶需要烘干。", "茶叶完成包装。"),
                               [["茶叶 烘干机 动画演示 揉捻"], ["茶叶 包装"]])
    selected, _ = planned(project, shots, group)
    assert set(selected) == {"茶叶 烘干", "茶叶 包装"}


def test_shared_specific_anchor_is_required():
    project, shots, group = case(("工人正在生产。", "工人操作设备。"))
    group["queries"] = ["设备生产加工制作完整流程纪录片"]
    assert planned(project, shots, group) == ({}, {})


@pytest.mark.parametrize("change", [
    {"narration": "不同原稿"}, {"visual_origin": "manual"}, {"locked": True},
    {"persons": [{"name": "张明", "aliases": []}]}, {"event": "指定事件", "event_scope": "exact"},
])
def test_stale_manual_named_and_event_groups_never_relax_contracts(change):
    project, shots, group = case()
    shots[0].update(change)
    assert planned(project, shots, group) == ({}, {})


def test_changed_global_requirements_invalidate_fallback():
    project, shots, group = case()
    project["requirements"] = "必须呈现指定设备，不得用其他型号。"
    assert planned(project, shots, group) == ({}, {})


def test_mandatory_visual_constraint_is_not_shortened():
    project, shots, group = case()
    project["requirements"] = "必须看清茶叶揉捻全过程。"
    rows = [{"ref": f"s{i+1}", "use": "illustration", "refs": [], "facts": [],
             "must": [["activity", "r", "必须看清茶叶揉捻全过程"]] if i == 0 else [],
             "queries": [s["narration"]]} for i, s in enumerate(shots)]
    bundle = editorial_contract.compile_responses(project, shots, [{"contracts": rows}])
    shots = editorial_contract.attach(project, shots, bundle)
    project["shots"] = shots
    assert planned(project, shots, group) == ({}, {})


def test_aircraft_model_literal_is_preserved_as_whole_original_identifier():
    project, shots, group = case(("C919飞机正在滑行。", "C919飞机随后起飞。"),
                               [["C919飞机 滑行"], ["C919飞机 起飞"]])
    group["queries"] = ["C919飞机滑行起飞全过程纪录片"]
    selected, details = planned(project, shots, group)
    assert selected and all(q.startswith("C919 ") for q in selected)
    assert all("c919" not in row["terms"] for row in details.values())


def test_sibling_identifiers_are_not_merged_and_omitted_model_is_restored():
    project, shots, group = case(("C919飞机的机翼接受检查。", "C929飞机的机翼需要维护。"),
                               [["飞机 机翼 检查"], ["C929飞机 机翼 维护"]])
    group["queries"] = ["飞机机翼检查维修完整过程纪录片", "C929飞机机翼维护纪录片"]
    selected, details = planned(project, shots, group)
    assert selected
    for query, row in details.items():
        if "shot-0" in row["shot_ids"]:
            assert "C919" in query and "C929" not in query
        if "shot-1" in row["shot_ids"]:
            assert "C929" in query and "C919" not in query


def test_explicit_source_reference_can_supply_missing_pronoun_identifier():
    project, shots, group = case(("C919飞机的机翼接受检查。", "这架飞机随后起飞。"),
                               [["飞机 机翼 检查"], ["飞机 起飞"]])
    rows = [{"ref": "s1", "use": "illustration", "refs": [], "facts": [], "must": [],
             "queries": ["飞机 机翼 检查"]},
            {"ref": "s2", "use": "illustration", "refs": ["s1"], "facts": [], "must": [],
             "queries": ["飞机 起飞"]}]
    shots = editorial_contract.attach(project, shots,
        editorial_contract.compile_responses(project, shots, [{"contracts": rows}]))
    project["shots"] = shots
    group["queries"] = ["飞机机翼检查与起飞全过程纪录片"]
    selected, details = planned(project, shots, group, remaining=[shots[1]])
    assert "C919 飞机 起飞" in selected
    assert details["C919 飞机 起飞"]["shot_ids"] == ["shot-1"]


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 8])
def test_broad_and_short_retries_share_one_limit_and_keep_original_groups(limit):
    project, shots, group = case()
    broad = {f"补搜长词{i}": [group] for i in range(8)}
    selected, details = planned(project, shots, group, broad=broad, limit=limit)
    assert len(selected) == limit and len(details) <= len(selected)
    assert all(groups[0] is group for groups in selected.values())
    if limit >= 2:
        assert list(selected)[1] == "补搜长词0"


def test_no_new_short_queries_preserves_existing_rescue_order():
    project, shots, group = case()
    broad = {f"补搜长词{i}": [group] for i in range(8)}
    selected, details = planned(project, shots, group, diagnostics=[], broad=broad, limit=3)
    assert list(selected) == list(broad)[:3] and not details


def test_already_searched_or_already_broad_query_is_not_reissued_as_provider_task():
    project, shots, group = case()
    selected, details = planned(project, shots, group,
        found={group["queries"][0]: [], "茶叶 揉捻": []}, broad={"茶叶 包装": [group]})
    assert list(selected) == ["茶叶 包装"] and not details


def test_remaining_subset_uses_stable_shot_ids():
    project, shots, group = case()
    selected, details = planned(project, shots, group, remaining=[shots[1]])
    assert set(selected) == {"茶叶 包装"}
    assert details["茶叶 包装"]["shot_ids"] == [shots[1]["id"]]
    assert planned(project, shots, group, remaining=[]) == ({}, {})


def test_broad_retry_does_not_consume_every_local_opportunity():
    project, shots, group = case(("茶叶需要揉捻烘干。", "茶叶完成包装。"),
                               [["茶叶 揉捻 烘干"], ["茶叶 包装"]])
    shots[0]["duration"] = 30
    selected, details = planned(project, shots, group, broad={"茶叶工序": [group]}, limit=3)
    assert list(selected)[1] == "茶叶工序"
    assert set(sid for row in details.values() for sid in row["shot_ids"]) == {"shot-0", "shot-1"}


def test_match_routes_short_queries_only_to_empty_provider_with_project_shot_indices(tmp_path, monkeypatch):
    from jinggao import config, library_match, montage_match, paths, planner, pooled_match, scene_recall, store
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    lines = ["开头已经检索过。", "茶叶正在揉捻。", "茶叶随后包装。"]
    project = store.new_project("渠道补搜接线", "\n".join(lines), media_policy="video_only")
    shots = [planner.new_shot(line, query=line) | {"visual_origin": "generated"} for line in lines]
    rows = [{"ref": f"s{i+1}", "use": "illustration", "refs": [], "facts": [], "must": [],
             "queries": [["开头检索"], ["茶叶 揉捻"], ["茶叶 包装"]][i]} for i, line in enumerate(lines)]
    attached = editorial_contract.attach(project, shots,
        editorial_contract.compile_responses(project, shots, [{"contracts": rows}]))
    store.mutate(project["id"], lambda p: p.update(shots=attached))
    project = store.get_project(project["id"])
    group = {"label": "茶叶制作", "indices": [0, 1], "queries": ["茶叶加工制作完整流程纪录片"]}
    monkeypatch.setattr(library_match, "match", lambda *a: ({shots[0]["id"]},
        {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {"cache_hit": True}))
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    monkeypatch.setattr(scene_recall, "candidates", lambda *a: {})
    monkeypatch.setattr(montage_match, "match", lambda *a, **kw: (set(), {"model_requests": 0, "failures": []}))
    from jinggao import source_metadata
    monkeypatch.setattr(source_metadata, "rank", lambda p, s, tasks, check: (
        [rows for _, rows in tasks], {"model_requests": 0, "failures": [], "completed_groups": list(range(len(tasks)))}))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("A query fallback must not buy model output"))
    calls, sampled = [], []

    def search(query, **kwargs):
        diagnostic = kwargs["diagnostics"]
        calls.append((query, kwargs))
        diagnostic.update(feedback(query))
        if diagnostic["phase"] == "initial":
            return [], []
        assert kwargs["channel_names"] == {"中新网"}
        assert kwargs["use_index"] is False and kwargs["include_stock"] is False
        if "stock_query_hint" in kwargs:
            assert kwargs["stock_query_hint"] is None
        return [{"id": query, "kind": "video", "title": query, "description": query,
                 "source_url": "https://www.chinanews.com.cn/"+query, "provider": "chinanews",
                 "duration": 50, "width": 1280, "height": 720, "download_status": "available"}], []

    def no_proposals(project, group_rows, *args, **kwargs):
        sampled.extend(c["id"] for _, candidates in group_rows for c in candidates)
        reviewed = {s["id"]: {c["id"] for c in candidates} for rows, candidates in group_rows for s in rows}
        return {}, reviewed, {"model_requests": 0, "sampled_bytes": 0, "failures": []}

    monkeypatch.setattr(pooled_match.video_sources, "search_videos", search)
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "discover_windows", no_proposals)
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate",
                        lambda *a, **kw: pytest.fail("Search metadata does not approve a video window"))
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    assert result["search_queries"] == len(calls) == 3
    assert result["online_rescue_queries"] == result["publisher_fallback_queries"] == 2
    assert result["model_requests"] == result["downloads"] == 0
    assert set(sampled) == {"茶叶 揉捻", "茶叶 包装"}
    diagnostic = {row["query"]: row for row in result["search_tasks"] if row["phase"] == "rescue"}
    assert diagnostic["茶叶 揉捻"]["shot_indices"] == [1]
    assert diagnostic["茶叶 包装"]["shot_indices"] == [2]
    final = store.get_project(project["id"])
    assert all(not s["approved"] and not s["selected_asset"] for s in final["shots"])
