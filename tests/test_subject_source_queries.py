from copy import deepcopy

import pytest

from jinggao import config, editorial_contract, library_match, montage_candidates, paths, planner, pooled_match, project_pool, store, video_sources


def scenario():
    lines = ["公交到站，乘客走近车门。", "公交车门打开，人们上车。", "公交车厢内有座椅和扶手。", "公交车乘客坐下休息。"]
    p = {"script": "".join(lines), "requirements": "", "min_video_height": 720}
    shots = [planner.new_shot(line, visual_origin="generated") for line in lines]
    response = {"contracts": [{"ref": f"s{i}", "use": "illustration", "refs": [], "facts": [], "must": [], "queries": []}
                              for i in range(1, 5)]}
    shots = editorial_contract.attach(p, shots, editorial_contract.compile_responses(p, shots, [response]))
    groups = [{"indices": [0, 1], "queries": ["公交车门打开乘客上车", "公交停靠"], "label": "上下车"},
              {"indices": [2, 3], "queries": ["公交车厢座椅扶手乘客", "公交乘客休息"], "label": "车内"}]
    return p, shots, groups


def test_generic_article_adds_one_shared_source_query_without_changing_contracts():
    p, shots, groups = scenario()
    before = deepcopy((p, shots, groups))
    assert pooled_match.subject_source_queries(p, groups, shots) == [{"query": "公交 实拍", "indices": [0, 1, 2, 3]}]
    assert (p, shots, groups) == before


@pytest.mark.parametrize("change", [{"persons": [{"name": "张三", "aliases": []}]},
    {"event": "2026年某次活动", "event_scope": "exact"}, {"locked": True}, {"visual_origin": "manual"}])
def test_specific_or_user_owned_shots_are_not_broadened(change):
    p, shots, groups = scenario()
    shots[0].update(change)
    assert not pooled_match.subject_source_queries(p, groups, shots)


def test_shared_search_never_displaces_existing_primary_queries(monkeypatch):
    p, shots, groups = scenario()
    monkeypatch.setattr(pooled_match, "VIDEO_QUERY_BUDGET", 2)
    assert not pooled_match.subject_source_queries(p, groups, shots)


def test_normal_match_routes_shared_results_into_each_related_group(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p, shots, groups = scenario()
    project = store.new_project("source-query test", script=p["script"])
    project = store.mutate(project["id"], lambda value: value.update(shots=shots))
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "video"})
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: (deepcopy(groups), {}))
    monkeypatch.setattr(project_pool, "collect", lambda *a: ([], {}))
    monkeypatch.setattr(library_match, "match", lambda *a, **k: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(montage_candidates, "match", lambda *a, **k: (set(), {"model_requests": 0, "failures": []}))
    monkeypatch.setattr(planner, "chat", lambda *a, **k: pytest.fail("No new model request for subject search"))
    queries = []
    def search(query, **kwargs):
        queries.append(query)
        return [{"id": "shared-bus", "kind": "video", "title": "公交运行实拍", "description": "",
                 "provider": "bilibili", "duration": 420, "width": 1280, "height": 720,
                 "source_url": "https://www.bilibili.com/video/fixture"}] if query == "公交 实拍" else [], []
    monkeypatch.setattr(video_sources, "search_videos", search)
    class ReachedRanking(BaseException):
        pass
    def inspect(project, group, targets, candidates, limit):
        assert [row["id"] for row in candidates] == ["shared-bus"]
        assert "公交 实拍" in queries and len(queries) == 3
        assert not any(s.get("selected_asset") or s.get("approved") for s in store.get_project(project["id"])["shots"])
        raise ReachedRanking
    monkeypatch.setattr(pooled_match, "primary_candidate_pool", inspect)
    with pytest.raises(ReachedRanking):
        pooled_match.match(project, shots, lambda: None, lambda *a: None, workers=1)
