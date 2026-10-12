"""Bilingual metadata grants inspection opportunities, never scene approval."""
from copy import deepcopy
import socket

import pytest

from jinggao import library, paths, planner, pooled_match, source_index, stock_sources, store, video_sources


def shot(query, **fields):
    return {"id": "s1", "query": query, "search_queries": [], "duration": 8,
            "persons": [], "event": "", "event_scope": "background", **fields}


def video(cid, title, **fields):
    return {"id": cid, "kind": "video", "provider": "nasa", "title": title,
            "description": "", "width": 1280, "height": 720, "status": "unverified", **fields}


@pytest.fixture(autouse=True)
def no_network_or_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bilingual metadata ranking must not use network or models")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(planner, "chat", forbidden)


def test_real_nasa_metadata_survives_recall_to_primary_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    monkeypatch.setattr(stock_sources, "registry", lambda: {"nasa": {"status": "configured"}})
    nasa = video_sources._candidate("nasa", "TROPICS Rocket Launch",
        "https://images.nasa.gov/details/KSC-20230507-MH-RLS01-TROPICS_Rocket_Launch-Rocket_Lab-WON_3330130",
        "", description="TROPICS Rocket Launch", source_verified=True)
    local = [video_sources._candidate("tencent", "火箭发射现场",
        f"https://v.qq.com/x/page/{i:011d}.html", "") for i in range(39)]
    library.remember_sources(local + [nasa])
    source_index.record_check(nasa | {"download_status": "available"})
    candidates = source_index.recall("火箭发射")
    assert len(candidates) == 40 and candidates[0]["id"] == nasa["id"]
    before = deepcopy(candidates)
    rows, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": ["火箭发射"]},
        [shot("火箭发射")], candidates, 24)
    assert nasa["id"] in {c["id"] for c in rows}
    entry = next(r for r in trace if r["candidate_id"] == nasa["id"])
    assert entry["reason"] == "primary_query_terms" and entry["matched_terms"] == []
    assert entry["bilingual_matches"] == [{"query": "火箭发射", "concepts": ["火箭发射"]}]
    assert entry["metadata_only"] and candidates == before
    assert all(not c.get("approved") and not c.get("verification") for c in rows)


@pytest.mark.parametrize("query,title,concept", [
    ("火箭发射现场实拍", "ROCKET-LAUNCH footage", "火箭发射"),
    ("发射火箭", "launching rockets", "火箭发射"),
    ("数据中心", "a data centre", "数据中心"),
    ("机场", "an airport terminal", "机场"),
])
def test_complete_phrase_and_known_aliases_get_current_query_bound_opportunities(query, title, concept):
    group = {"indices": [0], "queries": ["普通资料"]}
    shots = [shot(query)]
    candidates = [video("unrelated", "A scenic view"), video("mapped", title, index_coverage=0)]
    before = deepcopy((shots, candidates))
    rows, trace = pooled_match.primary_candidate_pool({}, group, shots, candidates, 1)
    assert rows[0]["id"] == "mapped"
    assert trace[0]["bilingual_matches"] == [{"query": query, "concepts": [concept]}]
    assert trace[0]["reason"] == "primary_query_terms"
    assert (shots, candidates) == before and not rows[0].get("approved")


@pytest.mark.parametrize("query,title", [
    ("火箭发射失败", "rocket launch footage"),
    ("2024年火箭发射", "rocket launch footage"),
    ("中国火箭发射", "rocket launch footage"),
    ("马龙机场", "an airport terminal"),
    ("不是火箭发射", "rocket launch footage"),
    ("火箭发射 空间站", "rocket launch footage"),
    ("火箭发射", "a rocket launcher"),
    ("火箭发射", "a phone launch beside a model rocket"),
    ("茶叶摊放", "tea, cup, tea light"),
])
def test_stale_coverage_cannot_erase_qualifiers_or_complete_a_partial_phrase(query, title):
    candidate = video("weak", title, index_coverage=1, from_source_index=True)
    before = deepcopy(candidate)
    rows, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": ["普通资料"]},
        [shot(query)], [candidate], 1)
    assert rows == [before] and trace[0]["reason"] == "group_metadata_fallback"
    assert "bilingual_matches" not in trace[0]


def test_group_translation_does_not_upgrade_incomplete_primary():
    candidate = video("related", "火箭 rocket launch footage", index_coverage=1)
    _, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": ["火箭发射"]},
        [shot("火箭发射失败")], [candidate], 1)
    assert trace[0]["reason"] == "group_metadata_fallback"
    assert trace[0]["bilingual_matches"] == [{"query": "火箭发射", "concepts": ["火箭发射"]}]


def test_all_concepts_remain_required_beyond_four_exported_english_queries():
    query = "火箭发射、空间站、航天员训练、卫星发射、地球观测"
    partial = video("partial", "rocket launch, space station, astronaut training, satellite launch")
    complete = video("complete", partial["title"] + ", Earth observation")
    _, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": ["普通资料"]},
        [shot(query)], [partial, complete], 2)
    assert trace[0]["candidate_id"] == "complete" and trace[0]["reason"] == "primary_query_terms"
    assert len(trace[0]["bilingual_matches"][0]["concepts"]) == 5
    assert trace[1]["candidate_id"] == "partial" and trace[1]["reason"] == "group_metadata_fallback"


def test_translation_does_not_bypass_exact_event_resolution_or_identity_priority():
    required = shot("火箭发射", persons=[{"name": "张三", "aliases": []}], event="2023年火箭发射", event_scope="exact")
    candidates = [video("wrong-year", "2024 张三 TROPICS Rocket Launch"),
                  video("low", "2023 张三 TROPICS Rocket Launch", height=480),
                  video("unnamed", "2023 TROPICS Rocket Launch"),
                  video("named", "2023 张三 TROPICS Rocket Launch")]
    before = deepcopy((required, candidates))
    rows, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": ["火箭发射"]},
        [required], candidates, 2)
    assert [c["id"] for c in rows] == ["named", "unnamed"]
    assert all(r["reason"] == "primary_query_terms" for r in trace)
    assert (required, candidates) == before and not any(c.get("approved") for c in rows)


def test_matched_mapping_is_not_carried_to_a_different_query():
    candidate = video("candidate", "TROPICS Rocket Launch", index_coverage=1)
    for query, reason in (("火箭发射", "primary_query_terms"), ("空间站", "group_metadata_fallback")):
        _, trace = pooled_match.primary_candidate_pool({}, {"indices": [0], "queries": [query]},
            [shot(query)], [candidate], 1)
        assert trace[0]["reason"] == reason
        assert bool(trace[0].get("bilingual_matches")) == (query == "火箭发射")
