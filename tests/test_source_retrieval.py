"""Source-format hints affect inspection opportunities, never visual approval."""
from copy import deepcopy
import json

import pytest

from jinggao import config, editorial_contract, library_match, paths, planner, pooled_match, store, video_sources


def bound(project, shots, *, must=(), use="illustration", queries=("抽象文案", "方便生活")):
    response = {"contracts": [{"ref": f"s{i}", "use": use, "refs": [], "facts": [],
                              "must": list(must), "queries": list(queries)}
                             for i, _ in enumerate(shots, 1)]}
    bundle = editorial_contract.compile_responses(project, shots, [response])
    return editorial_contract.attach(project, shots, bundle)


def scenario():
    project = {"script": "工人在车间生产产品。", "requirements": "", "min_video_height": 720}
    shots = [planner.new_shot(project["script"], query="工人 加工 产品", visual_origin="generated")]
    shots = bound(project, shots, queries=("车间生产", "工人加工"))
    group = {"indices": [0], "queries": ["车间 工人 工作记录", "车间 生产 加工"]}
    return project, shots, group


def video(cid, title, duration=30, **fields):
    return {"id": cid, "kind": "video", "title": title, "duration": duration, "description": "",
            "provider": "bilibili", "width": 1280, "height": 720, "status": "unverified", **fields}


def test_source_grouping_uses_original_activity_without_repeating_untested_queries(monkeypatch):
    project, shots, _ = scenario()
    before = deepcopy(shots)
    captured = []
    def chat(system, user, **options):
        captured.append((system, json.loads(user), options))
        return {"groups": [{"label": "车间工作", "indices": [0], "queries": ["工人 车间工作记录", "生产流程 实拍"]}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    pooled_match.retrieval_groups(project, shots)
    system, payload, options = captured[0]
    row = payload["shots"][0]
    assert payload["full_script"] == project["script"]
    assert row["narration"] == project["script"]
    assert not {"query", "search_queries", "query_hints"} & row.keys()
    assert "query_hints" not in row["editorial_contract"]
    assert "数据年份不是拍摄年份" in system and "当日实录" in system
    assert "不得为搜素材添加原稿没有的人物/具体事件/地点" in system
    assert len(captured) == 1 and options["timeout"] == 45 and shots == before


def test_longer_recording_gets_one_probe_without_replacing_direct_actions(monkeypatch):
    project, shots, group = scenario()
    candidates = [video("direct", "工人在车间加工产品"), video("other", "车间生产产品"),
                  video("recording", "探访工人生产车间", 900),
                  video("wrong-topic", "探访新型动车组", 200, description="工人 车间 生产 加工"),
                  video("low", "探访工人生产车间", 400, height=480),
                  video("unknown-duration", "工人车间工作记录", "unknown")]
    before = deepcopy((shots, candidates))
    pool, audit = pooled_match.primary_candidate_pool(project, group, shots, candidates, 3)
    assert pool[0]["id"] == "recording"
    assert 'direct' in {c['id'] for c in pool}
    assert 'wrong-topic' not in {c['id'] for c in pool}
    assert audit[0]["reason"] == "source_recording_opportunity" and audit[0]["metadata_only"]
    assert not any(row["candidate_id"] == "wrong-topic" and row["reason"] == "source_recording_opportunity"
                   for row in audit)
    assert not any(c.get("verification") or c.get("approved") for c in pool)
    assert all(c["status"] == "unverified" for c in pool) and (shots, candidates) == before


@pytest.mark.parametrize("change", [
    {"persons": [{"name": "张三", "aliases": []}]},
    {"event": "2026年某次开工仪式", "event_scope": "exact"},
    {"locked": True}, {"visual_origin": "manual"},
])
def test_named_event_and_user_edit_keep_existing_retrieval_path(change):
    project, shots, group = scenario()
    assert pooled_match.source_pool_group(group, shots, project)
    shots[0].update(change)
    original = deepcopy(shots)
    assert not pooled_match.source_pool_group(group, shots, project)
    candidates = [video("direct", "车间生产 工人加工", 30), video("recording", "探访工人生产车间", 900)]
    _, audit = pooled_match.primary_candidate_pool(project, group, shots, candidates, 2)
    assert all(row["reason"] != "source_recording_opportunity" for row in audit)
    assert shots == original


def test_explicit_visual_command_keeps_its_retrieval_path():
    project, shots, group = scenario()
    project["requirements"] = "必须拍到工人操作车床。"
    shots = bound(project, shots, must=[["activity", "r", "必须拍到工人操作车床"]])
    assert editorial_contract.for_shot(shots[0], project)["must"]
    assert not pooled_match.source_pool_group(group, shots, project)


def test_rescue_uses_alternate_source_query_not_abstract_contract_hints():
    project, shots, group = scenario()
    shots = bound(project, shots, queries=("普通产品", "现代生活方便"))
    before = deepcopy(shots)
    rescue = pooled_match.online_rescue_queries([group], shots, shots, {group["queries"][0]: []})
    assert list(rescue) == [group["queries"][1]]
    assert not {"普通产品", "现代生活方便"} & rescue.keys() and shots == before


def test_source_rescue_retains_broad_alternative_then_actual_uncovered_activity():
    project, shots, group = scenario()
    before = deepcopy(shots)
    rescue = pooled_match.online_rescue_queries([group], shots, shots, {group['queries'][0]: []})
    # Source-pool groups no longer re-add per-shot contract hints (see online_rescue_queries).
    assert list(rescue) == [group['queries'][1]]
    assert not {'车间生产', '工人加工'} & rescue.keys()
    assert shots == before
    assert pooled_match.online_rescue_queries([group], shots, [], {group['queries'][0]: []}) == {}


def test_source_gap_does_not_create_an_animation_or_new_action_requirement():
    project = {'script': '工人用高温处理原料。', 'requirements': ''}
    shots = bound(project, [planner.new_shot(project['script'], visual_origin='generated')],
                  queries=('高温处理原理动画', '高温处理设备'))
    original = deepcopy(shots)
    assert pooled_match.source_gap_queries(shots[0]) == ['高温处理设备']
    assert shots == original and not shots[0].get('approved')


def test_generic_metadata_path_skips_cover_model_but_never_downloads_blindly(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("工作过程", "工人在车间生产产品。", media_policy="video_only")
    shots = bound(project, [planner.new_shot(project["script"], visual_origin="generated")])
    store.mutate(project["id"], lambda p: p.update(shots=shots))
    project = store.get_project(project["id"])
    group = {"label": "车间工作", "indices": [0], "queries": ["车间 工人 工作记录"]}
    rows = [video(str(i), "实拍工人生产车间", 300,
                  source_url=f"https://www.bilibili.com/video/test{i}", download_status="available") for i in range(3)]
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {}))
    monkeypatch.setattr(video_sources, "search_videos", lambda *a, **kw: (rows, []))
    monkeypatch.setattr(video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "discover_windows", lambda *a, **kw: ({}, {}, {
        "model_requests": 0, "sampled_bytes": 0, "failures": []}))
    def unexpected(*a, **kw):
        pytest.fail("Metadata must not authorize a cover model or blind window download")
    monkeypatch.setattr(planner, "rank_candidates", unexpected)
    from jinggao import source_metadata
    def chat(system, content, **options):
        # The only permitted model call is the text-only source_ranking shortlist.
        if system != source_metadata.PROTOCOL or options.get("vision"):
            unexpected()
        return ({"groups": [{"group_ref": "g0", "candidate_refs": ["g0:c0", "g0:c1", "g0:c2"]}]},
                {"request_count": 1})
    monkeypatch.setattr(planner, "chat", chat)
    monkeypatch.setattr(video_sources, "download_video_candidate", unexpected)
    stats = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    assert stats["model_requests"] == 2  # Grouping plus one text-only source_ranking batch.
    assert stats["downloads"] == stats["selected"] == 0
    audit = stats["candidate_screening"][0]
    assert audit["cover_limit"] == 0 and audit["method"] == "source_metadata_to_actual_frames"
    assert all(not row["cover"] for row in audit["selections"])


def test_discovery_keeps_failed_paid_attempts_and_successful_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    candidate = {"id": "shared"}
    shots = [{"id": "failed"}, {"id": "okay"}]
    monkeypatch.setattr(pooled_match.segment_discovery, "sample_candidate", lambda *a, **kw: {
        "candidate_id": "shared", "frames": [{"id": "f0"}], "downloaded_bytes": 17})
    monkeypatch.setattr(pooled_match.segment_discovery, "observe_samples", lambda *a: ({}, {"model_requests": 0, "failures": []}))
    def locate(rows, *a, **kw):
        if rows[0]["id"] == "failed":
            error = ValueError("定位响应失败；原始记录已保存")
            error.model_requests, error.summary_path = 2, "failed-summary.json"
            raise error
        return {"proposals": {"okay": []}, "model_info": {}, "model_requests": 1,
                "summary_path": "okay-summary.json"}
    monkeypatch.setattr(pooled_match.segment_discovery, "rank_windows_batch", locate)
    proposals, reviewed, info = pooled_match.discover_windows({}, [([s], [candidate]) for s in shots],
                                                             lambda: None, lambda *a: None)
    assert info["model_requests"] == 3 and info["sampled_bytes"] == 17
    assert set(info["summaries"]) == {"failed-summary.json", "okay-summary.json"}
    assert info["failures"] and proposals == {"okay": []} and reviewed == {"okay": {"shared"}}


def test_resumed_source_queries_are_bound_and_never_change_review_conditions():
    project, shots, group = scenario()
    shot = shots[0]
    before_context = editorial_contract.context(shot)
    original = editorial_contract.retrieval_fields(shot)['query_hints']
    shot['retrieval_hints'] = {'source_checksum': shot['editorial_contract']['checksum'],
                              'queries': group['queries']}
    assert editorial_contract.retrieval_fields(shot)['query_hints'][:2] == group['queries']
    assert editorial_contract.context(shot) == before_context
    shot['retrieval_hints']['source_checksum'] = 'different-script'
    assert editorial_contract.retrieval_fields(shot)['query_hints'] == original
