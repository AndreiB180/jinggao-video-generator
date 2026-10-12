import json
from collections import defaultdict

import pytest

from jinggao import identity, media, network, paths, planner, pooled_match


def test_online_rescue_only_retries_failed_index_only_queries_and_deduplicates():
    shots=[{'id':'a'},{'id':'b'},{'id':'c'}]
    groups=[{'indices':[0],'queries':['马龙']},{'indices':[1],'queries':['马龙','厨房']},
            {'indices':[2],'queries':['雷军']}]
    indexed={'kind':'video','index_only_search':True}
    found={'马龙':[indexed], '厨房':[indexed,{'kind':'video'}], '雷军':[indexed]}
    tasks=pooled_match.online_rescue_queries(groups,shots,shots[:2],found)
    assert list(tasks)==['马龙'] and len(tasks['马龙'])==2
    assert pooled_match.online_rescue_queries(groups,shots,[],found)=={}


def test_unsuccessful_live_search_gets_an_unused_concrete_action_query():
    shots = [{"id": "a", "query": "物流", "search_queries": ["物流", "快递员 扫码"]},
             {"id": "b", "query": "快递员 电话", "search_queries": ["快递员 电话"]}]
    group = {"indices": [0, 1], "queries": ["物流"]}
    found = {"物流": [{"kind": "video", "index_only_search": False}]}
    rescue = pooled_match.online_rescue_queries([group], shots, shots[:1], found)
    assert list(rescue) == ["快递员 扫码"]
    assert all("快递员 电话" != query for query in rescue)


def test_sampled_allocation_never_shifts_to_a_different_scene():
    c = {"id": "clip", "source_url": "https://v.qq.com/example", "duration": 120}
    proposal = {"clip_start": 68, "sample_at": 72}
    used = defaultdict(list)
    start, size = pooled_match.reserve_sampled(c, proposal, 8, used)
    assert start == 68 and start <= 72 < start+size
    before = used.copy()
    assert pooled_match.reserve_sampled(c, proposal, 8, used) is None
    assert used == before
    assert pooled_match.reserve_sampled(c, {"clip_start": 200, "sample_at": 220}, 8, used) is None
    assert pooled_match.reserve_sampled(c | {"duration": 4}, proposal, 8, used) is None


def test_shared_discovery_samples_source_once_without_approving_any_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    shots = [{"id": "one"}, {"id": "two"}]
    candidate = {"id": "source", "source_url": "https://v.qq.com/example"}
    calls = []
    def sample(c, *args, **kwargs):
        calls.append(c["id"])
        return {"candidate_id": c["id"], "frames": [{"id": "f0"}], "downloaded_bytes": 17, "errors": []}
    def rank(rows, samples, check, **kwargs):
        return {"proposals": {s["id"]: [{"candidate_id": "source", "sampling_only": True}] for s in rows},
                "model_info": {"model": "test"}}
    monkeypatch.setattr(pooled_match.segment_discovery, "sample_candidate", sample)
    monkeypatch.setattr(pooled_match.segment_discovery, "observe_samples", lambda *a: ({}, {"model_requests": 0, "failures": []}))
    monkeypatch.setattr(pooled_match.segment_discovery, "rank_windows_batch", rank)
    suggestions, reviewed, stats = pooled_match.discover_windows({}, [([shots[0]], [candidate]), ([shots[1]], [candidate])], lambda: None, lambda *a: None)
    assert calls == ["source"] and stats["sampled_bytes"] == 17 and stats["model_requests"] == 2
    assert set(suggestions) == {"one", "two"}
    assert reviewed == {"one": {"source"}, "two": {"source"}}
    assert not any(s.get("selected_asset") or s.get("approved") for s in shots)


def test_discovery_does_not_swallow_account_quota_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(pooled_match.segment_discovery, "sample_candidate", lambda *a, **kw: {
        "candidate_id": "source", "frames": [{"id": "f0"}], "downloaded_bytes": 0})
    monkeypatch.setattr(pooled_match.segment_discovery, "observe_samples", lambda *a: ({}, {"model_requests": 0, "failures": []}))
    def unavailable(*a, **kw):
        raise planner.ModelUnavailable("HTTP 402")
    monkeypatch.setattr(pooled_match.segment_discovery, "rank_windows_batch", unavailable)
    with pytest.raises(planner.ModelUnavailable, match="402"):
        pooled_match.discover_windows({}, [([{"id": "one"}], [{"id": "source"}])], lambda: None, lambda *a: None)


def test_discovery_batches_eight_shots_without_resampling_shared_source(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    shots = [{"id": str(i)} for i in range(18)]
    candidate = {"id": "shared"}
    sampled, batches = [], []
    def sample(c, *args, **kwargs):
        sampled.append(c["id"])
        return {"candidate_id": c["id"], "frames": [{"id": "f0"}], "downloaded_bytes": 10, "errors": []}
    def rank(rows, samples, check, **kwargs):
        batches.append([s["id"] for s in rows])
        assert [s["candidate_id"] for s in samples] == ["shared"]
        return {"proposals": {s["id"]: [] for s in rows}, "model_info": {"model": "test"},
                "summary_path": "summary-" + rows[0]["id"]}
    monkeypatch.setattr(pooled_match.segment_discovery, "sample_candidate", sample)
    monkeypatch.setattr(pooled_match.segment_discovery, "observe_samples", lambda *a: ({}, {"model_requests": 0, "failures": []}))
    monkeypatch.setattr(pooled_match.segment_discovery, "rank_windows_batch", rank)
    proposed, reviewed, stats = pooled_match.discover_windows({}, [(shots, [candidate])], lambda: None, lambda *a: None)
    assert sampled == ["shared"] and sorted(map(len, batches)) == [2, 8, 8]
    assert set(proposed) == set(reviewed) == {s["id"] for s in shots}
    assert stats["model_requests"] == 3 and stats["sampled_bytes"] == 10 and len(stats["summaries"]) == 3


def test_remote_titles_without_relevant_sampled_frames_never_trigger_clip_download(tmp_path, monkeypatch):
    from jinggao import config, library_match, montage_match, store
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    p = store.new_project("无合适实帧", "工作人员扫描包裹。", media_policy="video_only")
    shot = planner.new_shot("工作人员扫描包裹。", query="扫描包裹")
    store.mutate(p["id"], lambda p: p.update(shots=[shot]))
    p = store.get_project(p["id"])
    candidates = [{"id": f"candidate-{i}", "kind": "video", "title": "扫描包裹相关标题",
                   "source_url": f"https://v.qq.com/{i}", "provider": "tencent", "duration": 90,
                   "width": 1280, "height": 720, "download_status": "available"} for i in range(5)]
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{"label": "扫码", "indices": [0], "queries": ["扫描包裹"]}], {}))
    monkeypatch.setattr(pooled_match.video_sources, "search_videos", lambda *a, **kw: (candidates, []))
    monkeypatch.setattr(planner, "rank_candidates", lambda p, s, rows, check: [{"candidate_id": c["id"]} for c in rows])
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    sampled, stages = [], []
    def no_related_frames(project, group_rows, *a):
        stages.append("online_video")
        sampled.append({c["id"] for _, rows in group_rows for c in rows})
        reviewed = {s["id"]: {c["id"] for c in rows} for shots, rows in group_rows for s in shots}
        return {}, reviewed, {"model_requests": 1, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", no_related_frames)
    def montage(*a, **kw):
        stages.append("montage")
        return set(), {"model_requests": 0, "failures": []}
    monkeypatch.setattr(montage_match, "match", montage)
    def unexpected_download(*a, **kw):
        pytest.fail("Title-only candidate must not cause blind full-clip download")
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate", unexpected_download)
    result = pooled_match.match(p, p["shots"], lambda: None, lambda *a: None)
    assert result["downloads"] == result["selected"] == 0
    assert len(sampled) == 2 and len(sampled[0]) == 3 and len(sampled[1]) == 2
    assert not sampled[0] & sampled[1]
    assert stages == ["online_video", "online_video", "montage"]
    final = store.get_project(p["id"])
    assert not final["shots"][0]["selected_asset"] and not final["shots"][0]["approved"]
    assert len(pooled_match.prior_probes(final, final["shots"][0])) == 5


def test_resume_probes_prioritize_unseen_sources_across_candidate_aliases():
    shot = {"id": "one", "narration": "工作人员分拣包裹", "query": "分拣包裹", "duration": 6}
    project = {"shots": [shot], "min_video_height": 720, "script": "工作人员分拣包裹"}
    group = {"indices": [0], "queries": ["分拣包裹"]}
    candidates = [{"id": f"new-alias-{i}", "kind": "video", "source_url": f"https://v.qq.com/{i}",
                   "title": "工作人员分拣包裹", "width": 1280, "height": 720,
                   "duration": 90, "download_status": "available"} for i in range(6)]
    seen = {c["source_url"] for c in candidates[:3]}
    shot["retrieval_probes"] = {"context": pooled_match.probe_context(project, shot), "sources": list(seen)}
    assert pooled_match.prior_probes(project, shot) == seen
    groups = {(0,): {"group": group, "shots": [shot], "choices": {c["id"]: c for c in candidates}}}
    chosen, budgeted = pooled_match.allocate_sampling_candidates(project, groups, [shot], seen_sources={"one": seen})
    assert {c["source_url"] for c in chosen[(0,)]} == {c["source_url"] for c in candidates[3:]}
    assert len(budgeted) == 3 and not shot.get("approved")
    assert pooled_match.prior_probes(project | {"requirements": "必须展示门口交付"}, shot) == set()
    assert pooled_match.prior_probes(project, shot | {"narration": "快递员拨打电话"}) == set()


def test_shared_intervals_do_not_overlap_even_across_groups():
    used = defaultdict(list)
    c = {"id": "a", "source_url": "https://example.com/video", "duration": 160}
    slots = [pooled_match.reserve(c, 5, used) for _ in range(16)]
    assert all(b[0] >= a[0]+a[1]-.001 for a, b in zip(slots, slots[1:], strict=False))
    assignments = [{"candidate": c, "start": start, "size": size} for start, size in slots]
    blocks = pooled_match.block_requests(assignments, limit=45)
    assert len(blocks) == 2
    assert all(b["end"]-b["start"] <= 45 for b in blocks)
    assert sum(len(b["items"]) for b in blocks) == 16


def test_rescue_keeps_inspected_but_unsampled_group_sources_eligible(tmp_path, monkeypatch):
    from jinggao import config, library_match, store
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    p = store.new_project("继续检查候选", "工作人员扫描包裹。", media_policy="video_only")
    shot = planner.new_shot("工作人员扫描包裹。", query="扫描包裹")
    store.mutate(p["id"], lambda p: p.update(shots=[shot]))
    p = store.get_project(p["id"])
    def candidates(prefix, count):
        return [{"id": f"{prefix}-{i}", "kind": "video", "title": "扫描包裹",
                 "source_url": f"https://v.qq.com/{prefix}/{i}", "provider": "tencent", "duration": 90,
                 "width": 1280, "height": 720, "download_status": "available"} for i in range(count)]
    initial, rescue = candidates("initial", 5), candidates("rescue", 10)
    group = {"label": "扫码", "indices": [0], "queries": ["扫描包裹"]}
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {}))
    monkeypatch.setattr(pooled_match.video_sources, "search_videos",
                        lambda q, **kw: (initial if q == "扫描包裹" else rescue, []))
    monkeypatch.setattr(planner, "rank_candidates", lambda p, s, rows, check: [{"candidate_id": c["id"]} for c in rows])
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {"扫码工作": [group]})
    sampled = []
    def no_related_frames(project, group_rows, *a):
        sampled.append({c["id"] for _, rows in group_rows for c in rows})
        return {}, {}, {"model_requests": 1, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", no_related_frames)
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate",
                        lambda *a, **kw: pytest.fail("No observed scene authorizes a download"))
    result = pooled_match.match(p, p["shots"], lambda: None, lambda *a: None)
    untried = {c["id"] for c in initial} - sampled[0]
    assert len(sampled) == 2 and all(len(rows) <= 3 for rows in sampled)
    assert untried <= sampled[1]
    assert not sampled[0] & sampled[1]
    assert result["downloads"] == result["selected"] == 0


def test_inspected_sources_do_not_occupy_fresh_inspection_queue_slots(tmp_path, monkeypatch):
    from jinggao import config, library_match, montage_match, scene_recall, store
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("补搜新来源可解析", "工作人员扫描包裹。", media_policy="video_only")
    shot = planner.new_shot("工作人员扫描包裹。", query="扫描包裹")
    store.mutate(project["id"], lambda p: p.update(shots=[shot]))
    project = store.get_project(project["id"])

    def candidates(prefix, count, providers):
        return [{"id": f"{prefix}-{i}", "kind": "video", "title": "扫描包裹",
                 "source_url": f"https://example.com/{prefix}/{i}",
                 "provider": providers[i % len(providers)], "duration": 90,
                 "width": 1280, "height": 720, "download_status": "available"} for i in range(count)]

    initial = candidates("initial", 10, ["tencent", "bilibili"])
    rescue = candidates("rescue", 6, ["chinanews"])
    group = {"label": "扫码", "indices": [0], "queries": ["扫描包裹"]}
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {"cache_hit": True}))
    monkeypatch.setattr(scene_recall, "candidates", lambda *a: {})
    monkeypatch.setattr(pooled_match.video_sources, "search_videos",
                        lambda q, **kw: (initial if q == "扫描包裹" else rescue, []))
    monkeypatch.setattr(planner, "rank_candidates", lambda p, s, rows, check: [{"candidate_id": c["id"]} for c in rows])
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {"包裹扫码动作": [group]})
    monkeypatch.setattr(montage_match, "match", lambda *a, **kw: (set(), {"model_requests": 0, "failures": []}))
    inspected, sampled = [], []

    def inspect(candidate, *args):
        inspected.append(candidate["id"])
        return candidate

    def no_related_frames(project, group_rows, *args):
        sampled.append({c["id"] for _, rows in group_rows for c in rows})
        return {}, {}, {"model_requests": 0, "sampled_bytes": 0, "failures": []}

    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", inspect)
    monkeypatch.setattr(pooled_match, "discover_windows", no_related_frames)
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate",
                        lambda *a, **kw: pytest.fail("Metadata inspection never authorizes a clip download"))
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    # The old providers used to occupy six of the ten second-pass lookup slots,
    # then get discarded. Two fresh URLs were never inspected, even with only
    # six new sources available and ten free lookup slots.
    assert len(inspected) == len(set(inspected)) == 16
    assert {c["id"] for c in rescue} <= set(inspected)
    assert all(len(rows) <= 3 for rows in sampled) and len(sampled) == 2
    assert result["downloads"] == result["selected"] == 0


def test_existing_used_source_ranges_remain_reserved():
    project = {"assets": [{"id": "x", "kind": "video", "source_url": "https://e.test/1", "source_clip_start": 12}],
               "shots": [{"selected_asset": "x", "clip_start": 0, "duration": 20}]}
    used = pooled_match.intervals(project)
    start, size = pooled_match.reserve({"id": "another-candidate-id", "source_url": "https://e.test/1", "duration": 100}, 8, used)
    assert start >= 32
    assert pooled_match.reserve({"id": "short", "duration": 3}, 4, used) is None


def test_prior_library_video_without_candidate_id_authorizes_only_neighbour_samples():
    good = {"verdict": "match", "selection_blocked": False, "needs_source_check": False,
            "visual_quality": {"quality_status": "not_detected", "representation": "live_action",
                               "frame_sha256": [f"{i:064x}" for i in range(5)]}}
    asset = {"id": "cached", "kind": "video", "source_url": "https://e.test/1", "source_clip_start": 12}
    project = {"assets": [asset], "shots": [{"id": "previous", "selected_asset": "cached", "clip_start": 2,
        "duration": 8, "verification": good}]}
    saved = pooled_match.verified_source_windows(project)
    assert saved == {"https://e.test/1": [(14, 22)]}
    candidate = {"id": "new-provider-id", "source_url": "https://e.test/1", "duration": 100}
    points = pooled_match.adjacent_sample_points(candidate, saved[candidate["source_url"]],
                                                pooled_match.intervals(project), [8])
    assert points and all(not 14 <= point <= 22 for point in points)
    project["shots"][0]["verification"] = good | {"needs_source_check": True}
    assert not pooled_match.verified_source_windows(project)
    project["shots"][0]["verification"] = good | {"selection_blocked": True}
    assert not pooled_match.verified_source_windows(project)


def test_generic_shared_pool_requires_topic_overlap_before_frame_review():
    assert pooled_match.same_topic(["乒乓球 练习", "table tennis training"], {"title": "Table tennis game.webm"})
    assert not pooled_match.same_topic(["马龙 乒乓球", "table tennis"], {"title": "Tennis court aerial view"})
    assert not pooled_match.same_topic(["training highlights"], {"title": "Random training highlights"})


def test_visual_ranking_is_not_overridden_by_publisher_badge():
    shot = {"persons": [{"name": "马龙", "aliases": []}], "event_scope": "background"}
    def candidate(cid, title, verified):
        return {"id": cid, "kind": "video", "title": title, "publisher_verified": verified,
            "source_evidence": {"readable": True, "title": title, "url": "https://v.qq.com/"+cid,
                                "evidence_origin": "publisher_search"}, "source_url": "https://v.qq.com/"+cid}
    interview = candidate("interview", "马龙采访", True)
    action = candidate("action", "马龙比赛回合", False)
    order = {"action": 0, "interview": 1}
    def key(c):
        return pooled_match.candidate_priority(shot, c, order, set(), "16:9")
    assert sorted([interview, action], key=key)[0] == action


@pytest.mark.parametrize("indices", [[0, 1, 2], [False, 1]])
def test_model_grouping_cannot_drop_duplicate_or_invent_shots(monkeypatch, indices):
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"groups": [
        {"label": "topic", "indices": indices, "queries": ["test"]}]}, {}))
    with pytest.raises(ValueError):
        pooled_match.retrieval_groups({}, [{"id": "a"}, {"id": "b"}])


def test_omitted_search_group_member_keeps_its_original_review_conditions(monkeypatch):
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"groups": [
        {"label": "topic", "indices": [0], "queries": ["test"]}]}, {}))
    shots = [{"id": "a", "event": "event A"}, {"id": "b", "event": "event B"}]
    groups, _ = pooled_match.retrieval_groups({}, shots)
    assert groups[0]["indices"] == [0, 1]
    assert shots[1]["event"] == "event B"


def test_overlapping_search_groups_are_deduplicated_without_repeated_work(monkeypatch):
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"groups": [
        {"label": "topic", "indices": [0, 0, 1], "queries": ["test"]},
        {"label": "topic2", "indices": [1, 2], "queries": ["test2"]}]}, {}))
    groups, _ = pooled_match.retrieval_groups({}, [{"id": str(i)} for i in range(3)])
    assert [i for g in groups for i in g["indices"]] == [0, 1, 2]


def test_evidence_gate_is_identical_for_batched_and_single_reviews():
    shot = {"persons": [{"name": "梁靖崑", "aliases": ["Liang Jingkun"]}],
            "event": "赛事：中国大满贯；年份：2026；轮次：R32；对阵：梁靖崑 vs 郭冠宏",
            "event_scope": "exact"}
    title = "2025 中国大满贯 梁靖崑 vs 郭冠宏 R16"
    asset = {"kind": "video", "source_url": "https://example.com/1", "source_evidence": {
        "url": "https://example.com/1", "title": title, "readable": True, "evidence_origin": "platform_metadata"}}
    result = {"verdict": "match", "reason": "模型误判", "conflicts": [], "needs_source_check": False,
              "visual_evidence": [{"kind": "scoreboard", "frame_index": 0, "quote": "LIANG Jingkun", "context": "计分牌"}],
              "event_evidence": [{"quote": title, "source_url": asset["source_url"], "relation": "same_event"}]}
    reviewed = planner.review_result(shot, asset, result, 5, {})
    assert reviewed["verdict"] == "mismatch" and reviewed["selection_blocked"]
    assert identity.assess(shot, asset, result, frame_count=5, has_segment_frames=True)["status"] == "conflict"


@pytest.mark.parametrize("scene,expected,blocked", [("mismatch", "mismatch", True), ("uncertain", "uncertain", False), ("match", "match", False)])
def test_correct_person_cannot_override_action_scene_mismatch(scene, expected, blocked):
    shot = {"narration": "看马龙的回合，也要看看脚下怎样调整。", "persons": [{"name": "马龙", "aliases": ["Ma Long"]}],
            "event_scope": "background", "overlay_label": "资料画面"}
    asset = {"kind": "video", "source_url": "https://example.com/interview", "source_evidence": {
        "url": "https://example.com/interview", "title": "马龙接受采访", "readable": True, "evidence_origin": "platform_metadata"}}
    observation = {"verdict": "match", "reason": "人物正确", "conflicts": [], "needs_source_check": False,
        "scene_alignment": scene, "scene_reason": "实际为坐着回答问题的采访，没有比赛或训练动作",
        "visual_evidence": [{"kind": "caption", "frame_index": 0, "quote": "马龙", "context": "发言人姓名字幕"}]}
    result = planner.review_result(shot, asset, observation, 5, {})
    assert result["identity_evidence"]["status"] == "supported"
    assert result["verdict"] == expected and result["selection_blocked"] == blocked
    if scene == "match":
        observation.pop("scene_alignment")
        assert planner.review_result(shot, asset, observation, 5, {})["verdict"] == "uncertain"


def test_batch_review_rejects_wrong_ids_and_keeps_frames_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    pid = "a"*32
    f = tmp_path/"frame.jpg"
    from PIL import Image
    Image.new("RGB", (64, 64), "white").save(f)
    monkeypatch.setattr(pooled_match, "sample_frames", lambda *a: ([f]*5, {"motion_detected": True}))
    seen = []
    def chat(system, content, **kwargs):
        seen.extend(content)
        return {"reviews": [{"shot_id": "first"}, {"shot_id": "invented"}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    project = {"id": pid}
    rows = [{"shot": {"id": sid, "clip_start": 0, "duration": 3},
             "asset": {"title": sid, "kind": "video"}} for sid in ("first", "second")]
    with pytest.raises(ValueError, match="编号"):
        pooled_match.review_batch(project, rows, lambda: None)
    labels = [json.loads(c["text"]) for c in seen if c["type"] == "text" and "frame_index" in c["text"]]
    assert [r["shot_id"] for r in labels] == ["first"]*5+["second"]*5
    assert [r["frame_index"] for r in labels] == list(range(5))*2


def test_actual_frame_sampling_uses_one_decode_and_correct_offsets(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    pid = "b"*32
    target = paths.project_dir(pid)/"moving.mp4"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=320x180:rate=25:duration=5", "-c:v", "libx264", "-preset", "ultrafast", str(target)])
    calls = []
    run = media.run
    def counted(args, **kw):
        calls.append(args)
        return run(args, **kw)
    monkeypatch.setattr(media, "run", counted)
    frames, preview = pooled_match.sample_frames(pid, {"duration": 3, "clip_start": 1},
        {"duration": 5, "path": "moving.mp4", "source_clip_start": 20}, lambda: None)
    assert len(calls) == 1 and len(frames) == 5 and preview["motion_detected"]
    assert all(21 <= f["at"] < 24 for f in preview["frames"])


def test_cancel_before_decode_is_propagated(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    def cancel(*a, **kw):
        raise media.Cancelled()
    monkeypatch.setattr(media, "run", cancel)
    with pytest.raises(media.Cancelled):
        pooled_match.sample_frames("c"*32, {"duration": 3, "clip_start": 0},
                                    {"duration": 5, "path": "sample.mp4"}, cancel)


def test_network_does_not_retry_timeout_or_server_errors(monkeypatch):
    import httpx
    calls = []
    def failed(*a, **kw):
        calls.append(1)
        raise httpx.ReadTimeout("timeout after request sent")
    monkeypatch.setattr(network.httpx, "post", failed)
    with pytest.raises(httpx.ReadTimeout):
        network.post("https://example.test/charge")
    assert len(calls) == 1


def test_network_recovers_only_stale_source_connect_error(monkeypatch):
    import httpx
    monkeypatch.setattr(network, "routed_address", lambda host: "10.0.0.5")
    def failed(*a, **kw):
        raise httpx.ConnectError("[Errno 49] Can't assign requested address")
    monkeypatch.setattr(network.httpx, "post", failed)
    calls = []
    class Client:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return "recovered"
    monkeypatch.setattr(network.httpx, "Client", Client)
    assert network.post("https://e.test/a", json={"ok": True}) == "recovered"
    assert calls == [("https://e.test/a", {"json": {"ok": True}})]
