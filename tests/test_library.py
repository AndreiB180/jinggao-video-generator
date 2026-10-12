import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jinggao import config, domestic, library, library_collect, library_match, media, neutral_observer, paths, planner, pooled_match, store
from jinggao.api import create_app


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "索引 测试")
    store.init()
    p = store.new_project("原项目")
    target = paths.project_dir(p["id"]) / "assets" / "moving.mp4"
    target.parent.mkdir(exist_ok=True)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=320x180:rate=25:duration=6", "-c:v", "libx264", "-preset", "ultrafast", str(target)])
    asset = media.inspect_asset(p["id"], target, "温室番茄采摘")
    asset.update(source_url="https://v.qq.com/x/page/test.html", provider="tencent", source_clip_start=40,
                 approved=True, verification={"verdict": "match"})
    store.mutate(p["id"], lambda p: p["assets"].append(asset))
    clip_id = library.remember_window(p["id"], asset, 1, 4, description="手从枝头摘下红色番茄", visible_text="智慧农业")
    return p, asset, library.get(clip_id)


def test_chinese_search_persists_and_deduplicates_bytes(indexed):
    p, asset, clip = indexed
    assert library.search("枝头 采摘")[0]["id"] == clip["id"]
    assert library.search("智慧农业")[0]["visible_text"] == "智慧农业"
    assert (clip["start"], clip["offset"], clip["duration"]) == (41, 1, 4)
    assert library.search("不存在的火箭") == []
    assert library.search("番茄", min_height=720) == []
    assert library.search("番茄", min_duration=5) == []
    again = library.remember_window(p["id"], asset, 1, 4)
    library.remember_window(p["id"], asset, 0, 3, description="另一个时间段")
    assert again == clip["id"]
    assert library.get(again)["description"] == clip["description"]
    assert library.stats()["clips"] == 2 and library.stats()["blob_count"] == 1


def test_cache_reuse_preserves_provenance_but_not_approval(indexed):
    p, asset, clip = indexed
    other = store.new_project("新项目")
    reused = library.materialize(other["id"], clip)
    assert reused["download_mode"] == "library_cache" and reused["downloaded_bytes"] == 0
    assert reused["source_clip_start"] + clip["offset"] == 41
    assert "approved" not in reused and "verification" not in reused
    assert library.materialize(other["id"], clip)["id"] == reused["id"]
    # Identical video bytes from a different source must not inherit the first source's identity evidence.
    asset = asset | {"source_url": "https://v.qq.com/x/page/another.html", "source_clip_start": 80}
    different = library.get(library.remember_window(p["id"], asset, 1, 4))
    reused_other = library.materialize(other["id"], different)
    assert reused_other["id"] != reused["id"]
    assert reused_other["source_url"] == asset["source_url"] and reused_other["source_clip_start"] == 80
    assert library.stats()["blob_count"] == 1


def test_damaged_or_missing_cache_is_not_silently_reused(indexed):
    _, _, clip = indexed
    target = library.file_path(clip["path"])
    content = target.read_bytes()
    target.write_bytes(b"x" + content[1:])  # Same length; checksum must still reject it.
    with pytest.raises(ValueError, match="变更"):
        library.materialize(store.new_project("新项目")["id"], clip)
    target.unlink()
    assert library.search("番茄") == []


def test_original_duration_does_not_overwrite_decoded_clip_duration(indexed):
    p, asset, _ = indexed
    other_source = asset | {"source_url": "https://v.qq.com/x/page/long-original.html"}
    library.remember_sources([other_source | {"duration": 600}])
    clip = library.get(library.remember_window(p["id"], other_source, 1, 4))
    assert clip["source"]["duration"] == 600
    reused = library.materialize(store.new_project("目标")["id"], clip)
    assert reused["duration"] == asset["duration"] == 6


def test_unintegrated_sources_and_invalid_windows_rejected(indexed):
    p, asset, _ = indexed
    with pytest.raises(ValueError, match="已接入"):
        library.remember_window(p["id"], asset | {"source_url": "https://youtube.com/watch?v=a"})
    with pytest.raises(ValueError, match="超出"):
        library.remember_window(p["id"], asset, 5, 4)
    with pytest.raises(ValueError, match="路径"):
        library.file_path("../settings.json")
    library.remember_sources([{**asset, "source_url": "https://www.youtube.com/watch?v=a"}])
    assert library.stats()["sources"] == 1


def test_observed_cctv_cdn_does_not_allow_arbitrary_cloud_accounts():
    assert domestic.allowed("https://newcntv.qcloudcdn.com/video.mp4")
    assert not domestic.allowed("https://arbitrary.qcloudcdn.com/video.mp4")
    assert not domestic.allowed("https://newcntv.qcloudcdn.com.evil.test/video.mp4")


@pytest.mark.parametrize("verdict,matched", [("match", True), ("mismatch", False)])
def test_cache_requires_fresh_actual_frame_review(indexed, monkeypatch, verdict, matched, isolated_pixel_quality):
    _, _, _clip = indexed
    with library.connection() as con:
        con.execute("INSERT INTO clip_observation_methods VALUES (?,?)", (_clip["id"], "blind_actual_frames_v1"))
    p = store.new_project("当前文稿", min_video_height=0)
    shot = {"id": store.uid(), "duration": 3, "query": "番茄采摘", "subject": "番茄",
            "narration": "工人在温室里采摘番茄。", "event_scope": "background", "persons": [],
            "selected_asset": None, "clip_start": 0, "approved": False}
    store.mutate(p["id"], lambda p: p["shots"].append(shot))
    seen, blind_seen, judge_narrations = [], [], []
    monkeypatch.setattr(config, "read", lambda: {"model_name": "text-test", "vision_model": "vision-test",
        "model_api_key": "test-only", "model_base_url": "https://models.test"})
    def observe_request(content, cfg, check):
        blind_seen.extend(content)
        assert json.loads(content[0]["text"]) == {"observation_groups": [
            {"group_id": "g0", "frame_indices": [0, 1, 2, 3, 4]}]}
        frames = [json.loads(part["text"]) for part in content[1:] if part["type"] == "text"]
        assert len(frames) == 5 and {f["group_id"] for f in frames} == {"g0"}
        assert all(set(frame) == {"group_id", "frame_index"} for frame in frames)
        return json.dumps({"groups": [{"id": "g0", "scene_description": "温室内手从枝头采摘红色番茄。",
            "frames": [{"frame_index": i, "objects": "番茄植株和手", "action": "手从枝头摘下番茄", "text": []}
                       for i in range(5)]}]}, ensure_ascii=False), {}
    monkeypatch.setattr(neutral_observer, "_request", observe_request)
    def chat(system, content, **kwargs):
        if isinstance(content, str):
            row = json.loads(content)["shots"][0]
            return {"rankings": [{"shot_id": shot["id"],
                                  "candidate_refs": [row["candidates"][0]["candidate_ref"]]}]}, {}
        seen.extend(content)
        if "独立的片段内容复核员" in system:
            assert kwargs["vision"] is False and all(part["type"] == "text" for part in content)
            facts = json.loads(content[0]["text"])
            judge_narrations.append(facts["narration"])
            assert len(facts["frame_observations"]) == 5
            return {"reviews": [{"shot_id": shot["id"],
                "status": "conflict" if "浇水" in facts["narration"] else "supported",
                "reason": "复核实际温室采摘", "labels": [],
                "evidence": [{"frame_index": 0, "field": "action", "quote": "手从枝头摘下番茄"}]}]}, {"requests": 1}
        return {"reviews": [{"shot_id": shot["id"], "verdict": verdict, "reason": "本次抽帧审核",
            "scene_alignment": verdict, "scene_reason": "本次动作核验", "needs_source_check": False,
            "conflicts": [], "visual_evidence": []}]}, {"requests": 1}
    monkeypatch.setattr(planner, "chat", chat)
    passed, stats = library_match.match(store.get_project(p["id"]), [shot], lambda: None, lambda *a: None)
    assert bool(passed) is matched
    actual = store.get_project(p["id"])["shots"][0]
    assert bool(actual.get("selected_asset")) is matched
    assert not actual["approved"]
    assert len([x for x in seen if x["type"] == "image_url"]) == 5
    assert len([x for x in blind_seen if x["type"] == "image_url"]) == (5 if matched else 0)
    assert stats["library_model_requests"] == (4 if matched else 2)
    if matched:
        # Reuse the same pixels under another script: blind observations can be
        # cached, but the current-script text judgment must still run and veto.
        another = store.new_project("另一份当前文稿", min_video_height=0)
        shot = shot | {"id": store.uid(), "narration": "工人在温室为番茄浇水。"}
        store.mutate(another["id"], lambda p: p["shots"].append(shot))
        passed, stats = library_match.match(store.get_project(another["id"]), [shot], lambda: None, lambda *a: None)
        assert not passed and not store.get_project(another["id"])["shots"][0]["selected_asset"]
        assert judge_narrations == ["工人在温室里采摘番茄。", "工人在温室为番茄浇水。"]
        assert len([x for x in seen if x["type"] == "image_url"]) == 10
        assert len([x for x in blind_seen if x["type"] == "image_url"]) == 5
        assert stats["library_model_requests"] == 3


@pytest.mark.parametrize("refs", [["S02-C01"], [["S01-C01"]], [True], ["S01-C01", "S01-C01"]])
def test_library_ranking_cannot_borrow_another_shots_candidates(indexed, monkeypatch, refs):
    _, _, clip = indexed
    options = {"one": [clip], "two": [clip | {"id": "different"}]}
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"rankings": [
        {"shot_id": "one", "candidate_refs": refs}, {"shot_id": "two", "candidate_refs": ["S02-C01"]}]}, {}))
    errors = []
    assert library_match.rank_windows([{"id": "one"}, {"id": "two"}], options, lambda: None, diagnostics=errors)
    assert options["one"] == []
    assert [r["id"] for r in options["two"]] == ["different"]
    assert len(errors) == 1 and "混用" in errors[0]


def test_library_ranking_can_reject_all_windows_without_rechecking_them(indexed, monkeypatch):
    _, _, clip = indexed
    options = {"one": [clip]}
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"rankings": [
        {"shot_id": "one", "candidate_refs": []}]}, {}))
    assert library_match.rank_windows([{"id": "one"}], options, lambda: None)
    assert options["one"] == []


def test_full_cache_hit_never_launches_online_search(indexed, monkeypatch):
    p = store.new_project("当前文稿")
    shot = {"id": store.uid(), "duration": 3, "selected_asset": None, "approved": False}
    p = store.mutate(p["id"], lambda p: p["shots"].append(shot))
    monkeypatch.setattr(library_match, "match", lambda *a: ({shot["id"]}, {
        "library_model_requests": 1, "library_failures": [], "library_matched": 1}))
    def online(*args, **kwargs):
        pytest.fail("cache hit launched online retrieval")
    monkeypatch.setattr(pooled_match, "retrieval_groups", online)
    stats = pooled_match.match(p, [shot], lambda: None, lambda *a: None)
    assert stats["downloads"] == 0 and stats["search_queries"] == 0


def test_repeated_cached_cuts_do_not_hide_an_older_independent_source(indexed, monkeypatch):
    source_project, asset, _ = indexed
    other_url = "https://v.qq.com/x/page/another-greenhouse.html"
    observation = {"description": "手从枝头采摘红色番茄", "visible_text": "智慧农业",
                   "replace_observations": True}
    library.remember_window(source_project["id"], asset | {"source_url": other_url}, 1, 4, **observation)
    for i in range(18):
        library.remember_window(source_project["id"], asset, 1 + i * .01, 4, **observation)
    # The former twelve-row lookup loses the other source before model ranking.
    assert other_url not in {r["source"]["source_url"] for r in library.search("番茄 采摘", limit=12)}
    project = store.new_project("新稿", min_video_height=0)
    shot = {"id": store.uid(), "duration": 3, "query": "番茄 采摘", "subject": "番茄",
            "selected_asset": None, "approved": False}
    project = store.mutate(project["id"], lambda p: p["shots"].append(shot))
    ranked = []
    def rank(shots, options, check, **kwargs):
        ranked.extend(options[shot["id"]])
        return True
    monkeypatch.setattr(library_match, "rank_windows", rank)
    monkeypatch.setattr(pooled_match, "review_batch", lambda p, rows, check: (
        [r | {"verification": {"verdict": "mismatch"}} for r in rows], [], {}))
    passed, stats = library_match.match(project, [shot], lambda: None, lambda *a: None)
    assert {r["source"]["source_url"] for r in ranked} == {asset["source_url"], other_url}
    assert len(ranked) <= 3 and stats["library_quick_reviewed"] == 1
    assert not passed and not store.get_project(project["id"])["shots"][0]["selected_asset"]


def test_caption_numbering_cannot_mix_observations(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), "red").save(frame)
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"model_api_key": "test", "vision_model": "fixture",
                                                "model_base_url": "https://example.invalid/v1"})
    monkeypatch.setattr(neutral_observer, "_request", lambda *a, **kw: (json.dumps({"groups": [
        {"id": "g0", "scene_description": "甲"}, {"id": "g0", "scene_description": "乙"}]}), {}))
    with pytest.raises(ValueError, match="编号"):
        library_collect.describe([{"frames": [frame]}, {"frames": [frame]}], lambda: None)


def test_collected_blind_observations_qualify_for_fast_lookup_without_old_approval(indexed, monkeypatch):
    p, asset, old = indexed
    frame = paths.project_dir(p["id"]) / "collection-frame.jpg"
    Image.new("RGB", (80, 80), "red").save(frame)
    monkeypatch.setattr(config, "read", lambda: {"model_api_key": "test", "vision_model": "fixture",
                                                "model_base_url": "https://example.invalid/v1"})
    def request(content, cfg, check):
        assert asset["title"] not in json.dumps(content, ensure_ascii=False)
        return json.dumps({"groups": [{"id": "g0", "scene_description": "人物的手从枝头采摘番茄",
            "frames": [{"frame_index": 0, "objects": "番茄植株与人的手", "action": "手正在采摘番茄",
                        "text": ["智慧农业"]}]}]}, ensure_ascii=False), {}
    monkeypatch.setattr(neutral_observer, "_request", request)
    description = library_collect.describe([{"frames": [frame]}], lambda: None)[0]
    cid = library.remember_window(p["id"], asset, 1, 4, **description)
    assert cid == old["id"]
    assert library_match._blind_clip_ids([cid]) == {cid}
    clip = library.get(cid)
    shot = {"id": "fresh", "duration": 4, "query": "番茄 采摘", "subject": "番茄", "visual": "人物采摘番茄",
            "visual_plan": {"mode": "single", "primary": {"subject": "番茄", "action": "采摘", "scene": "温室"}}}
    stats = {}
    selected, _ = library_match.quick_candidates({"shots": [shot]}, [shot], {shot["id"]: [clip]}, stats)
    assert selected == [shot]
    reused = library.materialize(store.new_project("另一个原稿")["id"], clip)
    assert "verification" not in reused and "approved" not in reused


def test_library_api_bounds_and_collection_project_visibility(indexed):
    p, _, clip = indexed
    with TestClient(create_app()) as client:
        r = client.get("/api/library?query=番茄&min_height=0")
        assert r.status_code == 200 and len(r.json()["clips"]) == 1
        assert client.get("/api/library?min_height=-1").status_code == 400
        assert client.post("/api/library/build", json={"queries": ["番茄"], "max_sources": 100}).status_code == 422
        assert client.post("/api/library/build", json={"urls": ["https://www.youtube.com/test"]}).status_code == 400
        assert client.post("/api/library/build", json={"action": "rebuild"},
                           headers={"Origin": "https://evil.example"}).status_code == 403
        job = client.post("/api/library/build", json={"action": "rebuild"}).json()
        assert job["kind"] == "library" and job["status"] == "queued"
        assert {x["id"] for x in store.list_projects()} == {p["id"]}
        assert client.post("/api/library/build", json={"action": "rebuild"}).status_code == 400
        assert client.get(f"/api/library/clips/{clip['id']}/thumbnail").status_code == 200
        store.enqueue(p["id"], "match", {})
        assert client.post(f"/api/library/clips/{clip['id']}/use", json={"project_id": p["id"]}).status_code == 400


def test_rebuild_does_not_index_requested_scene_as_observation(indexed):
    p, asset, _ = indexed
    shot = {"id": store.uid(), "duration": 3, "clip_start": 0, "selected_asset": asset["id"],
            "narration": "火箭发射", "visual": "火箭发射", "verification": {"reason": "火箭发射"}}
    store.mutate(p["id"], lambda p: p["shots"].append(shot))
    assert library.rebuild()["processed"] == 1
    assert library.search("火箭") == []
    with library.connection() as con:
        body = json.loads(con.execute("SELECT body FROM sources").fetchone()[0])
    assert "verification" not in body and "approved" not in body


def test_rebuild_uses_the_review_for_the_preview_not_an_older_selection(indexed):
    p, asset, _ = indexed
    other = asset | {"id": store.uid(), "source_url": "https://v.qq.com/x/page/different.html"}
    shot = {"id": store.uid(), "duration": 3, "clip_start": 0, "selected_asset": asset["id"],
            "match_preview": {"asset_id": other["id"], "clip_start": 2, "duration": 3},
            "verification": {"observations": {"scene_description": "火箭发射"}},
            "match_result": {"observations": {"scene_description": "温室栽培",
                "visual_evidence": [{"kind": "caption", "frame_index": 0, "quote": "可读字幕"}]}}}
    store.mutate(p["id"], lambda p: (p["assets"].append(other), p["shots"].append(shot)))
    library.rebuild()
    assert library.search("火箭") == []
    rows = library.search("可读字幕")
    assert len(rows) == 1 and rows[0]["source"]["source_url"] == other["source_url"]
    assert rows[0]["start"] == 42


def test_planning_keeps_explicit_paragraph_scene_boundaries():
    script = "夜幕降临，城市车灯连成线。\n走进温室，栽培装置整齐排列。\n回到球场，运动员来回击球。"
    shots = planner.rule_shots(script)
    assert [s["narration"] for s in shots] == script.splitlines()
    units = [u for batch in planner.source_unit_batches(script) for u in batch]
    assert [u["narration"] for u in units] == script.splitlines()


@pytest.mark.parametrize("coverage", ["brief", "none"])
def test_brief_action_cannot_pass_as_a_continuous_action_clip(coverage):
    shot = {"narration": "运动员移动步伐，来回击球。", "persons": [], "event_scope": "background"}
    asset = {"kind": "video"}
    observation = {"verdict": "match", "reason": "场馆相关", "conflicts": [], "needs_source_check": False,
                   "scene_alignment": "match", "scene_reason": "只有开头在击球，后段一直擦脸等候",
                   "action_coverage": coverage}
    result = planner.review_result(shot, asset, observation, 5, {})
    assert result["verdict"] == "mismatch" and result["selection_blocked"]
    assert result["scene_alignment"] == "mismatch"


def test_absent_actions_do_not_earn_observed_retrieval_terms():
    description = '画面展示仓库内静置的纸箱。未出现人物在楼道或小区内行走、核对地址或拨打电话的动作。'
    terms = library.observed_terms(description)
    assert '仓库' in terms and '纸箱' in terms
    assert not {'电话', '地址', '楼道', '小区'} & terms
    assert '电话' in library.observed_terms('工作人员正在打电话。')
    assert '电话' in library.observed_terms(description, '电话服务说明')
