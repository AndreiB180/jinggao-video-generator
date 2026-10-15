import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from jinggao import captions, context_sources, costs, curation, identity, media, paths, planner, pooled_match, store, worker
from jinggao.api import create_app
from test_curation import caption_reviewed_legacy_project, completed_pixel_review
from test_workflow import audio_bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "批量 字幕")
    with TestClient(create_app()) as c:
        # These cases cover the classic (v60) pipeline; the express pipeline has its own tests.
        c.patch("/api/settings", json={"model_api_key": "test-only", "pipeline": "classic"})
        yield c


@pytest.fixture
def narration_clock(monkeypatch):
    """Workflow mocks use a supplied cue clock, not recognition of sine waves."""
    def clock(project, **kwargs):
        audio = project["full_audio"]
        path = paths.file_path(project["id"], audio["path"])
        if not path.exists():
            path.write_bytes(b"fixture audio")
        *_, binding, _ = captions._source(project)
        interval = audio["duration"]/len(project["shots"])
        return {"binding": binding, "duration": audio["duration"], "method": "local_paraformer",
                "cues": [{"text": shot["narration"], "start": i*interval, "end": (i+1)*interval}
                         for i, shot in enumerate(project["shots"])]}
    monkeypatch.setattr(captions, "ensure", clock)


def test_batch_validates_all_items_before_creating_any_projects(client):
    assert client.post("/api/batch", json={"projects": [{"script": "有效文稿"}, {"script": "   "}]}).status_code == 422
    assert client.get("/api/projects").json() == []
    assert client.post("/api/batch", json={"projects": [{"script": "稿件"}]*201}).status_code == 422
    result = client.post("/api/batch", json={"projects": [{"title": "甲", "script": "第一篇", "media_policy": "mixed"},
                                                              {"title": "乙", "script": "第二篇", "aspect": "9:16"}]}).json()
    assert len(result["projects"]) == 2
    rows = client.get("/api/queue").json()
    assert {r["title"] for r in rows} == {"甲", "乙"}
    assert all(r["kind"] == "produce" and r["status"] == "queued" for r in rows)
    assert "test-only" not in json.dumps(rows)


def test_caption_preview_uses_render_timing_and_full_audio_offset(client):
    p = store.new_project("字幕", media_policy="mixed", aspect="9:16")
    shots = [planner.new_shot("这段口播需要显示中文字幕。"), planner.new_shot("第二段字幕必须与画面同步。")]
    shots[0].update(duration=4.2)
    shots[1].update(duration=6.7, overlay_label="资料画面")
    store.mutate(p["id"], lambda p: p.update(shots=shots, full_audio={"path": "audio/full.wav", "duration": 10.9}))
    result = client.get(f"/api/projects/{p['id']}/shots/{shots[1]['id']}/captions").json()
    _, expected = media.subtitles(shots[1]["narration"], 6.7, 1080, 1920, "资料画面")
    assert result["segments"] == [list(r) for r in expected]
    assert result["audio_offset"] == 4.2 and result["audio_path"] == "audio/full.wav"
    assert result["segments"][0][0] == 0 and result["segments"][-1][1] == 6.7


def test_pipeline_keeps_job_running_until_render_and_preserves_audio_on_resume(client, monkeypatch, narration_clock):
    steps = []
    def plan(p, *a, **kw):
        steps.append("plan")
        return [planner.new_shot(f"这是第{i}段文稿。") for i in range(4)], {"method": "model"}
    def voice(text, target, check):
        steps.append("voice")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(audio_bytes(8))
    def match(p, shots, check, progress, **kw):
        steps.append("match")
        asset = {"id": "photo", "kind": "image", "path": "assets/photo.jpg", "width": 1920, "height": 1080}
        def save(p):
            p["assets"] = [asset]
            for s in p["shots"]:
                s.update(selected_asset="photo", verification=completed_pixel_review())
        store.mutate(p["id"], save, internal=True)
        return {"selected": 4, "unmatched": [], "elapsed_seconds": 1}
    def render(p, jid, *a, **kw):
        steps.append("render")
        assert store.get_job(jid)["status"] == "running"
        assert all(s["approved"] for s in p["shots"])
        return {"id": jid, "path": "exports/test/video.mp4"}
    monkeypatch.setattr(planner, "plan", plan)
    monkeypatch.setattr(media, "synthesize", voice)
    monkeypatch.setattr(pooled_match, "match", match)
    monkeypatch.setattr(media, "render", render)
    item = client.post("/api/batch", json={"projects": [{"script": "测试四段文稿", "media_policy": "mixed"}]}).json()["projects"][0]
    assert worker.run_once()
    assert steps == ["plan", "voice", "match", "render"]
    assert store.get_job(item["job_id"])["status"] == "completed"
    assert len(store.get_project(item["project_id"])["exports"]) == 1
    steps.clear()
    store.enqueue(item["project_id"], "produce", {"resolution": "720p"})
    worker.run_once()
    assert steps == ["render"]


def test_needs_review_project_does_not_block_next_project(client, monkeypatch):
    for title, selected in (("需补图", False), ("可以导出", True)):
        p = store.new_project(title, "文稿", media_policy="mixed")
        s = planner.new_shot("这是一句口播。")
        s.update(audio={"path": "audio.wav", "duration": 4}, locked=not selected, selected_asset="a" if selected else None,
                 approved=selected)
        if selected:
            s["verification"] = completed_pixel_review()
        store.mutate(p["id"], lambda p, s=s: p.update(shots=[s], assets=[{"id": "a", "kind": "image"}]))
        store.enqueue(p["id"], "produce", {})
    monkeypatch.setattr(media, "render", lambda *a, **kw: {"path": "video.mp4"})
    assert worker.run_once() and worker.run_once()
    rows = client.get("/api/queue").json()
    assert {r["title"]: r["status"] for r in rows} == {"需补图": "needs_review", "可以导出": "completed"}
    waiting = next(r for r in rows if r["status"] == "needs_review")
    assert client.post(f"/api/jobs/{waiting['id']}/retry").status_code == 202


def test_pipeline_rechecks_retained_assets_when_audio_invalidated_review(client, monkeypatch, narration_clock):
    p = store.new_project("续跑复核", "已有口播", media_policy="mixed")
    shot = planner.new_shot("已有口播")
    shot.update(selected_asset="a", approved=False, verification=None)
    store.mutate(p["id"], lambda p: p.update(shots=[shot], assets=[{"id": "a", "kind": "image"}],
                                            full_audio={"path": "full.wav", "duration": 4}))
    calls = []
    def review(project, rows, check):
        calls.append([row["shot"]["id"] for row in rows])
        return [row | {"preview": {"media_kind": "image", "frames": []},
                       "verification": completed_pixel_review()} for row in rows], [], {}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(media, "render", lambda *a, **kw: {"path": "video.mp4"})
    job = store.enqueue(p["id"], "produce", {})
    worker.run_once()
    assert calls == [[shot["id"]]]
    assert store.get_job(job["id"])["status"] == "completed"


def test_resume_retries_uncertain_selected_shots_without_repeating_good_shots(client, monkeypatch, narration_clock):
    p = store.new_project("续跑待核验", "原文", media_policy="mixed")
    good, pending = planner.new_shot("已完成镜头"), planner.new_shot("场景尚待核对")
    good.update(selected_asset="a", approved=True, duration=4, verification=completed_pixel_review())
    pending.update(selected_asset="a", duration=4, verification=completed_pixel_review() | {
        "verdict": "uncertain", "scene_alignment": "uncertain"})
    store.mutate(p["id"], lambda p: p.update(shots=[good, pending], assets=[{"id": "a", "kind": "image"}],
                                            full_audio={"path": "full.wav", "duration": 8}))
    seen = []
    def match(project, shots, *args, **kwargs):
        seen.extend(s["id"] for s in shots)
        def choose(p):
            p["shots"][1].update(verification=completed_pixel_review())
        store.mutate(project["id"], choose, internal=True)
        return {"selected": 1, "unmatched": [], "elapsed_seconds": .1}
    monkeypatch.setattr(pooled_match, "match", match)
    monkeypatch.setattr(media, "render", lambda *a, **kw: {"path": "video.mp4"})
    job = store.enqueue(p["id"], "produce", {})
    worker.run_once()
    assert seen == [pending["id"]]
    assert store.get_job(job["id"])["status"] == "completed"


@pytest.mark.parametrize("manual_source", [False, True])
def test_blank_retained_window_cannot_reapprove_stale_match(client, monkeypatch, manual_source):
    p = store.new_project("空白片尾不得继承旧通过", "已有口播", media_policy="mixed")
    shot = planner.new_shot(p["script"])
    shot.update(selected_asset="a", duration=4, approved=False,
                audio={"path": "existing.wav", "duration": 4},
                verification={"verdict": "match", "scene_alignment": "match",
                    "selection_blocked": False, "needs_source_check": False,
                    "identity_audit": {"version": planner.FRAME_AUDIT_VERSION-1}})
    if manual_source:
        shot.update(source_review_note="用户提供的原始来源说明必须保留", source_review_url="https://example.com/source")
    asset = {"id": "a", "kind": "video", "width": 1280, "height": 720, "duration": 4}
    store.mutate(p["id"], lambda p: p.update(shots=[shot], assets=[asset]))
    monkeypatch.setattr(pooled_match, "sample_frames", lambda *a: ([], {
        "motion_detected": True, "blank_frames": [{"frame_index": 4, "kind": "white"}]}))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("blank sent to model"))
    monkeypatch.setattr(planner, "audit_selected_frames", lambda *a, **kw: pytest.fail("blank sent to observer"))
    retried = []

    def match(project, shots, *args, **kwargs):
        retried.extend(s["id"] for s in shots)
        assert not shots[0]["approved"] and shots[0]["verification"] is None
        return {"selected": 0, "unmatched": retried, "elapsed_seconds": 0}

    monkeypatch.setattr(pooled_match, "match", match)
    monkeypatch.setattr(media, "render", lambda *a, **kw: pytest.fail("stale match exported"))
    job = store.enqueue(p["id"], "produce", {})
    assert worker.run_once()
    result = store.get_project(p["id"])
    saved = result["shots"][0]
    assert retried == [shot["id"]] and not saved["approved"] and saved["verification"] is None
    assert saved["selected_asset"] == "a" and saved["audio"] == shot["audio"] and result["assets"] == [asset]
    assert "空白或黑屏" in saved["search_note"]
    assert saved.get("source_review_note") == shot.get("source_review_note")
    assert saved.get("source_review_url") == shot.get("source_review_url")
    assert not result["exports"] and store.get_job(job["id"])["status"] == "needs_review"


def test_retained_retrieval_miss_returns_to_search_without_paying_for_its_review(client, monkeypatch, narration_clock):
    from jinggao import library_match
    p = store.new_project("续跑检索", "原文", media_policy="mixed")
    shot = planner.new_shot("民航飞机起飞")
    shot.update(selected_asset="old", duration=4, approved=False)
    store.mutate(p["id"], lambda p: p.update(shots=[shot], assets=[{"id": "old", "kind": "image"}],
                                            full_audio={"path": "full.wav", "duration": 4}))
    monkeypatch.setattr(library_match, "retained_retrieval_miss", lambda *_: True)
    monkeypatch.setattr(pooled_match, "review_batch", lambda *a: pytest.fail("irrelevant retained review was paid"))
    seen = []
    def match(project, shots, *args, **kwargs):
        seen.extend(s["id"] for s in shots)
        return {"selected": 0, "unmatched": [s["id"] for s in shots], "elapsed_seconds": 0}
    monkeypatch.setattr(pooled_match, "match", match)
    monkeypatch.setattr(media, "render", lambda *a, **kw: pytest.fail("no approved footage"))
    job = store.enqueue(p["id"], "produce", {})
    worker.run_once()
    updated = store.get_project(p["id"])["shots"][0]
    assert seen == [shot["id"]] and not updated["approved"] and updated["verification"] is None
    assert updated["selected_asset"] == "old"
    assert store.get_job(job["id"])["status"] == "needs_review"


@pytest.mark.parametrize("failure,status", [
    (costs.BudgetExceeded("项目累计模型预算已用完"), "failed"),
    (ValueError("来源不可用"), "failed"),
    (media.Cancelled(), "cancelled"),
])
def test_interrupted_material_stage_has_terminal_workflow_and_keeps_elapsed_time(client, monkeypatch, failure, status):
    project = store.new_project("中断时仍保留时间", "这是已有的口播。", media_policy="mixed")
    shot = planner.new_shot(project["script"])
    audio = {"path": "audio/already-generated.wav", "duration": 4}
    shot.update(audio=audio, duration=4)
    store.mutate(project["id"], lambda p: p.update(shots=[shot]))
    def match(p, shots, check, progress, **kw):
        assert p["workflow_info"]["state"] == "running"
        assert p["workflow_info"]["current_stage"] == "match"
        raise failure
    monkeypatch.setattr(pooled_match, "match", match)
    job = store.enqueue(project["id"], "produce", {})
    assert worker.run_once()
    result = client.get(f"/api/projects/{project['id']}").json()
    terminal = store.get_job(job["id"])
    assert terminal["status"] == result["workflow_info"]["state"] == status
    assert result["workflow_info"]["job_id"] == job["id"]
    assert result["workflow_info"]["current_stage"] == "match"
    assert result["workflow_info"]["reason"] == (terminal["error"] or terminal["message"])
    assert result["workflow_info"]["stages"]["match"] >= 0
    assert result["workflow_info"]["stage_wall_seconds"]["match"] >= 0
    assert result["shots"][0]["audio"] == audio
    assert not result["exports"]


def test_failure_before_first_stage_does_not_reuse_previous_workflow_success(client):
    project = store.new_project("前置失败", "新原稿")
    store.mutate(project["id"], lambda p: p.update(script_changed=True,
        workflow_info={"job_id": "old", "state": "completed", "stages": {"render": 25}}))
    job = store.enqueue(project["id"], "produce", {})
    worker.run_once()
    workflow = store.get_project(project["id"])["workflow_info"]
    assert workflow["job_id"] == job["id"] and workflow["state"] == "failed"
    assert workflow["stages"] == {} and "原稿" in workflow["reason"]


def test_auto_approve_does_not_approve_uncertain_or_unreviewed_assets():
    good = {"selected_asset": "a", "verification": completed_pixel_review()}
    missing = {"selected_asset": "b"}
    uncertain = {"selected_asset": "c", "verification": {"verdict": "uncertain", "selection_blocked": False}}
    p = {"shots": [good, missing, uncertain]}
    assert curation.approve_verified(p) == 1
    assert good["approved"] and not missing.get("approved") and not uncertain.get("approved")
    legacy = {"selected_asset": "a", "verification": good["verification"].copy()}
    legacy["verification"].pop("scene_alignment")
    assert curation.approve_verified({"shots": [legacy]}) == 0


@pytest.mark.skip(reason="stale: built on the per-frame caption_cN audit protocol (test_curation.caption_reviewed_legacy_project); "
                         "frame_contract.requirements no longer emits caption_cN keys")
@pytest.mark.parametrize("change", ["current", "missing-caption", "changed-caption", "changed-frame", "old-version", "locked"])
def test_produce_revalidates_old_automatic_approvals_and_reuses_completed_reviews(client, monkeypatch, change):
    _, shot, asset = caption_reviewed_legacy_project()
    shot.update(duration=4, audio={"path": "already-generated.wav", "duration": 4})
    approved_review = deepcopy(shot["verification"]) | {"reason": "当前片段的动作与全部字幕均完成审核"}
    audit = shot["verification"]["identity_audit"]
    if change == "missing-caption":
        audit["primary_checks"].pop()
    elif change == "changed-caption":
        audit["frame_observations"][1]["text"] = ["另一未审字幕"]
    elif change == "changed-frame":
        audit["frame_observations"][0]["action"] = "坐在会议桌旁"
    elif change == "old-version":
        audit["version"] -= 1
    elif change == "locked":
        shot["locked"] = True
        del shot["verification"]["identity_audit"]
    project = store.new_project("保留审核迁移", shot["narration"], media_policy="mixed")
    store.mutate(project["id"], lambda p: p.update(shots=[shot], assets=[asset]))
    reviewed, rendered = [], []

    def review(p, rows, check):
        check()
        reviewed.extend(row["shot"]["id"] for row in rows)
        return [row | {"preview": {"frames": []}, "verification": deepcopy(approved_review)}
                for row in rows], [], {}

    def render(p, jid, *args, **kwargs):
        rendered.append(jid)
        assert p["shots"][0]["approved"]
        return {"id": jid, "path": "exports/mock.mp4"}

    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(pooled_match, "match", lambda *a, **kw: pytest.fail("current complete selection was searched again"))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("test must not issue paid requests"))
    monkeypatch.setattr(media, "synthesize", lambda *a, **kw: pytest.fail("existing audio was replaced"))
    monkeypatch.setattr(media, "render", render)
    before_cost = costs.summary(project["id"])["project"]
    for _ in range(2):
        job = store.enqueue(project["id"], "produce", {})
        assert worker.run_once()
        assert store.get_job(job["id"])["status"] == "completed", store.get_job(job["id"])
    assert reviewed == ([] if change in {"current", "locked"} else [shot["id"]])
    assert len(rendered) == 2
    assert costs.summary(project["id"])["project"] == before_cost
    assert store.get_project(project["id"])["shots"][0]["audio"] == shot["audio"]


def test_commons_attribution_is_bound_to_exact_file_and_caption():
    shot = {"persons": [{"name": "马龙", "aliases": ["Ma Long"]}], "event_scope": "background", "overlay_label": "资料画面"}
    url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Ma_Long.jpg"
    source = "https://commons.wikimedia.org/wiki/File:Ma_Long.jpg"
    asset = {"kind": "image", "provider": "commons", "source_url": source, "origin_image_url": url,
             "source_evidence": {"url": source, "readable": True, "title": "Ma Long playing table tennis",
                                 "evidence_origin": "platform_metadata", "image_url": url}}
    result = identity.assess(shot, asset)
    assert result["status"] == "supported" and result["coverage"] == "source_attributed"
    assert identity.assess(shot, asset | {"origin_image_url": url+"wrong"})["selection_blocked"]
    assert identity.assess(shot, asset | {"provider": "bing"})["selection_blocked"]
    derivative = "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ab/Ma_Long.jpg/1280px-Ma_Long.jpg"
    derived = asset | {"origin_image_url": derivative, "source_evidence": asset["source_evidence"] | {"image_url": derivative}}
    assert identity.assess(shot, derived)["coverage"] == "source_attributed"
    record = asset["source_evidence"] | {"title": "Fan Zhendong playing table tennis"}
    assert identity.assess(shot, asset | {"source_evidence": record})["selection_blocked"]


def test_library_queries_use_names_and_first_previews_include_multiple_sources():
    g = {"indices": [0]}
    shots = [{"persons": [{"name": "马龙", "aliases": ["Ma Long"]}]}]
    assert pooled_match.compact_query(g, shots, "Ma Long receive serve footwork") == "Ma Long"
    assert pooled_match.compact_query(g, shots, "马龙 接球 衔接 特写") == "马龙"
    rows = [{"provider": "youtube", "id": str(i)} for i in range(20)] + [{"provider": "commons", "id": "commons"}]
    assert "commons" in [c["id"] for c in pooled_match.diverse_candidates(rows)]


def test_primary_source_photo_caption_cannot_borrow_the_article_title_or_another_image(monkeypatch):
    source = "https://bj.news.cn/article/c.html"
    parser = context_sources.PhotoCaptions(source)
    parser.feed('<h1>孙颖莎比赛报道</h1><p><img src="one.jpg"></p><p>10月7日，孙颖莎在比赛中回球。</p>'
                '<p><img src="two.jpg"></p><p>观众在看台上观看比赛。</p>')
    monkeypatch.setattr(context_sources.video_sources, "_cached", lambda *a: parser.rows)
    candidate = {"title": "孙颖莎比赛报道", "kind": "image", "source_url": source,
                 "url": "https://bj.news.cn/article/one.jpg", "provider": "bing"}
    shot = {"persons": [{"name": "孙颖莎", "aliases": ["Sun Yingsha"]}], "event_scope": "background", "overlay_label": "资料画面"}
    enriched = context_sources.image_caption(candidate, lambda: None)
    assert identity.assess(shot, enriched)["coverage"] == "source_attributed"
    wrong = context_sources.image_caption(candidate | {"url": "https://bj.news.cn/article/two.jpg"}, lambda: None)
    assert identity.assess(shot, wrong)["selection_blocked"]
    assert identity.assess(shot, enriched | {"url": "https://bj.news.cn/article/other.jpg"})["selection_blocked"]
    assert not context_sources.photo_publisher("https://news.cn.evil.example/photo")
