"""Regressions discovered while auditing the full-length user workflow."""
import pytest
from fastapi.testclient import TestClient

from jinggao import config, context_sources, curation, library_match, media, paths, planner, pooled_match, sources, store, video_sources, worker
from jinggao.api import create_app
from test_curation import completed_pixel_review
from test_workflow import audio_bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with TestClient(create_app()) as client:
        yield client


def test_editor_full_payload_cannot_restore_stale_source_review(client):
    p = store.new_project("人物片段")
    s = planner.new_shot("刘翔接受采访。", persons=[{"name": "刘翔", "aliases": []}], evidence_mode="person")
    s.update(selected_asset="video", approved=True, source_review_note="原片此时间点姓名字幕确实写有刘翔", source_review_url="https://tv.cctv.com/interview")
    store.mutate(p["id"], lambda p: p.update(shots=[s]))
    fields = {k: s[k] for k in ("narration", "visual", "persons", "duration", "clip_start", "fit", "source_review_note", "source_review_url")}
    fields["clip_start"] = 10
    route = f"/api/projects/{p['id']}/shots/{s['id']}"
    result = client.patch(route, json=fields)
    assert result.status_code == 200
    changed = result.json()["shots"][0]
    assert changed["source_review_note"] == changed["source_review_url"] == ""
    assert not changed["approved"]
    assert client.patch(route, json={"approved": True}).status_code == 400


def test_locked_shot_survives_rejected_full_voice_import(client):
    p = store.new_project("锁定镜头")
    s = planner.new_shot("已有配音")
    s.update(locked=True, duration=2, audio={"path": "original.wav", "duration": 2})
    store.mutate(p["id"], lambda p: p.update(shots=[s]))
    r = client.post(f"/api/projects/{p['id']}/upload", data={"role": "narration"},
                    files={"file": ("整段.wav", audio_bytes(4), "audio/wav")})
    assert r.status_code == 400 and "解锁" in r.json()["detail"]
    current = store.get_project(p["id"])
    assert current["shots"][0] == s and not current.get("full_audio") and not current["assets"]


def test_auto_matching_preserves_full_scene_across_portrait_and_landscape():
    wide = {"width": 1920, "height": 1080}
    tall = {"width": 1080, "height": 1920}
    assert curation.preferred_fit({"fit": "cover"}, wide, "9:16") == "contain"
    assert curation.preferred_fit({"fit": "cover"}, tall, "16:9") == "contain"
    assert curation.preferred_fit({"fit": "cover"}, wide, "16:9") == "cover"
    assert curation.preferred_fit({"fit": "cover", "fit_origin": "manual"}, wide, "9:16") == "cover"


def test_produce_resumes_partial_voice_without_replacing_existing_audio(client, monkeypatch):
    p = store.new_project("续配", "第一段。第二段。", media_policy="mixed")
    first, second = [planner.new_shot(t) for t in ("第一段。", "第二段。")]
    first.update(duration=4, audio={"path": "original.wav", "duration": 4}, locked=True)
    for shot in (first, second):
        shot.update(selected_asset="photo", approved=True)
    second["duration"] = 4
    store.mutate(p["id"], lambda p: p.update(shots=[first, second], assets=[{"id": "photo", "kind": "image"}]))
    calls = []
    def synthesize(text, path, check):
        calls.append(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes(4))
    monkeypatch.setattr(media, "synthesize", synthesize)
    reviewed = []
    def review(project, rows, check):
        reviewed.extend(row["shot"]["id"] for row in rows)
        return [row | {"preview": {}, "verification": completed_pixel_review()} for row in rows], [], {}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("lifecycle test must not call a model"))
    monkeypatch.setattr(pooled_match, "match", lambda *a, **kw: pytest.fail("retained selection passed current review"))
    monkeypatch.setattr(media, "render", lambda *a, **kw: {"path": "test.mp4"})
    job = store.enqueue(p["id"], "produce")
    worker.run_once()
    assert store.get_job(job["id"])["status"] == "completed"
    assert calls == ["第二段。"]
    assert reviewed == [second["id"]]
    assert store.get_project(p["id"])["shots"][0]["audio"]["path"] == "original.wav"


@pytest.mark.parametrize("provider", ["local", "video"])
def test_shared_pool_respects_search_provider(client, monkeypatch, provider):
    config.update({"search_provider": provider})
    p = store.new_project("设置范围", "一段", media_policy="mixed")
    shot = planner.new_shot("一段", event="2024年8月1日比赛", evidence_mode="event")
    store.mutate(p["id"], lambda p: p.update(shots=[shot]))
    p = store.get_project(p["id"])
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{"label": "比赛", "indices": [0], "queries": ["比赛"]}], {}))
    def forbidden(*a, **kw):
        pytest.fail("queried a source disabled by the user's search setting")
    monkeypatch.setattr(sources, "search_images", forbidden)
    monkeypatch.setattr(video_sources, "search_videos", forbidden if provider == "local" else lambda *a, **kw: ([], []))
    monkeypatch.setattr(library_match, "match", forbidden if provider == "local" else lambda *a: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(context_sources, "search", forbidden if provider == "local" else lambda *a: [])
    result = pooled_match.match(p, p["shots"], lambda: None, lambda *a: None)
    assert result["selected"] == 0


def test_shared_pool_image_budget_keeps_gap_without_crashing(client, monkeypatch):
    config.update({"search_provider": "local"})
    p = store.new_project("视频预算", media_policy="video_preferred", min_video_ratio=.8)
    shots = [planner.new_shot(f"普通街景{i}") for i in range(5)]
    for s in shots:
        s["duration"] = 5
    previous_review = {"verdict": "mismatch", "selection_blocked": True, "reason": "旧视频不匹配"}
    shots[1].update(selected_asset="old-video", verification=previous_review)
    image = {"id": "photo", "kind": "image", "title": "普通街景", "path": "photo.jpg", "width": 1920, "height": 1080}
    store.mutate(p["id"], lambda p: p.update(shots=shots, assets=[image]))
    p = store.get_project(p["id"])
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{"label": "街景", "indices": list(range(5)), "queries": ["街景"]}], {}))
    monkeypatch.setattr(planner, "rank_candidates", lambda p, s, rows, check: [{"candidate_id": r["id"]} for r in rows])
    def review(p, rows, check):
        verdict = {"selection_blocked": False, "verdict": "match", "needs_source_check": False, "scene_alignment": "match", "reason": "街景匹配", "identity_evidence": {"reason": "通用画面"}}
        return [r | {"verification": verdict, "preview": {}} for r in rows], [], {}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    result = pooled_match.match(p, shots, lambda: None, lambda *a: None)
    assert result["selected"] == 1 and result["composition"]["image_seconds"] == 5
    final = store.get_project(p["id"])
    assert len([s for s in final["shots"] if s.get("selected_asset") == "photo"]) == 1
    assert all("图片预算" in s["search_note"] for s in final["shots"] if not s.get("selected_asset"))
    assert final["shots"][1]["verification"] == previous_review
