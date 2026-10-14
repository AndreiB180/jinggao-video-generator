import io
import json
import wave

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops, ImageDraw, ImageStat

from jinggao import curation, media, paths, planner, store, worker
from jinggao.api import create_app


def test_composition_counts_real_video_duration_and_rejects_static_substitution():
    project = {"media_policy": "video_preferred", "min_video_ratio": .8, "min_video_height": 720,
               "shots": [{"duration": 10, "selected_asset": "v", "clip_start": 1, "locked": True},
                         {"duration": 2, "selected_asset": "im", "locked": True}],
               "assets": [{"id": "v", "kind": "video", "duration": 5, "width": 1280, "height": 720},
                          {"id": "im", "kind": "image"}]}
    report = curation.composition(project)
    assert report["video_seconds"] == 4 and report["frozen_seconds"] == 6
    assert report["image_seconds"] == 2 and report["short_video_shots"] == [1]
    with pytest.raises(ValueError, match="图片运镜不计作视频"):
        curation.validate_export(project)
    project["assets"][0]["duration"] = 15
    assert curation.validate_export(project)["meets_target"]
    with pytest.raises(ValueError, match="开启运镜"):
        curation.validate_export(project, image_motion=False)


def test_person_manual_evidence_cannot_survive_changed_clip_or_source(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with TestClient(create_app()) as client:
        p = client.post("/api/projects", json={"title": "人物采访"}).json()
        assert p["media_policy"] == "video_preferred" and p["min_video_height"] == 720
        shot = planner.new_shot("陈明介绍这次实验。", persons=[{"name": "陈明", "aliases": []}], evidence_mode="person")
        shot["selected_asset"] = "v"
        store.mutate(p["id"], lambda item: item.update(shots=[shot], assets=[{
            "id": "v", "kind": "video", "title": "采访", "duration": 20, "width": 1280, "height": 720}]))
        route = f"/api/projects/{p['id']}/shots/{shot['id']}"
        assert client.patch(route, json={"approved": True}).status_code == 400
        approved = client.patch(route, json={"approved": True, "source_review_url": "https://example.org/interview",
                "source_review_note": "片段字幕写有全名，发布者原文介绍同一实验和受访者。"})
        assert approved.status_code == 200
        assert curation.identity_ready(approved.json()["shots"][0])
        changed = client.patch(route, json={"clip_start": 5}).json()["shots"][0]
        assert not changed["approved"] and not curation.identity_ready(changed)
        client.patch(route, json={"approved": True, "source_review_url": "https://example.org/interview",
                                "source_review_note": "新的时间段仍有完整姓名字幕及实验介绍。"})
        changed = client.patch(f"/api/projects/{p['id']}/assets/v", json={"source_url": "https://example.org/other"}).json()
        assert not changed["shots"][0]["approved"] and not curation.identity_ready(changed["shots"][0])


def test_old_project_keeps_mixed_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("旧稿")
    del p["media_policy"]
    with store.connection() as con:
        con.execute("UPDATE projects SET body=? WHERE id=?", (json.dumps(p), p["id"]))
    assert store.get_project(p["id"])["media_policy"] == "mixed"


def test_used_video_intervals_do_not_repeat_and_must_fit():
    selected = planner.new_shot("已用过这段")
    selected.update(duration=10, clip_start=2, selected_asset="old")
    shot = planner.new_shot("寻找另一个时间段")
    shot["duration"] = 10
    project = {"shots": [selected, shot], "assets": [
        {"id": "old", "candidate_id": "remote", "source_clip_start": 10, "duration": 30}]}
    starts = worker.clip_starts(project, shot, {"id": "remote", "duration": 60})
    assert starts and all(t+10 <= 12 or t >= 22 for t in starts)
    assert worker.clip_starts(project, shot, {"id": "remote", "duration": 8}) == []


def wav_bytes(duration=6):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\0\0"*int(duration*8000))
    return out.getvalue()


@pytest.mark.parametrize("action", ["voice", "narration", "synthesize", "verify"])
def test_audio_or_new_review_cannot_reuse_old_person_approval(tmp_path, monkeypatch, action):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with TestClient(create_app()) as client:
        p = store.new_project("采访", "陈明介绍这次实验。")
        shot = planner.new_shot(p["script"], persons=[{"name": "陈明", "aliases": []}], evidence_mode="person")
        shot.update(duration=2, selected_asset="v", approved=True,
                    verification={"identity_evidence": {"status": "supported"}, "selection_blocked": False},
                    source_review_url="https://example.org/interview", source_review_note="原片中字幕有完整姓名与实验地点，已核对。")
        store.mutate(p["id"], lambda row: row.update(shots=[shot], assets=[{
            "id": "v", "kind": "video", "duration": 30, "width": 1280, "height": 720}]))
        if action in ("voice", "narration"):
            response = client.post(f"/api/projects/{p['id']}/upload", data={"role": action, "shot_id": shot["id"]},
                                   files={"file": ("配音.wav", wav_bytes(), "audio/wav")})
            assert response.status_code == 200, response.text
        else:
            def synthesize(text, path, check):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(wav_bytes())
            monkeypatch.setattr(media, "synthesize", synthesize)
            monkeypatch.setattr(planner, "verify", lambda *args, **kwargs: {
                "verdict": "mismatch", "identity_evidence": {"status": "conflict"}, "selection_blocked": True})
            # The real video verify returns a code-owned reviewed_selection binding;
            # this fake has no file, so install the reviewed window unchanged.
            from jinggao import frame_sampling
            monkeypatch.setattr(frame_sampling, "verified_selection",
                                lambda p, s, asset, result: (asset, s.get("clip_start", 0), s.get("match_preview")))
            response = client.post(f"/api/projects/{p['id']}/jobs", json={
                "kind": "voice" if action == "synthesize" else "verify", "shot_id": shot["id"]})
            assert response.status_code == 202
            worker.run_once()
            assert store.get_job(response.json()["id"])["status"] == "completed"
        updated = store.get_project(p["id"])["shots"][0]
        assert not updated["approved"] and not curation.identity_ready(updated)
        assert updated["source_review_note"] == updated["source_review_url"] == ""
        if action != "verify":
            assert updated["duration"] == 6 and updated["verification"] is None


def test_audio_duration_cannot_shorten_review_or_changed_script_export(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with TestClient(create_app()) as client:
        p = store.new_project("时长检查", "原文")
        shot = planner.new_shot("原文")
        shot.update(duration=15, audio={"duration": 15, "path": "audio.wav"}, approved=True)
        store.mutate(p["id"], lambda row: row.update(shots=[shot]))
        r = client.patch(f"/api/projects/{p['id']}/shots/{shot['id']}", json={"duration": 5})
        assert r.status_code == 400 and "不能只缩短审核区间" in r.json()["detail"]
        r = client.patch(f"/api/projects/{p['id']}", json={"script": "新的文稿"})
        assert r.status_code == 200 and not r.json()["shots"][0]["approved"]
        with pytest.raises(ValueError, match="原稿已修改"):
            curation.validate_export(r.json())


def test_contain_photograph_really_moves_and_preserves_all_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("运镜测试", media_policy="mixed")
    image = Image.new("RGB", (900, 900), "#2c6578")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 899, 899), outline="#ff0000", width=30)
    for x in range(100, 800, 100):
        draw.line((x, 40, x, 860), fill="white", width=10)
    path = paths.project_dir(p["id"]) / "image.jpg"
    image.save(path)
    asset = media.inspect_asset(p["id"], path, "测试图")
    shot = planner.new_shot("运镜检查")
    # This checks rendering of a hand-selected test image, not auto approval.
    shot.update(duration=2, fit="contain", selected_asset=asset["id"], approved=True, locked=True)
    p.update(shots=[shot], assets=[asset])
    result = media.render(p, store.uid(), {"resolution": "720p", "allow_silent": True, "image_motion": True},
                          lambda *_: None, lambda: None)
    video = paths.file_path(p["id"], result["path"])
    frames = []
    for i, at in enumerate((.12, 1.8)):
        out = tmp_path / f"image-motion-{i}.png"
        media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", str(at), "-i", str(video),
                   "-frames:v", "1", str(out)])
        frames.append(Image.open(out).convert("RGB"))
    assert sum(ImageStat.Stat(ImageChops.difference(*frames)).mean) > 4
    for frame in frames:
        # All four red edges remain visible inside the 16:9 canvas after zooming.
        red = [(x, y) for y in range(0, 650, 5) for x in range(1280)
               if frame.getpixel((x, y))[0] > 180 and frame.getpixel((x, y))[1] < 75]
        assert red and min(x for x, _ in red) > 200 and max(x for x, _ in red) < 1080
        assert min(y for _, y in red) >= 0
