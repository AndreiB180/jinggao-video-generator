import io
import json
import math
import wave
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jinggao import config, media, paths, planner, sources, store, worker
from jinggao.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "中文 项目数据")
    with TestClient(create_app()) as c:
        yield c


def create(client, text="第一句是关于钢轨的介绍。第二句解释热胀冷缩。"):
    r = client.post("/api/projects", json={"title": "测试：中文与空格", "script": text})
    assert r.status_code == 201, r.text
    return r.json()


def plan(client, p):
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "plan", "mode": "rules"})
    assert r.status_code == 202
    assert worker.run_once()
    assert store.get_job(r.json()["id"])["status"] == "completed"
    return store.get_project(p["id"])


def audio_bytes(duration=1.2):
    output = io.BytesIO()
    import struct
    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"".join(struct.pack("<h", round(2000*math.sin(2*math.pi*330*i/24000)))
                                for i in range(round(duration*24000))))
    return output.getvalue()


def test_caption_breaks_preserve_text_and_prefer_sentence_boundaries():
    text = '如果你常坐高铁，可能会发现，车轮经过轨道接头的“哐当”声少了。轨温32.5度。'
    rows = media.subtitle_parts(text)
    assert ''.join(rows) == text
    assert all(len(row) <= 18 and not row.endswith('“') for row in rows)
    assert '轨温32.5度。' in rows


def test_script_preserved_and_queue_recovery(client):
    p = create(client, "2026年，轨温为32.5度。钢轨仍会热胀冷缩！\n工程师检查轨道。")
    j = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "plan", "mode": "rules"}).json()
    assert client.patch(f"/api/projects/{p['id']}", json={"title": "不能改"}).status_code == 400
    assert client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "plan"}).status_code == 400
    claimed = store.claim()
    assert claimed["id"] == j["id"]
    store.recover()
    assert store.get_job(j["id"])["status"] == "interrupted"
    assert client.post(f"/api/jobs/{j['id']}/retry").status_code == 202
    worker.run_once()
    p2 = store.get_project(p["id"])
    assert planner.normalized("".join(s["narration"] for s in p2["shots"])) == planner.normalized(p["script"])


def test_model_cannot_silently_rewrite(client, monkeypatch):
    p = create(client)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"shots": [{"narration": "编造的新内容"}]}, {}))
    with pytest.raises(ValueError, match="原文片段编号"):
        planner.plan(p, "model", False)


def test_secrets_origin_and_path_validation(client):
    assert client.patch("/api/settings", json={"model_api_key": "test-secret"}).status_code == 200
    body = client.get("/api/settings").json()
    assert body["model_api_key_set"] is True and "test-secret" not in json.dumps(body)
    client.patch("/api/settings", json={"model_name": "deepseek-flash"})
    assert config.read()["model_api_key"] == "test-secret"
    assert client.post("/api/projects", headers={"Origin": "https://evil.example"}, json={}).status_code == 403
    p = create(client)
    with pytest.raises(ValueError):
        paths.file_path(p["id"], "../../settings.json")
    with pytest.raises(ValueError):
        sources.public_url("http://127.0.0.1/private")


def test_upload_lock_and_voice_invalidation(client):
    p = plan(client, create(client))
    sid = p["shots"][0]["id"]
    r = client.post(f"/api/projects/{p['id']}/upload", data={"role": "voice", "shot_id": sid},
                    files={"file": ("中文 音频.wav", audio_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["shots"][0]["audio"]
    r = client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"narration": "新的口播需要重做音频。"})
    assert r.json()["shots"][0]["audio"] is None
    client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"locked": True})
    assert client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"query": "change"}).status_code == 400


def test_generate_voice_preserves_locked_shots(client, monkeypatch):
    p = client.post("/api/demo").json()
    client.patch(f"/api/projects/{p['id']}/shots/{p['shots'][0]['id']}", json={"locked": True})
    generated = []

    def synthesize(text, target, check):
        generated.append(text)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(audio_bytes())

    monkeypatch.setattr(media, "synthesize", synthesize)
    job = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "voice", "replace": True}).json()
    worker.run_once()
    result = store.get_project(p["id"])
    assert store.get_job(job["id"])["status"] == "completed"
    assert result["shots"][0]["audio"] is None
    assert generated == [s["narration"] for s in p["shots"][1:]]


def test_changed_requirements_or_crop_invalidate_old_approval(client):
    p = client.post("/api/demo").json()
    assert all(s["approved"] for s in p["shots"])
    changed = client.patch(f"/api/projects/{p['id']}", json={"requirements": "只用指定地区资料"}).json()
    assert all(not s["approved"] and s["verification"] is None for s in changed["shots"])
    sid = p["shots"][0]["id"]
    client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"approved": True})
    changed = client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"fit": "contain"}).json()
    assert changed["shots"][0]["approved"] is False


def test_real_mp4_export_with_audio_subtitles_and_sources(client, monkeypatch):
    from jinggao import captions
    # This renderer fixture contains sine waves, not spoken Mandarin. Supply
    # timestamps explicitly; real ASR alignment has separate regression tests.
    monkeypatch.setattr(captions, "ensure", lambda project, shot, *a, **k:
                        {"cues": [{"start": .1, "end": 1.1, "text": shot["narration"]}]})
    p = plan(client, create(client, "钢轨会热胀冷缩。工程师检查轨道。"))
    client.patch(f"/api/projects/{p['id']}", json={"media_policy": "mixed"})
    output = io.BytesIO()
    Image.new("RGB", (480, 270), "#355866").save(output, "PNG")
    r = client.post(f"/api/projects/{p['id']}/upload", files={"file": ("轨道示意.png", output.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    aid = r.json()["assets"][0]["id"]
    for shot in p["shots"]:
        sid = shot["id"]
        assert client.post(f"/api/projects/{p['id']}/shots/{sid}/asset/{aid}").status_code == 200
        assert client.patch(f"/api/projects/{p['id']}/shots/{sid}",
                            json={"approved": True, "overlay_label": "通用铁路工程资料图"}).status_code == 200
        assert client.post(f"/api/projects/{p['id']}/upload", data={"role": "voice", "shot_id": sid},
                           files={"file": ("voice.wav", audio_bytes(), "audio/wav")}).status_code == 200
        # Renderer fixtures use explicitly locked manual selections, without
        # pretending the solid-color image passed automatic pixel review.
        assert client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"approved": True, "locked": True}).status_code == 200
    config.update({"encoder": "cpu"})
    job = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "render", "resolution": "720p"}).json()
    worker.run_once()
    result = client.get(f"/api/jobs/{job['id']}").json()
    assert result["status"] == "completed", result["error"]
    export = store.get_project(p["id"])["exports"][0]
    video = paths.file_path(p["id"], export["path"])
    streams = media.probe(video)["streams"]
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    assert next(s for s in streams if s["codec_type"] == "video")["width"] == 1280
    with zipfile.ZipFile(paths.file_path(p["id"], export["bundle"])) as z:
        assert set(z.namelist()) == {"video.mp4", "sources.json", "subtitles.srt", "project.json"}
        assert "钢轨" in z.read("subtitles.srt").decode("utf-8-sig")
        assert json.loads(z.read("sources.json"))[0]["overlay_label"] == "通用铁路工程资料图"
    frame = video.with_suffix(".png")
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-frames:v", "1", str(frame)])
    with Image.open(frame) as im:
        # A visible label must be burned into the upper right, not only recorded in JSON.
        pixels = im.convert("RGB").crop((640, 0, 1280, 110)).getdata()
        assert sum(min(pixel) > 200 for pixel in pixels) > 100
    served = client.get(f"/api/projects/{p['id']}/files/{export['path']}", headers={"Range": "bytes=0-127"})
    assert served.status_code == 206 and len(served.content) == 128


def test_full_audio_alignment_and_render_guard(client):
    p = plan(client, create(client, "第一段口播讲工程。\n第二段口播讲维护。"))
    assert len(p["shots"]) == 2
    r = client.post(f"/api/projects/{p['id']}/upload", data={"role": "narration"},
                    files={"file": ("narration.wav", audio_bytes(4.1), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["full_audio"]["alignment"] == "pending"
    assert [s["duration"] for s in r.json()["shots"]] == [s["duration"] for s in p["shots"]]
    job = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "render"}).json()
    worker.run_once()
    assert store.get_job(job["id"])["status"] == "failed"
    assert "选择素材" in store.get_job(job["id"])["error"]


def test_auto_match_rejects_mismatch_and_leaves_review_pending(client, monkeypatch):
    p = client.post("/api/demo").json()
    shot = p["shots"][0]
    config.update({"model_api_key": "test-only", "search_provider": "local"})
    candidates = sources.local_candidates(p, shot["query"])
    monkeypatch.setattr(planner, "rank_candidates", lambda *args: [
        {"candidate_id": c["id"], "verdict": "match", "reason": "test"} for c in candidates[:2]])
    calls = []
    def verify(pid, s, a):
        calls.append(a["id"])
        return {"verdict": "mismatch" if len(calls) == 1 else "uncertain", "reason": "需确认来源"}
    monkeypatch.setattr(planner, "verify", verify)
    job = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "match", "shot_id": shot["id"], "replace": True}).json()
    worker.run_once()
    assert store.get_job(job["id"])["status"] == "completed"
    chosen = store.get_project(p["id"])["shots"][0]
    assert len(calls) == 2 and chosen["selected_asset"] == candidates[1]["asset_id"]
    assert chosen["approved"] is False and chosen["verification"]["verdict"] == "uncertain"


def test_visual_ranking_rejects_invented_candidates(client, monkeypatch):
    p = client.post("/api/demo").json()
    shot = p["shots"][0]
    seen = []
    def chat(system, content, **kwargs):
        seen.extend(content)
        return {"ranking": [{"candidate_id": "invented", "verdict": "match", "reason": "test"}]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="无效候选"):
        planner.rank_candidates(p, shot, sources.local_candidates(p, shot["query"]), lambda: None)
    assert any(c["type"] == "image_url" and c["image_url"]["url"].startswith("data:image/jpeg") for c in seen)


@pytest.mark.parametrize("conflicts,needs_source,expected", [
    (["实际为有砟轨道，与指定结构不符"], False, "mismatch"),
    ([], True, "uncertain"),
])
def test_verification_cannot_approve_its_own_conflicts(client, monkeypatch, conflicts, needs_source, expected):
    p = client.post("/api/demo").json()
    shot = p["shots"][0]
    asset = next(a for a in p["assets"] if a["id"] == shot["selected_asset"])
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: (
        {"verdict": "match", "reason": "模型矛盾结论", "conflicts": conflicts, "needs_source_check": needs_source}, {}))
    result = planner.verify(p["id"], shot, asset)
    assert result["verdict"] == expected


def test_cancelling_encoder_terminates_process(client):
    import sys
    def cancelled():
        raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        media.run([sys.executable, "-c", "import time;time.sleep(60)"], check=cancelled)


def test_single_instance_lock_releases(client):
    from jinggao import instance
    lock = instance.acquire()
    assert lock is not None
    assert instance.acquire() is None
    lock.close()
    reopened = instance.acquire()
    assert reopened is not None
    reopened.close()


def test_portrait_video_clip_freeze_and_music_mix(client, monkeypatch):
    from jinggao import captions
    monkeypatch.setattr(captions, "ensure", lambda project, shot, *a, **k:
                        {"cues": [{"start": .1, "end": 1.2, "text": shot["narration"]}]})
    p = plan(client, create(client, "检查轨道。"))
    client.patch(f"/api/projects/{p['id']}", json={"media_policy": "mixed"})
    source = paths.project_dir(p["id"]) / "short.mp4"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25",
               "-t", "0.8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
    r = client.post(f"/api/projects/{p['id']}/upload", files={"file": ("视频.mp4", source.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    aid, sid = r.json()["assets"][0]["id"], p["shots"][0]["id"]
    client.patch(f"/api/projects/{p['id']}", json={"aspect": "9:16"})
    client.post(f"/api/projects/{p['id']}/shots/{sid}/asset/{aid}")
    client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"approved": True, "clip_start": .4, "fit": "contain"})
    for role in ("voice", "music"):
        r = client.post(f"/api/projects/{p['id']}/upload", data={"role": role, "shot_id": sid},
                        files={"file": ("audio.wav", audio_bytes(1.5), "audio/wav")})
        assert r.status_code == 200, r.text
    assert client.patch(f"/api/projects/{p['id']}/shots/{sid}", json={"approved": True, "locked": True}).status_code == 200
    j = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "render", "resolution": "720p"}).json()
    worker.run_once()
    assert store.get_job(j["id"])["status"] == "completed", store.get_job(j["id"])["error"]
    p = store.get_project(p["id"])
    assert (p["exports"][0]["width"], p["exports"][0]["height"]) == (720, 1280)
    info = media.probe(paths.file_path(p["id"], p["exports"][0]["path"]))
    assert abs(float(info["format"]["duration"])-1.52) < .1
