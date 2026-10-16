"""Text burned into footage (contact details, subtitles, title cards, scrolling lines), the audio batch and the
adaptive second candidate."""
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from jinggao import config, express_clean, express_match, paths, planner, screen_text, store
from jinggao.api import create_app

REPO_MODELS = Path(__file__).resolve().parents[1] / "data" / "models" / "ocr"
needs_models = pytest.mark.skipif(not all((REPO_MODELS / n).exists() for n in screen_text.FILES),
                                  reason="文字识别模型未安装（scripts/setup_captions.py）")


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    return tmp_path


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(screen_text, "model_dir", lambda: REPO_MODELS)


def line(text, box=(0.3, 0.45, 0.7, 0.5), score=0.95):
    return {"text": text, "score": score, "box": list(box)}


@pytest.mark.parametrize("text, expected", [
    ("需要视频，私信或看签名", True), ("费用10米有需要再联系", True), ("2695274503", True), ("w15817463073", True),
    ("加V：abc123", True), ("vx同号", True), ("订房电话56621306", True),
    ("航哥素材库", False), ("bili_20132629499", False), ("古比雪天水电话bilibili", False), ("MSKU1234567", False),
    ("这件事引发了广泛关注", False), ("1955年10月8日", False), ("CMA CGM", False), ("牛肉面12元", False)])
def test_contact_and_selling_text(text, expected):
    assert screen_text.contact(line(text)) is expected


def test_seven_digits_count_only_at_the_edges_where_uploaders_put_their_ids():
    assert screen_text.contact(line("TC8888802", (0.02, 0.02, 0.2, 0.06)))
    assert not screen_text.contact(line("TC8888802", (0.3, 0.45, 0.6, 0.5)))


def test_classify_crops_captions_and_titles_and_flags_text_across_the_picture():
    frames = [[line("我回去试了一下，满脑壳油", (0.25, 0.86, 0.75, 0.92)),
               line("美食台", (0.8, 0.03, 0.95, 0.08)),
               line("兰州", (0.4, 0.4, 0.5, 0.45))]]
    result = screen_text.classify(frames)
    assert result["contact"] == [] and not result["dense"]
    assert sorted(m["kind"] for m in result["marks"]) == ["caption", "logo"]   # a two-character sign is scenery
    dense = screen_text.classify([[line("工业高度集中，加工工业发达", (0.2, 0.4, 0.8, 0.46))]])
    assert dense["dense"] and dense["marks"][0]["kind"] == "center"
    ad = screen_text.classify([[line("需要视频，私信或看签名", (0.1, 0.1, 0.5, 0.14))]])
    assert ad["contact"] == ["需要视频，私信或看签名"]


def test_a_line_moving_sideways_between_frames_is_scrolling_text():
    a = [line("关注我们获取更多精彩内容", (0.5, 0.5, 0.9, 0.55))]
    b = [line("关注我们获取更多精彩内容", (0.2, 0.505, 0.6, 0.555))]
    assert screen_text.classify([a, b])["moving"]
    assert not screen_text.classify([a, a])["moving"]


def _picture(lines, size=(960, 540)):
    image = Image.new("RGB", size, (40, 90, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 300, 960, 540), fill=(70, 110, 60))
    font = ImageFont.truetype(str(paths.font_path()), 30)
    for text, xy in lines:
        draw.text(xy, text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


@needs_models
def test_the_local_model_reads_contact_text_and_subtitles(models):
    image = _picture([("需要视频，私信或看签名", (300, 120)), ("这里是原视频的字幕", (330, 480))])
    lines = screen_text.read(image)
    texts = " ".join(l["text"] for l in lines)
    assert "私信" in texts and "原视频的字幕" in texts
    result = screen_text.classify([lines])
    assert result["contact"] and any(m["kind"] == "caption" for m in result["marks"])


@needs_models
def test_frames_of_a_real_clip_in_one_ffmpeg_run(models, tmp_path):
    still = tmp_path / "still.png"
    _picture([("加微信看完整版", (330, 250))]).save(still)
    clip = tmp_path / "clip.mp4"
    subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-loop", "1", "-i", str(still), "-t", "2",
                    "-pix_fmt", "yuv420p", "-r", "25", str(clip)], check=True)
    frames = screen_text.frames_of(clip, 2.0)
    assert [f.size for f in frames] == [(960, 540), (960, 540)]
    assert screen_text.classify([screen_text.read(f) for f in frames])["contact"]


def _matcher(shots):
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    return express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)


def test_contact_text_rejects_the_clip_and_the_whole_source(data, monkeypatch):
    shot = planner.new_shot("港口。") | {"duration": 3.0}
    m = _matcher([shot])
    source = {"provider": "bilibili", "id": "BV1", "duration": 60.0}
    monkeypatch.setattr(screen_text, "installed", lambda: True)
    monkeypatch.setattr(screen_text, "frames_of", lambda path, duration: [Image.new("RGB", (64, 36))])
    monkeypatch.setattr(screen_text, "read", lambda image: [line("加微信看原片", (0.1, 0.1, 0.4, 0.15))])
    with pytest.raises(ValueError, match="联系方式"):
        m.read_text("j1", m.shots[0], {"tile": {"source": source}}, Path("clip.mp4"))
    assert source["contact"] and m.text_found["contact"] == 1
    # Such a source is never allocated again.
    tiles = [{"source": source, "t": 20.0}, {"source": {"provider": "bilibili", "id": "BV2", "duration": 60.0}, "t": 20.0}]
    assigned = m.allocate({m.shots[0]["id"]: [(t, "pick") for t in tiles]}, {}, {}, m.shots)
    assert assigned[m.shots[0]["id"]]["tile"]["source"]["id"] == "BV2"


def test_captions_found_by_reading_are_cropped_away(data, monkeypatch):
    shot = planner.new_shot("港口。") | {"duration": 3.0}
    m = _matcher([shot])
    m.clean["j1"] = {"content": [0, 0, 1, 1], "overlays": [], "central": [], "crop": [0, 0, 1, 1], "zoom": 1.0,
                     "residual": [], "src_aspect": 16 / 9}
    monkeypatch.setattr(screen_text, "installed", lambda: True)
    monkeypatch.setattr(screen_text, "frames_of", lambda path, duration: [Image.new("RGB", (64, 36))])
    monkeypatch.setattr(screen_text, "read", lambda image: [line("这是原视频的字幕", (0.3, 0.88, 0.7, 0.94))])
    m.read_text("j1", m.shots[0], {"tile": {"source": {"provider": "bilibili", "id": "BV1"}}}, Path("clip.mp4"))
    crop = m.clean["j1"]["crop"]
    assert crop[1] + crop[3] <= 0.88 and m.clean["j1"]["residual"] == []
    assert not express_clean.dirty(m.clean["j1"]) and m.text_found["caption"] == 1


def test_stock_footage_is_not_read(data, monkeypatch):
    m = _matcher([planner.new_shot("港口。") | {"duration": 3.0}])
    monkeypatch.setattr(screen_text, "installed", lambda: True)
    monkeypatch.setattr(screen_text, "frames_of", lambda *a: (_ for _ in ()).throw(AssertionError("read stock")))
    m.read_text("j1", m.shots[0], {"tile": {"source": {"provider": "pixabay", "id": "1"}}}, Path("clip.mp4"))


def test_preview_screening_flags_sources_once_and_caches(data, monkeypatch):
    m = _matcher([planner.new_shot("港口。") | {"duration": 3.0}])
    calls = []
    monkeypatch.setattr(screen_text, "installed", lambda: True)
    def read(image):
        calls.append(1)
        return [line("这是原视频的字幕", (0.3, 0.88, 0.7, 0.94))]
    monkeypatch.setattr(screen_text, "read", read)
    images = [Image.new("RGB", (480, 270)) for _ in range(8)]
    row = {"provider": "bilibili", "id": "BVsub"}
    m.screen_source(row, images)
    assert row.get("captioned") and len(calls) == express_match.Matcher.SCREEN_FRAMES
    again = {"provider": "bilibili", "id": "BVsub"}
    m.screen_source(again, images)
    assert again.get("captioned") and len(calls) == express_match.Matcher.SCREEN_FRAMES    # from the cache
    tiny = {"provider": "bilibili", "id": "BVtiny"}
    m.screen_source(tiny, [Image.new("RGB", (160, 90))])
    assert not tiny.get("captioned")      # 160x90 previews are too small to read


def test_second_candidate_only_where_it_pays_off_when_several_videos_run(data):
    m = _matcher([planner.new_shot("港口。") | {"duration": 3.0}])
    clean = {"tile": {"source": {"provider": "bilibili", "id": "BV1"}}, "lane": "pick"}
    plain, person = {"id": "a"}, {"id": "b", "persons": [{"name": "钱学森"}]}
    assert m.wants_backup(plain, clean, videos=1)                 # one video alone: always (faster)
    assert not m.wants_backup(plain, clean, videos=3)             # confident pick from a clean source
    assert m.wants_backup(person, clean, videos=3)                # named people fail the check more often
    assert m.wants_backup(plain, clean | {"lane": "topic"}, videos=3)
    flagged = {"tile": {"source": {"provider": "bilibili", "id": "BV2", "captioned": True}}, "lane": "pick"}
    assert m.wants_backup(plain, flagged, videos=3)


def _wav(path: Path, seconds=4):
    subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    str(path)], check=True)
    return path


def test_audio_batch_makes_one_video_per_file_in_one_batch(data, tmp_path):
    with TestClient(create_app()) as client:
        assert client.patch("/api/settings", json={"model_api_key": "sk-test"}).status_code == 200
        first = _wav(tmp_path / "第一集 港口.wav")
        r = client.post("/api/batch/audio", data={"aspect": "9:16", "burn_subtitles": "0"},
                        files={"file": (first.name, first.read_bytes(), "audio/wav")})
        assert r.status_code == 202, r.text
        batch, pid = r.json()["batch"], r.json()["project_id"]
        assert r.json()["title"] == "第一集 港口"
        p = store.get_project(pid)
        assert p["source_audio"] and p["aspect"] == "9:16" and p["burn_subtitles"] is False and p["script"] == ""
        second = _wav(tmp_path / "第二集.mp3")
        r2 = client.post("/api/batch/audio", data={"batch": batch},
                         files={"file": (second.name, second.read_bytes(), "audio/mpeg")})
        assert r2.json()["batch"] == batch
        rows = [row for row in client.get("/api/queue").json() if row["batch"] == batch]
        assert len(rows) == 2 and {row["kind"] for row in rows} == {"produce"}
        # A file without sound leaves nothing behind.
        before = len(store.list_projects())
        bad = client.post("/api/batch/audio", data={"batch": batch},
                          files={"file": ("坏文件.mp3", b"not audio", "audio/mpeg")})
        assert bad.status_code == 400 and len(store.list_projects()) == before


def test_audio_batch_needs_the_model_key(data, tmp_path):
    with TestClient(create_app()) as client:
        f = _wav(tmp_path / "a.wav")
        r = client.post("/api/batch/audio", files={"file": (f.name, f.read_bytes(), "audio/wav")})
        assert r.status_code == 400 and "API Key" in r.json()["detail"]
        assert config.read()["model_api_key"] == ""


def test_short_shots_join_a_neighbour_after_alignment_until_none_flashes(data, monkeypatch):
    from jinggao import captions, curation, express_plan, media, worker
    shots = [planner.new_shot(t) for t in ("第一句很长很长的口播内容。", "很短。", "第三句同样很长的口播内容。")]
    p = store.new_project("闪", "x")
    store.mutate(p["id"], lambda q: q.update(shots=shots, full_audio={"path": "a.wav", "duration": 9.0},
                                              plan_info={"mode": "express", "groups": []}), internal=True)
    monkeypatch.setattr(curation, "express_project", lambda project: True)
    def prepare(project, check, progress):
        durations = {"第一句很长很长的口播内容。": 4.0, "很短。": 0.9 + 0.3, "第三句同样很长的口播内容。": 3.8}
        values = [sum(durations.get(part, 0) for part in durations if part in s["narration"]) for s in project["shots"]]
        return {"expected": None, "timing": {"durations": values}}
    monkeypatch.setattr(media, "prepare_narration_alignment", prepare)
    monkeypatch.setattr(media, "apply_narration_alignment", lambda project, proposal: project.update(aligned=proposal))
    result = worker.ensure_narration_alignment(p["id"])
    durations = result["aligned"]["timing"]["durations"]
    assert len(result["shots"]) == 2 and min(durations) >= express_plan.MIN_SECONDS
    assert "".join(s["narration"] for s in result["shots"]) == "".join(s["narration"] for s in shots)


def test_a_line_without_usable_footage_borrows_a_nearby_clip_instead_of_a_card(data, monkeypatch, tmp_path):
    from jinggao import express_sources as es
    shots = [planner.new_shot(f"第{i}句。") | {"duration": 3.0} for i in range(6)]
    m = _matcher(shots)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(es, "probe_duration", lambda path: 3.5)
    source = {"provider": "bilibili", "id": "BV1", "title": "港口"}
    final = {m.shots[i]["id"]: {"tile": {"source": source}, "lane": "pick", "job": f"j{i}"} for i in (0, 4)}
    files = {sid: clip for sid in final}
    assert m.borrow(m.shots[3], final, files) == m.shots[0]["id"]     # the next shot's clip would play twice in a row
    assert m.borrow(m.shots[2], final, files) in (m.shots[0]["id"], m.shots[4]["id"])
    person = m.shots[2] | {"persons": [{"name": "潘水苗", "aliases": []}]}
    assert m.borrow(person, final, files) is None        # never someone else's footage for a named person
    title = Image.open(m.title_card(m.shots[2]))
    assert len(set(title.convert("L").getdata())) < 60   # a plain gradient: no name written on it
