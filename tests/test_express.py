"""Offline checks for the express pipeline (no network, no paid calls)."""
import struct

import pytest

from jinggao import captions, config, curation, express_clean, express_match, express_plan, express_sources as es, paths, planner, store


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    return tmp_path


def test_spoken_numbers_for_alignment():
    assert captions._spoken(captions._normal("1955年10月8日"))[0] == "一九五五年十月八日"
    assert captions._spoken("享年98岁")[0] == "享年九十八岁"
    assert captions._spoken("105")[0] == "一百零五"
    spoken, owner = captions._spoken("约20人")
    assert spoken == "约二十人" and owner == [0, 1, 2, 3]


def test_alignment_maps_spoken_digits_back_to_source_characters():
    text = "他在1955年回国。"
    words = [{"text": ch, "start": i * .3, "end": i * .3 + .25} for i, ch in enumerate("他在一九五五年回国")]
    track = captions.align(text, words, 4)
    assert track["coverage"] == 1 and len(track["character_timing"]["spans"]) == len(captions._normal(text))
    shots = {"shots": [{"narration": "他在1955年"}, {"narration": "回国。"}], "full_audio": {"duration": 4}}
    timing = captions.shot_timing(shots, track | {"duration": 4})
    assert timing["boundaries"][0]["method"] == "token_edges"


def _box(kind, body):
    return struct.pack(">I4s", 8 + len(body), kind.encode()) + body


def test_dash_index_parses_sidx_references(monkeypatch):
    moov = _box("moov", b"\0" * 20)
    refs = [(1000, 16000 * 5), (2000, 16000 * 5), (1500, 16000 * 3)]
    body = bytes([0, 0, 0, 0]) + struct.pack(">III", 1, 16000, 0) + struct.pack(">I", 0)
    body += struct.pack(">HH", 0, len(refs)) + b"".join(struct.pack(">III", size, dur, 0) for size, dur in refs)
    head = _box("ftyp", b"isom") + moov + _box("sidx", body)
    monkeypatch.setattr(es, "fetch_range", lambda client, url, start, end, check=None: head[start:end + 1])
    index = es.dash_index(None, "u")
    assert index["init"] == head[:len(_box("ftyp", b"isom")) + len(moov)]
    first = len(head)
    assert [(r[0], r[1], r[2], r[3]) for r in index["refs"]] == [
        (first, 1000, 0.0, 5.0), (first + 1000, 2000, 5.0, 5.0), (first + 3000, 1500, 10.0, 3.0)]


def _matcher(data, shots):
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    return express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)


def test_allocation_never_overlaps_windows_in_one_source(data):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 8.0} for i in range(4)]
    m = _matcher(data, shots)
    source = {"provider": "bilibili", "id": "BV1", "duration": 40.0}
    tiles = [{"source": source, "t": t} for t in (6.0, 16.0, 26.0, 34.0)]
    options = {s["id"]: [(tile, "pick") for tile in tiles] for s in m.shots}
    assigned = m.allocate(options, {}, {}, m.shots)
    windows = sorted((c["start"], c["start"] + 8.0 + express_match.MARGIN) for c in assigned.values())
    assert len(windows) >= 3
    # Every window contains the frame the model actually judged.
    assert all(c["start"] <= c["tile"]["t"] <= c["start"] + 8.0 for c in assigned.values())
    assert all(b[0] >= a[1] + express_match.WINDOW_GAP - 1e-6 for a, b in zip(windows, windows[1:], strict=False))
    assert all(0 <= start and end <= 40.0 for start, end in windows)


def test_person_identity_requires_name_in_source_text():
    shot = {"persons": [{"name": "钱学森", "aliases": []}]}
    assert express_match.Matcher.names_in_source(shot, {"title": "钱学森讲课影像"})
    assert not express_match.Matcher.names_in_source(shot, {"title": "邮轮 航行", "tags": "ship"})


def test_express_identity_gate_accepts_only_supported_or_not_shown():
    shot = {"persons": [{"name": "钱学森", "aliases": []}], "evidence_mode": "person"}
    for status, ready in (("supported", True), ("not_shown", True), ("", False)):
        shot["verification"] = {"express": {"identity": status}}
        assert curation.identity_ready(shot) is ready


def test_express_selection_error_checks_asset_and_file(data):
    project = store.new_project("t", "x")
    shot = {"verification": {"express": {"lane": "pick"}}}
    assert curation.selection_error(project, shot, {}) == "所选素材已失效，请重新匹配"
    asset = {"kind": "video", "path": "assets/missing.mp4"}
    assert curation.selection_error(project, shot, asset) == ""
    assert curation.selection_error(project, shot, asset, verify_files=True) == "素材文件缺失，请重新匹配"


def test_plan_cross_checks_known_names_and_drops_invented_ones(data, monkeypatch):
    shots = [planner.new_shot(t) for t in ("钱学森回到了祖国。", "他的老师也很有名。", "后来钱学森开始讲课。")]
    reply = {"topic": {"zh": ["航天 发射 实拍"], "en": ["rocket launch"]},
             "groups": [{"id": "G1", "zh": ["历史影像"], "en": ["x"], "person": "钱学森"},
                        {"id": "G2", "zh": ["教室 实拍"], "en": ["classroom"], "person": ""}],
             "shots": [{"id": "S1", "g": "G1", "v": "归国", "p": ["钱学森"]},
                       {"id": "S2", "g": "G2", "v": "老师", "p": ["冯·卡门"]},
                       {"id": "S3", "g": "G2", "v": "讲课", "p": []}]}
    monkeypatch.setattr(planner, "chat", lambda *a, **k: (reply, {"usage": {}}))
    info = express_plan.annotate(shots, "".join(s["narration"] for s in shots))
    assert [p["name"] for p in shots[0]["persons"]] == ["钱学森"]
    assert shots[1]["persons"] == []                       # not in the narration: dropped
    assert [p["name"] for p in shots[2]["persons"]] == ["钱学森"]   # missed by the model: added locally
    assert info["groups"][0]["zh"] == ["钱学森 历史影像"] and info["groups"][0]["en"] == []


def test_pipeline_setting_is_validated(data):
    assert config.update({"pipeline": "classic"})["pipeline"] == "classic"
    with pytest.raises(ValueError):
        config.update({"pipeline": "fast"})


def test_named_person_sources_reject_dramatisations():
    group = {"person": "钱学森", "zh": ["钱学森 历史影像"], "en": [], "shots": []}
    m = express_match.Matcher.__new__(express_match.Matcher)
    def row(title, tags="历史", typename="人文历史"):
        return {"provider": "bilibili", "title": title, "tags": tags, "typename": typename, "duration": 120, "play": 10}
    assert m.score(row("55年钱学森回国，聂元帅对王诤说：你让学森先说"), group, {}) is None
    assert m.score(row("钱学森与好友这番对话", "电影,陈坤", "小剧场"), group, {}) is None
    assert m.score(row("冯·卡门讲课影像"), group, {}) is None                  # name missing
    assert m.score(row("钱学森在中科院力学研究所的上课影像"), group, {}) > 0


def test_safe_crop_avoids_corner_logo_and_subtitle_within_zoom_limit():
    from jinggao import express_clean as ec
    boxes = [{"kind": "logo", "box": [0.7, 0.0, 0.99, 0.09]}, {"kind": "subtitle", "box": [0.2, 0.86, 0.8, 0.95]}]
    rect, zoom, residual = ec.safe_crop(boxes, (0, 0, 1, 1), 16 / 9, 16 / 9)
    x, y, w, h = rect
    assert residual == [] and 1 < zoom <= ec.MAX_ZOOM
    assert y >= 0.09 and y + h <= 0.86 and abs(w / h - 1) < 0.02   # 16:9 source -> same-shaped rectangle
    prefix = ec.video_filter({"crop": rect, "residual": [{"kind": "subtitle", "box": [0.2, 0.8, 0.8, 0.9]}]})
    assert prefix.startswith("crop=") and "boxblur" not in prefix   # subtitles are never smeared


def test_motion_flags_slideshow_stills():
    from jinggao import express_clean as ec
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    photo = rng.uniform(0, 255, (90, 160)).astype("float32")
    other = rng.uniform(0, 255, (90, 160)).astype("float32")
    assert ec.motion([photo] * 5 + [other] * 5)["still"]
    moving = [rng.uniform(0, 255, (90, 160)).astype("float32") for _ in range(10)]
    assert not ec.motion(moving)["still"]


def test_platform_watermark_found_on_moving_footage(tmp_path):
    from PIL import Image as PILImage
    from jinggao import express_clean as ec
    np = pytest.importorskip("numpy")
    logo = ec._logo()
    big = PILImage.fromarray(logo.astype("uint8")).resize((96, 46))
    rng = np.random.default_rng(3)
    for i in range(12):
        base = (rng.uniform(40, 120, (36, 64, 3))).astype("uint8")
        frame = PILImage.fromarray(base).resize((1280, 720), PILImage.BILINEAR)
        mask = big.point(lambda v: 255 if v > 200 else 0)
        frame.paste(PILImage.new("RGB", big.size, (235, 235, 235)), (1162, 24), mask)
        frame.save(tmp_path / f"f{i:02}.png")
    clip = tmp_path / "clip.mp4"
    media_run = __import__("subprocess").run
    media_run([str(paths.binary("ffmpeg")), "-v", "error", "-y", "-framerate", "2", "-i", str(tmp_path / "f%02d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    found = ec.platform_watermark(clip, 6.0, (1280, 720))
    assert found and found[0]["box"][1] == 0 and 0.06 < found[0]["box"][3] < 0.13 and found[0]["box"][2] > 0.95
    plain = tmp_path / "plain.mp4"
    media_run([str(paths.binary("ffmpeg")), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=2:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)],
              check=True)
    assert ec.platform_watermark(plain, 6.0, (1280, 720)) == []


def test_subtitles_need_steady_strokes_not_moving_bright_texture():
    from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont
    from jinggao import express_clean as ec
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    font = PILFont.truetype(str(paths.font_path()), 22)
    def frame(text, shift):
        base = PILImage.fromarray(rng.uniform(30, 90, (27, 48)).astype("uint8")).resize((480, 270))
        draw = PILDraw.Draw(base)
        if text:
            draw.text((240, 240), text, font=font, fill=255, anchor="mm", stroke_width=2, stroke_fill=0)
        else:   # bright blobs drifting along the bottom, like foam or containers under a moving camera
            for k in range(14):
                x = (k * 37 + shift * 23) % 440
                draw.rectangle((x, 228 + (k % 3) * 6, x + 9, 236 + (k % 3) * 6), fill=250)
        return np.asarray(base, dtype="float32")
    lines = ["他就从第一块黑板的第一行写起"] * 5 + ["从最基础的知识教起"] * 5
    with_subs = [frame(t, i) for i, t in enumerate(lines)]
    found = ec.subtitles(with_subs, (0, 0, 1, 1))
    assert found and found[0]["box"][1] > 0.8
    assert ec.subtitles([frame("", i) for i in range(10)], (0, 0, 1, 1)) == []


def test_mid_picture_name_split_into_fragments_is_one_central_mark():
    from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont
    from jinggao import express_clean as ec
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    font = PILFont.truetype(str(paths.font_path()), 15)
    pictures = []
    for _ in range(45):
        base = PILImage.fromarray(rng.uniform(20, 200, (9, 16, 3)).astype("uint8")).resize((480, 270))
        draw = PILDraw.Draw(base)
        # One short name whose characters edge detection splits into two clusters (like "月色梦光影").
        draw.text((312, 205), "月色", font=font, fill=(245, 245, 245))
        draw.text((360, 205), "梦光", font=font, fill=(245, 245, 245))
        pictures.append(base)
    marks = ec.central_marks(ec.profile(pictures, (480, 270)))
    assert len(marks) == 1
    x0, y0, x1, y1 = marks[0]["box"]
    assert x0 <= 0.66 and x1 >= 0.8 and 0.7 < y0 < y1 < 0.9
    # Fragments near a frame edge are left alone: the crop removes those whole.
    assert ec._join_line([{"kind": "logo", "box": [0.0, 0.0, 0.05, 0.05]},
                          {"kind": "logo", "box": [0.07, 0.0, 0.1, 0.05]}]) == [
        {"kind": "logo", "box": [0.0, 0.0, 0.05, 0.05]}, {"kind": "logo", "box": [0.07, 0.0, 0.1, 0.05]}]


def test_person_without_any_footage_takes_background_fills(data):
    shots = [planner.new_shot("一位叫马保子的厨师。") | {"duration": 6.0, "persons": [{"name": "马保子", "aliases": []}]},
             planner.new_shot("面馆里坐满了人。") | {"duration": 6.0}]
    m = _matcher(data, shots)
    person, plain = m.shots
    m.unfilmed = {"马保子"}
    assert m.unfilmed_shot(person) and not m.unfilmed_shot(plain)
    source = {"provider": "bilibili", "id": "BV1", "duration": 60.0}
    topic = {"source": source, "t": 10.0}
    other = {"source": {"provider": "bilibili", "id": "BV2", "duration": 60.0}, "t": 10.0}
    options = {person["id"]: [(other, "near"), (topic, "topic")]}
    assigned = m.allocate(options, {}, {}, [person])
    assert assigned[person["id"]]["lane"] == "topic"   # no +5 person penalty when nobody has footage of him
    m.unfilmed = set()
    assigned = m.allocate(options, {}, {}, [person])
    assert assigned[person["id"]]["lane"] == "near"


def test_visual_check_marks_every_reported_position(data):
    m = _matcher(data, [planner.new_shot("港口。") | {"duration": 6.0}])
    base = {"content": [0, 0, 1, 1], "overlays": [], "central": [], "crop": [0, 0, 1, 1], "zoom": 1.0,
            "residual": [], "motion": {"still": False}, "src_aspect": 16 / 9}
    m.clean["a"] = dict(base)
    m.residual_mark("a", ["右上", "底部"])
    assert sorted(o["kind"] for o in m.clean["a"]["overlays"]) == ["logo", "subtitle"]
    x, y, w, h = m.clean["a"]["crop"]
    assert y + h <= 0.8 + 0.01   # the subtitle line is out of view
    # The corner too, unless that needs more zoom than allowed: then it is blurred and the clip flagged.
    residual = [r for r in m.clean["a"]["residual"] if r["kind"] == "logo"]
    assert ("wm" in m.soft.get("a", set())) == express_clean.dirty(m.clean["a"])
    assert residual or x + w <= 0.65 or y >= 0.15
    m.clean["b"] = dict(base)
    m.residual_mark("b", "中间")
    assert "wm" in m.soft["b"]   # cannot be cropped: prefer another clip


def test_portrait_stock_clips_are_skipped_in_landscape_projects(data):
    m = _matcher(data, [planner.new_shot("港口。") | {"duration": 6.0}])
    assert m.oriented({"rendition": {"width": 1920, "height": 1080}})
    assert not m.oriented({"rendition": {"width": 720, "height": 1280}})
    assert m.oriented({"provider": "pixabay"})   # unknown size: keep


def test_slow_downloads_carry_over_to_the_next_round(data, tmp_path):
    import threading
    import time as _time
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 4.0} for i in range(2)]
    m = _matcher(data, shots)
    gate = threading.Event()
    def fetch(shot, choice, streams, check=None):
        if choice["slow"]:
            while not gate.is_set():
                check()
                _time.sleep(0.01)
        path = tmp_path / f"{shot['id']}.mp4"
        path.write_bytes(b"x")
        return path
    m.fetch = fetch
    m.analyse = lambda shot, choice, path: {}
    source = {"provider": "pixabay", "id": "1"}
    jobs = [("a#0a", m.shots[0]["id"], {"tile": {"source": source}, "slow": False}),
            ("b#0a", m.shots[1]["id"], {"tile": {"source": source}, "slow": True})]
    got, failed, bad, running = m.download(jobs, {}, deadline=_time.monotonic() + 0.5)
    assert set(got) == {"a#0a"} and running == {"b#0a"} and not failed
    gate.set()
    got, failed, bad, running = m.download([], {}, deadline=_time.monotonic() + 5)
    assert set(got) == {"b#0a"} and not running   # the earlier download finished instead of restarting
    gate.clear()
    m.download([("c#1a", m.shots[0]["id"], {"tile": {"source": source}, "slow": True})], {},
               deadline=_time.monotonic() + 0.2)
    assert m.stop_downloads() == {"c#1a": "下载超时，已放弃该片段"} and not m.inflight
