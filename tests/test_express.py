"""Offline checks for the express pipeline (no network, no paid calls)."""
import hashlib
import struct
from pathlib import Path

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
    source = {"provider": "bilibili", "id": "BV1", "duration": 120.0}
    tiles = [{"source": source, "t": t} for t in (6.0, 16.0, 26.0, 34.0, 70.0, 100.0)]
    options = {s["id"]: [(tile, "pick") for tile in tiles] for s in m.shots}
    assigned = m.allocate(options, {}, {}, m.shots)
    windows = sorted((c["start"], c["start"] + 8.0 + express_match.MARGIN) for c in assigned.values())
    assert len(windows) == express_match.SOURCE_MAX   # one source fills at most this many shots of a film
    # Every window contains the frame the model actually judged.
    assert all(c["start"] <= c["tile"]["t"] <= c["start"] + 8.0 for c in assigned.values())
    # Another moment of one video is another scene only well apart, and never in the neighbouring shot.
    assert all(b[0] - a[0] >= express_match.SCENE_GAP - 1e-6 for a, b in zip(windows, windows[1:], strict=False))
    assert all(0 <= start and end <= 120.0 for start, end in windows)
    places = sorted(m.position[sid] for sid in assigned)
    assert all(b - a >= 2 for a, b in zip(places, places[1:], strict=False))


def test_allocation_spreads_shots_over_sources_and_skips_reuploaded_pictures(data):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 4.0} for i in range(12)]
    m = _matcher(data, shots)
    def look(name):    # a 64-bit picture hash, unrelated for different names
        return int(hashlib.md5(name.encode()).hexdigest()[:16], 16)
    big = {"provider": "bilibili", "id": "BV_big", "duration": 900.0}
    others = [{"provider": "bilibili", "id": f"BV_{n}", "duration": 120.0} for n in range(6)]
    # The big source is every shot's best-ranked option; the others are a little further down.
    options = {s["id"]: [({"source": big, "t": 20.0 + 60 * i, "hash": look(f"big{i}")}, "pick")]
               + [({"source": o, "t": t, "hash": look(f"{n}-{t}")}, "pick")
                  for n, o in enumerate(others) for t in (20.0, 60.0, 100.0)]
               for i, s in enumerate(m.shots)}
    assigned = m.allocate(options, {}, {}, m.shots)
    used = [c["tile"]["source"]["id"] for _, c in sorted(assigned.items(), key=lambda kv: m.position[kv[0]])]
    assert used.count("BV_big") <= 4                       # capped near its share instead of filling everything
    assert all(a != b or a != "BV_big" for a, b in zip(used, used[1:]))
    # A re-upload (same picture, another id) loses to fresh footage when there is any.
    twin = {"provider": "bilibili", "id": "BV_twin", "duration": 120.0}
    m2 = _matcher(data, shots[:2])
    first, second = m2.shots
    m2.looks[m2.position[first["id"]]] = look("shown")
    options = {second["id"]: [({"source": twin, "t": 30.0, "hash": look("shown") ^ 0b101}, "pick"),
                              ({"source": others[0], "t": 30.0, "hash": look("fresh")}, "pick")]}
    choice = m2.allocate(options, {}, {}, [second])[second["id"]]
    assert choice["tile"]["source"]["id"] == "BV_0"


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


def test_story_nicknames_are_ordinary_people_not_names_to_search(data, monkeypatch):
    shots = [planner.new_shot(t) for t in ("张女士和老周结婚十六年。", "老周天天抱着手机聊天。", "两人去老挝旅游。")]
    reply = {"domain": "婚姻家庭", "topic": {"zh": ["夫妻 生活"], "en": []},
             "groups": [{"id": "G1", "zh": ["张女士 老周 夫妻 吵架 电视剧片段"], "en": [], "person": "张女士"},
                        {"id": "G2", "zh": ["老周 丈夫 玩手机 短剧"], "en": [], "person": "老周"},
                        {"id": "G3", "zh": ["老挝 旅游 街头"], "en": [], "person": ""}],
             "shots": [{"id": "S1", "g": "G1", "v": "夫妻", "p": ["张女士", "老周"]},
                       {"id": "S2", "g": "G2", "v": "玩手机", "p": ["老周"]},
                       {"id": "S3", "g": "G3", "v": "旅游", "p": []}]}
    monkeypatch.setattr(planner, "chat", lambda *a, **k: (reply, {"usage": {}}))
    info = express_plan.annotate(shots, "".join(s["narration"] for s in shots))
    assert all(shot["persons"] == [] and shot["evidence_mode"] == "generic" for shot in shots)
    zh = {g["id"]: g["zh"] for g in info["groups"]}
    assert zh["G1"] == ["婚姻家庭 夫妻 吵架 电视剧片段"] and zh["G2"] == ["婚姻家庭 丈夫 玩手机 短剧"]
    assert "老挝" in zh["G3"][0]                                           # a place, kept
    assert shots[1]["search_queries"] == zh["G2"] and shots[1]["express"]["zh"] == zh["G2"]


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


def test_tv_broadcast_sources_crop_away_the_top_left_channel_logo(tmp_path):
    from jinggao import express_clean as ec
    pytest.importorskip("numpy")
    clip = tmp_path / "doc.mp4"
    __import__("subprocess").run([str(paths.binary("ffmpeg")), "-v", "error", "-y", "-f", "lavfi", "-i",
                                  "testsrc2=size=1280x720:rate=5:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                  str(clip)], check=True)
    plain = ec.analyse(clip, 3.0, (1280, 720), 16 / 9)
    broadcast = ec.analyse(clip, 3.0, (1280, 720), 16 / 9, channel_logo=True)
    assert not any(o["kind"] == "logo" and o["box"][0] < .05 and o["box"][1] < .05 for o in plain["overlays"])
    assert any(o["kind"] == "logo" and o["box"][:2] == [0.0, 0.0] for o in broadcast["overlays"])
    x, y, w, h = broadcast["crop"]
    assert (x >= ec.CHANNEL_LOGO[2] - .01 or y >= ec.CHANNEL_LOGO[3] - .01) and broadcast["zoom"] <= ec.MAX_ZOOM
    assert express_match.BROADCAST.search("【1080P】四集纪录片《追寻宋金时代的别样生活》【CCTV9-HD】")


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


def test_stock_needs_two_words_of_a_longer_query(data):
    m = _matcher(data, [planner.new_shot("红海。") | {"duration": 6.0}])
    yacht = {"provider": "pixabay", "tags": "monaco, wealth, yachts, sea, harbour"}
    warship = {"provider": "pixabay", "tags": "warship, navy, sea, military"}
    assert not m.stock_fits(yacht, "warship sailing sea")     # one word in common: off-topic
    assert m.stock_fits(warship, "warship sailing sea")
    assert m.stock_fits({"provider": "pixabay", "tags": "office, desk"}, "office")   # one-word query: one word


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


def test_shots_hold_one_subtitle_line_each():
    from jinggao import subtitle_style
    script = "第二局只拿到2分。\n这场胜利不能掩盖长时间脱离高强度赛事留下的隐患。\n比分定格在7比11、2比11、11比6、10比12。"
    shots = express_plan.line_shots(script, 20)
    texts = [s["narration"] for s in shots]
    assert "".join(texts) == script.replace("\n", "")
    assert all(subtitle_style.width(t) <= 20 for t in texts)
    # A sentence of about two seconds or less does not flash past alone: it joins the next piece of its line.
    assert texts[:2] == ["第二局只拿到2分。这场胜利不能掩盖长时间", "脱离高强度赛事留下的隐患。"]
    assert not any(t in "。，、" for t in texts)   # no shot is a lone punctuation mark
    assert subtitle_style.line_chars_for("16:9", {"subtitle_size": "84"}) == 20
    assert subtitle_style.line_chars_for("9:16", {"subtitle_size": "84"}) < 20


def test_long_shots_take_neighbouring_lines_of_a_paragraph():
    from jinggao import subtitle_style
    script = ("中东局势突然升级，多国紧急撤侨。航母编队已经进入相关海域，周边国家进入戒备状态。"
              "分析人士认为，这次冲突短期内难以平息，油价应声上涨。\n\n"
              "另一边，外交斡旋仍在继续。")
    lines = express_plan.line_shots(script, 20)
    shots = express_plan.line_shots(script, 20, express_plan.SHOT_WIDTHS["long"])
    texts = [s["narration"] for s in shots]
    assert "".join(texts) == script.replace("\n", "")
    assert len(shots) < len(lines)
    target, most = express_plan.SHOT_WIDTHS["long"]
    assert all(subtitle_style.width(t) <= most for t in texts)
    assert texts[-1] == "另一边，外交斡旋仍在继续。"     # a new paragraph starts a new shot
    assert express_plan.shot_widths({"shot_length": "line"}) is None
    assert express_plan.shot_widths({}) == express_plan.SHOT_WIDTHS["long"]   # the default


def test_shot_lines_are_timed_by_their_own_characters():
    # Uneven speech: the first five characters of the second sentence take 2.0 s, the rest 0.5 s.
    # A proportional (cue-only) split would cut at 1.25 s; a wrong char_offset would read the first sentence.
    text = "第一句话。第二句稍长一点的话。"
    normal = captions._normal(text)
    first = len(captions._normal("第一句话。"))
    spans, at = [], 0.0
    for i in range(len(normal)):
        step = 0.2 if i < first else (0.4 if i < first + 5 else 0.1)
        spans.append([at, at + step * .9, True, True])
        at += step
    track = {"cues": [{"start": 0, "end": spans[-1][1], "text": text}], "character_timing": {"spans": spans}}
    shots = [{"narration": "第一句话。"}, {"narration": "第二句稍长一点的话。"}]
    second_start = spans[first][0]
    lines = captions.segments(track, second_start, 2.5, 1280, 720, narration=shots[1]["narration"],
                              char_offset=captions.char_offset(shots, 1), line_chars=6)
    assert [t for *_, t in lines] == ["第二句稍长", "一点的话。"]
    assert lines[0][0] == 0 and lines[-1][1] == 2.5 and abs(lines[0][1] - 2.0) < .05 and lines[0][1] == lines[1][0]
    wrong = captions.segments(track, second_start, 2.5, 1280, 720, narration=shots[1]["narration"],
                              char_offset=0, line_chars=6)
    assert wrong != lines


def test_subtitle_style_falls_back_for_missing_glyphs():
    from jinggao import subtitle_style
    if not (subtitle_style.font_file("huangyou") and subtitle_style.font_file("black")):
        pytest.skip("optional fonts not fetched")
    st = subtitle_style.style(1280, 720, {"subtitle_font": "huangyou", "subtitle_size": "84", "subtitle_color": "yellow"})
    assert st["size"] == 56 and st["primary"] == "&H0000E6FF"
    text = subtitle_style.ass_text("吴晙诚{\\b1}", st)
    assert "{\\fnNoto Sans SC Black}晙{\\fnZCOOL QingKe HuangYou}" in text and "{\\b1}" not in text


def test_alignment_survives_digits_read_as_one_syllable():
    text = "比分来到10比11。"
    spoken, _ = captions._spoken(captions._normal(text))
    words = [{"text": ch, "start": i * .3, "end": i * .3 + .25} for i, ch in enumerate(spoken)]
    track = captions.align(text, words, len(spoken) * .3)
    spans = track["character_timing"]["spans"]
    assert len(spans) == len(captions._normal(text))
    assert all(b[0] >= a[1] - .001 and b[0] <= b[1] for a, b in zip(spans, spans[1:]))


def test_black_screens_and_fades_are_flagged_dark_but_night_scenes_are_not():
    import numpy as np
    rng = np.random.default_rng(1)
    black = [np.zeros((27, 48), dtype=np.float32)] * 2 + [rng.random((27, 48), dtype=np.float32) * 255 for _ in range(4)]
    night = [np.full((27, 48), 35, dtype=np.float32) + rng.random((27, 48), dtype=np.float32) * 30 for _ in range(6)]
    assert express_clean.motion(black)["dark"] and not express_clean.motion(night)["dark"]


def _clip(tmp_path, name, graph, seconds=4):
    out = tmp_path / f"{name}.mp4"
    __import__("subprocess").run([str(paths.binary("ffmpeg")), "-v", "error", "-y", "-f", "lavfi", "-i",
                                  f"mandelbrot=size=1280x720:rate=10,trim=duration={seconds}", "-filter_complex", graph,
                                  "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_a_vertical_video_inside_a_landscape_canvas_is_shown_whole_without_the_canvas(tmp_path):
    from jinggao import express_clean as ec
    pytest.importorskip("numpy")
    dark = _clip(tmp_path, "dark", "color=c=0x303030:s=1280x720:d=4[bg];[0:v]crop=405:720[fg];"
                                   "[bg]drawbox=x=40:y=40:w=420:h=60:color=white:t=fill[bg2];[bg2][fg]overlay=700:0:shortest=1")
    # (A blurred-copy canvas is checked on real footage; this synthetic source is too smooth at its edges.)
    for clip, (x0, x1) in ((dark, (700 / 1280, 1105 / 1280)),):
        result = ec.analyse(clip, 4.0, (1280, 720), 16 / 9)
        assert result["narrow"] and abs(result["inner_picture"][0] - x0) < 0.02 and abs(result["inner_picture"][2] - x1) < 0.02
        assert result["crop"][2] < 0.4                         # the picture alone; the render fills the sides
    plain = ec.analyse(_clip(tmp_path, "plain", "[0:v]null"), 4.0, (1280, 720), 16 / 9)
    assert plain["inner_picture"] is None and not plain["layout_jump"] and not plain["narrow"]


def test_an_intro_card_giving_way_to_footage_is_a_jump(tmp_path):
    from jinggao import express_clean as ec
    pytest.importorskip("numpy")
    clip = _clip(tmp_path, "jump", "[0:v]split[a][b];"
                 "color=c=0x303030:s=1280x720:d=4[bg];[a]crop=405:720[fg];[bg][fg]overlay=700:0:shortest=1,trim=0:1.8,setpts=PTS-STARTPTS[p];"
                 "[b]trim=1.8:4,setpts=PTS-STARTPTS[q];[p][q]concat=n=2:v=1")
    assert ec.analyse(clip, 4.0, (1280, 720), 16 / 9)["layout_jump"]


def test_footage_fills_the_frame_unless_its_shape_is_far_off():
    from jinggao import curation
    video = {"kind": "video"}
    assert curation.preferred_fit({}, video | {"width": 1440, "height": 1080}, "16:9") == "cover"     # 4:3
    assert curation.preferred_fit({}, video | {"width": 1920, "height": 800}, "16:9") == "cover"      # 2.4:1
    assert curation.preferred_fit({}, video | {"width": 1080, "height": 1920}, "16:9") == "contain"   # vertical
    assert curation.preferred_fit({}, {"kind": "image", "width": 1440, "height": 1080}, "16:9") == "cover"    # 4:3 photo
    assert curation.preferred_fit({}, {"kind": "image", "width": 1080, "height": 1440}, "16:9") == "contain"  # tall
    assert express_match.intro_seconds(60) == 6 and express_match.intro_seconds(900) == 8


def test_the_opening_shot_chooses_its_footage_first(data):
    shots = [planner.new_shot(t) | {"duration": 5.0} for t in ("开头一句钩子。", "第二句讲细节。")]
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    best = {"source": {"provider": "bilibili", "id": "BV1best00000", "duration": 300.0}, "t": 100.0}
    filler = [{"source": {"provider": "bilibili", "id": f"BV1fill0000{n}", "duration": 300.0}, "t": 50.0} for n in range(3)]
    first, second = m.shots[0]["id"], m.shots[1]["id"]
    # The second line has fewer options and used to choose first, taking the one good clip.
    options = {first: [(best, "pick")] + [(t, "near") for t in filler], second: [(best, "pick")]}
    assigned = m.allocate(options, {}, {}, m.shots)
    assert assigned[first]["tile"]["source"]["id"] == "BV1best00000" and assigned[first]["lane"] == "pick"


def test_a_bilibili_refusal_waits_and_makes_the_video_again(data, monkeypatch):
    from jinggao import worker
    project = store.new_project("片子", "第一句。")
    store.enqueue(project["id"], "produce", {})
    def refused(job, stop=None):
        raise express_match.BiliUnavailable("B站的 6 次搜索全部没有结果")
    monkeypatch.setattr(worker, "process", refused)
    assert worker.run_once()
    jobs = store.jobs(project["id"])
    assert jobs[0]["status"] == "queued" and jobs[0]["payload"]["bili_retries"] == 1 and "分钟后自动重做" in jobs[0]["message"]
    assert jobs[1]["status"] == "interrupted"
    assert store.claim() is None                                   # it waits its ten minutes
    other = store.new_project("另一条", "第一句。")
    store.enqueue(other["id"], "produce", {})
    assert not worker.run_once()                                   # and no other video starts meanwhile
    assert store.jobs(other["id"])[0]["status"] == "queued"
    later = jobs[0]["payload"]["not_before"] + 1
    monkeypatch.setattr(worker.time, "time", lambda: later)
    monkeypatch.setattr(store.time, "time", lambda: later)
    assert store.claim()["id"] == jobs[0]["id"]
    assert later - worker.bili_refused_at() < worker.BILI_CALM     # one video at a time for the next hour


def test_searches_that_all_come_back_empty_stop_the_film(data):
    shots = [planner.new_shot(t) | {"duration": 5.0} for t in ("一。", "二。")]
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    class Bili:
        def search(self, q):
            return []
    m.bili, m.pixa = Bili(), None
    groups = [{"zh": ["甲 乙", "丙 丁"], "en": []}, {"zh": ["戊 己", "庚 辛"], "en": []}]
    with pytest.raises(express_match.BiliUnavailable):
        m.search(groups)


def test_refused_searches_go_on_with_the_local_library_or_by_setting(data, monkeypatch):
    from jinggao import config, footage_index
    shots = [planner.new_shot(t) | {"duration": 5.0} for t in ("一。", "二。")]
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    class Bili:
        asked = 0
        def search(self, q):
            Bili.asked += 1
            return []
    groups = [{"zh": ["甲 乙", "丙 丁"], "en": []}, {"zh": ["戊 己", "庚 辛"], "en": []}]
    monkeypatch.setattr(footage_index, "search", lambda q, limit=8: [{"source": {"id": q}}])
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    m.bili, m.pixa = Bili(), None
    results = m.search(groups)                     # the library has these: the film goes on
    assert results[("local", "甲 乙")] and Bili.asked == 4
    m.search(groups)                               # and the next films leave Bilibili search alone a while
    assert Bili.asked == 4 and any("本地素材库" in n for n in m.notes)
    monkeypatch.setattr(footage_index, "search", lambda q, limit=8: [])
    with pytest.raises(express_match.BiliUnavailable):
        m.search(groups)
    config.update({"bili_limit_mode": "continue"})
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    m.bili, m.pixa = Bili(), None
    m.search(groups)                               # set to go on regardless: no stop


def test_small_edge_logos_left_after_the_crop_are_painted_out_and_text_in_the_middle_is_not():
    import subprocess
    from jinggao import express_clean, paths
    corner = {"crop": [0.0, 0.1, 0.9, 0.9], "size": [1280, 720], "zoom": 1.11,
              "residual": [{"kind": "logo", "box": [0.82, 0.12, 0.9, 0.2]}]}
    vf = express_clean.video_filter(corner)
    assert vf.startswith("crop=") and ",delogo=x=" in vf and not express_clean.dirty(corner)
    middle = corner | {"residual": [{"kind": "center", "box": [0.3, 0.4, 0.7, 0.55]}]}
    assert "delogo" not in express_clean.video_filter(middle) and express_clean.dirty(middle)
    big = corner | {"residual": [{"kind": "logo", "box": [0.5, 0.1, 0.9, 0.4]}]}
    assert "delogo" not in express_clean.video_filter(big) and express_clean.dirty(big)
    old = {"crop": corner["crop"], "residual": corner["residual"]}       # saved before sizes were kept
    assert "delogo" not in express_clean.video_filter(old)
    done = subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720",
                           "-vf", vf, "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_vertical_and_half_black_pictures_are_recognised():
    from PIL import Image, ImageDraw
    from jinggao import express_clean
    import numpy as np
    wide = Image.new("RGB", (480, 270), (90, 120, 150))
    assert not express_match._boxed(wide)
    assert express_match._boxed(Image.new("RGB", (270, 480), (90, 120, 150)))       # a vertical video
    boxed = Image.new("RGB", (480, 270), (0, 0, 0))
    ImageDraw.Draw(boxed).rectangle((165, 0, 315, 270), fill=(140, 120, 100))         # vertical inside 16:9
    assert express_match._boxed(boxed)
    # The same frames in a clip: the content is found between bars wider than 30% each, and is narrow.
    frames = [np.array(boxed.convert("L"), dtype=np.uint8) for _ in range(4)]
    x0, y0, x1, y1 = express_clean.content_rect(frames)
    assert 0.3 < x0 < 0.36 and 0.64 < x1 < 0.7


def test_a_new_copy_of_the_app_keeps_the_api_key_login_and_assist_key(tmp_path, monkeypatch):
    from jinggao import paths
    old, new, keep = tmp_path / "Jinggao-0.4.9" / "Jinggao-Windows-x64", tmp_path / "Jinggao-0.5.0" / "Jinggao-Windows-x64", tmp_path / "AppData"
    (old / "data" / "assist").mkdir(parents=True)
    (old / "data" / "settings.json").write_text('{"model_api_key": "sk-old"}', "utf-8")
    (old / "data" / "bilibili-cookies.json").write_text('{"SESSDATA": "x"}', "utf-8")
    (old / "data" / "assist" / "assist.key").write_text("KEY", "utf-8")
    # First start of 0.5.0 unpacked next to 0.4.9: nothing kept yet, so it starts from the old copy.
    monkeypatch.setattr(paths, "ROOT", new)
    monkeypatch.setattr(paths, "DATA", new / "data")
    monkeypatch.setattr(paths, "KEEP", keep)
    (new / "data").mkdir(parents=True)
    paths.keep_sync()
    assert (new / "data" / "settings.json").read_text("utf-8") == '{"model_api_key": "sk-old"}'
    assert (new / "data" / "assist" / "assist.key").read_text("utf-8") == "KEY"
    paths.keep_sync()                                   # and keeps them in the user's folder
    assert (keep / "bilibili-cookies.json").exists()
    # A later copy anywhere starts from the kept files.
    other = tmp_path / "elsewhere" / "data"
    other.mkdir(parents=True)
    monkeypatch.setattr(paths, "DATA", other)
    paths.keep_sync()
    assert (other / "settings.json").read_text("utf-8") == '{"model_api_key": "sk-old"}'


def test_meme_captions_and_vertical_videos_on_blurred_sides_are_caught():
    import random
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    from jinggao import express_clean, screen_text
    # A huge red caption at the bottom is 花字, not a subtitle to crop away; a normal subtitle line is not.
    big = screen_text.classify([[{"text": "不要视奸我", "score": 0.95, "box": [0.05, 0.82, 0.9, 0.95]}]])
    assert big["dense"] and big["marks"][0]["kind"] == "center"
    small = screen_text.classify([[{"text": "他说今天的会议推迟了", "score": 0.95, "box": [0.2, 0.88, 0.8, 0.93]}]])
    assert not small["dense"] and small["marks"][0]["kind"] == "caption"
    random.seed(2)
    sharp = Image.new("L", (480, 270))
    d = ImageDraw.Draw(sharp)
    for _ in range(500):
        x, y = random.randrange(480), random.randrange(270)
        d.rectangle((x, y, x + random.randrange(3, 20), y + random.randrange(3, 20)), fill=random.randrange(256))
    mix = sharp.filter(ImageFilter.GaussianBlur(14))
    mix.paste(sharp.crop((165, 0, 315, 270)), (165, 0))
    assert express_clean.blurred_sides([np.array(mix)] * 3, (0, 0, 1, 1))
    assert not express_clean.blurred_sides([np.array(sharp)] * 3, (0, 0, 1, 1))
    assert express_match._boxed(mix.convert("RGB")) and not express_match._boxed(sharp.convert("RGB"))


def test_pasted_keys_are_cleaned_and_a_rejected_key_is_refused_and_requeued(data, monkeypatch):
    from fastapi.testclient import TestClient
    from jinggao import api, config
    assert config.clean_key(" sk-abc​123 \n") == "sk-abc123" and config.clean_key("“sk-1”") == "sk-1"
    config.update({"model_api_key": " sk-abc123 \n"})
    assert config.read()["model_api_key"] == "sk-abc123"
    def refuse(base, key):
        raise ValueError("模型服务不认这个 API Key（HTTP 401）")
    monkeypatch.setattr(config, "check_model_key", refuse)
    with pytest.raises(ValueError):
        config.update({"model_api_key": "wrong"})
    assert config.read()["model_api_key"] == "sk-abc123"          # the old key stays
    # A failed queue from a refused key goes back when a new key is saved.
    monkeypatch.setattr(config, "check_model_key", lambda *a, **k: None)
    project = store.new_project("片子", "第一句。")
    job = store.enqueue(project["id"], "produce", {})
    store.job_update(job["id"], status="failed", error="模型 API Key 无效或已过期，请在设置中更换（HTTP 401）")
    with TestClient(api.create_app()) as client:
        r = client.patch("/api/settings", json={"model_api_key": "sk-new"})
    assert r.status_code == 200 and r.json()["requeued"] == 1
    assert store.jobs(project["id"])[0]["status"] == "queued"


def test_big_letters_in_the_middle_of_the_scene_are_not_花字():
    from jinggao import screen_text
    card = screen_text.classify([[{"text": "社会保障卡", "score": 0.95, "box": [0.3, 0.4, 0.7, 0.52]}]])
    assert not card["dense"]                     # a 社保卡 held up to the camera is footage


def test_lines_without_footage_get_a_fresh_library_video_at_its_matching_moment(data, monkeypatch):
    from jinggao import express_plan, footage_index
    shots = [planner.new_shot(t) | {"duration": 5.0, "visual": "超市 收银台 顾客"} for t in ("一。", "二。")]
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    used = {"tile": {"source": {"provider": "bilibili", "id": "BVused"}}}
    rows = {"收银员 扫码": [{"provider": "bilibili", "id": "BVnew1", "duration": 300, "moments": [90.0, 12.0]},
                          {"provider": "bilibili", "id": "BVshort", "duration": 6, "moments": [3.0]}],
            "超市": [{"provider": "bilibili", "id": "BVnew2", "duration": 300, "moments": [55.0]},
                    {"provider": "bilibili", "id": "BVnew3", "duration": 300, "moments": [20.0]}]}
    asked = []
    def find(words, limit=6, exclude=None, penalize=None, foreign_ok=False, people=None):
        asked.append((" ".join(words), set(exclude or ())))
        return [dict(r) for r in rows.get(" ".join(words), []) if f"bilibili:{r['id']}" not in (exclude or ())]
    monkeypatch.setattr(footage_index, "find", find, raising=False)
    monkeypatch.setattr(express_plan, "library_words", lambda shot: [["收银员", "扫码"], ["超市"], ["收银台"]], raising=False)
    monkeypatch.setattr(m, "resolve", lambda sources: {k: {"available": True} for k in sources})
    jobs = m.library_jobs(project["shots"], {"x": used}, {}, {})
    picked = [(j[1], j[2]["tile"]["source"]["id"], j[2]["start"]) for j in jobs]
    first, second = (s["id"] for s in project["shots"])
    # Videos the film does not show yet, at the matching moment (a second early, the frame inside the window); a video
    # too short for the line is skipped; every line takes different videos.
    assert picked == [(first, "BVnew1", 89.0), (first, "BVnew2", 54.0), (second, "BVnew3", 19.0)]
    assert all(j[2]["lane"] == "library" and j[2]["moment"] == j[2]["tile"]["t"] for j in jobs)
    assert "bilibili:BVused" in asked[0][1]


def test_lines_without_their_own_words_take_single_words_then_the_film_topic(data, monkeypatch):
    from jinggao import express_plan
    shot = planner.new_shot("一。") | {"duration": 5.0}
    m = _matcher(data, [shot])
    m.plan = {"topic": {"zh": ["家庭教育 孩子 手机 电视剧片段"]}, "domain": "家庭教育"}
    monkeypatch.setattr(express_plan, "library_words", lambda s: [["中年男子", "摔手机"], ["男子", "愤怒"], ["手机"]],
                        raising=False)
    assert m.library_words(m.shots[0], (0, 1)) == [["中年男子", "摔手机"], ["男子", "愤怒"]]
    wide = m.library_words(m.shots[0], "wide")
    # Alone, a thing finds phone reviews and a vague person any stranger: single words are people doing something.
    assert ["手机"] not in wide and ["男子"] not in wide and ["摔手机"] in wide and ["中年男子"] in wide
    assert m.library_words(m.shots[0], "topic")[0] == ["家庭教育", "孩子"]


def test_the_finished_film_is_checked_for_alike_shots_and_overused_videos(data, monkeypatch, tmp_path):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 5.0} for i in range(6)]
    m = _matcher(data, shots)
    ids = [s["id"] for s in m.shots]
    def clip(n, video, lane="pick"):
        return {"tile": {"source": {"provider": "bilibili", "id": video}, "t": 10.0 * n}, "lane": lane, "job": f"j{n}"}
    final = {ids[0]: clip(0, "BVa"), ids[1]: clip(1, "BVb"), ids[2]: clip(2, "BVc", "library"),
             ids[3]: clip(3, "BVd", "repeat"), ids[4]: clip(4, "BVe", "near"), ids[5]: clip(5, "BVf")}
    files = {sid: tmp_path / f"{sid}.mp4" for sid in ids}
    for job in ("j0", "j1", "j2", "j3", "j4", "j5"):
        m.clean[job] = {"crop": [0, 0, 1, 1], "zoom": 1.0}
    # Shots 0, 2, 3 and 4 show one picture (a drama's girl in pink); 1 and 5 are other footage.
    girl, other = [0b1010] * 3, [(1 << 63) - 1] * 3
    looks = {ids[0]: girl, ids[1]: other, ids[2]: girl, ids[3]: girl, ids[4]: girl, ids[5]: [123456789] * 3}
    monkeypatch.setattr(express_match.express_clean, "available", lambda: True)
    monkeypatch.setattr(express_match.Matcher, "looks_of", staticmethod(lambda path, d: looks[Path(path).stem]))
    redo = m.audit(final, files)
    # The pick and the library window stay; the repeat and the near window of that picture go.
    assert set(redo) == {ids[3], ids[4]}
    assert m.similar == [4]
    # One video in more than SOURCE_MAX shots: the extra ones go too.
    final = {sid: clip(n, "BVsame") for n, sid in enumerate(ids[:3])}
    looks.update({ids[0]: [0] * 3, ids[1]: [(1 << 64) - 1] * 3, ids[2]: [0xAAAAAAAAAAAAAAAA] * 3})
    m.looked.clear()
    assert set(m.audit(final, files)) == {ids[1], ids[2]}    # never next to itself either
    assert m.similar == []


def test_windows_taken_from_the_library_teach_it_what_they_showed(data, monkeypatch):
    from jinggao import footage_index
    shot = planner.new_shot("一。") | {"duration": 5.0}
    m = _matcher(data, [shot])
    source = {"provider": "bilibili", "id": "BVx", "indexed": True, "moments": [40.0, 80.0]}
    m.job_choice = {"a": {"tile": {"source": source, "t": 40.0}, "start": 39.0, "moment": 40.0},
                    "b": {"tile": {"source": source, "t": 80.5}, "start": 79.0},
                    "c": {"tile": {"source": source, "t": 120.0}, "start": 119.0},
                    "d": {"tile": {"source": source, "t": 40.0}, "start": 39.0, "moment": 40.0}}
    m.clean = {"a": {"crop": [0, 0, 1, 1], "zoom": 1.0}, "b": {"crop": [0, 0, 1, 1], "zoom": 1.0, "narrow": True},
               "c": {"crop": [0, 0, 1, 1], "zoom": 1.0}}
    m.fill_errors = {"d": "ConnectError: timed out"}
    monkeypatch.setattr(express_match.screen_text, "installed", lambda: True)
    written = []
    monkeypatch.setattr(footage_index, "record_checks", lambda items: written.extend(items) or len(items), raising=False)
    assert m.record_windows({"a": "画面出现主持人对着镜头讲话"}) == 2
    assert {(w["t"], w["status"]) for w in written} == {(40.0, "presenter"), (80.0, "narrow")}


def test_marked_clips_are_never_used_unless_the_setting_allows_them(data):
    assert config.read()["marked_clips"] == "never"
    config.update({"marked_clips": "last_resort"})
    assert config.read()["marked_clips"] == "last_resort"
    with pytest.raises(ValueError):
        config.update({"marked_clips": "sometimes"})


def test_lines_without_footage_take_a_clean_clip_never_a_title_card(data, monkeypatch, tmp_path):
    shots = [planner.new_shot(t) | {"duration": 5.0} for t in ("一。", "二。", "三。", "四。")]
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    m = express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    ids = [s["id"] for s in project["shots"]]
    clip = lambda sid, lane="pick": {"tile": {"source": {"provider": "bilibili", "id": "BV" + sid[:4]}}, "lane": lane, "qa": "pass"}
    final = {ids[0]: clip(ids[0]), ids[1]: clip(ids[1], "pick") | {"qa": "rejected_kept:有去不掉的水印或文字"}}
    files = {ids[0]: tmp_path / "a.mp4", ids[1]: tmp_path / "b.mp4"}
    monkeypatch.setattr(express_match.es, "probe_duration", lambda path: 3.0)    # shorter than the 5 s line
    assert m.any_clean(project["shots"][3], final, files) == (ids[0], 1.7)     # slowed to fill it; never the marked clip
    assert m.any_clean(project["shots"][3], {ids[1]: final[ids[1]]}, files) is None
    monkeypatch.setattr(express_match.es, "probe_duration", lambda path: 2.0)    # would need more than half speed
    assert m.any_clean(project["shots"][3], final, files) is None


def test_a_slowed_clip_counts_its_slowed_length():
    from jinggao import curation, media
    project = {"media_policy": "video_preferred", "assets": [{"id": "a", "kind": "video", "duration": 3.0, "width": 1920, "height": 1080}],
               "shots": [{"duration": 5.0, "selected_asset": "a", "clip_start": 0,
                          "verification": {"visual_quality": {"representation": "live_action"}}}]}
    assert curation.composition(project)["short_video_shots"] == [1]
    project["shots"][0]["playback_slow"] = 1.7
    assert curation.composition(project)["short_video_shots"] == [] and media.playback_slow(project["shots"][0]) == 1.7
    assert media.playback_slow({"playback_slow": 9}) == 2.0 and media.playback_slow({}) == 1.0


def test_a_film_short_of_footage_waits_alone_without_pausing_the_computer(data, monkeypatch):
    from jinggao import worker
    project = store.new_project("片子", "第一句。")
    store.enqueue(project["id"], "produce", {})
    def starved(job, stop=None):
        raise express_match.FootageStarved("只拿到 2 段可用画面")
    monkeypatch.setattr(worker, "process", starved)
    assert worker.run_once()
    jobs = store.jobs(project["id"])
    assert jobs[0]["status"] == "queued" and "重新找一次" in jobs[0]["message"]
    assert worker.bili_refused_at() == 0.0                         # other videos keep going


def test_painting_follows_the_rendered_files_size_and_never_runs_past_its_edge():
    import subprocess
    from jinggao import express_clean, paths
    result = {"crop": [0.0, 0.0, 1.0, 1.0], "size": [1920, 1080], "residual": [{"kind": "logo", "box": [0.86, 0.02, 0.98, 0.1]}]}
    vf = express_clean.video_filter(result, (854, 480))           # the rendered file is smaller than the one analysed
    done = subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=854x480",
                           "-vf", vf, "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
    assert "delogo" in vf and done.returncode == 0, done.stderr


def test_video_info_and_play_addresses_are_asked_for_once(data, monkeypatch):
    calls = []
    def api(self, url, params, paced=False):
        calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("view"):
            return {"code": 0, "data": {"cid": 7, "pages": [{"duration": 300}], "duration": 300, "title": "t"}}
        return {"code": 0, "data": {"dash": {"video": [{"width": 1280, "height": 720, "codecs": "avc1", "bandwidth": 1,
                                                        "baseUrl": "https://u/v.m4s", "SegmentBase": {}}]}}}
    monkeypatch.setattr(es.Bilibili, "_api", api)
    bili = es.Bilibili()
    first = bili.stream("BVcache1", 720)
    again = bili.stream("BVcache1", 720)
    assert first["available"] and again == first
    assert calls == ["view", "playurl"]          # a film made again after a refusal asks B站 nothing new
    bili.stream("BVcache1", 480)
    assert calls == ["view", "playurl", "playurl"]  # another sharpness needs its own address, not the video's info


def test_stock_site_watermarks_and_corner_pictures_are_caught():
    import numpy as np
    from jinggao import screen_text
    assert screen_text.contact({"text": "光厂素材", "box": [0.4, 0.45, 0.6, 0.52]})
    assert not screen_text.contact({"text": "望京SOHO", "box": [0.02, 0.05, 0.2, 0.15]})
    # A picture in the top-left fifth of a black frame (the 望京SOHO template of a dorm test film).
    frames = []
    for i in range(6):
        f = np.zeros((270, 480), dtype=np.float32)
        f[5:60, 5:110] = 120 + 10 * i
        frames.append(f)
    assert express_clean.small_picture(frames)
    full = [np.full((270, 480), 90.0, dtype=np.float32) for _ in range(6)]
    assert not express_clean.small_picture(full)
    letterbox = [np.zeros((270, 480), dtype=np.float32) for _ in range(6)]
    for f in letterbox:
        f[35:235, :] = 100       # a wide film's bars top and bottom: content_rect trims those
    assert not express_clean.small_picture(letterbox)


def test_groups_the_library_already_covers_search_bilibili_less(data, monkeypatch):
    shots = [planner.new_shot(t) | {"duration": 5.0} for t in ("一。", "二。", "三。")]
    m = _matcher(data, shots)
    ids = [s["id"] for s in m.shots]
    asked = []
    class Bili:
        def search(self, q):
            asked.append(q)
            return [{"provider": "bilibili", "id": "BV" + q}]
    m.bili, m.pixa = Bili(), None
    groups = [{"id": "G1", "zh": ["甲 乙", "丙 丁"], "en": [], "shots": ids[:2]},
              {"id": "G2", "zh": ["戊 己", "庚 辛"], "en": [], "shots": ids[2:]}]
    row = lambda n: {"provider": "bilibili", "id": f"BVlib{n}", "duration": 120}
    monkeypatch.setattr(m, "library_options", lambda gs: {"G1": [row(1), row(2)]})
    m.search(groups)
    assert sorted(asked) == ["庚 辛", "戊 己", "甲 乙"]     # every line of G1 has its own library video: one query
    assert m.library_rich == {"G1"}


def test_the_final_review_looks_at_every_shot_and_names_the_ones_to_replace(data, monkeypatch, tmp_path):
    from PIL import Image as _Image
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 5.0} for i in range(12)]
    m = _matcher(data, shots)
    ids = [s["id"] for s in m.shots]
    final = {sid: {"tile": {"source": {"provider": "bilibili", "id": f"BV{n}"}, "t": 10.0}, "lane": "library", "job": f"j{n}"}
             for n, sid in enumerate(ids)}
    final[ids[5]]["lane"] = "image"
    files = {sid: tmp_path / f"{sid}.mp4" for sid in ids}
    monkeypatch.setattr(m, "qa_frames", lambda job, path, shot: [_Image.new("RGB", (320, 180))] * 3)
    asked = []
    def chat(system, content, **kw):
        text = content[0]["text"]
        asked.append(text.count("口播："))
        return ({"bad": {"S2": "浴缸里的亲热镜头", "S9": "外国人"}} if text.count("口播：") == 10 else {"bad": {}}), {}
    monkeypatch.setattr(m, "chat", chat)
    verdict = m.final_review(final, files)
    assert sorted(asked) == [1, 10]                    # 11 shots (the picture is left out) on two sheets
    assert set(verdict) <= set(ids) and len(verdict) == 2 and "浴缸里的亲热镜头" in verdict.values()


def test_a_line_about_a_person_counts_as_shown_only_with_their_own_footage(data):
    shot = planner.new_shot("赵长鹏被罚款。") | {"duration": 5.0, "persons": [{"name": "赵长鹏", "aliases": []}]}
    m = _matcher(data, [shot])
    line = m.shots[0]
    own = {"provider": "bilibili", "id": "BV1", "title": "赵长鹏 采访 新闻"}
    other = {"provider": "bilibili", "id": "BV2", "title": "比特币 行情 素材"}
    assert m.shows_person(line, {"tile": {"source": own}, "lane": "pick"})
    assert m.shows_person(line, {"tile": {"source": own}, "lane": "person_near"})
    assert m.shows_person(line, {"tile": {"source": {"provider": "web_image", "id": "x"}}, "lane": "image",
                                 "base_lane": "person_photo"})
    assert not m.shows_person(line, {"tile": {"source": other}, "lane": "ctx"})     # background, not the person
    assert not m.shows_person(line, {"tile": {"source": other}, "lane": "library"})
    assert not m.shows_person(line, None)


def test_the_topic_pool_takes_long_videos_about_the_story_itself(data, monkeypatch):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 6.0} for i in range(3)]
    m = _matcher(data, shots)
    m.plan = {"pool": ["勇士 森林狼 集锦", "库里 44分 集锦"]}
    rows = {"勇士 森林狼 集锦": [{"provider": "bilibili", "id": "BVgame", "title": "勇士vs森林狼 全场集锦 库里44分", "tags": "",
                               "duration": 600, "play": 50000},
                              {"provider": "bilibili", "id": "BVshort", "title": "勇士 森林狼", "tags": "", "duration": 30},
                              {"provider": "bilibili", "id": "BVcook", "title": "家常菜 做法", "tags": "", "duration": 300}],
            "库里 44分 集锦": [{"provider": "bilibili", "id": "BVcurry", "title": "库里44分集锦 勇士客场", "tags": "",
                             "duration": 240, "play": 9000}]}
    class Bili:
        def search(self, q):
            return [dict(r) for r in rows.get(q, [])]
    m.bili = Bili()
    pool = m.pool_rows()
    assert [r["id"] for r in pool] == ["BVgame", "BVcurry"] and all(r["pool"] for r in pool)


def test_open_lines_walk_through_the_pool_and_a_pool_video_gives_many_windows(data):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 6.0} for i in range(6)]
    m = _matcher(data, shots)
    game = {"provider": "bilibili", "id": "BVgame", "duration": 600.0, "pool": True}
    tiles = {"BVgame": [{"source": game, "t": float(t)} for t in (30, 35, 60, 90, 120, 150, 180)]}
    jobs = m.pool_sweep(m.shots[:4], {}, {}, {"BVgame": {"available": True}}, tiles)
    starts = [j[2]["start"] for j in jobs]
    assert len(jobs) == 4 and starts == sorted(starts)                       # in the video's own order
    assert all(b - a >= express_match.POOL_GAP for a, b in zip(starts, starts[1:], strict=False))
    # The picker's pool tiles: one long video fills more lines than SOURCE_MAX, neighbours included.
    options = {s["id"]: [({"source": game, "t": 40.0 * (i + 1)}, "pool")] for i, s in enumerate(m.shots)}
    assigned = m.allocate(options, {}, {}, m.shots)
    assert len(assigned) == 5        # one window per two minutes of it: a 10-minute video gives five
    game["duration"] = 7236.0        # a full game replay gives every line
    assert len(m.allocate(options, {}, {}, m.shots)) == 6


def test_pool_covers_person_lines_with_story_footage_and_skips_talk_videos(data):
    shots = [planner.new_shot("库里这场轰下44分。") | {"duration": 6.0, "persons": [{"name": "库里"}]},
             planner.new_shot("乔治防得很凶。") | {"duration": 6.0, "persons": [{"name": "乔治"}]},
             planner.new_shot("比赛进入第四节。") | {"duration": 6.0}]
    m = _matcher(data, shots)
    replay = {"provider": "bilibili", "id": "BVreplay", "duration": 7236.0, "pool": True, "pool_talk": False,
              "title": "勇士vs森林狼完整全场回放！库里44分"}
    talk = {"provider": "bilibili", "id": "BVtalk", "duration": 797.0, "pool": True, "pool_talk": True,
            "title": "库里常规赛44分赛后采访"}
    other = {"provider": "bilibili", "id": "BVother", "duration": 900.0, "pool": True, "pool_talk": False,
             "title": "黄子韬直播首秀"}
    m.pool_sources = [replay, talk]
    curry, george, plain = m.shots
    assert m.pool_covers(curry, replay) and m.pool_covers(curry, talk)        # its title names him
    assert m.pool_covers(george, replay) and not m.pool_covers(george, talk)  # the game he played in, not an interview
    assert m.pool_covers(plain, talk)
    m.pool_sources = [replay, other]
    assert not m.pool_covers(curry, other)          # another star's video while a pool video names him
    assert m.shows_person(george, {"tile": {"source": replay, "t": 10.0}, "lane": "sweep", "base_lane": "sweep"})
    # The sweep walks the replay for every line, never the interview.
    tiles = {"BVreplay": [{"source": replay, "t": float(t)} for t in range(60, 600, 30)],
             "BVtalk": [{"source": talk, "t": float(t)} for t in range(30, 300, 30)]}
    jobs = m.pool_sweep(m.shots, {}, {}, {"BVreplay": {"available": True}, "BVtalk": {"available": True}}, tiles)
    assert len(jobs) == 3 and {j[2]["tile"]["source"]["id"] for j in jobs} == {"BVreplay"}


def test_pool_rows_prefer_footage_over_talk_and_need_the_story_name(data, monkeypatch):
    m = _matcher(data, [planner.new_shot("勇士赢了。") | {"duration": 6.0}])
    m.plan["pool"] = ["勇士 森林狼 全场回放", "遥望科技 上市公司"]
    rows = [{"provider": "bilibili", "id": "BVt", "title": "库里赛后采访：勇士森林狼一战", "duration": 800, "play": 900000},
            {"provider": "bilibili", "id": "BVr", "title": "勇士vs森林狼完整全场回放", "duration": 7200, "play": 20000},
            {"provider": "bilibili", "id": "BVd", "title": "十分钟读懂上市公司：德福科技", "duration": 600, "play": 5000},
            {"provider": "bilibili", "id": "BVy", "title": "遥望科技上市公司新闻", "duration": 400, "play": 5000}]
    class Bili:
        logged_in = True
        def search(self, q):
            return [dict(r) for r in rows]
    m.bili = Bili()
    monkeypatch.setattr(express_match, "_stamp", lambda name: 0)
    found = m.pool_rows()
    ids = [r["id"] for r in found]
    assert ids[0] == "BVr" and "BVd" not in ids              # 科技 alone is no name of the story
    assert next(r for r in found if r["id"] == "BVt")["pool_talk"] if "BVt" in ids else True


def test_pool_takes_a_video_naming_the_person_and_a_noise_word_the_story_is_about(data, monkeypatch):
    m = _matcher(data, [planner.new_shot("贾乃亮单日卖5.7亿。") | {"duration": 6.0}])
    m.plan["pool"] = ["贾乃亮 直播间 现场", "遥望科技 发布会 新闻"]
    rows = [{"provider": "bilibili", "id": "BVj", "title": "贾乃亮10.17早9点骆驼羽绒服底价大场", "duration": 1347, "play": 9000},
            {"provider": "bilibili", "id": "BVc", "title": "贾乃亮在车里直播15分钟就睡着了", "duration": 900, "play": 9000},
            {"provider": "bilibili", "id": "BVg", "title": "直播游戏实况", "duration": 900, "play": 9000}]
    class Bili:
        logged_in = True
        def search(self, q):
            return [dict(r) for r in rows]
    m.bili = Bili()
    monkeypatch.setattr(express_match, "_stamp", lambda name: 0)
    assert {r["id"] for r in m.pool_rows()} == {"BVj", "BVc"}


def test_pool_keeps_to_the_event_itself_when_its_footage_is_long_enough(data, monkeypatch):
    from PIL import Image
    m = _matcher(data, [planner.new_shot("勇士赢了。") | {"duration": 6.0}])
    m.plan["pool"] = ["勇士 森林狼 全场回放", "库里 比赛 集锦"]
    rows = [{"provider": "bilibili", "id": "BVold", "title": "2016年勇士vs雷霆中文超清回放 库里比赛", "duration": 7000, "play": 900000},
            {"provider": "bilibili", "id": "BVr", "title": "勇士vs森林狼完整全场回放", "duration": 7200, "play": 2000},
            {"provider": "bilibili", "id": "BVc", "title": "库里比赛集锦", "duration": 300, "play": 90000}]
    class Bili:
        logged_in = True
        def search(self, q):
            return [dict(r) for r in rows]
    m.bili = Bili()
    monkeypatch.setattr(express_match, "_stamp", lambda name: 0)
    candidates = m.pool_rows()
    assert [r["id"] for r in candidates][0] == "BVr" and len(candidates) == 3   # the event's own first
    image = Image.new("RGB", (160, 90))
    def tiles_of(rows_):
        return {("bilibili", r["id"]): [{"source": r, "t": float(t), "image": image} for t in range(10, 70, 10)]
                for r in rows_}
    m.presenter_sources = set()
    # Without a picture answer: the videos naming the event, which are long enough, are the whole pool.
    def down(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(m, "chat", down)
    fresh = [dict(r) for r in candidates]
    assert [r["id"] for r in m.vet_pool(fresh, tiles_of(fresh))] == ["BVr"]
    # The picture check: the 2016 game is only the field; with the event's own footage long enough it stays out
    # and loses its pool mark.
    fresh = [dict(r) for r in candidates]
    order = [r["id"] for r in fresh]
    label = {bvid: f"V{i + 1}" for i, bvid in enumerate(order)}
    monkeypatch.setattr(m, "chat", lambda *a, **k: ({"same": [label["BVr"], label["BVc"]], "field": [label["BVold"]]}, {}))
    pool = m.vet_pool(fresh, tiles_of(fresh))
    assert [r["id"] for r in pool] == ["BVr", "BVc"]
    assert "pool" not in next(r for r in fresh if r["id"] == "BVold")
    assert m.pool_sources == pool and "同题素材：2 部" in m.notes[-1]
    # A short event: same-kind footage joins after it.
    fresh = [dict(r) for r in candidates]
    next(r for r in fresh if r["id"] == "BVr")["duration"] = 600
    monkeypatch.setattr(m, "chat", lambda *a, **k: ({"same": [label["BVr"]], "field": [label["BVold"]]}, {}))
    assert [r["id"] for r in m.vet_pool(fresh, tiles_of(fresh))] == ["BVr", "BVold"]


def test_the_sweep_walks_a_long_pool_video_past_its_previews(data):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 6.0} for i in range(30)]
    m = _matcher(data, shots)
    replay = {"provider": "bilibili", "id": "BVreplay", "duration": 7236.0, "pool": True, "title": "全场回放"}
    m.pool_sources = [replay]
    tiles = {"BVreplay": [{"source": replay, "t": float(t)} for t in (600, 1200, 1800)]}
    jobs = m.pool_sweep(m.shots, {}, {}, {"BVreplay": {"available": True}}, tiles)
    starts = [j[2]["start"] for j in jobs]
    assert len(jobs) == 30 and starts == sorted(starts)           # one window a line, in the video's order
    assert all(b - a >= express_match.POOL_GAP for a, b in zip(starts, starts[1:], strict=False))


def test_state_leaders_footage_only_when_the_script_names_them(data):
    m = _matcher(data, [planner.new_shot("特朗普和泽连斯基见面。") | {"duration": 6.0}])
    leader = {"title": "习近平主席阅兵讲话视频，振奋人心！", "tags": ""}
    assert m.off_limits(leader) and not m.off_limits({"title": "特朗普会见泽连斯基", "tags": ""})
    m.project["script"] = "习近平主席在阅兵式上讲话。"
    assert not m.off_limits(leader)
    assert m.off_limits({"title": "ASMR 助眠", "tags": ""})
