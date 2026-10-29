"""干净体检 (footage_check): synthetic 480p windows judged by production's own checks, a fake Bilibili and a stubbed
download; the dorm script's order, pauses and batched writes against a temporary index."""
import importlib.util
import shutil
import subprocess
import time

import pytest

from jinggao import footage_check as fc, footage_index as fi, paths, screen_text

OCR_DIR = screen_text.model_dir()    # the real data folder's text-reading model (tests move DATA elsewhere)
W, H = 854, 480


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(screen_text, "model_dir", lambda: OCR_DIR)
    for cache in (fc._streams, fc._sheets, fc._profiles):
        cache.clear()
    yield
    for cache in (fc._streams, fc._sheets, fc._profiles):
        cache.clear()


def _ffmpeg(folder, *args):
    subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", *args], cwd=folder, check=True, timeout=120)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """Seven-second 480p windows of every kind the check must tell apart (relative names: Windows paths carry a
    colon that FFmpeg's filter syntax would split on)."""
    folder = tmp_path_factory.mktemp("clips")
    shutil.copy(paths.ROOT / "assets" / "fonts" / "NotoSansSC-Black.otf", folder / "font.otf")
    (folder / "big.txt").write_text("震惊！父亲怒摔手机儿子拒绝上学", encoding="utf-8")
    (folder / "sub.txt").write_text("他说今天的会议推迟到下午三点", encoding="utf-8")
    (folder / "short.txt").write_text("太扎心了", encoding="utf-8")
    (folder / "five.txt").write_text("笑死我了！", encoding="utf-8")
    source = f"mandelbrot=size={W}x{H}:rate=25,trim=duration=7"
    encode = ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast"]
    graphs = {
        "clean": "null",
        # A vertical video posted inside a landscape frame: black bars, or an enlarged blurred copy of itself.
        "bars": f"crop=270:{H},pad={W}:{H}:292:0:black",
        "blur": f"crop=270:{H},split[a][b];[a]scale={W}:1518,crop={W}:{H},gblur=sigma=30[bg];[bg][b]overlay=292:0",
        # 花字 across the middle (the client's biggest complaint) and an ordinary subtitle line at the bottom.
        "caption": "drawtext=fontfile=font.otf:textfile=big.txt:fontsize=52:fontcolor=yellow:borderw=4:bordercolor=red:"
                   "x=(w-text_w)/2:y=(h-text_h)/2",
        "subtitle": "drawtext=fontfile=font.otf:textfile=sub.txt:fontsize=20:fontcolor=white:borderw=2:"
                    "bordercolor=black:x=(w-text_w)/2:y=h*0.87",
        # Short 花字 the old rules passed as clean (review 10-28): four big characters in the middle, five shown
        # for three seconds only, and a faint account name across the picture.
        "huazi": "drawtext=fontfile=font.otf:textfile=short.txt:fontsize=64:fontcolor=yellow:borderw=4:bordercolor=red:"
                 "x=(w-text_w)/2:y=(h-text_h)/2",
        "popup": "drawtext=fontfile=font.otf:textfile=five.txt:fontsize=56:fontcolor=yellow:borderw=4:bordercolor=red:"
                 "x=(w-text_w)/2:y=h*0.4:enable='between(t,1,4)'",
        "account": "drawtext=fontfile=font.otf:text='@情感剧场':fontsize=44:fontcolor=white@0.35:x=(w-text_w)/2:y=h*0.55",
    }
    out = {}
    for name, graph in graphs.items():
        _ffmpeg(folder, "-f", "lavfi", "-i", source, "-filter_complex", graph, *encode, f"{name}.mp4")
        out[name] = folder / f"{name}.mp4"
    _ffmpeg(folder, "-f", "lavfi", "-i", source, "-frames:v", "1", "still.png")
    _ffmpeg(folder, "-loop", "1", "-framerate", "25", "-i", "still.png", "-t", "7", *encode, "still.mp4")
    _ffmpeg(folder, "-f", "lavfi", "-i", f"color=c=0x060606:s={W}x{H}:r=25:d=7", "-vf", "noise=alls=3:allf=t",
            *encode, "dark.mp4")
    out |= {"still": folder / "still.mp4", "dark": folder / "dark.mp4"}
    return out


class FakeBili:
    """stream() answers like express_sources.Bilibili.stream (or raises its errors); previews are absent."""
    def __init__(self, stream=None, error=None, previews=None):
        self._stream, self.error, self._previews = stream, error, previews
        self.calls = 0

    def stream(self, bvid, short_edge):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(self._stream)

    def previews(self, bvid, duration):
        if isinstance(self._previews, Exception):
            raise self._previews
        return self._previews or []


def landscape(clip="clean", duration=60.0):
    return {"available": True, "width": W, "height": H, "duration": duration, "clip": clip}


def row(bvid="BV1check0001", duration=60.0, title="父亲摔手机 情景短剧"):
    return {"provider": "bilibili", "id": bvid, "title": title, "tags": "", "author": "up", "duration": duration}


@pytest.fixture
def downloads(clips, monkeypatch):
    """The download step copies the synthetic clip the fake stream names; records the windows asked for."""
    asked = []
    def fetch(stream, start, seconds, out, *, short_edge, check=None):
        asked.append((round(start, 2), seconds, short_edge))
        shutil.copy(clips[stream["clip"]], out)
    monkeypatch.setattr(fc, "fetch_window", fetch)
    return asked


@pytest.mark.parametrize("clip, status", [("clean", "clean"), ("bars", "narrow"), ("blur", "narrow"),
                                          ("caption", "text"), ("subtitle", "crop"), ("still", "still"),
                                          ("dark", "dark"), ("huazi", "text"), ("popup", "text"),
                                          ("account", "mark")])
def test_windows_are_judged_like_production(clip, status, downloads, tmp_path):
    if clip in ("caption", "subtitle", "huazi", "popup", "account") and not screen_text.installed():
        pytest.skip("text-reading model not installed in the data folder")
    work = tmp_path / "work"
    result = fc.check_window(FakeBili(landscape(clip)), row(), 30.0, work=work)
    assert result["status"] == status, result
    assert set(result) == {"status", "crop", "note", "seconds"} and result["seconds"] >= 0
    assert downloads == [(29.0, 7.0, 480)]                      # [t-1, t-1+7] at a low resolution
    assert list(work.iterdir()) == []                            # the clip is deleted afterwards
    if status == "crop":
        x, y, w, h = result["crop"]
        assert y + h <= 0.87 and "字幕" in result["note"]          # cut above the subtitle line
    elif status == "clean":
        assert result["crop"] is None and result["note"] == "无预览帧"
    else:
        assert result["crop"] is None and result["note"]


def test_unavailable_lowres_portrait_and_refused_videos_need_no_download(downloads):
    gone = FakeBili(error=ValueError("B站视频信息不可用：啥都木有"))
    assert fc.check_window(gone, row("BV1gone000001"), 10)["status"] == "gone"
    assert fc.check_window(gone, row("BV1gone000001"), 40)["status"] == "gone" and gone.calls == 1   # cached
    low = fc.check_window(FakeBili({"available": False, "best_short_edge": 360}), row("BV1low0000001"), 10)
    assert low["status"] == "lowres" and "360p" in low["note"]
    # No video stream at all is B站 under load, not a low-resolution video: nothing to record, asked again.
    empty = FakeBili({"available": False, "best_short_edge": 0})
    with pytest.raises(fc.NoStream):
        fc.check_window(empty, row("BV1empty00001"), 10)
    with pytest.raises(fc.NoStream):
        fc.check_window(empty, row("BV1empty00001"), 40)
    assert empty.calls == 2
    with pytest.raises(fc.Refused):
        fc.check_window(FakeBili({"available": False, "best_short_edge": 0, "voucher": True}), row("BV1vouch00001"), 10)
    tall = fc.check_window(FakeBili({"available": True, "width": 480, "height": 854, "duration": 60}),
                           row("BV1tall000001"), 10)
    assert tall["status"] == "narrow" and tall["note"] == "竖屏视频"
    late = fc.check_window(FakeBili(landscape(duration=30.0)), row("BV1short00001", duration=90.0), 45)
    assert late["status"] == "outside" and "超出" in late["note"]  # the library's length was the whole series
    assert "outside" not in fi.CHECK_VIDEO                         # that moment, not the video
    with pytest.raises(fc.Refused):
        fc.check_window(FakeBili(error=ValueError("B站接口暂时限制访问（风控），请稍后重试或扫码登录B站")),
                        row("BV1risk000001"), 10)
    with pytest.raises(ConnectionError):                         # a network failure is not a verdict
        fc.check_window(FakeBili(error=ConnectionError("reset")), row("BV1net0000001"), 10)
    assert downloads == []
    assert ("BV1risk000001", 480) not in fc._streams and ("BV1net0000001", 480) not in fc._streams
    assert ("BV1empty00001", 480) not in fc._streams and ("BV1vouch00001", 480) not in fc._streams


def test_one_address_per_video_and_a_failed_download_asks_again(clips, monkeypatch, tmp_path):
    bili = FakeBili(landscape())
    calls = []
    def fetch(stream, start, seconds, out, *, short_edge, check=None):
        calls.append(start)
        if len(calls) == 2:
            raise ValueError("片段下载失败：HTTP 403")
        shutil.copy(clips["clean"], out)
    monkeypatch.setattr(fc, "fetch_window", fetch)
    assert fc.check_window(bili, row(), 10, work=tmp_path)["status"] == "clean"
    with pytest.raises(ValueError):
        fc.check_window(bili, row(), 20, work=tmp_path)
    assert bili.calls == 1
    fc.check_window(bili, row(), 30, work=tmp_path)
    assert bili.calls == 2                                       # the address may have expired: a fresh one
    assert list(tmp_path.glob("*.mp4")) == []


def test_window_fits_the_video(downloads):
    bili = FakeBili(landscape(duration=60.0))
    for t in (0.4, 10.0, 59.0):
        fc.check_window(bili, row(), t)
    assert [start for start, _, _ in downloads] == [0.0, 9.0, 52.9]


def test_refused_previews_pause_instead_of_judging_without_them(downloads, tmp_path):
    bili = FakeBili(landscape(), previews=ValueError("B站预览接口暂时限制访问（风控）"))
    with pytest.raises(fc.Refused):
        fc.check_window(bili, row(), 10, work=tmp_path)
    assert list(tmp_path.iterdir()) == []
    broken = FakeBili(landscape(), previews=OSError("sprite"))   # anything else: the clip alone decides
    assert fc.check_window(broken, row("BV1other00001"), 10, work=tmp_path)["status"] == "clean"


def test_the_local_model_only_looks_at_otherwise_clean_windows(downloads):
    seen = []
    def presenter(r, images):
        seen.append(len(images))
        return "presenter"
    assert fc.check_window(FakeBili(landscape()), row(), 10, describe=presenter)["status"] == "presenter"
    assert seen == [3]
    assert fc.check_window(FakeBili(landscape("bars")), row("BV1bars000001"), 10, describe=presenter)["status"] == "narrow"
    assert seen == [3]                                           # a narrow window needs no model
    def meme(r, images):
        return "meme"
    assert fc.check_window(FakeBili(landscape()), row("BV1meme000001"), 10, describe=meme)["status"] == "text"
    def down(r, images):
        raise ConnectionError("server gone")
    result = fc.check_window(FakeBili(landscape()), row("BV1down000001"), 10, describe=down)
    assert result["status"] == "clean" and "看图失败" in result["note"]


def test_model_answers_are_read_in_any_shape():
    assert fc.parse_kind('{"类型":"实拍"}') == "filmed"
    assert fc.parse_kind('<think>想想</think>{"类型": "主持人出镜"}') == "presenter"
    assert fc.parse_kind("类型：文字页") == "text"
    assert fc.parse_kind('{"类型":"动画"}') == "cartoon" and fc.parse_kind('{"类型":"表情包"}') == "meme"
    assert fc.parse_kind('{"类型":"花字"}') == "overlay"
    assert fc.parse_kind("说不清") is None


def test_production_flags_map_to_one_status_most_fundamental_first():
    base = {"motion": {"still": False, "dark": False}, "crop": [0, 0, 1, 1], "size": [854, 480], "overlays": [],
            "residual": [], "central": []}
    found = {"contact": [], "moving": False, "dense": False, "marks": []}
    assert fc.verdict(base | {"motion": {"still": True, "dark": True}, "narrow": True}, found, set())[0] == "dark"
    assert fc.verdict(base | {"narrow": True, "layout_jump": True}, found, set())[0] == "narrow"
    assert fc.verdict(base | {"layout_jump": True}, found, set())[0] == "jump"
    status, note = fc.verdict(base, found | {"contact": ["加微信"], "dense": True}, {"wm"})
    assert status == "mark" and note.startswith(fc.CONTACT)
    assert fc.verdict(base, found | {"moving": True}, set()) == ("text", "滚动字幕")
    assert fc.verdict(base | {"central": [{"kind": "center", "box": [.4, .4, .6, .5]}],
                              "residual": [{"kind": "center", "box": [.4, .4, .6, .5]}]}, found, set())[0] == "mark"
    big_logo = {"kind": "logo", "box": [0.3, 0.02, 0.7, 0.2]}
    assert fc.verdict(base | {"overlays": [big_logo], "residual": [big_logo]}, found, set())[0] == "mark"
    stuck = {"kind": "subtitle", "box": [0.2, 0.3, 0.8, 0.4]}
    assert fc.verdict(base | {"overlays": [stuck], "residual": [stuck]}, found, set())[0] == "text"
    corner = {"kind": "logo", "box": [0.9, 0.02, 0.98, 0.06]}       # small, at the edge: painted out
    assert fc.verdict(base | {"overlays": [corner], "residual": [corner]}, found, set())[0] == "crop"
    assert fc.verdict(base, found, set()) == ("clean", "")


def test_moments_worth_a_check_are_filmed_between_filmed_neighbours_and_spread_out():
    kinds = [(5.0, "主持人出镜"), (15.0, "实拍"), (25.0, "实拍"), (35.0, "影视剧"), (45.0, "图表文字"), (55.0, "实拍")]
    kinds += [(65.0 + 10 * i, "实拍") for i in range(10)]
    r = row(duration=200.0)
    picked = fc.moments_to_check(r, kinds, limit=4)
    assert 15.0 not in picked and 35.0 not in picked and 55.0 not in picked   # next to presenter / text frames
    assert picked == [25.0, 85.0, 125.0, 155.0]                    # spread over the 11 good ones, ends included
    assert 25.0 not in fc.moments_to_check(r, kinds, limit=4, checked=[25.0])
    assert fc.moments_to_check(row(duration=70.0), [(60.0, "实拍"), (69.5, "实拍")], limit=4) == [60.0]
    assert fc.worth_checking(r, kinds)
    commentary = [(t, "主持人出镜") for t in range(0, 40, 10)] + [(t, "实拍") for t in range(40, 100, 10)]
    assert not fc.worth_checking(r, commentary)                    # 40% presenter: a commentary video
    assert not fc.worth_checking(row(duration=12.0), kinds) and not fc.worth_checking(row(title="游戏实况"), kinds)


# ------------------------------------------------------------------ the dorm script
def _script():
    spec = importlib.util.spec_from_file_location("footage_check_script", paths.ROOT / "scripts" / "footage_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _video(bvid, topic, used=0, frames=8, kind="实拍"):
    fi.add(row(bvid, duration=300.0) | {"play": 10}, [{"t": 20.0 + 30 * i, "hash": str(i), "text": "男子 看手机｜男子 手机",
                                                       "kind": kind} for i in range(frames)], topic=topic,
           described=True)
    if used:
        fi.mark_used([f"bilibili:{bvid}"] * 1)


@pytest.fixture
def index(monkeypatch):
    """A small index: two topics the client writes about, one other; one video already partly checked."""
    _video("BV1other00001", "动物")
    _video("BV1family0001", "生活·家庭情感")
    _video("BV1family0002", "生活·家庭情感", used=1)
    _video("BV1cartoon001", "生活·家庭情感", kind="动画")             # nothing filmed: never checked
    _video("BV1work000001", "职场·教育", frames=3)
    written, stats = [], {"clean": 0}
    monkeypatch.setattr(fi, "checks_for", lambda keys, since=0.0: {k: {20.0: "clean"} for k in keys
                                                                    if k == "bilibili:BV1family0001"}, raising=False)
    monkeypatch.setattr(fi, "record_checks", lambda items: written.extend(items) or len(items), raising=False)
    monkeypatch.setattr(fi, "check_stats", lambda: stats, raising=False)
    monkeypatch.setattr(fi, "install_seed", lambda *a, **k: 0)
    monkeypatch.setattr(paths, "prepare", lambda: paths.DATA.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(screen_text, "installed", lambda: True)
    return written


def test_the_script_checks_the_clients_topics_first_and_writes_in_batches(index, monkeypatch, tmp_path):
    script = _script()
    calls = []
    def judge(bili, r, t, **kw):
        calls.append((r["id"], t))
        return {"status": "clean", "crop": None, "note": "", "seconds": 0.1}
    monkeypatch.setattr(fc, "check_window", judge)
    log = tmp_path / "run" / "footage-check.log"
    assert script.main(["--workers", "1", "--no-describe", "--max-windows", "12", "--log", str(log)]) == 0
    videos = list(dict.fromkeys(bvid for bvid, _ in calls))
    # Client topics in FIRST order; inside one, the video films used more first; other topics after.
    assert videos == ["BV1family0002", "BV1family0001", "BV1work000001", "BV1other00001"]
    assert len([c for c in calls if c[0] == "BV1family0002"]) == 4       # at most four moments a video
    assert ("BV1family0001", 20.0) not in calls                           # checked before: skipped
    assert [t for bvid, t in calls if bvid == "BV1family0002"] == [20.0, 80.0, 170.0, 230.0]   # spread out
    assert len(index) == len(calls) == 12 and index[0] == {"source": "bilibili:BV1family0002", "t": 20.0,
                                                           "status": "clean", "crop": None, "note": ""}
    text = log.read_text(encoding="utf-8")
    assert "题材：生活·家庭情感" in text and "stopped: 已到 --max-windows" in text and "/小时" in text
    assert (log.parent / "footage-check-status.json").exists()


def test_the_script_pauses_on_risk_control_and_marks_a_contact_video(index, monkeypatch, tmp_path):
    script = _script()
    monkeypatch.setattr(fi, "BLOCKED_PAUSE", 1)
    calls, began = [], time.monotonic()
    def judge(bili, r, t, **kw):
        calls.append((r["id"], t, round(time.monotonic() - began, 1)))
        if len(calls) == 1:
            raise fc.Refused("B站接口暂时限制访问（风控）")
        if r["id"] == "BV1family0001":
            return {"status": "mark", "crop": None, "note": f"{fc.CONTACT}：加微信", "seconds": 0.1}
        return {"status": "clean", "crop": None, "note": "", "seconds": 0.1}
    monkeypatch.setattr(fc, "check_window", judge)
    log = tmp_path / "footage-check.log"
    script.main(["--workers", "1", "--no-describe", "--max-windows", "8", "--log", str(log)])
    assert calls[0][:2] == calls[1][:2] and calls[1][2] >= 0.9           # refused: paused, then the same window
    assert "风控" in log.read_text(encoding="utf-8")
    contact = [item for item in index if item["source"] == "bilibili:BV1family0001"]
    assert [c[0] for c in calls].count("BV1family0001") == 1             # one window showed contact details …
    assert len(contact) == 4 and all(item["status"] == "mark" for item in contact)   # … the video's others follow


def test_a_stop_file_ends_the_script(index, monkeypatch, tmp_path):
    script = _script()
    log = tmp_path / "footage-check.log"
    def judge(bili, r, t, **kw):
        (tmp_path / "STOP").write_text("")
        return {"status": "clean", "crop": None, "note": "", "seconds": 0.1}
    monkeypatch.setattr(fc, "check_window", judge)
    script.main(["--workers", "2", "--no-describe", "--log", str(log)])
    assert 1 <= len(index) <= 2 and "stopped: 到点停止" in log.read_text(encoding="utf-8")


def test_moments_checked_by_older_rules_are_checked_again(monkeypatch):
    script = _script()
    _video("BV1family0001", "生活·家庭情感")
    since = script.rules_since(fi, 99)
    assert script.rules_since(fi, 99) == since                            # noted the first time only
    fi.record_checks([{"source": "bilibili:BV1family0001", "t": 20.0, "status": "text", "checked": since - 3600},
                      {"source": "bilibili:BV1family0001", "t": 50.0, "status": "clean"}])
    [(_, times)] = list(script.topic_videos(fi, fc, "生活·家庭情感", 4, 7.0, {}, since))
    assert 20.0 in times and 50.0 not in times                            # an older verdict is made again
    [(_, times)] = list(script.topic_videos(fi, fc, "生活·家庭情感", 4, 7.0, {}))
    assert 20.0 not in times and 50.0 not in times


def test_a_run_of_windows_without_any_stream_pauses_like_risk_control(index, monkeypatch, tmp_path):
    script = _script()
    monkeypatch.setattr(fi, "BLOCKED_PAUSE", 1)
    calls = []
    def judge(bili, r, t, **kw):
        calls.append((r["id"], t))
        if len(calls) <= script.NO_STREAM_RUN:
            raise fc.NoStream("B站没有给出视频流")
        return {"status": "clean", "crop": None, "note": "", "seconds": 0.1}
    monkeypatch.setattr(fc, "check_window", judge)
    log = tmp_path / "footage-check.log"
    script.main(["--workers", "1", "--no-describe", "--max-windows", "2", "--log", str(log)])
    text = log.read_text(encoding="utf-8")
    assert "没有视频流" in text and "全部暂停" in text
    assert calls[0] == calls[1] and calls[2] == calls[3]                  # tried twice, then the third is the run
    assert index[0]["t"] == calls[2][1] and all(item["status"] == "clean" for item in index)
