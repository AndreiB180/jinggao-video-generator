"""11-04 client feedback: shots seen again, 24-30 s runs of one fixed camera, 1-2 s flashes of another scene, and
related pictures instead of a repeat (offline; the cut finder runs FFmpeg on generated clips)."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from jinggao import express_match, express_sources as es, paths, planner, sources, store


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    return tmp_path


def _matcher(shots):
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=shots))
    return express_match.Matcher(project, project["shots"], lambda: None, lambda *a: None)


def _look(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:16], 16)


# ------------------------------------------------------------------ flashes at a window's ends

def test_a_window_moves_off_a_cut_near_its_start_or_end():
    # Nothing to move: no cut, or a cut leaving two long pieces.
    assert es.steady_shift([], 10.0, 6.0, 7.5, 12.5) == 0.0
    assert es.steady_shift([13.0], 10.0, 6.0, 7.5, 12.5) == 0.0
    # Half a second of the previous scene at the start: start just after the cut.
    assert es.steady_shift([10.5], 10.0, 6.0, 7.5, 12.5) == 0.7
    # 0.8 s of the next scene at the end: end before it (the smallest move that works).
    assert es.steady_shift([15.2], 10.0, 6.0, 7.5, 12.5) == -1.0
    # A frame or two of another scene right at the start counts too.
    assert es.steady_shift([10.1], 10.0, 6.0, 7.5, 12.5) == 0.3
    # Two cuts a second apart in the middle: no move within reach gets rid of the one-second piece (flagged instead).
    assert es.steady_shift([12.0, 13.0], 10.0, 6.0, 7.5, 12.5) == 0.0
    assert es.shortest_piece([12.0, 13.0], 10.0, 6.0) == 1.0
    # One cut 2.5 s in and 1.2 s of the next scene at the end: a later start ends past the second cut.
    assert es.shortest_piece([12.5, 14.8], 10.0 + es.steady_shift([12.5, 14.8], 10.0, 6.0, 7.5, 12.5), 6.0) >= 2.0
    # The footage fetched around the window bounds the move.
    assert es.steady_shift([10.5], 10.0, 6.0, 10.0, 10.3) == 0.0
    # A source cutting every second: the move that leaves the longest shortest piece, or none.
    choppy = [10.0 + k for k in range(-3, 10)]
    assert abs(es.steady_shift(choppy, 10.0, 6.0, 7.5, 12.5)) <= es.STEADY_LEAD


def _clip(path: Path, parts):
    ffmpeg = shutil.which("ffmpeg") or paths.binary("ffmpeg")
    inputs, labels = [], []
    for n, (source, seconds) in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", f"{source}=size=320x180:rate=25:duration={seconds}"]
        labels.append(f"[{n}:v]")
    graph = "".join(labels) + f"concat=n={len(parts)}:v=1:a=0[v]"
    subprocess.run([ffmpeg, "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]",
                    "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
    return str(path)


def test_the_cut_finder_sees_a_hard_cut_and_not_motion(tmp_path):
    try:
        cut = _clip(tmp_path / "cut.mp4", [("testsrc2", 3), ("smptehdbars", 3)])
        moving = _clip(tmp_path / "moving.mp4", [("testsrc2", 6)])
    except (OSError, RuntimeError, subprocess.SubprocessError):
        pytest.skip("no FFmpeg")
    found = es.scene_cuts(cut, 0.0, 6.0)
    assert len(found) == 1 and abs(found[0] - 3.0) <= 0.15
    # From the middle of the file: times stay the file's own.
    assert [round(t) for t in es.scene_cuts(cut, 1.5, 3.0)] == [3]
    assert es.scene_cuts(moving, 0.0, 6.0) == []


def test_dash_windows_fetch_around_and_report_the_move(monkeypatch, tmp_path):
    asked, cut_at = [], []
    refs = [(0, 10, 50.0 + 2 * i, 2.0) for i in range(12)]          # 2-second fragments from 50 s to 74 s
    def fetch(stream, url, start, duration, local, check):
        asked.append((round(start, 2), round(duration, 2)))
        local.write_bytes(b"x")
        return [r for r in refs if r[2] + r[3] > start - 0.05 and r[2] < start + duration + 0.05]
    monkeypatch.setattr(es, "_fetch_segments", fetch)
    monkeypatch.setattr(es, "scene_cuts", lambda path, start, length: [9.6])   # in file time
    monkeypatch.setattr(es, "cut_window", lambda local, start, duration, out, **kw: cut_at.append(round(start, 2)) or "cpu")
    stream = {"urls": ["u"], "headers": {}}
    steady = {}
    assert es.dash_window(stream, 60.0, 6.0, tmp_path / "o.mp4", steady=steady) == "cpu"
    assert asked == [(57.5, 11.0)]                       # 2.5 s more on each side
    # The file starts at 56 s (the fragment holding 57.5 s); its cut at 9.6 s is 65.6 s, 0.4 s before the end.
    assert steady["shift"] == -0.6 and cut_at == [round(60.0 - 0.6 - 56.0, 2)] and steady["shortest"] == 6.0
    # Without steady (the library's clean check) the window is fetched and cut as asked.
    asked.clear()
    cut_at.clear()
    es.dash_window(stream, 60.0, 6.0, tmp_path / "o.mp4")
    assert asked == [(60.0, 6.0)] and cut_at == [2.0]   # the fragment holding 60 s starts at 58 s


# ------------------------------------------------------------------ repeats and long runs

def test_lines_naming_a_person_keep_the_repeat_limits_for_footage_not_of_them(data):
    shots = [planner.new_shot(f"赵长鹏第{i}句。") | {"duration": 5.0, "persons": [{"name": "赵长鹏", "aliases": []}]}
             for i in range(4)]
    m = _matcher(shots)
    a, b, c, d = m.shots
    bridge = {"provider": "bilibili", "id": "BV_bridge", "duration": 600.0, "title": "海峡 大桥 航拍"}
    fresh = {"provider": "bilibili", "id": "BV_fresh", "duration": 600.0, "title": "城市 街景"}
    fresh2 = {"provider": "bilibili", "id": "BV_fresh2", "duration": 600.0, "title": "办公室"}
    m.looks[m.position[a["id"]]] = _look("bridge")
    m.placed[m.position[a["id"]]] = ("bilibili", "BV_bridge")
    # The same picture further on, and the same video right next to it: refused although the line names a person.
    options = {c["id"]: [({"source": bridge, "t": 300.0, "hash": _look("bridge") ^ 0b11}, "ctx"),
                         ({"source": fresh, "t": 100.0, "hash": _look("street")}, "ctx"),
                         ({"source": fresh2, "t": 100.0, "hash": _look("office")}, "ctx")],
               b["id"]: [({"source": bridge, "t": 200.0, "hash": _look("other")}, "ctx"),
                         ({"source": fresh, "t": 300.0, "hash": _look("street2")}, "ctx")]}
    got = m.allocate(options, {("bilibili", "BV_bridge"): [(10.0, 15.3)]}, {("bilibili", "BV_bridge"): 1}, [b, c])
    assert got[b["id"]]["tile"]["source"]["id"] == "BV_fresh"
    assert got[c["id"]]["tile"]["source"]["id"] == "BV_fresh2"   # not the bridge again, not b's video beside it
    # The person's own footage may show more of them (other moments), but never in neighbouring shots.
    own = {"provider": "bilibili", "id": "BV_own", "duration": 3600.0, "title": "赵长鹏 专访 全程"}
    m2 = _matcher(shots)
    options = {s["id"]: [({"source": own, "t": 100.0 + 300 * i, "hash": _look(f"own{i}")}, "pick")]
               for i, s in enumerate(m2.shots)}
    got = m2.allocate(options, {}, {}, m2.shots)
    used = sorted(m2.position[sid] for sid in got)
    assert len(used) >= 2 and all(y - x > 1 for x, y in zip(used, used[1:]))


def test_the_film_check_replaces_any_second_look_and_neighbouring_windows_of_one_video(data, monkeypatch, tmp_path):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 6.0} for i in range(6)]
    shots[4]["persons"] = shots[5]["persons"] = [{"name": "赵长鹏", "aliases": []}]
    m = _matcher(shots)
    ids = [s["id"] for s in m.shots]
    strait = {"provider": "bilibili", "id": "BV_strait", "pool": True, "duration": 3600.0, "title": "台海 直播"}
    def clip(n, source, lane):
        return {"tile": {"source": source, "t": 30.0 * n}, "lane": lane, "job": f"j{n}"}
    # 0-1-2: three neighbouring windows of one fixed camera on the strait (one 18 s take on screen).
    # 4: a generic clip on a line naming a person, the same look as 0 (it was let off before).
    # 5: the person's own interview.
    other = {"provider": "bilibili", "id": "BV_x"}
    final = {ids[0]: clip(0, strait, "sweep"), ids[1]: clip(1, strait, "sweep"), ids[2]: clip(2, strait, "sweep"),
             ids[3]: clip(3, other, "pick"),
             ids[4]: clip(4, {"provider": "bilibili", "id": "BV_sea", "title": "海边"}, "ctx"),
             ids[5]: clip(5, {"provider": "bilibili", "id": "BV_cz", "title": "赵长鹏 专访"}, "pick")}
    files = {sid: tmp_path / f"{sid}.mp4" for sid in ids}
    for n in range(6):
        m.clean[f"j{n}"] = {"crop": [0, 0, 1, 1], "zoom": 1.0}
    sea = [0b1010] * 3
    looks = {ids[0]: sea, ids[1]: [_look("w1")] * 3, ids[2]: [_look("w2")] * 3, ids[3]: [_look("x")] * 3,
             ids[4]: sea, ids[5]: [_look("cz")] * 3}
    monkeypatch.setattr(express_match.express_clean, "available", lambda: True)
    monkeypatch.setattr(express_match.Matcher, "looks_of", staticmethod(lambda path, d: looks[Path(path).stem]))
    m.plan = {"domain": "台海局势", "pool": ["台海 直播"]}
    redo = m.audit(final, files)
    assert redo.get(ids[1]) == "和上一镜是同一个视频" and redo.get(ids[2]) == "和上一镜是同一个视频"
    assert ids[4] in redo and ids[0] not in redo          # the second look goes, person line or not
    assert ids[3] not in redo and ids[5] not in redo
    # One game's replay: its neighbouring plays stay.
    m.plan = {"domain": "NBA篮球", "pool": ["勇士 森林狼 全场回放"]}
    m.looked.clear()
    redo = m.audit(final, files)
    assert ids[1] not in redo and ids[2] not in redo


def test_the_pool_walk_skips_the_line_next_to_the_same_video(data):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 6.0} for i in range(4)]
    m = _matcher(shots)
    pool = {"provider": "bilibili", "id": "BV_pool", "pool": True, "duration": 3600.0, "title": "台海 纪录片"}
    tiles = {"BV_pool": [{"source": pool, "t": float(t)} for t in range(60, 3500, 120)]}
    streams = {"BV_pool": {"available": True}}
    m.plan = {"domain": "台海局势", "pool": ["台海 纪录片"]}
    jobs = m.pool_sweep(m.shots, {}, {}, streams, tiles)
    assert sorted(m.position[sid] for _, sid, _ in jobs) == [0, 2]
    m.plan = {"domain": "NBA篮球", "pool": ["全场回放"]}
    jobs = m.pool_sweep(m.shots, {}, {}, streams, tiles)
    assert sorted(m.position[sid] for _, sid, _ in jobs) == [0, 1, 2, 3]


# ------------------------------------------------------------------ your own picture source

def test_your_picture_source_comes_in_as_rows_and_stays_out_when_empty(monkeypatch):
    from jinggao import picture_plugin
    sources._custom_cache.clear()
    assert sources.custom_images("台海") == []
    sources._custom_cache.clear()
    monkeypatch.setattr(picture_plugin, "search", lambda q: [
        {"url": "https://img.example.com/a.jpg", "title": "台海&amp;大桥", "width": 1600, "height": 900,
         "page": "https://news.example.com/1", "headers": {"Referer": "https://image.example.com/"}},
        {"url": "https://img.example.com/small.jpg", "width": 200, "height": 120},
        {"url": "javascript:alert(1)"}, "not a dict",
        {"url": "https://img.example.com/b.jpg", "fallback": "ftp://x"}])
    rows = sources.custom_images("台海")
    assert [r["url"] for r in rows] == ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"]
    assert rows[0]["title"] == "台海&大桥" and rows[0]["source_url"] == "https://news.example.com/1"
    assert rows[0]["headers"] == {"Referer": "https://image.example.com/"} and rows[1]["fallback"] == ""
    sources._custom_cache.clear()
    monkeypatch.setattr(picture_plugin, "search", lambda q: (_ for _ in ()).throw(RuntimeError("blocked")))
    with pytest.raises(RuntimeError):
        sources.custom_images("台海")
    assert sources.custom_images("台海") == []     # a search that raised is not asked again for a while
    sources._custom_cache.clear()


# ------------------------------------------------------------------ pictures: judged once a film, within a budget

def test_pictures_are_judged_once_a_film_and_searched_within_its_budget(data, monkeypatch):
    import io

    import numpy as np
    from PIL import Image

    from jinggao import picture_plugin, screen_text
    shots = [planner.new_shot("台海局势紧张。") | {"duration": 6.0}, planner.new_shot("军舰在附近巡航。") | {"duration": 6.0}]
    m = _matcher(shots)
    m.plan = {"groups": [{"id": "G1", "shots": [s["id"] for s in m.shots], "zh": ["台海 军舰"]}]}
    rng = np.random.default_rng(7)
    pictures = {}
    for n in range(12):
        buffer = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)).resize((1280, 720)).save(buffer, "JPEG")
        pictures[f"https://img.example.com/{n}.jpg"] = buffer.getvalue()
    monkeypatch.setattr(picture_plugin, "search", lambda q: [{"url": u, "width": 1280, "height": 720} for u in pictures])
    monkeypatch.setattr(screen_text, "installed", lambda: False)
    fetched = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            fetched.append(url)
            return type("R", (), {"status_code": 200, "content": pictures[url]})()
    monkeypatch.setattr(express_match.network, "client", lambda **kw: Client())
    monkeypatch.setattr(m, "chat", lambda system, content, **kw: ({"S1": 1, "S2": 1}, {}))
    first = m.image_fill(m.shots)
    assert len(first) == 2 and len(fetched) == len(set(fetched))
    final = {sid: choice for sid, (choice, _) in first.items()}
    second = m.image_fill(m.shots, final)
    # Nothing downloaded twice; the photos already in the film are not offered again.
    assert len(fetched) == len(set(fetched)) and m.picture_stats.get("known", 0) > 0
    shown = {c["tile"]["hash"] for c in final.values()}
    assert all(choice["tile"]["hash"] not in shown for choice, _ in second.values())
    # The film's picture time is used up: no more searching.
    m.picture_spent = express_match.PICTURE_FILM_SECONDS
    assert m.image_fill(m.shots) == {} and m.picture_stats.get("out_of_budget") == 1


def test_the_final_review_saying_someone_else_is_read_as_a_wrong_person():
    assert express_match.WRONG_PERSON.search("前两张截图为俄罗斯总统普京，口播讲菲律宾防长，明显张冠李戴")
    assert express_match.WRONG_PERSON.search("中间镜头人物非周群飞本人")
    assert not express_match.WRONG_PERSON.search("口播讲2015年手机玻璃，画面是古代城门建筑")


def test_a_plan_cut_off_at_the_output_limit_is_asked_once_more_with_more_room(monkeypatch):
    from jinggao import express_plan, planner
    asked = []
    def chat(system, content, *, max_tokens, timeout, task):
        asked.append(max_tokens)
        if len(asked) == 1:
            raise planner.ModelResponseError("模型响应达到输出长度上限，返回内容不完整，任务已停止")
        return {"rows": []}, {}
    monkeypatch.setattr(planner, "chat", chat)
    shots = [planner.new_shot(f"第{i}句。") for i in range(35)]
    assert express_plan._ask(shots, "") == ({"rows": []}, {})
    assert asked == [5750, 9775]
    asked.clear()
    monkeypatch.setattr(planner, "chat", lambda *a, **k: (_ for _ in ()).throw(planner.ModelResponseError("模型返回空内容，任务已停止")))
    with pytest.raises(planner.ModelResponseError):
        express_plan._ask(shots, "")


def test_your_picture_source_is_asked_once_a_search_and_never_holds_the_film(data, monkeypatch):
    import threading
    import time

    from jinggao import picture_plugin, publisher_videos, screen_text
    sources._custom_cache.clear()
    calls, lock = [], threading.Lock()
    delay = {"s": 0.3}
    def search(q):
        with lock:
            calls.append(q)
        time.sleep(delay["s"])
        return [{"url": f"https://img.example.com/{q}.jpg", "width": 1280, "height": 720}]
    monkeypatch.setattr(picture_plugin, "search", search)
    # Five callers of one search at once: it is asked once, everyone gets the answer.
    got = []
    threads = [threading.Thread(target=lambda: got.append(sources.custom_images("南海"))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["南海"] and len(got) == 5 and all(len(rows) == 1 for rows in got)
    # A film of 12 lines in two groups: the source is asked once for each group's search, not once a line.
    sources._custom_cache.clear()
    calls.clear()
    monkeypatch.setattr(sources, "so360_images", lambda q: [])
    monkeypatch.setattr(sources, "bing_images", lambda q: [])
    monkeypatch.setattr(publisher_videos, "chinanews_images", lambda q: [])
    monkeypatch.setattr(screen_text, "installed", lambda: False)
    monkeypatch.setattr(express_match.network, "client", lambda **kw: (_ for _ in ()).throw(OSError("offline")))
    shots = [planner.new_shot(f"第{i}句讲军舰。") | {"duration": 6.0} for i in range(12)]
    m = _matcher(shots)
    m.plan = {"groups": [{"id": "G1", "shots": [s["id"] for s in m.shots[:6]], "zh": ["南海 军舰"]},
                         {"id": "G2", "shots": [s["id"] for s in m.shots[6:]], "zh": ["海警 执法"]}]}
    m.image_fill(m.shots)
    assert sorted(calls) == ["南海 军舰", "海警 执法"] and m.picture_stats["custom_asked"] == 2
    m.image_fill(m.shots)                      # a later call in the film asks nothing again
    assert len(calls) == 2
    # A source that hangs: the film waits CUSTOM_SECONDS for all its searches together, then goes on without it.
    sources._custom_cache.clear()
    calls.clear()
    delay["s"] = 3.0
    monkeypatch.setattr(express_match, "CUSTOM_SECONDS", 1)
    m2 = _matcher(shots)
    m2.plan = m.plan
    began = time.monotonic()
    m2.image_fill(m2.shots)
    assert time.monotonic() - began < 2.5 and m2.own_off and m2.picture_stats["custom_late"] == 2
    began = time.monotonic()
    m2.image_fill(m2.shots)                    # not asked again in this film
    assert time.monotonic() - began < 1.0 and len(calls) == 2
    time.sleep(2.5)                            # the late answers still land for the next film
    assert sources.custom_images("南海 军舰") and len(calls) == 2
    sources._custom_cache.clear()


def test_a_clip_lent_out_once_is_lent_again_only_when_nothing_else_is_left(data, monkeypatch, tmp_path):
    shots = [planner.new_shot(f"镜头{i}。") | {"duration": 5.0} for i in range(8)]
    m = _matcher(shots)
    ids = [s["id"] for s in m.shots]
    monkeypatch.setattr(express_match.es, "probe_duration", lambda path: 9.0)
    def clip(video):
        return {"tile": {"source": {"provider": "bilibili", "id": video}, "t": 10.0}, "lane": "pick", "qa": "pass"}
    final = {ids[2]: clip("BVnear"), ids[0]: clip("BVfar"),
             ids[6]: clip("BVnear") | {"lane": "reuse", "repeat_of": ids[2]}}   # BVnear's clip is lent out already
    files = {sid: tmp_path / f"{sid}.mp4" for sid in final}
    shot = m.shots[4]
    assert m.borrow(shot, final, files) == ids[0]                  # further away, but not lent yet
    assert m.borrow(shot, final, files, again=True) == ids[2]      # nothing else: the nearest again
    del final[ids[0]]
    assert m.borrow(shot, final, files) is None
