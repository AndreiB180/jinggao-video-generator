"""The local footage index: offline, temporary data, fake Bilibili and a fake vision model."""
import json
import re

import pytest
from PIL import Image

from jinggao import config, costs, footage_index as fi, paths


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    return tmp_path


def row(bvid, title, tags="", duration=300.0):
    return {"provider": "bilibili", "id": bvid, "url": f"https://www.bilibili.com/video/{bvid}", "title": title,
            "tags": tags, "description": "", "duration": duration, "author": "up", "typename": "纪录片", "play": 1000,
            "pubdate": 0, "query": ""}


def test_terms_are_chinese_bigrams_and_words():
    assert fi.grams("宋朝汴京 4K航拍") == ["宋朝", "朝汴", "汴京", "航拍", "4k"]
    assert fi.grams("米") == ["米"]
    assert fi.grams("！？") == []


def test_described_moments_find_the_video_and_the_frame(data):
    fi.add(row("BV1dock", "江南古镇纪录片"), [
        {"t": 30, "hash": "1", "text": "古代码头工人搬运粮袋", "kind": "影视剧"},
        {"t": 90, "hash": "2", "text": "小桥流水古镇街道", "kind": "实拍"},
        {"t": 150, "hash": "3", "text": "主持人在演播室讲解", "kind": "主持人出镜"}], topic="历史", described=True)
    fi.add(row("BV1title", "古代码头的一天"), [{"t": 10, "hash": "4"}])
    hits = fi.search("码头 搬运 粮食")
    assert [h["id"] for h in hits][:2] == ["BV1dock", "BV1title"]
    assert hits[0]["moments"][0] == 30 and hits[0]["indexed"] and hits[0]["provider"] == "bilibili"
    assert hits[0]["avoid"] == [150.0]                            # the presenter frame is to be kept out
    assert fi.search("演播室 主持人") == []                       # unusable frames are not searchable
    assert fi.search("码头", exclude={"bilibili:BV1dock"})[0]["id"] == "BV1title"
    stats = fi.stats()
    assert stats["sources"] == 2 and stats["described"] == 1 and stats["described_moments"] == 3


def test_presenter_led_videos_are_kept_out_of_search(data):
    fi.add(row("BV1host", "宋朝 讲解"), [{"t": t, "text": "主持人讲宋朝", "kind": "主持人出镜"} for t in (10, 20, 30)]
           + [{"t": 40, "text": "宋朝宫殿", "kind": "实拍"}], described=True)
    assert fi.search("宋朝 宫殿") == []


def test_production_finds_grow_the_index_without_replacing_descriptions(data):
    described = row("BV1old", "汴京 繁华")
    fi.add(described, [{"t": 5, "text": "汴京城门车马", "kind": "实拍"}], described=True)
    seen = row("BV1new", "清明上河图 动态")
    tiles = [{"source": seen, "t": 12.0, "hash": 0xABC}, {"source": described, "t": 7.0, "hash": 1},
             {"source": seen | {"indexed": True}, "t": 3.0, "hash": 2}]
    assert fi.add_seen([seen, described], tiles, topic="历史") == 1           # only the video not yet indexed
    assert fi.search("清明上河图")[0]["id"] == "BV1new"
    assert fi.search("城门 车马")[0]["moments"] == [5.0]          # the described video kept its moments


def test_merge_takes_missing_or_better_described_videos(data, tmp_path, monkeypatch):
    fi.add(row("BV1aaaaaaaaa", "航母 编队"), [{"t": 1}])
    other = tmp_path / "other"
    monkeypatch.setattr(paths, "DATA", other)
    fi.add(row("BV1aaaaaaaaa", "航母 编队"), [{"t": 1, "text": "航母在海上航行", "kind": "实拍"}], described=True)
    fi.add(row("BV1bbbbbbbbb", "战斗机 起飞"), [{"t": 2, "text": "战斗机从甲板起飞", "kind": "实拍"}], described=True)
    monkeypatch.setattr(paths, "DATA", data / "data")
    assert fi.merge(other / "footage-index.sqlite") == 2
    assert fi.search("海上 航行")[0]["id"] == "BV1aaaaaaaaa" and fi.search("甲板 起飞")[0]["id"] == "BV1bbbbbbbbb"
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"not a database")
    with pytest.raises(ValueError):
        fi.merge(bad)


def test_build_searches_describes_and_skips_known_videos(data, monkeypatch):
    from jinggao import express_sources as es, planner
    found = {"码头 实拍": [row("BV1x", "码头 装卸 实拍"), row("BV1game", "原神 码头"), row("BV1short", "码头", duration=8)]}
    class Bili:
        blocked = search_refused = shots_blocked = 0
        offline_search = False
        counts = __import__("threading").Lock()
        def search(self, q, page=1):
            return found.get(q, [])
        def cached(self, q, page=1):
            return False
        def close(self):
            pass
    monkeypatch.setattr(es, "Bilibili", Bili)
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (40 + t, 90, 120)), "hash": t * 7919} for t in range(1, 5)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    calls = []
    def chat(system, content, **kw):
        calls.append(kw["task"])
        assert kw["vision"] and "联系表" in system
        return {"1": ["码头工人装卸货物", "实拍"], "2": ["集装箱吊车", "实拍"], "3": ["字幕页", "图表文字"]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    report = fi.build([("城市", "码头 实拍")], lambda: None, lambda *a: None, widen=False)
    assert report["found"] == 1 and report["indexed"] == 1 and report["described_frames"] == 4
    assert calls == ["footage_index", "footage_index"]      # frame 4 was skipped, so it is asked about once more
    assert fi.search("吊车 集装箱")[0]["moments"] == [2.0]
    assert fi.build([("城市", "码头 实拍")], lambda: None, lambda *a: None, widen=False)["indexed"] == 0   # already described


def test_library_building_has_its_own_daily_cap(data, monkeypatch):
    import httpx
    cfg = config.DEFAULTS | {"model_api_key": "k", "model_job_budget_cny": "0.00", "index_daily_budget_cny": "0.01"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    payload = {"model": "qwen3.7-flash", "max_tokens": 200, "messages": [{"role": "user", "content": "x"}]}
    with costs.scope("library", "job", "index"):
        with costs.request(cfg, payload, "footage_index") as charge:     # the model job cap of 0 does not apply
            charge.mark_sent()
            charge.response(httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 50}}))
        cfg["index_daily_budget_cny"] = "0.00"
        with pytest.raises(costs.BudgetExceeded, match="今日素材库"):
            with costs.request(cfg, payload, "footage_index"):
                pass
    report = costs.summary("library")
    assert report["index"]["daily_cny"] > 0 and report["model"]["daily_cny"] == 0


def test_matcher_prefers_indexed_videos_and_their_matching_frames(data, monkeypatch):
    from jinggao import express_match as em, planner, store
    paths.prepare()
    store.init()
    fi.add(row("BV1lib", "宋代 漕运 纪录片"), [{"t": 200.0, "text": "运粮船队在运河上航行", "kind": "影视剧"}],
           described=True)
    shot = planner.new_shot("粮船一艘接一艘，日夜兼程。") | {"duration": 3.0}
    project = store.new_project("t", "x")
    project = store.mutate(project["id"], lambda p: p.update(shots=[shot], plan_info={"groups": [
        {"id": "G1", "zh": ["运河 粮船", "漕运 船队"], "en": [], "person": "", "shots": [shot["id"]]}]}))
    m = em.Matcher(project, project["shots"], lambda: None, lambda *a: None)
    monkeypatch.setattr(m.bili, "search", lambda q, page=1: [])
    m.pixa = None
    groups = m.groups()
    chosen = m.candidates(groups, m.search(groups))
    picked = chosen["G1"]
    assert picked and picked[0]["id"] == "BV1lib" and picked[0]["moments"] == [200.0]
    previews = [{"t": float(t), "sheet": 0, "cell": 0} for t in range(0, 600, 10)]
    monkeypatch.setattr(m.bili, "previews", lambda bvid, duration: previews)
    import random
    def tile(bili, p):   # bright noise, different for every time
        rng = random.Random(p["t"])
        return Image.frombytes("RGB", (160, 90), bytes(rng.randrange(90, 255) for _ in range(160 * 90 * 3)))
    monkeypatch.setattr(em.es, "bilibili_tile", tile)
    monkeypatch.setattr(em.express_clean, "available", lambda: False)
    tiles = m.tiles_for(picked[0], 4)
    assert tiles[0]["t"] == 200.0
    tiles = m.tiles_for(picked[0] | {"avoid": [200.0, 450.0]}, 6)
    assert tiles and not any(abs(t["t"] - 200) < 6 or abs(t["t"] - 450) < 6 for t in tiles)


def test_tags_shared_by_a_topics_results_become_new_searches():
    rows = [row(f"BV{n}", "x", tags="漕运,运河,纪录片,宋朝,古代运输") for n in range(4)] + [row("BV9", "y", tags="漕运,游戏")]
    more = fi.expand({"历史": rows}, {"运河"})
    assert ("历史", "漕运") in more and ("历史", "古代运输") in more
    assert not any(q in ("运河", "纪录片", "游戏") for _, q in more)       # used, generic or skipped words
    assert [fi.frame_count(d) for d in (60, 600, 3600)] == [8, 12, 16]


def test_index_building_runs_in_its_own_worker_lane(data):
    from jinggao import store
    paths.prepare()
    store.init()
    film = store.new_project("片子")
    library = store.new_project("素材库采集")
    store.enqueue(library["id"], "footage_index", {"packs": ["动物"]})
    store.enqueue(film["id"], "produce", {})
    assert store.claim("main")["kind"] == "produce"
    assert store.claim("main") is None
    assert store.claim("background")["kind"] == "footage_index"


def test_account_errors_stop_the_build_and_rate_limits_wait(data, monkeypatch):
    from jinggao import planner
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(30)]
    calls = []
    def limited(system, content, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise planner.ModelUnavailable("模型服务限流或额度受限，请稍后再试（HTTP 429）")
        return {str(i + 1): ["画面", "实拍"] for i in range(24)}, {}
    monkeypatch.setattr(planner, "chat", limited)
    monkeypatch.setattr(fi, "_sleep", lambda seconds, check: None)
    report = {"indexed": 0, "described_frames": 0, "failed": 0}
    fi._build_chunk(None, [], lambda: None, lambda *a: None, report, True)
    assert fi._describe(frames[:24], lambda: None)[0] == ("画面", "实拍") and len(calls) == 2
    def broke(system, content, **kw):
        raise planner.ModelUnavailable("模型账户额度不足，请充值（HTTP 402）")
    monkeypatch.setattr(planner, "chat", broke)
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    with pytest.raises(planner.ModelUnavailable):
        fi._build_chunk(None, [("动物", row("BV1z", "狮子"))], lambda: None, lambda *a: None, report, True)
    assert fi.stats()["sources"] == 0                         # nothing half-described was saved
    def flaky(system, content, **kw):
        raise ValueError("模型服务返回 HTTP 400；请检查地址、模型名、额度和密钥")
    monkeypatch.setattr(planner, "chat", flaky)
    many = [("动物", row(f"BV{n}", "狮子")) for n in range(12)]
    with pytest.raises(RuntimeError, match="连续 8 张"):
        fi._build_chunk(None, many, lambda: None, lambda *a: None, report, True)


def test_seed_ships_the_described_index_and_installs_once(data, tmp_path, monkeypatch):
    for n in range(30):
        fi.add(row(f"BV1seed{n:05d}", f"码头 视频 {n}"),
               [{"t": float(t), "hash": "a", "text": f"码头工人装卸 {n}-{t}", "kind": "实拍"} for t in range(5)],
               topic="城市", described=True)
    fi.add(row("BV1bare000000", "只有标题"), [{"t": 1.0}])                    # not described: not shipped
    seed = tmp_path / "seed"
    parts = fi.export_seed(seed, part_bytes=600)
    assert len(parts) > 1 and all(p.name.startswith("part-") for p in parts)
    client = tmp_path / "client"
    monkeypatch.setattr(paths, "DATA", client)
    fi.add(row("BV1seed00003", "本机已有"), [{"t": 9.0, "text": "本机的描述", "kind": "实拍"}], described=True)
    assert fi.install_seed(seed) == 29                                           # its own described copy stays
    assert fi.search("本机的描述")[0]["id"] == "BV1seed00003"
    assert fi.search("码头工人装卸")[0]["indexed"] and fi.stats()["sources"] == 30
    assert fi.install_seed(seed) == 0                                            # once per seed version
    (seed / "part-99.jsonl.gz").write_bytes(__import__("gzip").compress(
        json.dumps({"row": {"provider": "bilibili", "id": "../../etc", "title": "x"}, "moments": []}).encode() + b"\n"))
    assert fi.install_seed(seed) == 0                                            # a malformed id is skipped
    assert fi.install_seed(tmp_path / "none") == 0


def test_reseeding_after_a_reset_and_damaged_parts(data, tmp_path, monkeypatch):
    for n in range(6):
        fi.add(row(f"BV1part{n:05d}", f"航母 {n}"), [{"t": 1.0, "text": f"航母甲板 {n}", "kind": "实拍"}], described=True)
    seed = tmp_path / "seed"
    fi.export_seed(seed, part_bytes=150)
    assert len(list(seed.glob("part-*.jsonl.gz"))) >= 2 and not list(seed.glob("*.tmp"))
    monkeypatch.setattr(paths, "DATA", tmp_path / "client")
    assert fi.install_seed(seed) == 6
    (tmp_path / "client" / "footage-index.sqlite").unlink()          # a reset index is seeded again
    for extra in (tmp_path / "client").glob("footage-index.sqlite-*"):
        extra.unlink()
    fi._ready.clear()
    assert fi.install_seed(seed) == 6
    first = sorted(seed.glob("part-*.jsonl.gz"))[0]
    first.write_bytes(first.read_bytes()[:20])                          # cut short: that part only
    monkeypatch.setattr(paths, "DATA", tmp_path / "other")
    with pytest.raises(ValueError, match="损坏"):
        fi.install_seed(seed)
    assert 0 < fi.stats()["described"] < 6


def test_readding_a_video_replaces_its_terms_and_installs_stay_fast(data, tmp_path, monkeypatch):
    import time as clock
    fi.add(row("BV1again0000", "片子"), [{"t": 3.0, "text": "码头吊车", "kind": "实拍"}], described=True)
    fi.add(row("BV1again0000", "片子"), [{"t": 3.0, "text": "森林瀑布", "kind": "实拍"}], described=True)
    assert fi.search("码头吊车") == [] and fi.search("森林瀑布")[0]["id"] == "BV1again0000"
    with fi._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM terms").fetchone()[0] == con.execute(
            "SELECT COUNT(*) FROM term_rows").fetchone()[0] == 2
    seed = tmp_path / "seed"
    seed.mkdir()
    lines = [json.dumps({"row": row(f"BV1{n:09d}", f"视频 {n}"), "topic": "城市",
                         "moments": [{"t": float(t), "hash": "", "text": f"城市 街道 车流 {n} {t}", "kind": "实拍"}
                                     for t in range(10)]}, ensure_ascii=False) for n in range(3000)]
    (seed / "part-01.jsonl.gz").write_bytes(__import__("gzip").compress("\n".join(lines).encode()))
    monkeypatch.setattr(paths, "DATA", tmp_path / "client")
    began = clock.monotonic()
    assert fi.install_seed(seed) == 3000
    assert clock.monotonic() - began < 30
    assert fi.search("城市 街道")[0]["indexed"]


def test_described_sheets_are_kept_when_a_later_sheet_stops_the_build(data, monkeypatch):
    from jinggao import planner
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(12)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    monkeypatch.setattr(fi, "DESCRIBE_WORKERS", 1)
    calls = []
    def chat(system, content, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise planner.ModelUnavailable("模型账户额度不足，请充值（HTTP 402）")
        return {str(i + 1): ["街道", "实拍"] for i in range(24)}, {}
    monkeypatch.setattr(planner, "chat", chat)
    videos = [("城市", row(f"BV1keep{n:05d}", "街道")) for n in range(4)]      # two videos to a sheet
    report = {"indexed": 0, "described_frames": 0, "failed": 0}
    with pytest.raises(planner.ModelUnavailable):
        fi._build_chunk(None, videos, lambda: None, lambda *a: None, report, True)
    assert fi.stats()["described"] == 2 and report["described_frames"] == 24


def test_videos_whose_sheets_keep_failing_are_left_out(data, monkeypatch):
    from jinggao import planner, express_sources as es
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(4)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    def refused(system, content, **kw):
        raise ValueError("模型服务返回 HTTP 400；请检查地址、模型名、额度和密钥")
    monkeypatch.setattr(planner, "chat", refused)
    class Bili:
        blocked = search_refused = shots_blocked = 0
        offline_search = False
        counts = __import__("threading").Lock()
        def search(self, q, page=1):
            return [row("BV1stubborn0", "码头")]
        def cached(self, q, page=1):
            return False
        def close(self):
            pass
    monkeypatch.setattr(es, "Bilibili", Bili)
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
    for _ in range(2):
        assert fi.build([("城市", "码头")], lambda: None, lambda *a: None, widen=0)["indexed"] == 0
    assert fi.stats()["sources"] == 1 and fi.stats()["described"] == 0
    calls = []
    monkeypatch.setattr(planner, "chat", lambda *a, **k: calls.append(1) or ({}, {}))
    fi.build([("城市", "码头")], lambda: None, lambda *a: None, widen=0)
    assert calls == []                                                   # two failed tries: not asked again


def test_refused_searches_do_not_block_previews_or_streams(data, monkeypatch):
    import httpx
    from jinggao import express_sources as es
    def handler(request):
        if "search/type" in request.url.path:
            return httpx.Response(200, json={"code": -412, "message": "risk"})
        if "videoshot" in request.url.path:
            return httpx.Response(200, json={"code": 0, "data": {"image": ["//i0.hdslb.com/x.jpg"], "img_x_len": 10,
                                                                  "img_y_len": 10, "img_x_size": 160, "img_y_size": 90}})
        return httpx.Response(200, text="ok")
    original = httpx.Client
    monkeypatch.setattr(es.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(es.time, "sleep", lambda s: None)
    monkeypatch.setattr(es, "saved_cookies", lambda: {})
    bili = es.Bilibili()
    for n in range(4):
        assert bili.search(f"码头 {n}") == []
    assert bili.search_refused == 4 and bili.blocked == 0
    assert bili.previews("BV1abcdefghi", 100.0)                        # previews still work
    bili.close()


def test_answers_survive_a_failed_follow_up_and_cut_off_answers_are_split(data, monkeypatch):
    from jinggao import planner
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(8)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    calls = []
    def partial(system, content, **kw):
        calls.append(1)
        if len(calls) == 1:
            return {str(i + 1): ["码头", "实拍"] for i in range(7)}, {}        # frame 8 skipped
        raise ValueError("模型服务返回 HTTP 400；请检查地址、模型名、额度和密钥")
    monkeypatch.setattr(planner, "chat", partial)
    report = {"indexed": 0, "described_frames": 0, "failed": 0}
    fi._build_chunk(None, [("城市", row("BV1partial00", "码头"))], lambda: None, lambda *a: None, report, True)
    assert fi.stats()["described"] == 1 and report["described_frames"] == 7
    sizes = []
    def cut(system, content, **kw):
        n = int(re.search(r"共 (\d+) 格", content[0]["text"]).group(1))
        sizes.append(n)
        if n > 4:
            raise planner.ModelResponseError("模型响应达到输出长度上限，返回内容不完整，任务已停止")
        return {str(i + 1): ["街道", "实拍"] for i in range(n)}, {}
    monkeypatch.setattr(planner, "chat", cut)
    assert len(fi._describe(frames, lambda: None)) == 8 and sizes == [8, 4, 4]


def test_service_wide_failures_do_not_count_against_videos(data, monkeypatch):
    from jinggao import planner
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(4)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    monkeypatch.setattr(fi, "_sleep", lambda seconds, check: None)
    def limited(system, content, **kw):
        raise planner.ModelUnavailable("模型服务限流或额度受限，请稍后再试（HTTP 429）")
    monkeypatch.setattr(planner, "chat", limited)
    report = {"indexed": 0, "described_frames": 0, "failed": 0}
    fi._build_chunk(None, [("城市", row(f"BV1limit{n:04d}", "码头")) for n in range(3)], lambda: None,
                    lambda *a: None, report, True)
    with fi._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM sources WHERE tries>0").fetchone()[0] == 0


def test_refused_searches_fall_back_to_todays_cached_results(data, monkeypatch):
    from jinggao import express_sources as es, planner
    calls = {"live": 0}
    class Bili:
        blocked = search_refused = shots_blocked = 0
        offline_search = False
        counts = __import__("threading").Lock()
        def search(self, q, page=1):
            if q == "已缓存":
                return [row("BV1cached000", "码头")]
            if self.offline_search:
                return []
            calls["live"] += 1
            self.search_refused += 1
            return []
        def cached(self, q, page=1):
            return False
        def close(self):
            pass
    monkeypatch.setattr(es, "Bilibili", Bili)
    monkeypatch.setattr(fi, "_sleep", lambda seconds, check: None)
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in range(4)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    monkeypatch.setattr(planner, "chat", lambda *a, **k: ({str(i + 1): ["码头", "实拍"] for i in range(4)}, {}))
    queries = [("城市", f"新词{n}") for n in range(100)] + [("城市", "已缓存")]
    report = fi.build(queries, lambda: None, lambda *a: None, widen=0)
    assert report["search_offline"] and report["indexed"] == 1 and calls["live"] < 70


def test_a_local_model_describes_one_video_per_request(data, monkeypatch):
    from jinggao import footage_local
    answer = ('<think></think>{"1":["夫妻在客厅激烈争吵","影视剧","吵架 争执 夫妻矛盾"],'
              '"图2":["男主播在演播室播报","主持人出镜","新闻 主播"],"3":"", "9":["多余","实拍"]}')
    parsed = footage_local.parse(answer, 3)
    assert parsed == {0: ("夫妻在客厅激烈争吵｜吵架 争执 夫妻矛盾", "影视剧"), 1: ("男主播在演播室播报｜新闻 主播", "主持人出镜")}
    assert footage_local.parse("说不清", 3) == {}
    frames = [{"t": float(t), "image": Image.new("RGB", (160, 90), (90, 120, 150)), "hash": t} for t in (10, 60, 120)]
    monkeypatch.setattr(fi, "_frames", lambda bili, r, count=10: frames)
    seen = []
    def local(r, images):
        seen.append((r["id"], len(images)))
        if r["id"] == "BV1bad":
            raise ValueError("本地模型没有给出可用的描述")
        return footage_local.parse(answer, len(images))
    report = {"indexed": 0, "described_frames": 0, "failed": 0}
    fi._build_chunk(None, [("家庭", row("BV1fight", "电视剧 吵架")), ("家庭", row("BV1bad", "x"))],
                    lambda: None, lambda *a: None, report, True, local)
    assert sorted(seen) == [("BV1bad", 3), ("BV1fight", 3)] and report["indexed"] == 1
    hit = fi.search("夫妻 争执")[0]
    assert hit["id"] == "BV1fight" and hit["moments"] == [10.0] and hit["avoid"] == [60.0]
    with fi._conn() as con:
        assert con.execute("SELECT tries, described FROM sources WHERE id='BV1bad'").fetchone()[:] == (1, 0)
    def down(r, images):
        raise OSError("connection refused")
    many = [("家庭", row(f"BV1down{n}", "x")) for n in range(9)]
    with pytest.raises(RuntimeError, match="连续 8 个视频"):
        fi._build_chunk(None, many, lambda: None, lambda *a: None, report, True, down)


def test_a_new_seed_brings_frame_by_frame_descriptions_of_known_videos(data, tmp_path, monkeypatch):
    fi.add(row("BV1rich00001", "夫妻 吵架"), [{"t": 5.0, "text": "夫妻在客厅争吵｜吵架 争执 夫妻矛盾", "kind": "影视剧"}],
           described=True)
    fi.add(row("BV1rich00002", "老街"), [{"t": 5.0, "text": "老街｜胡同 街巷", "kind": "实拍"}], described=True)
    seed = tmp_path / "seed"
    fi.export_seed(seed)
    monkeypatch.setattr(paths, "DATA", tmp_path / "client")
    fi.add(row("BV1rich00001", "夫妻 吵架"), [{"t": 5.0, "text": "两人在屋里说话", "kind": "影视剧"}], described=True)
    fi.add(row("BV1rich00002", "老街"), [{"t": 5.0, "text": "本机的描述｜本机", "kind": "实拍"}], described=True)
    fi.mark_used(["bilibili:BV1rich00001"])
    assert fi.install_seed(seed) == 1
    hit = fi.search("夫妻矛盾")[0]
    assert hit["id"] == "BV1rich00001" and hit["index_used"] == 1      # better lines, its use count kept
    assert fi.search("本机")[0]["id"] == "BV1rich00002"                 # already frame by frame: stays
