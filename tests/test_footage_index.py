"""The local footage index: offline, temporary data, fake Bilibili and a fake vision model."""
import json

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
    assert fi.add_seen([seen, described], tiles, topic="历史") == 2
    assert fi.search("清明上河图")[0]["id"] == "BV1new"
    assert fi.search("城门 车马")[0]["moments"] == [5.0]          # the described video kept its moments


def test_merge_takes_missing_or_better_described_videos(data, tmp_path, monkeypatch):
    fi.add(row("BV1a", "航母 编队"), [{"t": 1}])
    other = tmp_path / "other"
    monkeypatch.setattr(paths, "DATA", other)
    fi.add(row("BV1a", "航母 编队"), [{"t": 1, "text": "航母在海上航行", "kind": "实拍"}], described=True)
    fi.add(row("BV1b", "战斗机 起飞"), [{"t": 2, "text": "战斗机从甲板起飞", "kind": "实拍"}], described=True)
    monkeypatch.setattr(paths, "DATA", data / "data")
    assert fi.merge(other / "footage-index.sqlite") == 2
    assert fi.search("海上 航行")[0]["id"] == "BV1a" and fi.search("甲板 起飞")[0]["id"] == "BV1b"
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"not a database")
    with pytest.raises(ValueError):
        fi.merge(bad)


def test_build_searches_describes_and_skips_known_videos(data, monkeypatch):
    from jinggao import express_sources as es, planner
    found = {"码头 实拍": [row("BV1x", "码头 装卸 实拍"), row("BV1game", "原神 码头"), row("BV1short", "码头", duration=8)]}
    class Bili:
        blocked = 0
        def search(self, q, page=1):
            return found.get(q, [])
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
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
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
