"""Short-word library search (find), its synonym table, and the clean-up checks that travel with the seed."""
import gzip
import json
import sqlite3
import time

import pytest

from jinggao import footage_index as fi, paths


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    return tmp_path


def row(bvid, title="短剧片段", duration=300.0, tags=""):
    return {"provider": "bilibili", "id": bvid, "url": f"https://www.bilibili.com/video/{bvid}", "title": title,
            "tags": tags, "description": "", "duration": duration, "author": "up", "typename": "影视", "play": 1000,
            "pubdate": 0, "query": ""}


def video(bvid, *frames, title="短剧片段", duration=300.0, tags=""):
    """frames: (t, text, kind); filmed filler frames keep the neighbour rule out of the way unless given."""
    fi.add(row(bvid, title, duration, tags), [{"t": float(t), "hash": "", "text": text, "kind": kind}
                                              for t, text, kind in frames], described=True)


def ids(rows):
    return [r["id"] for r in rows]


def test_expand_uses_the_describers_words():
    assert fi.expand("父亲")[:3] == ["父亲", "爸爸", "老爸"] and "中年男" in fi.expand("父亲")
    assert "父亲" not in fi.expand("中年男子")                                # one way only
    boy = fi.expand("男孩")
    assert boy[0] == "男孩" and {"少年", "男生", "小男孩"} <= set(boy)
    assert fi.expand("男孩/少年")[:2] == ["男孩", "少年"]                    # the caller's own alternatives first
    assert fi.expand("哭")[0] == "哭泣" and "哭" not in fi.expand("哭")       # one character: only via the table
    assert fi.expand("猫") == [] and fi.expand("") == []
    assert fi.expand("码头 吊车，") == ["码头吊车"]                           # spaces and punctuation go
    assert "手机+柜台" in fi.expand("手机店")


def test_every_word_on_one_filmed_frame_synonyms_count_and_the_word_itself_ranks_first(data):
    video("BV1exact0001", (10, "男孩坐在沙发上玩手机｜男孩 玩手机 沙发", "实拍"), (40, "客厅一角", "实拍"))
    video("BV1synonym01", (10, "少年躺在床上看手机｜少年 看手机 卧室", "影视剧"), (40, "卧室", "影视剧"))
    video("BV1split0001", (10, "男孩在操场奔跑", "实拍"), (40, "桌上放着手机", "实拍"))   # words on two frames
    video("BV1cartoon01", (10, "动画男孩玩手机", "动画"), (40, "动画男孩玩手机", "动画"))
    rows = fi.find(["男孩", "玩手机"])
    assert ids(rows) == ["BV1exact0001", "BV1synonym01"]
    best = rows[0]
    assert best["indexed"] and best["provider"] == "bilibili" and best["moments"] == [10.0]
    assert best["match"] == {"t": 10.0, "text": "男孩坐在沙发上玩手机｜男孩 玩手机 沙发", "kind": "实拍", "check": ""}
    assert best["index_score"] > rows[1]["index_score"] and best["index_used"] == 0
    assert ids(fi.find(["男孩 玩手机"])) == ids(rows)                       # one string with spaces = two words
    assert fi.find(["男孩", "宇宙飞船"]) == [] and fi.find([]) == []


def test_only_filmed_frames_between_filmed_neighbours_from_usable_videos(data):
    video("BV1cutaway01", (10, "主持人讲解", "主持人出镜"), (20, "情侣拖着行李箱", "实拍"), (30, "主持人讲解", "主持人出镜"),
          (40, "街道", "实拍"), (50, "街道", "实拍"), (60, "街道", "实拍"), (70, "街道", "实拍"))
    video("BV1fine00001", (10, "街道", "实拍"), (20, "情侣拖着行李箱走进小区", "实拍"), (30, "小区", "实拍"),
          (40, "主持人讲解", "主持人出镜"))
    video("BV1talky0001", (10, "情侣拖着行李箱", "实拍"), (20, "文字页", "图表文字"), (30, "主持人", "主持人出镜"))
    video("BV1short0001", (5, "情侣拖着行李箱", "实拍"), duration=12)
    video("BV1skipped01", (10, "情侣拖着行李箱", "实拍"), title="情侣 游戏 实况")
    fi.add(row("BV1hosted001"), [{"t": t, "text": "主持人", "kind": "主持人出镜"} for t in (1, 2, 3)]
           + [{"t": 4.0, "text": "情侣拖着行李箱", "kind": "实拍"}], described=True)
    rows = fi.find(["情侣", "行李箱"])
    assert ids(rows) == ["BV1fine00001"]                                  # the cut-away between presenters is not
    assert rows[0]["avoid"] == [40.0]                                     # presenter frames to keep away from


def test_foreign_frames_and_foreign_videos_are_skipped_unless_asked(data):
    video("BV1foreign01", (10, "外国情侣在厨房做饭｜外国 情侣 做饭", "实拍"), (20, "厨房", "实拍"))
    video("BV1vlog00001", (10, "情侣在厨房做饭", "实拍"), (20, "厨房", "实拍"), title="美国自驾vlog 情侣日常")
    video("BV1home00001", (10, "年轻情侣在厨房做饭", "实拍"), (20, "厨房", "实拍"))
    assert ids(fi.find(["情侣", "做饭"])) == ["BV1home00001"]
    assert set(ids(fi.find(["情侣", "做饭"], foreign_ok=True))) == {"BV1foreign01", "BV1vlog00001", "BV1home00001"}
    assert ids(fi.find(["外国", "做饭"])) == ["BV1foreign01"]              # a foreign word asks for it


def test_exclude_penalize_people_and_period_pictures(data):
    video("BV1sea000001", (10, "大海海浪拍打礁石｜大海 海浪", "实拍"), (20, "礁石", "实拍"))
    video("BV1sea000002", (10, "女子站在大海边远眺｜大海 女子", "实拍"), (20, "沙滩", "实拍"))
    assert ids(fi.find(["大海"], people=True))[0] == "BV1sea000002"
    assert ids(fi.find(["大海"], people=False))[0] == "BV1sea000001"
    assert ids(fi.find(["大海"], exclude={"bilibili:BV1sea000001"})) == ["BV1sea000002"]
    assert ids(fi.find(["大海"], penalize={"bilibili:BV1sea000001": 5.0}, people=False))[0] == "BV1sea000002"
    video("BV1palace001", (10, "古装男子愤怒拍桌｜古装 愤怒", "影视剧"), (20, "宫殿", "影视剧"))
    video("BV1modern001", (10, "男子在客厅愤怒拍桌｜男子 愤怒", "影视剧"), (20, "客厅", "影视剧"))
    assert ids(fi.find(["男子", "愤怒"])) == ["BV1modern001", "BV1palace001"]
    assert ids(fi.find(["古装", "愤怒"])) == ["BV1palace001"]


def test_a_title_supplies_one_missing_word_only_when_frames_are_short(data):
    video("BV1frame0001", (10, "中年男子在客厅摔手机", "影视剧"), (20, "客厅", "影视剧"))
    video("BV1title0001", (10, "中年男子坐在沙发上", "影视剧"), (20, "客厅", "影视剧"), title="父亲一怒之下摔手机")
    video("BV1loose0001", (10, "中年男子坐在沙发上", "影视剧"), (20, "客厅", "影视剧"), title="矿难砸碎了他的腰")
    rows = fi.find(["中年男子", "摔手机"])
    assert ids(rows) == ["BV1frame0001", "BV1title0001"]                 # a title's 砸碎 is no 摔手机
    assert "title_word" not in rows[0]["match"] and rows[1]["match"]["title_word"] == "摔手机"
    assert ids(fi.find(["中年男子", "摔手机"], limit=1)) == ["BV1frame0001"]


def test_checks_rank_clean_moments_first_and_hide_bad_ones(data):
    video("BV1plain0001", (10, "学生走出校门｜学生 校门", "实拍"), (20, "校园", "实拍"))
    video("BV1clean0001", (10, "学生在校门口排队", "实拍"), (20, "校园", "实拍"), (30, "学生走出校门", "实拍"))
    video("BV1gone00001", (10, "学生走出校门｜学生 校门", "实拍"), (20, "校园", "实拍"))
    assert fi.record_checks([
        {"source": "bilibili:BV1clean0001", "t": 10.0, "status": "text", "crop": None, "note": "花字"},
        {"provider": "bilibili", "id": "BV1clean0001", "t": 30.004, "status": "crop",
         "crop": [0.0596, 0.16581, 0.6942, 0.6942]},
        {"source": "bilibili:BV1gone00001", "t": 10.0, "status": "gone", "note": "视频不见了"},
        {"source": "", "t": 1, "status": "clean"}, {"t": "x"}]) == 3   # malformed items are skipped
    rows = fi.find(["学生", "校门"])
    assert ids(rows) == ["BV1clean0001", "BV1plain0001"]
    assert rows[0]["moments"] == [30.0] and rows[0]["match"]["check"] == "crop" and 10.0 in rows[0]["avoid"]
    assert fi.checks_for(["bilibili:BV1clean0001", "bilibili:BV1none000000"]) == {
        "bilibili:BV1clean0001": {10.0: "text", 30.0: "crop"}}
    # Newer replaces older, an older answer does not.
    assert fi.record_checks([{"source": "bilibili:BV1clean0001", "t": 10, "status": "clean"}]) == 1
    assert fi.record_checks([{"source": "bilibili:BV1clean0001", "t": 10, "status": "dark", "checked": 1.0}]) == 0
    assert fi.checks_for(["bilibili:BV1clean0001"])["bilibili:BV1clean0001"][10.0] == "clean"
    stats = fi.check_stats()
    assert stats["windows"] == 3 and stats["videos"] == 2 and stats["clean_videos"] == 1
    assert stats["by_status"] == {"clean": 1, "crop": 1, "gone": 1}
    with fi._conn() as con:
        # Fractions of the source picture, as production crops (pixels are no crop).
        assert json.loads(con.execute("SELECT crop FROM checks WHERE t=30").fetchone()[0]) == [0.0596, 0.1658, 0.6942, 0.6942]
    assert fi.record_checks([{"source": "bilibili:BV1clean0001", "t": 50, "status": "crop", "crop": [0, 0, 640, 360]}]) == 1
    with fi._conn() as con:
        assert con.execute("SELECT crop FROM checks WHERE t=50").fetchone()[0] == ""
    assert fi.checks_for(["bilibili:BV1clean0001"], since=time.time() + 60) == {}     # only newer checks


def test_a_busy_index_skips_the_checks_instead_of_raising(data):
    video("BV1busy00001", (10, "学生走出校门", "实拍"))
    other = sqlite3.connect(paths.DATA / "footage-index.sqlite")
    other.execute("BEGIN IMMEDIATE")
    try:
        began = time.monotonic()
        assert fi.record_checks([{"source": "bilibili:BV1busy00001", "t": 10, "status": "clean"}]) == 0
        assert time.monotonic() - began < 10
    finally:
        other.rollback()
        other.close()
    assert fi.record_checks([{"source": "bilibili:BV1busy00001", "t": 10, "status": "clean"}]) == 1


def test_checks_travel_with_the_seed_and_merge_into_described_videos(data, tmp_path, monkeypatch):
    video("BV1seedck001", (10, "情侣拖着行李箱", "实拍"), (20, "街道", "实拍"))
    video("BV1seedck002", (10, "男孩玩手机", "实拍"), (20, "沙发", "实拍"))
    fi.record_checks([{"source": "bilibili:BV1seedck001", "t": 10, "status": "clean", "crop": [0.0, 0.05, 1.0, 0.9]},
                      {"source": "bilibili:BV1seedck002", "t": 10, "status": "mark", "checked": 1000.0}])
    seed = tmp_path / "seed"
    fi.export_seed(seed)
    lines = [json.loads(line) for line in gzip.open(seed / "part-01.jsonl.gz", "rt", encoding="utf-8")]
    first = next(item for item in lines if item["row"]["id"] == "BV1seedck001")
    assert [{k: c[k] for k in ("t", "status", "crop")} for c in first["checks"]] == [
        {"t": 10.0, "status": "clean", "crop": [0.0, 0.05, 1.0, 0.9]}] and first["checks"][0]["checked"] > 1000
    client = tmp_path / "client"
    monkeypatch.setattr(paths, "DATA", client)
    video("BV1seedck002", (10, "男孩玩手机", "实拍"), (20, "沙发", "实拍"))        # already described here: skipped,
    fi.record_checks([{"source": "bilibili:BV1seedck002", "t": 10, "status": "clean", "checked": 2000.0}])
    assert fi.install_seed(seed) == 1                                     # but its checks still come in
    got = fi.checks_for(["bilibili:BV1seedck001", "bilibili:BV1seedck002"])
    assert got == {"bilibili:BV1seedck001": {10.0: "clean"}, "bilibili:BV1seedck002": {10.0: "clean"}}  # newer kept
    assert fi.find(["情侣", "行李箱"])[0]["match"]["check"] == "clean"
    # A newer check in a new seed replaces the older one here; the new seed installs again.
    monkeypatch.setattr(paths, "DATA", data / "data")
    fi.record_checks([{"source": "bilibili:BV1seedck002", "t": 10, "status": "narrow", "checked": 3000.0}])
    fi.export_seed(seed)
    monkeypatch.setattr(paths, "DATA", client)
    assert fi.install_seed(seed) == 0
    assert fi.checks_for(["bilibili:BV1seedck002"])["bilibili:BV1seedck002"][10.0] == "narrow"
    assert fi.find(["男孩", "玩手机"]) == []
    # Another machine's index file brings its checks too.
    monkeypatch.setattr(paths, "DATA", tmp_path / "third")
    assert fi.merge(data / "data" / "footage-index.sqlite") == 2
    assert fi.check_stats()["windows"] == 2


def test_a_seed_without_checks_installs_as_before(data, tmp_path, monkeypatch):
    seed = tmp_path / "old-seed"
    seed.mkdir()
    line = {"row": row("BV1oldseed01"), "topic": "家庭", "moments": [{"t": 10.0, "hash": "", "text": "情侣拖着行李箱",
                                                                     "kind": "实拍"}]}
    (seed / "part-01.jsonl.gz").write_bytes(gzip.compress((json.dumps(line, ensure_ascii=False) + "\n").encode()))
    assert fi.install_seed(seed) == 1 and fi.check_stats()["windows"] == 0
    assert ids(fi.find(["情侣", "行李箱"])) == ["BV1oldseed01"]
