"""对标采集: author links, 好看 homepages read as a visitor sees them, the popular list, and the folder pipeline
(narration → rewrite → 待审 → batch)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from jinggao import benchmark, config, media, paths, store
from jinggao.api import create_app


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(media, "output_folder", lambda: tmp_path / "成片")
    monkeypatch.setattr(benchmark, "GAP", 0)
    return tmp_path


def page(name, videos):
    state = {"author_info": {"name": name, "fansCntText": "2250", "videoCntText": "235个视频"},
             "video": {"results": [{"tplName": "video", "content": {"vid": vid, "title": title, "publish_time": when,
                                                                     "playcnt": plays, "duration": "02:46"}}
                                   for vid, title, when, plays in videos]}}
    return f"<html><script>window.__PRELOADED_STATE__ = {json.dumps(state, ensure_ascii=False)};document.x=1</script></html>"


def test_author_links_and_relative_dates():
    assert benchmark.author_id("https://haokan.baidu.com/author/1872764043922508") == "1872764043922508"
    assert benchmark.author_id("https://haokan.baidu.com/author/1851836204565823?tab=video") == "1851836204565823"
    assert benchmark.author_id("1868148881032791") == "1868148881032791"
    with pytest.raises(ValueError, match="认不出"):
        benchmark.author_id("https://www.bilibili.com/video/BV1xx")
    now = time.mktime((2026, 11, 3, 15, 0, 0, 0, 0, -1))
    assert benchmark.published("昨天", now) == now - 86400
    assert benchmark.published("2天前", now) == now - 2 * 86400
    assert benchmark.published("3小时前", now) == now - 3 * 3600
    assert time.localtime(benchmark.published("10月28日", now))[:3] == (2026, 10, 28)
    assert time.localtime(benchmark.published("12月30日", now))[:3] == (2025, 12, 30)     # not a future date
    assert time.localtime(benchmark.published("2025-06-01", now))[:3] == (2025, 6, 1)
    assert benchmark.published("", now) == 0.0


def test_homepage_videos_and_the_check_page_left_alone():
    author, videos = benchmark.parse_author(page("科普瞭望台", [("1", "台北市长蒋万安曾表示", "昨天", "24564"),
                                                                ("2", "新版人民币正式发行", "2天前", "1.2万")]))
    assert author == {"name": "科普瞭望台", "fans": "2250", "count": "235个视频"}
    assert [(v["vid"], v["plays"], v["duration"]) for v in videos] == [("1", 24564, 166), ("2", 12000, 166)]
    assert videos[0]["url"] == "https://haokan.baidu.com/v?vid=1"
    with pytest.raises(PermissionError, match="不绕过"):
        benchmark.parse_author('<html><script src="/js/mkdcheck.7b1ef6.chunk.js"></script></html>')


def test_scan_keeps_what_was_copied_and_filters_by_age_and_plays(data, monkeypatch):
    pages = {"111111": page("甲", [("1", "爆款一", "昨天", "24564"), ("2", "冷门", "昨天", "44")]),
             "222222": page("乙", [("3", "老爆款", "2025-01-01", "99999"), ("4", "爆款二", "3小时前", "1.5万")])}
    monkeypatch.setattr(benchmark, "_fetch", lambda aid: pages[aid])
    benchmark.add_authors("https://haokan.baidu.com/author/111111\nhttps://haokan.baidu.com/author/222222 111111")
    state = benchmark.scan()
    assert [a["name"] for a in state["authors"]] == ["甲", "乙"] and state["known"] == 4
    assert [v["title"] for v in benchmark.popular(3, 10000)] == ["爆款一", "爆款二"]
    assert [v["title"] for v in benchmark.popular(0, 10000)] == ["老爆款", "爆款一", "爆款二"]
    benchmark.mark(["1"], "copied")
    pages["111111"] = page("甲", [("1", "爆款一", "昨天", "30000")])
    benchmark.scan()
    assert [v["title"] for v in benchmark.popular(3, 10000)] == ["爆款二"]            # copied ones hidden
    again = benchmark.popular(3, 10000, include_done=True)
    assert again[0]["plays"] == 30000 and again[0]["status"] == "copied"
    # A check page stops the round and says so on the author; nothing is worked around.
    monkeypatch.setattr(benchmark, "_fetch", lambda aid: "<script src='mkdcheck.js'></script>")
    state = benchmark.scan()
    assert "不绕过" in state["authors"][0]["error"] and state["authors"][1]["error"] == ""
    assert benchmark.remove_author("111111")["known"] == 2


def test_files_become_rewritten_scripts_to_review_then_a_batch(data, monkeypatch):
    from jinggao import rewrite
    monkeypatch.setattr(rewrite, "rewrite", lambda text, **kw: {"title": "改写后的标题", "script": "改写：" + text})
    folders = benchmark.folders()
    original = "台北市长蒋万安曾表示，如果大陆继续在台海进行军机绕台和军演，台北将会怎么样。这是原视频里的口播。"
    (data / "成片" / "对标采集" / "1-放视频" / "台北市长蒋万安曾表示.txt").write_text(original, "utf-8")
    assert benchmark.waiting()[0].name == "台北市长蒋万安曾表示.txt"
    out = benchmark.process_one(benchmark.waiting()[0])
    assert out.name == "改写后的标题.txt" and out.read_text("utf-8").startswith("改写：台北市长")
    assert (data / "成片" / "对标采集" / "原文案" / "台北市长蒋万安曾表示.txt").read_text("utf-8").strip() == original
    assert not benchmark.waiting() and any((data / "成片" / "对标采集" / "已转写").iterdir())
    assert set(folders) == {"1-放视频", "2-待审", "3-已出片", "原文案", "已转写", "封面"}
    draft = benchmark.review()[0]
    assert draft["title"] == "改写后的标题" and draft["chars"] > 30
    benchmark.save_review(draft["name"], "改过的稿子：" + original)
    with pytest.raises(ValueError):
        benchmark.save_review("../settings.txt", "x")

    config.update({"model_api_key": "sk-test"})
    with TestClient(create_app()) as client:
        r = client.post("/api/benchmark/produce", json={"names": [draft["name"]], "resolution": "720p"})
        assert r.status_code == 202, r.text
        body = r.json()
    project = store.get_project(body["projects"][0]["project_id"])
    assert project["title"] == "改写后的标题" and project["script"].startswith("改过的稿子：")
    assert project["burn_subtitles"] is False and "对标批量" in project["save_folder"]
    assert body["review"] == [] and (data / "成片" / "对标采集" / "3-已出片" / "改写后的标题.txt").exists()


def test_too_short_narration_is_reported(data):
    benchmark.folders()
    path = data / "成片" / "对标采集" / "1-放视频" / "短.txt"
    path.write_text("太短了", "utf-8")
    with pytest.raises(ValueError, match="太短"):
        benchmark.process_one(path)
    assert path.exists()        # left where it was for the customer to see


def test_picked_covers_are_saved_by_title(data, monkeypatch):
    pages = {"111111": page("甲", [("1", "爆款一", "昨天", "24564")])}
    monkeypatch.setattr(benchmark, "_fetch", lambda aid: pages[aid])
    benchmark.add_authors("111111")
    benchmark.scan()
    stored = benchmark._load()
    stored["videos"]["1"]["cover"] = "https://f7.baidu.com/it/u=1,2&fm=222&app=106&f=JPEG@s_0,w_660,h_370,q_80,f_auto"
    benchmark._save(stored)
    asked = []

    class Response:
        def __init__(self, url):
            self.status_code, self.headers = (200, {"content-type": "image/jpeg"}) if "@" not in url else (404, {})
            self.content = b"\xff\xd8" + b"0" * 5000
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            asked.append(url)
            return Response(url)
    monkeypatch.setattr(benchmark.network, "client", lambda **kw: Client())
    assert benchmark.save_covers(["1", "missing"]) == 1
    assert asked == ["https://f7.baidu.com/it/u=1,2&fm=222&app=106&f=JPEG"]       # the full picture first
    assert (data / "成片" / "对标采集" / "封面" / "爆款一.jpg").stat().st_size > 2000
