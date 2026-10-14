"""Offline checks for news topics, the AI rewrite (洗稿) and express workbench edits."""
import json
import sqlite3
import sys

import pytest

from jinggao import config, curation, model_routing, news, news_crawler, paths, planner, rewrite, store, worker

ARTICLE = ("今年9月，浙江宁波一对95后新人办了一场没有接亲、没有婚车、没有司仪的婚礼。"
           "新娘父亲上台致辞41秒，一句吃好喝好之后宴席直接开席。每桌成本约2000元。")


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    return tmp_path


def test_clean_script_keeps_numbers_and_puts_one_sentence_per_line():
    text = "## 标题\n1.8场比赛。\n- 第一点：好！第二点。\n\n\n2. 他说：“好。”然后走了😀 #话题\n**加粗**"
    assert rewrite.clean_script(text) == "标题\n1.8场比赛。\n第一点：好！\n第二点。\n\n他说：“好。”\n然后走了 话题\n加粗"


def test_target_length_follows_the_source_and_the_wanted_minutes():
    assert rewrite.target_chars("字" * 1000, 4) == 900
    assert rewrite.target_chars("字" * 5000, 2) == 470
    assert rewrite.target_chars("字" * 40) == 150


def test_rewrite_applies_only_the_fact_fixes_and_reports_unsupported_numbers(monkeypatch):
    calls = []
    def chat(system, content, **kw):
        calls.append(kw["task"])
        if len(calls) == 1:
            return {"title": "宁波四无婚礼", "script": ["去年9月，宁波新人办了场四无婚礼。", "父亲致辞41秒就开席。",
                                                      "每桌花了3000元，菜品分量都很足。", "没有仪式，也没有繁文缛节。", "你会这样办自己的婚礼吗？"]}, {"model": "m", "usage": {"total_tokens": 10}}
        return {"fixes": [{"line": "去年9月宁波新人办了场四无婚礼", "fix": "今年9月，宁波新人办了场四无婚礼。"},
                          {"line": "不存在的句子。", "fix": "无关"}]}, {"usage": {"total_tokens": 5}}
    monkeypatch.setattr(planner, "chat", chat)
    result = rewrite.rewrite(ARTICLE, project_id="p")
    assert calls == ["rewrite", "rewrite"]
    assert result["script"].splitlines()[0] == "今年9月，宁波新人办了场四无婚礼。"
    assert result["fixes"] == ["去年9月，宁波新人办了场四无婚礼。 → 今年9月，宁波新人办了场四无婚礼。"]
    assert result["check_numbers"] == ["3000"] and result["usage"]["total_tokens"] == 15


def test_rewrite_retries_once_then_fails_clearly(monkeypatch):
    replies = iter([({"script": ""}, {}), ({"script": ""}, {})])
    monkeypatch.setattr(planner, "chat", lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match="没有返回可用的口播稿"):
        rewrite.rewrite(ARTICLE)
    with pytest.raises(ValueError, match="素材太短"):
        rewrite.rewrite("太短")


def test_rewrite_uses_a_stronger_default_model_only_on_dashscope():
    cfg = config.DEFAULTS | {"model_name": "qwen3.7-flash"}
    assert model_routing.model(cfg, "rewrite") == "qwen3.7-plus"
    assert model_routing.model(cfg | {"rewrite_model": "qwen3.7-flash"}, "rewrite") == "qwen3.7-flash"
    assert model_routing.model(cfg | {"model_base_url": "https://api.deepseek.com"}, "rewrite") == "qwen3.7-flash"
    assert model_routing.model(cfg, "planning") == "qwen3.7-flash"


PAGE = """<html><head><title>站点首页</title><meta property="og:title" content="宁波“四无婚礼”火了">
<meta property="og:site_name" content="中新网"></head><body><nav><p>导航链接一 导航链接二</p></nav>
<h1>宁波“四无婚礼”火了</h1><div class="left_zw"><p>没有接亲，没有婚车，没有司仪，连伴郎伴娘都没请。</p>
<p>今年9月，浙江宁波一对95后新人的婚礼迅速出圈，被网友称作四无婚礼。</p><p>（完）【编辑:张三】</p></div>
<footer><p>版权所有 联系我们</p></footer></body></html>"""


def test_extract_takes_the_article_block_and_drops_page_furniture():
    found = news.extract(PAGE)
    assert found["title"] == "宁波“四无婚礼”火了" and found["source"] == "中新网"
    assert found["text"].startswith("没有接亲") and "导航" not in found["text"] and "编辑" not in found["text"]
    assert "版权" not in found["text"]


def test_fetch_refuses_private_addresses(monkeypatch):
    with pytest.raises(ValueError, match="http"):
        news.fetch("file:///etc/passwd")
    with pytest.raises(ValueError, match="公网"):
        news.fetch("http://127.0.0.1:8766/api/settings")


def test_news_items_are_deduplicated_and_become_auto_projects(data):
    rows = [{"title": "四无婚礼", "content": ARTICLE, "url": "https://example.com/a"},
            {"title": "重复", "text": ARTICLE, "url": "https://example.com/a"},
            {"title": "太短", "text": "短"}]
    stored, new = news.add(rows, "crawler")
    assert new == 1 and len(news.items()) == 1 and stored[0]["source"] == "example.com"
    made = news.make_project(stored[0]["id"], minutes=2, resolution="720p")
    project = store.get_project(made["project_id"])
    assert project["script"] == "" and project["news"]["text"] == ARTICLE and project["news"]["minutes"] == 2
    assert made["job_id"] and news.get(stored[0]["id"])["projects"] == [made["project_id"]]
    manual = news.make_project(stored[0]["id"], auto=False)
    assert store.get_project(manual["project_id"])["script"] == ARTICLE and manual["job_id"] is None


def test_rewrite_stage_fills_the_script_and_title(data, monkeypatch):
    stored, _ = news.add([{"title": "四无婚礼", "text": ARTICLE}], "text")
    made = news.make_project(stored[0]["id"], auto=False)
    store.mutate(made["project_id"], lambda p: p.update(script=""))
    monkeypatch.setattr(rewrite, "rewrite", lambda *a, **k: {"title": "宁波95后的四无婚礼", "script": "第一句。\n第二句。",
        "chars": 8, "fixes": [], "check_numbers": [], "model": "m"})
    job = store.enqueue(made["project_id"], "plan", {})
    worker.process(job | {"kind": "rewrite", "payload": {}}, finalize=False)
    project = store.get_project(made["project_id"])
    assert project["script"] == "第一句。\n第二句。" and project["title"] == "宁波95后的四无婚礼"
    assert project["news"]["rewrite"]["chars"] == 8


def test_crawler_output_files_are_read_in_any_supported_format(tmp_path):
    rows = [{"title": "甲", "content": ARTICLE, "url": "https://e.com/1"}]
    (tmp_path / "a.json").write_text(json.dumps({"data": rows}, ensure_ascii=False), "utf-8")
    (tmp_path / "a.jsonl").write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", "utf-8")
    (tmp_path / "a.csv").write_text("title,content\n甲," + ARTICLE + "\n", "utf-8")
    with sqlite3.connect(tmp_path / "a.db") as con:
        con.execute("CREATE TABLE news (id INTEGER PRIMARY KEY, title TEXT, content TEXT, url TEXT)")
        con.execute("INSERT INTO news (title, content, url) VALUES (?,?,?)", ("甲", ARTICLE, "https://e.com/1"))
    for name in ("a.json", "a.jsonl", "a.csv", "a.db"):
        found = news_crawler._records(tmp_path / name)
        assert found[0]["title"] == "甲" and news_crawler._text_field(found[0])["text"] == ARTICLE


def test_crawl_runs_the_configured_command_and_reports_failures(tmp_path, monkeypatch):
    script = tmp_path / "crawl.py"
    script.write_text("import json;print(json.dumps([{'title':'甲','content':%r}]))" % ARTICLE, "utf-8")
    cfg = config.DEFAULTS | {"news_crawler_dir": str(tmp_path), "news_crawler_command": f'"{sys.executable}" crawl.py'}
    monkeypatch.setattr(config, "read", lambda: cfg)
    assert news_crawler.status()["ready"]
    assert news_crawler.crawl()[0]["text"] == ARTICLE
    script.write_text("import sys;print('boom', file=sys.stderr);sys.exit(3)", "utf-8")
    with pytest.raises(ValueError, match="退出码 3.*boom"):
        news_crawler.crawl()
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS)
    with pytest.raises(ValueError, match="尚未接入"):
        news_crawler.crawl()


def _express_shot(**extra):
    return {"id": "s", "selected_asset": "a", "approved": True, "persons": [], "evidence_mode": "generic", "event": "",
            "verification": {"verdict": "match", "visual_quality": {"representation": "live_action"},
                             "express": {"lane": "pick", "identity": ""}}} | extra


def test_framing_edits_keep_an_express_pick_confirmed():
    project = {"plan_info": {"mode": "express"}}
    before = _express_shot()
    shot = dict(before, fit="contain", verification=None, approved=False)
    assert curation.express_edit(project, shot, before, {"fit", "clip_start"})
    assert shot["approved"] and curation.express_result(shot)["lane"] == "pick"
    shot = dict(before, narration="新口播")
    assert curation.express_edit(project, shot, before, {"narration"}) and not shot["approved"]
    assert curation.selection_error(project | {"id": "x"}, shot, {"kind": "video"}) == ""
    assert not curation.express_edit({"plan_info": {"mode": "model"}}, dict(before), before, {"fit"})


def test_changed_person_target_needs_new_evidence():
    project = {"plan_info": {"mode": "express"}}
    before = _express_shot(persons=[{"name": "梁靖崑", "aliases": []}], evidence_mode="person",
                           verification={"express": {"lane": "person", "identity": "supported"}})
    assert curation.identity_ready(before)
    shot = dict(before, persons=[{"name": "郭冠宏", "aliases": []}])
    curation.express_edit(project, shot, before, {"persons"})
    assert not shot["approved"] and not curation.identity_ready(shot)
    shot.update(source_review_note="比赛官方直播回放中确认", source_review_url="https://example.com/v")
    assert curation.identity_ready(shot)


def test_manual_pick_in_an_express_project_is_the_users_confirmation():
    project = {"plan_info": {"mode": "express"}}
    shot = {"id": "s", "persons": [], "evidence_mode": "generic", "event": "", "verification": None, "approved": False}
    curation.manual_pick(project, shot, {"kind": "video"})
    assert shot["approved"] and curation.selection_error(project | {"id": "x"}, shot, {"kind": "video"}) == ""
    person = dict(shot, persons=[{"name": "梁靖崑", "aliases": []}], evidence_mode="person", approved=False)
    curation.manual_pick(project, person, {"kind": "video"})
    assert not person["approved"]
    classic = dict(shot, verification=None, approved=False)
    curation.manual_pick({"plan_info": {"mode": "model"}}, classic, {"kind": "video"})
    assert classic["verification"] is None and not classic["approved"]


def test_changed_person_drops_the_old_persons_source_note():
    project = {"plan_info": {"mode": "express"}}
    note = {"source_review_note": "官方直播回放中确认是张三本人", "source_review_url": "https://example.com/zhang"}
    before = _express_shot(persons=[{"name": "张三", "aliases": []}], evidence_mode="person",
                           verification={"express": {"lane": "person", "identity": "supported"}}, **note)
    shot = dict(before, persons=[{"name": "李四", "aliases": []}])
    curation.express_edit(project, shot, before, {"persons"})
    assert shot["source_review_note"] == "" and not curation.identity_ready(shot)
    typed = dict(before, persons=[{"name": "李四", "aliases": []}], source_review_note="新闻发布会视频里确认李四本人")
    curation.express_edit(project, typed, before, {"persons", "source_review_note"})
    assert typed["source_review_note"].startswith("新闻发布会") and curation.identity_ready(typed)


def test_narration_edit_is_not_reapproved_automatically():
    project = {"id": "x", "plan_info": {"mode": "express"}, "assets": [{"id": "a", "kind": "video"}]}
    before = _express_shot(verification={"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
                                         "needs_source_check": False, "express": {"lane": "pick", "identity": ""}})
    shot = dict(before, narration="新口播")
    curation.express_edit(project, shot, before, {"narration"})
    project["shots"] = [shot]
    assert curation.approve_verified(project) == 0 and not shot["approved"]
    framed = dict(before, fit="contain")
    curation.express_edit(project, framed, before, {"fit"})
    assert framed["approved"] and not framed["verification"]["needs_source_check"]


def test_manual_choice_is_recognised_and_survives_an_ai_check():
    shot = {"persons": [], "evidence_mode": "generic", "event": "", "verification": None, "selected_asset": "a"}
    curation.manual_pick({"plan_info": {"mode": "express"}}, shot, {"id": "a", "kind": "video"})
    assert curation.manual_choice(shot) and not curation.manual_choice(_express_shot())
    shot["verification"] = {"verdict": "mismatch"}          # a classic AI check replaced the verification
    assert curation.manual_choice(shot)
    shot["selected_asset"] = "b"                            # a different clip is no longer the user's pick
    assert not curation.manual_choice(shot)


def test_a_second_different_fix_for_one_line_is_flagged_not_merged(monkeypatch):
    line = "这款车的销量达到了14万辆，同比增长25倍。"
    script = "\n".join([line, "这背后是整个汽车行业的变化与竞争格局的重塑。", "每一家车企都在重新思考未来的路线。"])
    replies = iter([({"script": script}, {}),
                    ({"fixes": [{"line": line, "fix": "这款车的销量达到了1.4万辆，同比增长25倍。"},
                                {"line": line, "fix": "这款车的销量达到了14万辆，同比增长2.5倍。"}]}, {})])
    monkeypatch.setattr(planner, "chat", lambda *a, **k: next(replies))
    result = rewrite.rewrite("这款车销量1.4万辆，同比增长2.5倍，背后是整个行业的变化与竞争格局的重塑，每一家车企都在重新思考。")
    assert result["script"].splitlines()[0] == "这款车的销量达到了1.4万辆，同比增长25倍。" and result["unmatched_fixes"] == 1


def test_a_failed_fact_check_keeps_the_paid_rewrite(monkeypatch):
    script = ["今年9月，宁波新人办了场四无婚礼。", "父亲致辞41秒就开席，没有任何仪式。", "每桌成本约2000元，菜品分量很足。"]
    calls = []
    def chat(system, content, **kw):
        calls.append(1)
        if len(calls) == 1:
            return {"title": "t", "script": script}, {"usage": {}}
        raise ValueError("模型返回的不是 JSON")
    monkeypatch.setattr(planner, "chat", chat)
    result = rewrite.rewrite(ARTICLE)
    assert result["checked"] is False and result["script"].splitlines() == script


def test_malformed_fix_rows_never_delete_or_garble_lines(monkeypatch):
    script = ["今年9月，宁波新人办了场四无婚礼。", "父亲致辞41秒就开席，没有任何仪式。", "每桌成本约2000元，菜品分量很足。"]
    replies = iter([({"script": "\n".join(script)}, {}),
                    ({"fixes": [{"line": script[0], "correction": "改名的键"},
                                {"line": script[1], "fix": ["父亲致辞41秒", "就直接开席。"]},
                                {"line": script[2], "fix": ""}, "不是对象", {"line": "对不上的句子", "fix": "x"}]}, {})])
    monkeypatch.setattr(planner, "chat", lambda *a, **k: next(replies))
    result = rewrite.rewrite(ARTICLE)
    assert result["script"].splitlines() == [script[0], "父亲致辞41秒就直接开席。"]
    assert result["unmatched_fixes"] == 3 and result["checked"] is True


def test_list_markers_are_not_confused_with_months():
    assert rewrite.clean_script("3、4月份天气转暖。\n1、第一点。") == "3、4月份天气转暖。\n第一点。"
    assert rewrite.target_chars("字" * 5000) == 15 * rewrite.CHARS_PER_MINUTE


def test_extractor_keeps_unclosed_paragraphs_forms_and_headline_dashes():
    page = """<html><head><meta property="og:title" content="国足1-2负于泰国 无缘世界杯 - 中新网">
    <meta property="og:site_name" content="中新网"></head><body><form id="aspnetForm"><div class="article">
    <p>第一段内容足够长，用于测试段落提取是否完整。<p>第二段也要足够长，确保不会被丢弃掉。
    <p>第三<wbr>段落，含有一个分词标签但仍属于同一段。<p>第四段同样足够长，用来凑够篇幅。<div>其他</div></div></form></body></html>"""
    found = news.extract(page)
    assert found["title"] == "国足1-2负于泰国 无缘世界杯"
    assert found["text"].splitlines() == ["第一段内容足够长，用于测试段落提取是否完整。", "第二段也要足够长，确保不会被丢弃掉。",
                                          "第三段落，含有一个分词标签但仍属于同一段。", "第四段同样足够长，用来凑够篇幅。"]


def test_link_guard_rejects_cgnat_and_tailscale_addresses():
    assert not news._public("100.100.100.100") and not news._public("10.0.0.1") and news._public("8.8.8.8")


def test_crawler_output_after_log_lines_and_bad_commands(tmp_path, monkeypatch):
    assert news_crawler._printed('[INFO] start\n[{"title": "a", "content": ["p1", "p2"]}]')[0]["title"] == "a"
    assert len(news_crawler._printed('{"title": "a"}\n{"title": "b"}')) == 2
    assert news_crawler._printed("null") == [] and news_crawler._text_field({"content": ["p1", "p2"]})["text"] == "p1\np2"
    cfg = config.DEFAULTS | {"news_crawler_dir": str(tmp_path), "news_crawler_command": "no-such-crawler-binary --run"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    with pytest.raises(ValueError, match="找不到爬虫运行命令"):
        news_crawler.crawl()


def test_film_timeline_matches_the_export_cuts(data):
    from fastapi.testclient import TestClient
    from jinggao.api import create_app
    project = store.new_project("片", "第一句。第二句话更长一些。")
    def shots(p):
        p["shots"] = [planner.new_shot("第一句。") | {"duration": 1.234}, planner.new_shot("第二句话更长一些。") | {"duration": 2.5}]
    store.mutate(project["id"], shots)
    film = TestClient(create_app()).get(f"/api/projects/{project['id']}/film").json()
    assert [s["start"] for s in film["shots"]] == [0, 1.24] and film["total"] == 3.76   # 25 fps frames, as exported
    assert film["shots"][1]["segments"][0][2] == "第二句话更长一些。" and film["audio_path"] is None


def _express_project(shot, asset_duration=30):
    return {"id": "x", "plan_info": {"mode": "express"}, "shots": [shot],
            "assets": [{"id": "a", "kind": "video", "duration": asset_duration}]}


def test_retimed_or_revoiced_express_pick_keeps_its_footage():
    from jinggao import media
    shot = _express_shot(duration=4, clip_start=0, verification={
        "verdict": "match", "scene_alignment": "match", "selection_blocked": False, "needs_source_check": False,
        "express": {"lane": "manual", "identity": ""}})
    project = _express_project(shot)
    media._invalidate_narration_window(shot, project)
    assert curation.manual_choice(shot) and not shot["approved"]
    assert curation.approve_verified(project) == 1          # produce re-approves it instead of re-matching
    shot["duration"] = 40                                     # longer than the 30 s clip
    curation.keep_express(project, shot)
    assert shot["verification"]["needs_source_check"] and not shot["approved"]
    classic = {"id": "y", "plan_info": {"mode": "model"}, "assets": []}
    old = {"id": "s", "verification": {"verdict": "match"}, "approved": True}
    media._invalidate_narration_window(old, classic)
    assert old["verification"] is None


def test_metadata_edit_keeps_express_picks_and_requirements_edit_asks_for_a_new_look(data):
    from fastapi.testclient import TestClient
    from jinggao.api import create_app
    project = store.new_project("片", "第一句。")
    def setup(p):
        p["plan_info"] = {"mode": "express"}
        p["assets"] = [{"id": "a", "kind": "video", "duration": 30, "path": "assets/a.mp4", "title": "t"}]
        p["shots"] = [planner.new_shot("第一句。") | _express_shot(duration=3)]
    store.mutate(project["id"], setup)
    client = TestClient(create_app())
    shot = client.patch(f"/api/projects/{project['id']}/assets/a", json={"license": "CC BY 4.0"}).json()["shots"][0]
    assert shot["approved"] and curation.express_result(shot)
    shot = client.patch(f"/api/projects/{project['id']}", json={"requirements": "只要夜景"}).json()["shots"][0]
    assert not shot["approved"] and shot["verification"]["needs_source_check"] and curation.express_result(shot)


def test_a_fix_that_only_moves_a_decimal_point_is_applied(monkeypatch):
    script = ["这款车的销量达到了14万辆。", "比去年增长了25倍，非常惊人。", "这背后是整个汽车行业的变化与竞争格局的重塑。"]
    replies = iter([({"script": "\n".join(script)}, {}),
                    ({"fixes": [{"line": script[0], "fix": "这款车的销量达到了1.4万辆。"},
                                {"line": script[1], "fix": "比去年增长了2.5倍，非常惊人。"}]}, {})])
    monkeypatch.setattr(planner, "chat", lambda *a, **k: next(replies))
    result = rewrite.rewrite("这款车销量1.4万辆，同比增长2.5倍，背后是整个行业的变化与竞争格局的重塑。")
    assert result["script"].splitlines()[:2] == ["这款车的销量达到了1.4万辆。", "比去年增长了2.5倍，非常惊人。"]
    assert len(result["fixes"]) == 2


def test_crawler_wrapper_objects_and_trailing_logs():
    assert [r["title"] for r in news_crawler._printed('[INFO] crawl\n{"count": 2, "articles": [{"title": "a"}, {"title": "b"}]}')] == ["a", "b"]
    assert news_crawler._printed('[{"title": "a"}]\n[INFO] done in 3.2s')[0]["title"] == "a"
    assert [r["title"] for r in news_crawler._printed('{"data": [{"title": "a"}]}\n{"data": [{"title": "b"}]}')] == ["a", "b"]


def test_extractor_ignores_side_columns_and_keeps_column_headlines():
    page = ('<div class="main"><div class="left"><div class="content"><p>正文第一段，足够长的新闻内容在这里。</p>'
            '<p>正文第二段，同样足够长的新闻内容。</p></div></div><div class="right"><div class="hot">'
            '<p>热门推荐：完全无关的另一条新闻标题一</p><p><a href="#">热门推荐：另一条，带链接的新闻标题二。</a></p></div></div></div>')
    assert news.extract(page)["text"].startswith("正文第一段")
    title = lambda t: news.extract(f'<meta property="og:title" content="{t}"><p>x</p>')["title"]
    assert title("热点｜网友热议宁波四无婚礼") == "热点｜网友热议宁波四无婚礼"
    assert title("浙江四无婚礼火了_新浪新闻") == "浙江四无婚礼火了"


def test_a_large_crawl_keeps_the_newest_and_reports_what_was_stored(data, monkeypatch):
    monkeypatch.setattr(news, "MAX_ITEMS", 3)
    dated = [{"title": f"第{i}条", "text": ARTICLE + str(i), "url": f"https://e.com/{i}",
              "published": f"2026年10月{10 + i}日 08:00"} for i in range(5)]       # oldest first in the file
    stored, new = news.add(dated, "crawler")
    assert new == 3 and sorted(r["title"] for r in news.items()) == ["第2条", "第3条", "第4条"]
    news._save([])
    undated = [{"title": f"无日期{i}", "text": ARTICLE + "x" + str(i), "url": f"https://f.com/{i}"} for i in range(5)]
    stored, new = news.add(undated, "crawler")       # no dates: keep the crawler's own (newest-first) order
    assert new == 3 and sorted(r["title"] for r in news.items()) == ["无日期0", "无日期1", "无日期2"]


def test_full_link_guard_rejects_invalid_hosts():
    with pytest.raises(ValueError, match="有效的新闻链接"):
        news.fetch("https://ｗｗｗ.example.com/a")


def test_film_reports_unaligned_narration(data):
    from fastapi.testclient import TestClient
    from jinggao.api import create_app
    project = store.new_project("片", "第一句。")
    store.mutate(project["id"], lambda p: p.update(shots=[planner.new_shot("第一句。")],
                                                     full_audio={"path": "audio/x.wav", "duration": 9, "alignment": "pending"}))
    film = TestClient(create_app()).get(f"/api/projects/{project['id']}/film").json()
    assert film["aligned"] is False and film["alignment_note"] and film["audio_path"] == "audio/x.wav"


def test_short_lines_join_a_neighbour_within_the_paragraph():
    from jinggao import express_plan
    script = "这胆子也太大了。\n\n要知道，\n悬空寺始建于北魏时期，\n距今已有1500多年历史。"
    lines = [s["narration"] for s in express_plan.line_shots(script, 20)]
    assert lines == ["这胆子也太大了。", "要知道，悬空寺始建于北魏时期，", "距今已有1500多年历史。"]


def test_shots_under_a_second_after_alignment_are_merged_and_aligned_again(data, monkeypatch):
    from jinggao import captions, express_plan, media
    project = store.new_project("片", "第一句比较长的话。好。第三句也比较长。")
    def setup(p):
        p["plan_info"] = {"mode": "express", "groups": []}
        p["shots"] = [planner.new_shot(t) for t in ("第一句比较长的话。", "好。", "第三句也比较长。")]
        p["full_audio"] = {"path": "audio/x.wav", "duration": 5}
    store.mutate(project["id"], setup)
    calls = []
    def prepare(p, check, progress):
        calls.append(len(p["shots"]))
        if len(p["shots"]) == 3:
            raise captions.ShotLengthOutOfRange([1], [])
        return {"ok": True}
    monkeypatch.setattr(media, "prepare_narration_alignment", prepare)
    monkeypatch.setattr(media, "apply_narration_alignment", lambda p, proposal: None)
    result = worker.ensure_narration_alignment(project["id"])
    assert calls == [3, 2] and [s["narration"] for s in result["shots"]] == ["第一句比较长的话。", "好。第三句也比较长。"]


def test_the_built_in_crawler_batch_and_its_scripts_are_imported(tmp_path, monkeypatch, data):
    crawler = tmp_path / "news-crawler"
    (crawler / "core").mkdir(parents=True)
    (crawler / "platforms").mkdir()
    (crawler / "main.py").write_text("", "utf-8")
    (crawler / "core" / "crawler.py").write_text("", "utf-8")
    batch = crawler / "data" / "采集" / "采集_10-14_12-28"
    batch.mkdir(parents=True)
    (batch / "宁波四无婚礼_31万.txt").write_text(
        "标题：宁波四无婚礼\n类别：社会\n平台：tencent\n浏览量：31.1万\n作者/来源：中新网\n发布时间：2026-10-14 07:00\n"
        "抓取时间：2026-10-14 12:30:51\n原文链接：https://example.com/a\n" + "-" * 50 + "\n" + ARTICLE + "\n", "utf-8")
    scripts = crawler / "koubo" / "口播" / "口播_10-14_12-40"
    scripts.mkdir(parents=True)
    (scripts / "宁波四无婚礼_311000.txt").write_text(
        "没有接亲，没有婚车，也没有司仪，这场婚礼火了。新娘父亲上台致辞41秒，一句吃好喝好就开席了。"
        "每桌成本约2000元，菜品分量很足，宾客到齐就开饭。\n"
        "百家号标题：宁波四无婚礼火了，父亲致辞41秒，你会这样办吗？\n", "utf-8")
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {"news_crawler_dir": str(crawler)})
    assert news_crawler.status()["builtin"] and news_crawler.status()["latest"]["count"] == 1
    rows = news_crawler.import_latest()
    assert rows[0]["url"] == "https://example.com/a" and rows[0]["source"] == "中新网" and rows[0]["text"] == ARTICLE
    assert rows[0]["koubo"].startswith("没有接亲") and rows[0]["koubo_title"].startswith("宁波四无婚礼火了")
    stored, new = news.add(rows, "crawler")
    assert new == 1 and stored[0]["koubo"] and news.add(rows, "crawler")[0][0]["id"] == stored[0]["id"]


def test_rewrite_stage_uses_the_crawlers_script_and_only_fact_checks_it(data, monkeypatch):
    stored, _ = news.add([{"title": "四无婚礼", "text": ARTICLE, "url": "https://e.com/x",
                           "koubo": "没有接亲，没有婚车，也没有司仪，这场婚礼火了。新娘父亲上台致辞41秒，一句吃好喝好就开席了。每桌成本约2000元。",
                           "koubo_title": "宁波四无婚礼火了，父亲致辞41秒，你会这样办吗？"}], "crawler")
    made = news.make_project(stored[0]["id"], auto=True)
    calls = []
    monkeypatch.setattr(planner, "chat", lambda system, content, **k: (calls.append(system[:6]), ({"fixes": []}, {}))[1])
    job = store.get_job(made["job_id"])
    worker.process(job | {"kind": "rewrite", "payload": {}}, finalize=False)
    project = store.get_project(made["project_id"])
    assert len(calls) == 1 and project["script"].splitlines()[0] == "没有接亲，没有婚车，也没有司仪，这场婚礼火了。"
    assert project["title"] == "宁波四无婚礼火了，父亲致辞41秒，你会这样办吗？" and project["news"]["rewrite"]["model"] == "crawler"


def test_crawler_scripts_run_under_an_isolated_embedded_python(tmp_path):
    import subprocess
    (tmp_path / "config.py").write_text("VALUE = 'ok'\n", "utf-8")
    (tmp_path / "main.py").write_text("import sys, config\nprint(config.VALUE, sys.argv[1:])\n", "utf-8")
    args = news_crawler._script_args(sys.executable, "main.py", ["--limit", "3"])
    args.insert(1, "-I")                      # what a ._pth file does to the Windows embedded runtime
    out = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "ok ['--limit', '3']"


def test_crawler_scripts_match_only_their_own_article_and_drop_label_lines(tmp_path):
    folder = tmp_path
    scripts = folder / "koubo" / "口播" / "口播_10-14_12-40"
    scripts.mkdir(parents=True)
    (scripts / "我军匕首枪曝光_123.txt").write_text("正文一。" * 20 + "\n**百家号标题：** 甲，乙，丙？\n关键词：匕首枪", "utf-8")
    found = news_crawler._koubo(folder)
    assert found["我军匕首枪曝光"][1] == "甲，乙，丙？" and "关键词" not in found["我军匕首枪曝光"][0]
    batch = folder / "data" / "采集" / "采集_10-15_08-00"
    batch.mkdir(parents=True)
    (batch / "a.txt").write_text("标题：我军匕首枪曝光后，外媒称其射程只有十米\n" + "-" * 50 + "\n" + ARTICLE, "utf-8")
    (batch / "b.txt").write_text("标题：我军匕首枪曝光\n" + "-" * 50 + "\n" + ARTICLE, "utf-8")
    rows = {r["title"]: r for r in news_crawler.read_batch(folder, batch)}
    assert "koubo" not in rows["我军匕首枪曝光后，外媒称其射程只有十米"] and rows["我军匕首枪曝光"]["koubo"].startswith("正文一")


def test_crawler_platform_names_accept_chinese_commas_and_reject_typos(tmp_path, monkeypatch):
    for name in ("main.py", "core/crawler.py"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("", "utf-8")
    (tmp_path / "platforms").mkdir()
    cfg = config.DEFAULTS | {"news_crawler_dir": str(tmp_path), "news_crawler_platforms": "toutiao，sohu tencnet"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    with pytest.raises(ValueError, match="未知的爬虫平台：tencnet"):
        news_crawler.crawl()
