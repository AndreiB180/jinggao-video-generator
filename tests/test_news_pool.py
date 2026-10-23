from jinggao import news_pool


def words(text, start, step=0.25):
    return [{"text": ch, "start": start + i * step, "end": start + (i + 1) * step} for i, ch in enumerate(text)]


def test_phrases_split_at_pauses_and_length():
    rows = news_pool.segments(words("胡塞武装发射导弹", 0) + words("总统府遭到袭击", 5))
    assert [r["text"] for r in rows] == ["胡塞武装发射导弹", "总统府遭到袭击"]
    assert rows[1]["start"] == 5


def test_lines_land_where_the_report_says_the_same_thing():
    report = {"id": "BV1", "provider": "bilibili", "duration": 300}
    other = {"id": "BV2", "provider": "bilibili", "duration": 300}
    spoken = {
        "BV1": (report, news_pool.segments(
            words("各位观众大家好欢迎收看今天的节目", 0) + words("也门南部城市亚丁的总统府", 20)
            + words("今天凌晨遭到六枚弹道导弹袭击", 27) + words("内阁会议当时正在楼内举行", 40))),
        "BV2": (other, news_pool.segments(words("沙特主导的联军当天对萨那发动空袭", 100)
                                           + words("拦截了三枚导弹和四架无人机", 110))),
    }
    shots = [{"id": "a", "narration": "凌晨的亚丁，六枚弹道导弹砸向总统府，内阁会议正在开。", "duration": 6},
             {"id": "b", "narration": "沙特联军当天拦截了三枚弹道导弹和四架无人机。", "duration": 6},
             {"id": "c", "narration": "这事您怎么看，评论区聊聊。", "duration": 3}]
    found = news_pool.match(shots, spoken, ["亚丁", "总统府", "沙特", "联军"])
    row, t, score = found["a"][0]
    assert row["id"] == "BV1" and 20 <= t <= 45 and score >= news_pool.MIN_SCORE
    assert found["b"][0][0]["id"] == "BV2" and found["b"][0][1] >= 100
    assert "c" not in found                       # nothing in the reports says this: no forced match


def test_only_report_length_videos_are_used():
    class Bili:
        def search(self, query):
            return [{"id": "short", "duration": 20}, {"id": "ok", "duration": 240}, {"id": "stream", "duration": 9000},
                    {"id": "ok2", "duration": 600}]
    assert [r["id"] for r in news_pool.find_sources(Bili(), ["也门 袭击"])] == ["ok", "ok2"]
    assert [r["id"] for r in news_pool.find_sources(Bili(), ["也门 袭击"], used={"ok"})] == ["ok2"]


def test_names_the_speech_model_misspells_still_match_by_sound():
    report = {"id": "BV1", "provider": "bilibili", "duration": 300}
    spoken = {"BV1": (report, news_pool.segments(words("今天凌晨胡赛武装向亚丁发射导弹", 30)))}
    shots = [{"id": "a", "narration": "胡塞武装凌晨朝亚丁打了导弹。", "duration": 6}]
    found = news_pool.match(shots, spoken, ["胡塞", "亚丁"])
    assert found["a"][0][0]["id"] == "BV1" and 30 <= found["a"][0][1] <= 40
    assert news_pool._named("胡赛来了", ["胡塞"]) == set()                  # by characters alone: not found
    assert news_pool._named("胡赛来了", ["胡塞"], news_pool._sounds("胡赛来了")) == {"胡塞"}
    assert news_pool._named("葫芦赛跑", ["胡塞"], news_pool._sounds("葫芦赛跑")) == set()   # syllables must be adjacent


def test_windows_move_onto_the_scene_and_presenter_only_windows_are_dropped(monkeypatch):
    from PIL import Image
    from jinggao import express_sources as es, planner
    report = {"id": "BV1", "provider": "bilibili", "duration": 300}
    class Bili:
        def previews(self, bvid, duration):
            return [{"t": float(t)} for t in range(0, 300, 3)]
    monkeypatch.setattr(es, "bilibili_tile", lambda bili, p: Image.new("RGB", (160, 90), (int(p["t"]) % 255, 0, 0)))
    studio = set(range(18, 29)) | set(range(99, 120))           # the presenter talks here; the rest is on site
    sheets = []
    def chat(system, content, **kw):
        sheets.append(content[0]["text"])
        return {str(i + 1): ("主播" if t in studio else "现场") for i, t in enumerate(chat.times)}, {}
    real = news_pool.onsite
    def spy_sheet(frames, cols):
        chat.times = [int(f["t"]) for f in frames]
        return Image.new("RGB", (10, 10))
    from jinggao import footage_index
    monkeypatch.setattr(footage_index, "_sheet", spy_sheet)
    monkeypatch.setattr(planner, "chat", chat)
    notes = []
    out = real(Bili(), {"a": [(report, 20.0, 0.3)], "b": [(report, 100.0, 0.2)], "c": [(report, 200.0, 0.2)]},
               notes=notes)
    assert out["a"][0][1] == 30.0 and out["c"][0][1] == 201.0      # from the first on-site frame
    assert "b" not in out                                    # only the presenter in the 14 s after the words
    assert len(sheets) == 1 and "1 个挪到现场画面" in notes[-1] and "1 个只有主播" in notes[-1]
    class Blocked(Bili):
        def previews(self, bvid, duration):
            raise ValueError("B站预览接口暂时限制访问（风控）")
    kept = real(Blocked(), {"a": [(report, 20.0, 0.3)]}, notes=notes)
    assert kept == {"a": [(report, 20.0, 0.3)]}                # unreadable previews: unchecked, kept


def test_news_mode_is_turned_on_once_for_settings_saved_while_it_was_an_experiment(tmp_path, monkeypatch):
    import json
    from jinggao import config, paths
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    paths.DATA.mkdir(parents=True)
    saved = dict(config.DEFAULTS, news_mode="0", tts_migrated="1", tts_migrated_edge="1")
    saved.pop("news_migrated")
    (paths.DATA / "settings.json").write_text(json.dumps(saved), "utf-8")      # as 0.4.3 saved it
    config.migrate()
    assert config.read()["news_mode"] == "1"
    config.update({"news_mode": "0"})                                           # switched off afterwards: stays off
    config.migrate()
    assert config.read()["news_mode"] == "0"
    monkeypatch.setattr(paths, "DATA", tmp_path / "new")
    assert config.read()["news_mode"] == "1"                                    # a new install has it on
