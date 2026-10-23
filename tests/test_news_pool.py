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
