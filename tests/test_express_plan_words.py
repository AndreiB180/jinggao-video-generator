"""Library word sets ("k") in the express plan: prompt, parsing, fallback (offline, no paid calls)."""
import json
import re

from jinggao import express_plan, footage_index, planner


def _plan(monkeypatch, lines, rows, groups=None, seen=None):
    shots = [planner.new_shot(t) for t in lines]
    reply = {"domain": "教育", "topic": {"zh": ["学生 上学"], "en": []},
             "groups": groups or [{"id": "G1", "zh": ["男孩 玩手机 短剧"], "en": [], "person": ""}],
             "shots": rows}
    def chat(system, content, **kwargs):
        if seen is not None:
            seen.append({"system": system, "content": content} | kwargs)
        return reply, {"usage": {}}
    monkeypatch.setattr(planner, "chat", chat)
    info = express_plan.annotate(shots, "".join(lines))
    return shots, info


def test_prompt_asks_for_k_in_the_describers_words():
    system = express_plan.SYSTEM
    assert '"k":["..","..",".."]' in system and "12. k：" in system
    listed = (express_plan.LIBRARY_PEOPLE + " " + express_plan.LIBRARY_ACTS + " " + express_plan.LIBRARY_THINGS).split()
    assert all(re.fullmatch(r"[一-鿿]{2,4}", w) for w in listed) and len(set(listed)) == len(listed)
    assert all(w in system for w in listed)
    # Relation words stay (footage_index widens them); the prompt no longer maps 大学生 to 学生 or 父亲 to 中年男子.
    assert "大学生→学生" not in system and "父亲→中年男子" not in system and "男友" in system
    # The worked examples (from the client's scripts) survive the parser as written, and none is a lone thing.
    examples = re.findall(r"“([^”]+)”→\"k\":(\[[^\]]+\])", system)
    assert len(examples) == 3 and "男子 摔手机" in system and "情侣 行李箱" in system
    for line, k in examples:
        sets = [s.split() for s in json.loads(k)]
        assert express_plan.library_sets(sets, narration=line) == sets
        assert not any(express_plan.only_things(s) for s in sets)


def test_library_sets_cleans_what_cheap_models_send():
    sets = express_plan.library_sets
    assert sets(["男子 摔手机", "摔手机", "父亲 愤怒"]) == [["男子", "摔手机"], ["摔手机"], ["父亲", "愤怒"]]
    assert sets([["情侣", "行李箱"], ["情侣 交谈"]]) == [["情侣", "行李箱"], ["情侣", "交谈"]]
    assert sets("情侣 行李箱，行李箱 行走；情侣") == [["情侣", "行李箱"], ["行李箱", "行走"], ["情侣"]]
    assert sets(["情侣、行李箱", "情侣+做饭"]) == [["情侣", "行李箱"], ["情侣", "做饭"]]
    # At most 3 sets of 3 words; repeats (also reordered) go.
    assert sets(["男孩 书包 卧室 床上", "书包 男孩 卧室", "男孩", "男孩", "书包", "手机"]) == \
        [["男孩", "书包", "卧室"], ["男孩"], ["书包"]]
    assert sets(None) == [] and sets({"a": 1}) == [] and sets(["", "  "]) == []


def test_library_sets_drop_names_numbers_ideas_and_search_words():
    sets = express_plan.library_sets
    # Stand-ins go, even unreported; what their title says of the person stays (张女士 is a woman).
    assert sets(["小夏 做饭", "张女士 哭泣", "老周 打电话"]) == [["做饭"], ["女士", "哭泣"], ["打电话"]]
    assert sets(["小孩 老人", "小区 老师"]) == [["小孩", "老人"], ["小区", "老师"]]  # ordinary 小X/老X words stay
    assert sets(["阿姨 做饭", "老板 办公室", "老妈 哭泣"]) == [["阿姨", "做饭"], ["老板", "办公室"], ["老妈", "哭泣"]]
    assert sets(["钱学森 讲课"], avoid={"钱学森"}) == [["讲课"]]
    assert sets(["十六岁 男孩", "九千块 手机", "三人 交谈"]) == [["男孩"], ["手机"], ["三人", "交谈"]]
    assert sets(["16岁男孩", "iPhone手机"]) == [["男孩"], ["手机"]]              # library words inside
    assert sets(["三对情侣 校园", "两个学生 交谈", "八千块钱"]) == [["情侣", "校园"], ["学生", "交谈"]]
    assert sets(["教育 底线", "情侣 同居", "空镜 校园"]) == [["情侣", "同居"], ["校园"]]   # 同居: footage_index knows it
    # Search wording and scenery go, also inside a word.
    assert sets(["校园空镜", "城市航拍", "出租屋风景", "航拍", "风景 夜景"]) == []
    assert sets(["短剧片段 情侣", "新闻现场", "城市 航拍"]) == [["情侣"], ["城市"]]
    assert sets(["外国男子 争吵"]) == [["男子", "争吵"]]                         # Chinese story: no foreigners
    assert sets(["外国男子 争吵"], narration="两名外国游客在街头争吵") == [["外国男子/男子", "争吵"]]


def test_library_sets_drop_lone_things_on_lines_with_people_or_figures():
    lone = express_plan.only_things
    assert lone(["手机"]) and lone(["城市", "街道"]) and lone(["江西", "校园"]) and lone(["出租屋"])
    assert not lone(["男孩", "手机"]) and not lone(["摔手机"]) and not lone(["同居"]) and not lone(["中年夫妻/夫妻"])
    sets = express_plan.library_sets
    line = "父亲一时火起把手机摔了个稀碎。"
    assert sets(["手机", "城市 街道", "男子 摔手机", "父亲 愤怒"], narration=line) == [["男子", "摔手机"], ["父亲", "愤怒"]]
    assert sets(["江西 校园", "学生 校门"], narration="孩子背起书包走进校门。") == [["学生", "校门"]]
    assert sets(["城市", "情侣 争吵"], narration="一线城市高校更是高达9.7%。") == [["情侣", "争吵"]]
    assert sets(["码头", "老人 码头"], avoid={"钱学森"}, narration="钱学森回到了祖国。") == [["老人", "码头"]]
    # A line about a place, with no one in it, may ask for the place.
    assert sets(["港口 集装箱", "集装箱"], narration="港口的集装箱堆成了山。") == [["港口", "集装箱"], ["集装箱"]]


def test_library_sets_keep_words_footage_index_widens_and_split_joined_words():
    sets = express_plan.library_sets
    # 男友 → 情侣/恋人…, 丈夫 → 夫妻/夫妇, 家长 → 父母/母子…: rewriting them (男友 → 年轻男子) would lose the people.
    for word, wider in (("男友", "情侣"), ("丈夫", "夫妻"), ("家长", "父母"), ("大学生", "年轻人"), ("父亲", "中年男")):
        assert wider in footage_index.expand(word)
    assert sets(["男友 做饭", "丈夫 争吵", "大学生 学习"]) == [["男友", "做饭"], ["丈夫", "争吵"], ["大学生", "学习"]]
    assert sets(["父亲 打电话", "家长 无奈"]) == [["父亲", "打电话"], ["家长", "无奈"]]
    assert sets(["男孩子 书包", "女大学生 宿舍"]) == [["男孩", "书包"], ["大学生", "宿舍"]]   # roles its table lacks
    # find() wants each word whole on one frame: library words written together are split.
    assert sets(["男孩玩手机", "情侣争吵", "女生学习"]) == [["男孩", "玩手机"], ["情侣", "争吵"], ["女生", "学习"]]
    assert sets(["中年男子摔手机", "男孩在卧室哭泣"]) == [["中年男子", "摔手机"], ["男孩", "卧室", "哭泣"]]
    assert sets(["摔手机 男子"]) == [["摔手机", "男子"]]
    # A person with a phone is seen looking at it ("中年男子 手机" finds phone reviews).
    assert sets(["中年男子 手机", "男子 摔手机 手机", "手机 柜台"]) == \
        [["中年男子", "看手机"], ["男子", "摔手机", "手机"], ["手机", "柜台"]]
    # A phrase around one library word: the phrase, or else that word.
    assert sets(["中年夫妻 争吵", "校外同居"]) == [["中年夫妻/夫妻", "争吵"], ["校外同居/同居"]]
    assert sets(["男孩/少年 看手机", "父亲/中年男子"]) == [["男孩/少年", "看手机"], ["父亲/中年男子"]]
    assert sets(["床 a", "好"]) == []                                           # under 2 characters


def test_annotate_stores_k_without_names(monkeypatch):
    lines = ["江西赣州一个十六岁的男孩沉迷游戏。", "父亲一时火起把手机摔了个稀碎。", "老周说孩子背起书包走进校门。"]
    rows = [{"id": "S1", "g": "G1", "v": "男孩玩手机", "k": ["男孩 玩手机", "少年 卧室", "手机"], "p": []},
            {"id": "S2", "g": "G1", "v": "父亲摔手机", "k": ["男子 摔手机", "摔手机", "父亲 愤怒"], "p": []},
            {"id": "S3", "g": "G1", "v": "学生进校门", "k": ["老周 男孩 书包", "学生 校门", "江西 校园"], "p": ["老周"]}]
    seen = []
    shots, _ = _plan(monkeypatch, lines, rows, seen=seen)
    assert shots[0]["express"]["k"] == [["男孩", "玩手机"], ["少年", "卧室"]]          # a lone 手机 finds phone reviews
    assert shots[1]["express"]["k"] == [["男子", "摔手机"], ["摔手机"], ["父亲", "愤怒"]]
    assert shots[2]["express"]["k"] == [["男孩", "书包"], ["学生", "校门"]]
    assert seen[0]["max_tokens"] >= 900 + 90 * len(lines)                     # room for the extra field


def test_annotate_keeps_named_people_out_of_k(monkeypatch):
    lines = ["钱学森回到了祖国。", "后来钱学森开始讲课。"]
    rows = [{"id": "S1", "g": "G1", "v": "归国", "k": ["钱学森 码头", "老人 码头"], "p": ["钱学森"]},
            {"id": "S2", "g": "G1", "v": "讲课", "k": ["钱学森讲课", "老师 黑板"], "p": []}]
    groups = [{"id": "G1", "zh": ["历史影像"], "en": [], "person": "钱学森"}]
    shots, _ = _plan(monkeypatch, lines, rows, groups)
    assert shots[0]["express"]["k"] == [["老人", "码头"]]                       # an empty 码头 is scenery
    assert shots[1]["express"]["k"] == [["讲课"], ["老师", "黑板"]]              # name found locally, still dropped


def test_missing_k_falls_back_to_the_visual(monkeypatch):
    lines = ["开学在即，越来越多的大学生情侣拖着行李箱搬进校外出租屋。", "到底谁说得对？今天不站队，只把这件事掰开揉碎了讲清楚。"]
    rows = [{"id": "S1", "g": "G1", "v": "大学生情侣拖行李箱", "p": []},
            {"id": "S2", "g": "G1", "v": "", "k": "无", "p": []}]
    shots, _ = _plan(monkeypatch, lines, rows)
    assert shots[0]["express"]["k"] == [] and shots[1]["express"]["k"] == []
    assert express_plan.library_words(shots[0]) == [["情侣", "拖行李箱"], ["情侣"]]   # a pair, not 大学生
    # No visual (the narration stands in) and no library word in it: nothing rather than scenery.
    assert express_plan.library_words(shots[1]) == []
    assert express_plan.library_words({"narration": "说到底，大学是人生最后一段低成本试错的阶段。"}) == []
    assert express_plan.library_words({"narration": "一线城市高校更是高达9.7%。"}) == []


def test_library_words_tops_up_and_follows_a_typed_visual():
    shot = {"narration": "父亲一时火起把手机摔了个稀碎。", "visual": "父亲摔手机", "visual_origin": "express",
            "express": {"k": [["男子", "愤怒"]]}}
    assert express_plan.library_words(shot) == [["男子", "愤怒"], ["父亲", "摔手机"], ["父亲"]]
    shot["express"]["k"] = [["男子", "摔手机"], ["摔手机"], ["父亲", "愤怒"]]
    assert express_plan.library_words(shot) == shot["express"]["k"]
    shot |= {"visual": "男孩在卧室哭泣", "visual_origin": "manual"}
    assert express_plan.library_words(shot) == [["男孩", "哭泣", "卧室"], ["男孩", "哭泣"], ["男孩"]]
    # A typed visual may ask for a place.
    typed = {"narration": "孩子背起书包走进校门。", "visual": "校门口", "visual_origin": "manual"}
    assert express_plan.library_words(typed) == [["校门口"], ["孩子", "书包"], ["孩子"]]


def test_plans_made_before_k_get_words_from_text():
    old = {"narration": "孩子才重新背起书包走进校门。", "visual": "高中生背书包走进校门"}
    assert express_plan.library_words(old) == [["高中生", "书包", "校门"], ["高中生", "书包"], ["高中生"]]
    old = {"narration": "家长夹在中间左右为难。", "visual": "", "persons": [{"name": "张三"}]}
    assert express_plan.library_words(old) == [["家长"]]
    old = {"narration": "很多家长为了图个清静，不得不配上一部手机。"}
    assert express_plan.library_words(old) == [["家长", "看手机"], ["家长"]]
    old = {"narration": "她每天和男友在出租屋里一起做饭。"}
    assert express_plan.library_words(old) == [["男友", "做饭"], ["男友"]]
    assert express_plan.library_words({"narration": "", "express": "x"}) == []
