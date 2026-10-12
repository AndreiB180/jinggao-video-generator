"""Source subjects prevent unrelated cache footage from consuming review budget."""
import pytest

from jinggao import retrieval_terms


@pytest.mark.parametrize("source,queries,wrong,right", [
    ("公交车门打开，乘客走下台阶，离开公交车厢。",
     ["公交车门打开 乘客走下台阶离开车厢"],
     "快递员打开车门，在车厢内搬运纸箱。",
     "公交车在路边停靠，车门打开，乘客下车。"),
    ("公交的座椅沿着过道排列。公交车乘客握住扶手。",
     ["公交车厢内部 座椅扶手立柱 乘客找座位坐下", "公交车内 乘客站立扶稳"],
     "女子拆下座椅，弯腰清洁白色货车的车厢地板。",
     "公交车厢内，多名乘客坐在座椅上。"),
    ("茶叶的加工批次需要记录，茶叶的设备使用前要清洁。",
     ["茶叶烘干生产线", "茶叶筛分包装"],
     "塑料灯具的生产线经传送带送入烘干设备。",
     "茶叶经生产线输送，人员检查包装。"),
    ("高铁连接车站，旅客进入车厢。高铁每天承载乘客出行。",
     ["高铁顺畅旅行体验车站列车连接", "高铁日常出行细节交通记录"],
     "快递员进入车厢，搬运货物并走出装车区。",
     "高铁站内人群排队，列车停靠在站台旁。"),
])
def test_same_action_or_environment_without_subject_defers_spending(source, queries, wrong, right):
    profile = retrieval_terms.source_profile(queries, source)
    assert profile["subject_terms"]
    assert not retrieval_terms.source_support(profile, wrong)["eligible"]
    match = retrieval_terms.source_support(profile, right)
    assert match["eligible"] and match["observed_subjects"]
    assert not {"approved", "verification", "verdict"}.intersection(match)


def test_compound_query_accepts_repeated_literal_subject_without_exact_compound():
    profile = retrieval_terms.source_profile(
        ["城市公交站台 乘客看站牌等车过程"], "城市公交到站。公交乘客排队。公交每天运送人们。")
    assert profile["subject_terms"] == ["公交"]
    assert retrieval_terms.source_support(profile, "城市道路上有公交车行驶。")["eligible"]
    assert not retrieval_terms.source_support(profile, "城市道路上有私家车行驶。")["eligible"]


def test_query_compound_not_in_source_uses_its_literal_subject_before_incidental_object():
    profile = retrieval_terms.source_profile(
        ["公交站牌 查看手机线路信息"], "公交到站前，乘客查看手机。公交连接不同的地方。")
    assert profile["subject_terms"] == ["公交"]
    assert not retrieval_terms.source_support(profile, "快递员查看手机线路信息。")["eligible"]
    assert retrieval_terms.source_support(profile, "路边停靠着公交车。")["eligible"]


def test_caption_only_and_explicitly_absent_subjects_do_not_earn_review():
    profile = retrieval_terms.source_profile(["公交车厢 乘客下车"], "公交车厢内乘客下车。")
    assert not retrieval_terms.source_support(profile, "货车中有工人搬箱子，字幕讲公交车厢。", "公交车厢")["eligible"]
    assert not retrieval_terms.source_support(profile, "没有公交车厢，画面中是货车。")["eligible"]


def test_subject_is_bound_to_source_and_action_only_queries_keep_existing_path():
    profile = retrieval_terms.source_profile(["名牌手机打电话"], "男子打电话。")
    assert not profile["subject_terms"]
    assert retrieval_terms.source_support(profile, "男子打电话。")["eligible"]
    assert not retrieval_terms.source_support(profile, "名牌手机静物展示。")["eligible"]


def test_named_person_does_not_require_blind_observer_to_identify_a_face():
    profile = retrieval_terms.source_profile(["刘翔赛道冲刺"], "刘翔在赛道冲刺。",
        people=[{"name": "刘翔", "aliases": []}])
    assert not profile["subject_terms"]
    assert retrieval_terms.source_support(profile, "运动员在赛道冲刺。")["eligible"]
