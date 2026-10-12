from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from jinggao import retrieval_terms


@pytest.mark.parametrize("query,scene,text", [
    ("民航运行系统架构", "仓库内的分拣系统正在运行。", ""),
    ("不同岗位协同工作", "多名工作人员正在整理包裹。", "工作中的平凡坚守"),
    ("工厂实物制造检验", "一名男子坐在沙发上接受采访，字幕介绍智能制造工厂。", "制造检验工厂"),
    ("飞机起飞", "未观察到飞机起飞，实际画面是一间快递仓库。", "飞机起飞"),
    ("资料画面 工作场景", "实际工作人员在工作场景中操作系统。", ""),
    ("A320飞机", "画面为一架客机。", "A3200"),
])
def test_weak_terms_caption_subjects_and_absent_objects_do_not_earn_paid_review(query, scene, text):
    assert not retrieval_terms.observed_match([query], scene, text)


@pytest.mark.parametrize("query,scene", [
    ("家具组装过程", "工作人员正在组装一张木桌。"),
    ("打电话", "男子拿着手机打电话，底部叠有白色字幕。"),
    ("运输货物", "货车正在运输货物。"),
    ("手工制作", "工匠正在手工制作一件木器。"),
    ("工厂设备运转", "工厂内设备运转，画面上方有字幕。"),
    ("民航飞机起飞", "民航飞机沿跑道滑行后起飞。"),
])
def test_observed_objects_and_verbs_remain_eligible(query, scene):
    assert retrieval_terms.observed_match([query], scene)


def test_literal_model_or_requested_name_in_ocr_is_only_a_retrieval_clue():
    assert retrieval_terms.observed_match(["A320飞机"], "一架客机。", "型号A320客机")
    assert retrieval_terms.observed_match(["刘翔赛事"], "运动员在场内移动。", "刘翔", people=[
        {"name": "刘翔", "aliases": []}])
    assert not retrieval_terms.observed_match(["刘翔赛事"], "运动员在场内移动。", "刘先生", people=[
        {"name": "刘翔", "aliases": []}])


def test_missing_segmenter_does_not_restore_han_bigram_recall(monkeypatch):
    retrieval_terms.anchors.cache_clear()
    monkeypatch.setitem(sys.modules, "jieba", None)
    assert not retrieval_terms.observed_match(["工作中的系统运行"], "物流系统运行工作。")
    assert retrieval_terms.observed_match(["A320"], "客机。", "A320")
    retrieval_terms.anchors.cache_clear()


@pytest.mark.parametrize("source,queries,scene,text", [
    ("茶叶需要机械揉捻。", ["机械揉捻机内部"], "机械设备正在分拣包裹。", ""),
    ("加工产品需要专门设备。", ["加工设备"], "车间的设备在运转。", ""),
    ("关注茶叶包装上的产品名称。", ["食品生产许可证SC标志"], "食品生产车间。", "SC12345"),
    ("茶叶摊放需要通风。", ["鲜叶摊放"], "这组画面未观察到鲜叶摊放。", "鲜叶摊放"),
    ("民航飞机起飞。", ["民航飞机起飞"], "仓库纸箱旁的字幕介绍民航飞机起飞。", "民航飞机起飞"),
    ("绿茶的包装需要密封。", ["茶叶包装"], "人员打开快递包裹，字幕询问寄的是否为茶叶。", "寄的是茶叶是吗"),
    ("鲜叶离开茶树后需要及时运输。", ["鲜叶运输"], "快递物流场景没有鲜叶，人员搬运纸箱。", ""),
])
def test_source_bound_spending_rejects_generic_invented_negative_and_caption_clues(source, queries, scene, text):
    profile = retrieval_terms.source_profile(queries, source)
    assert not retrieval_terms.source_support(profile, scene, text)["eligible"]


@pytest.mark.parametrize("source,query,scene,clue", [
    ("全国快递业务量达到一千亿件。", "快递分拣", "快递分拣中心，包裹在传送带上移动。", "快递"),
    ("服务细节关系到快递行业的发展。", "快递日常劳动", "快递员正在步行送件。", "快递"),
    ("制度让民航发展更稳健。", "民航飞机起飞", "民航飞机停在机场。", "民航"),
    ("家具产业逐步发展。", "家具组装", "工厂里摆放着家具。", "家具"),
    ("工人揉捻鲜叶。", "机械揉捻机内部", "手掌在揉捻鲜叶。", "揉捻"),
    ("制茶人员需要检查原料。", "手工制茶", "制茶人员在车间。", "制茶"),
    ("叶子要在通风的地方铺开。", "薄摊通风技巧", "厂房通风管道。", "通风"),
    ("杀青可用滚筒。", "滚筒杀青设备", "纸箱沿滚筒滑动。", "滚筒"),
])
def test_source_bound_industry_and_specific_clues_earn_inspection_not_approval(source, query, scene, clue):
    profile = retrieval_terms.source_profile([query], source)
    support = retrieval_terms.source_support(profile, scene)
    assert support["eligible"] and clue in support["observed_terms"]
    assert not {"approved", "verification", "verdict"} & support.keys()


def test_source_profile_restores_only_literal_unknown_compounds_and_keeps_hint_provenance():
    profile = retrieval_terms.source_profile(
        ["手工揉茶动作 机械揉捻机内部", "食品生产许可证SC标志"],
        "手工揉捻能看见手掌推动茶叶，机械揉捻把动作交给设备。")
    assert "揉捻" in profile["source_terms"]
    assert not {"手工", "机械", "动作", "设备", "食品", "sc"} & set(profile["source_terms"])
    assert {"手工", "机械", "动作"} <= set(profile["generic_terms"])
    assert {"食品", "sc"} <= set(profile["query_only_terms"])
    # Do not recreate arbitrary bigrams across already segmented known words.
    assert "业生" not in retrieval_terms.source_profile(["工业生产"], "工业生产")['source_terms']


def test_source_bound_identifier_and_requested_full_name_ocr_only_earn_inspection():
    profile = retrieval_terms.source_profile(["C919客机"], "国产C919客机已交付。")
    support = retrieval_terms.source_support(profile, "一架客机。", "型号C919客机")
    assert support["eligible"] and support["ocr_identifiers"] == ["c919"]
    assert not retrieval_terms.source_support(profile, "停机坪。", "C9190")["eligible"]
    invented = retrieval_terms.source_profile(["C919客机"], "一架普通客机。")
    assert not retrieval_terms.source_support(invented, "停机坪。", "C919")["eligible"]
    people = [{"name": "刘翔", "aliases": ["翔哥"]}, {"name": "张三", "aliases": []}]
    profile = retrieval_terms.source_profile(["训练现场"], "刘翔参加了训练。", people=people)
    assert profile["people"] == [{"name": "刘翔", "aliases": []}]
    support = retrieval_terms.source_support(profile, "运动员在场地内移动。", "刘翔")
    assert support["eligible"] and support["ocr_names"] == ["刘翔"]
    assert not retrieval_terms.source_support(profile, "运动员在场地内移动。", "翔哥 张三")["eligible"]


def test_source_profile_without_segmenter_defers_han_but_keeps_literal_model_ids(monkeypatch):
    retrieval_terms._source_terms.cache_clear()
    monkeypatch.setitem(sys.modules, "jieba", None)
    try:
        profile = retrieval_terms.source_profile(["制茶设备"], "制茶设备工作。")
        assert not retrieval_terms.source_support(profile, "制茶设备。")["eligible"]
        profile = retrieval_terms.source_profile(["C919"], "C919客机。")
        assert retrieval_terms.source_support(profile, "客机。", "C919")["eligible"]
    finally:
        retrieval_terms._source_terms.cache_clear()


def test_real_58_library_rank_inputs_defer_old_selected_windows_without_relabeling_them():
    fixture = json.loads((Path(__file__).parent / "fixtures/library_tea_rank_inputs.json").read_text())
    original = deepcopy(fixture)
    assert hashlib.sha256(fixture["script"].encode()).hexdigest() == fixture["script_sha256"]
    assert len(fixture["rows"]) == 10
    kept, old_selected, count = [], set(), 0
    for row in fixture["rows"]:
        profile = retrieval_terms.source_profile(row["queries"],
            fixture["script"] + "\n" + fixture["requirements"],
            people=[*row["persons"], *row["event_persons"]])
        old_selected.update((row["shot_index"], cid) for cid in row["selected_clip_ids"])
        for candidate in row["candidates"]:
            count += 1
            assert retrieval_terms.observed_match(row["queries"], candidate["observed_scene"],
                                                  candidate["visible_text"])
            support = retrieval_terms.source_support(profile, candidate["observed_scene"], candidate["visible_text"])
            if support["eligible"]:
                kept.append((row["shot_index"], row["candidate_clip_ids"][candidate["candidate_ref"]]))
    assert count == 58 and len(old_selected) == 7
    assert set(kept) == {(9, "b67dca18728b45e4b25227488d16f43c"),
                         (12, "c9b888fc8fa7d96e72234cf747d5ad5d")}
    assert not old_selected.intersection(kept)
    assert fixture == original
