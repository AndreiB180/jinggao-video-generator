from copy import deepcopy
import json

from PIL import Image
import pytest

from jinggao import config, editorial_contract, identity, paths, planner, store
from planner_helpers import consistency_response, source_contract_response


def test_relative_source_statistic_cannot_become_the_narrated_year_via_date_badge():
    from jinggao.planner import statistical_caption_conflict
    shot = {"narration": "这说的是那一年的业务量，不是人口数。"}
    context = [{"position": "before", "narration": "二〇二四年我国快递业务量完成一千七百五十点八亿件。"}]
    labels = [{"frame_index": 0, "quote": "今年中国快递业务量达300亿件"}]
    assert "2023" in statistical_caption_conflict(shot, context, labels, report_year="2023")
    assert not statistical_caption_conflict({"narration": "运输环节由不同岗位接力。"}, context, labels, report_year="2023")

def named_shot(**changes):
    return {"persons": [{"name": "梁靖崑", "aliases": ["LIANG Jingkun"]}],
            "evidence_mode": "person", "event": "", "event_scope": "exact", **changes}


def sourced_video(title="[乒乓球]梁靖崑晋级男单八强", description="梁靖崑在男单比赛中战胜陈垣宇。"):
    url = "https://tv.cctv.com/2026/01/09/VIDEO.shtml"
    return {"kind": "video", "title": title, "source_url": url,
            "source_evidence": {"url": url, "title": title, "description": description,
                                "publisher": "央视", "readable": True, "evidence_origin": "publisher_page"}}


def name_labels(**changes):
    return {"visual_evidence": [{"kind": "scoreboard", "quote": "LIANG Jingkun  CHEN Yuanyu",
                                "frame_index": 0, "context": "乒乓球比赛完整计分牌列出双方全名"}],
            "context_conflicts": [], "event_evidence": [], **changes}


def test_title_name_alone_never_confirms_person_and_requests_segment_preview():
    result = identity.assess(named_shot(), sourced_video(), {}, frame_count=1)
    assert result["status"] == "insufficient"
    assert result["matched_names"] == []
    assert result["selection_blocked"] and result["needs_segment_preview"]


def test_corresponding_source_and_readable_scoreboard_support_named_participant():
    result = identity.assess(named_shot(), sourced_video(), name_labels(), frame_count=5, has_segment_frames=True)
    assert result["status"] == "supported"
    assert result["matched_names"] == ["梁靖崑"]
    assert result["coverage"] == "named_participant"
    assert not result["selection_blocked"]


def test_same_name_but_different_non_biometric_context_is_conflict():
    shot = named_shot(persons=[{"name": "李明", "aliases": []}])
    candidate = sourced_video(title="李明医师讲座", description="李明医师在医学会议上演讲。")
    observations = name_labels(visual_evidence=[{"kind": "nameplate", "quote": "李明 医师", "frame_index": 0,
                                                "context": "演讲台胸牌写的是医师"}],
                               context_conflicts=["来源与胸牌均为医师，分镜要求的是同名乒乓球选手。"])
    assert identity.assess(shot, candidate, observations, frame_count=3)["status"] == "conflict"


def test_same_person_different_event_cannot_be_passed_as_exact_even_if_model_says_match():
    video = sourced_video(title="2026年WTT新加坡大满贯 梁靖崑男单比赛", description="梁靖崑对阵陈垣宇。")
    observations = name_labels(event_evidence=[{"quote": video["title"], "relation": "same_event",
                                                "source_url": video["source_url"]}])
    shot = named_shot(event="2026年WTT中国大满贯男单", evidence_mode="event")
    result = identity.assess(shot, video, observations, frame_count=5, has_segment_frames=True)
    assert result["status"] == "conflict" and result["event_status"] == "conflict"
    assert result["selection_blocked"]


def test_wrong_year_cannot_be_overruled_by_model_and_background_needs_label():
    video = sourced_video(title="2025年WTT中国大满贯 梁靖崑男单比赛", description="梁靖崑对阵陈垣宇。")
    observations = name_labels(event_evidence=[{"quote": video["title"], "relation": "same_event",
                                                "source_url": video["source_url"]}])
    shot = named_shot(event="2026年WTT中国大满贯男单", evidence_mode="event")
    assert identity.assess(shot, video, observations, frame_count=5)["status"] == "conflict"
    shot.update(event_scope="background", evidence_mode="person")
    assert identity.assess(shot, video, observations, frame_count=5)["selection_blocked"]
    shot["overlay_label"] = "资料画面 · 非本场赛事"
    result = identity.assess(shot, video, observations, frame_count=5)
    assert result["status"] == "supported" and result["event_status"] == "background"


def test_unreadable_sample_or_model_face_guess_never_establishes_identity():
    observations = name_labels(visual_evidence=[{"kind": "face", "quote": "LIANG Jingkun", "frame_index": 0,
                                                "context": "我认出了他的脸"}])
    observations["identity_evidence"] = {"status": "supported"}
    result = identity.assess(named_shot(), sourced_video(), observations, frame_count=5, has_segment_frames=True)
    assert result["status"] == "insufficient" and result["selection_blocked"]
    assert not result["needs_segment_preview"]


def test_invalid_frame_and_surname_only_are_not_name_evidence():
    for quote, index in (("LIANG Jingkun", 8), ("LIANG", 0), ("梁", 0)):
        observations = name_labels(visual_evidence=[{"kind": "scoreboard", "quote": quote,
                                                    "frame_index": index, "context": "计分牌"}])
        assert identity.assess(named_shot(), sourced_video(), observations, frame_count=3)["selection_blocked"]


def test_model_invented_publisher_or_event_quote_does_not_upgrade_evidence():
    video = sourced_video()
    video["source_evidence"]["url"] = "https://cctv.com.attacker.invalid/2026/01/09/VIDEO.shtml"
    result = identity.assess(named_shot(), video, name_labels(), frame_count=3)
    assert result["status"] == "insufficient"
    assert not any(row["kind"] == "publisher_domain" for row in result["evidence"])
    video = sourced_video(title="[乒乓球]梁靖崑男单比赛")
    observations = name_labels(event_evidence=[{"quote": "2026中国大满贯男单32强", "relation": "same_event",
                                                "source_url": video["source_url"]}])
    assert identity.assess(named_shot(event="2026中国大满贯男单32强"), video, observations,
                           frame_count=3)["status"] == "insufficient"


def test_other_query_of_same_video_host_does_not_authenticate_source():
    video = sourced_video()
    video["source_url"] = "https://www.youtube.com/watch?v=target"
    video["source_evidence"]["url"] = "https://www.youtube.com/watch?v=unrelated"
    assert identity.assess(named_shot(), video, name_labels(), frame_count=3)["selection_blocked"]


def test_user_edited_title_never_substitutes_for_retrieval_evidence():
    video = sourced_video(title="其他人的比赛", description="队员参加比赛。")
    video["title"] = "梁靖崑比赛 CCTV官方"
    assert identity.assess(named_shot(), video, name_labels(), frame_count=3)["selection_blocked"]


def test_generic_objects_remain_compatible_with_old_shots():
    result = identity.assess({"subject": "铁路扣件"}, {"kind": "image", "title": "扣件"})
    assert result["status"] == "supported" and not result["selection_blocked"]
    assert result["coverage"] == "generic"


def test_rule_split_preserves_text_and_defaults_to_video():
    text = "梁靖崑回到赛场。随后进入第二轮。"
    shots = planner.rule_shots(text)
    assert "".join(row["narration"] for row in shots) == text
    assert all(row["preferred_media"] == "video" for row in shots)


def test_independent_record_is_not_mutated_by_assessment():
    video = sourced_video()
    original = deepcopy(video)
    identity.assess(named_shot(), video, name_labels(), frame_count=3)
    assert video == original


def test_rank_reads_multiple_previews_and_keeps_title_only_video_for_segment_check(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    project = {"id": "a" * 32, "requirements": "梁靖崑本人，优先视频", "aspect": "16:9"}
    folder = paths.project_dir(project["id"])
    for index in range(3):
        Image.new("RGB", (640, 360), (index * 50, 0, 0)).save(folder / f"{index}.jpg")
    video = sourced_video() | {"id": "candidate", "preview_paths": [
        {"path": f"{i}.jpg", "at": i * 5, "kind": "video_frame"} for i in range(3)]}
    seen = []

    def chat(system, content, **kwargs):
        seen.extend(content)
        assert "禁止通过脸" in system
        return {"ranking": [{"candidate_id": "candidate", "verdict": "match", "reason": "标题相关"}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    ranking = planner.rank_candidates(project, named_shot(), [video], lambda: None)
    assert len([row for row in seen if row["type"] == "image_url"]) == 3
    assert ranking[0]["selection_blocked"] and ranking[0]["preview_frame_count"] == 3
    assert not ranking[0]["needs_segment_preview"]
    video.pop("preview_paths")
    video["thumbnail_path"] = "0.jpg"
    ranking = planner.rank_candidates(project, named_shot(), [video], lambda: None)
    assert ranking[0]["selection_blocked"] and ranking[0]["needs_segment_preview"]


def test_verify_reads_actual_frame_and_overrules_unsubstantiated_model_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    pid = "b" * 32
    Image.new("RGB", (640, 360)).save(paths.project_dir(pid) / "photo.jpg")
    video = sourced_video() | {"kind": "image", "path": "photo.jpg"}
    monkeypatch.setattr(store, "get_project", lambda _: {"requirements": "必须有梁靖崑"})
    observed = []

    def chat(system, content, **kwargs):
        observed.extend(content)
        return {"verdict": "match", "reason": "标题说是梁靖崑", "needs_source_check": False,
                "identity_evidence": {"status": "supported"}}, {}

    monkeypatch.setattr(planner, "chat", chat)
    result = planner.verify(pid, named_shot(), video)
    assert result["verdict"] == "uncertain" and result["selection_blocked"]
    data = json.loads(observed[0]["text"])
    assert data["source_evidence"]["title"] == video["source_evidence"]["title"]
    assert result["frame_count"] == 1


def event_observation(video, **changes):
    return name_labels(event_evidence=[{"quote": video["source_evidence"]["title"], "relation": "same_event",
                                       "source_url": video["source_url"]}], **changes)


def test_real_quote_for_same_tournament_cannot_hide_wrong_round_or_opponent():
    video = sourced_video(title="2026 WTT中国大满贯 MS R64 梁靖崑 vs 陈垣宇", description="梁靖崑与陈垣宇交手。")
    shot = named_shot(event="赛事：WTT中国大满贯；轮次：R32；对阵：梁靖崑 vs 郭冠宏；2026年", evidence_mode="event")
    result = identity.assess(shot, video, event_observation(video), frame_count=5)
    assert result["status"] == "conflict"
    fields = {row["field"]: row["status"] for row in result["event_checks"]}
    assert fields["轮次"] == "conflict" and fields["对阵"] == "insufficient"


def test_same_round_and_tournament_but_other_opponent_stays_blocked():
    video = sourced_video(title="2026 WTT中国大满贯 MS R32 梁靖崑 vs 陈垣宇", description="梁靖崑与陈垣宇交手。")
    shot = named_shot(event="赛事：WTT中国大满贯；轮次：R32；对阵：梁靖崑 vs 郭冠宏；2026年", evidence_mode="event")
    result = identity.assess(shot, video, event_observation(video), frame_count=5)
    assert result["status"] == "insufficient" and result["selection_blocked"]
    assert "对阵" in result["reason"]


def test_exact_date_is_required_and_publication_date_does_not_establish_event_date():
    video = sourced_video(title="2026 WTT中国大满贯 梁靖崑男单比赛", description="梁靖崑对阵陈垣宇。")
    video["source_evidence"]["published_at"] = "2026-10-07"
    shot = named_shot(event="赛事：WTT中国大满贯；日期：2026年10月7日", evidence_mode="event")
    result = identity.assess(shot, video, event_observation(video), frame_count=5)
    assert result["status"] == "insufficient" and "日期" in result["reason"]
    video["source_evidence"]["title"] += " 2026-10-06"
    result = identity.assess(shot, video, event_observation(video), frame_count=5)
    assert result["status"] == "conflict"


def test_missing_year_cannot_be_invented_from_same_event_claim():
    video = sourced_video(title="WTT中国大满贯 梁靖崑比赛", description="梁靖崑对阵陈垣宇。")
    result = identity.assess(named_shot(event="2026 WTT中国大满贯"), video, event_observation(video), frame_count=5)
    assert result["status"] == "insufficient" and "年份" in result["reason"]


def test_model_cannot_quote_only_safe_excerpt_to_skip_missing_date_and_round():
    video = sourced_video(title="2026 WTT中国大满贯 梁靖崑 MS R32 2026年10月7日", description="其他比赛在10月6日举行。")
    observations = name_labels(event_evidence=[{"quote": "2026 WTT中国大满贯 梁靖崑", "relation": "same_event",
                                                "source_url": video["source_url"]}])
    shot = named_shot(event="赛事：WTT中国大满贯；日期：2026年10月7日；轮次：R32")
    result = identity.assess(shot, video, observations, frame_count=5)
    assert result["status"] == "insufficient"
    assert "日期" in result["reason"] and "轮次" in result["reason"]


def test_round_notation_does_not_confuse_qualifiers_with_final_or_advancement():
    assert identity.event_rounds("MS R32 1/16决赛") == {("size", 32)}
    assert identity.event_rounds("quarter final 八强赛") == {("size", 8)}
    assert identity.event_rounds("半决赛") == {("size", 4)}
    assert identity.event_rounds("决赛") == {("size", 2)}
    assert identity.event_rounds("晋级男单八强") == {("size", 16)}
    checks = identity.event_checks("轮次：第二轮", "男单 MS R32", [])
    assert checks[0]["status"] == "insufficient"


def test_non_sports_conference_nameplate_and_dated_primary_text_can_support_event():
    url = "https://conference.example.org/session/42"
    title = "2026年10月7日，开源技术论坛举行，李明作主题演讲"
    video = {"kind": "video", "title": title, "source_url": url,
             "source_evidence": {"url": url, "title": title, "description": "论坛嘉宾李明讨论工程实践。",
                                 "readable": True, "evidence_origin": "publisher_page"}}
    shot = named_shot(persons=[{"name": "李明", "aliases": []}], evidence_mode="event",
                      event="活动：开源技术论坛；日期：2026-10-07")
    observations = {"visual_evidence": [{"kind": "nameplate", "quote": "李明 主题演讲", "frame_index": 1,
                                        "context": "讲台姓名牌关联正在发言的嘉宾"}],
                    "event_evidence": [{"quote": title, "relation": "same_event", "source_url": url}]}
    result = identity.assess(shot, video, observations, frame_count=3, has_segment_frames=True)
    assert result["status"] == "supported"
    assert not any(row["kind"] == "publisher_domain" for row in result["evidence"])


def test_full_names_and_provided_aliases_match_source_and_scoreboard_without_face_recognition():
    video = sourced_video(title="LIANG Jingkun vs KUO Guan-Hong | MS R32 | China Smash 2026", description="")
    shot = named_shot(persons=[{"name": "梁靖崑", "aliases": ["梁靖昆", "LIANG Jingkun"]},
                               {"name": "郭冠宏", "aliases": ["KUO Guan-Hong"]}],
                      event="赛事：WTT中国大满贯；轮次：R32；对阵：梁靖崑 vs 郭冠宏；2026年")
    observations = event_observation(video, visual_evidence=[{"kind": "scoreboard", "quote": "LIANG JINGKUN / KUO GUAN-HONG",
                                                             "frame_index": 0, "context": "完整计分牌列出参赛者"}])
    result = identity.assess(shot, video, observations, frame_count=5)
    assert result["status"] == "supported" and result["matched_names"] == ["梁靖崑", "郭冠宏"]


def test_model_proposed_nickname_surname_and_title_aliases_are_not_person_confirmation():
    for alias in ("LIANG", "大胖", "梁老师", "靖崑"):
        shot = named_shot(persons=[{"name": "梁靖崑", "aliases": [alias]}])
        observations = name_labels(visual_evidence=[{"kind": "caption", "quote": alias, "frame_index": 0,
                                                    "context": "节目字幕"}])
        assert identity.assess(shot, sourced_video(), observations, frame_count=1)["selection_blocked"]


def long_plan_project():
    paragraphs = [f"第{number}段，梁靖崑继续比赛。比分来到10比8，随后双方来到10比10。"
                  "解说提到“资料画面”与本场录像需要区分。郭冠宏等待下一次发球，不能添加原稿没有的动作。"
                  for number in range(1, 39)]
    return {"script": "\n\n".join(paragraphs), "style": "体育解说", "requirements": "保持每个字和标点",
            "shots": [{"id": "old-shot", "narration": "原有计划保持完整"}]}


def test_long_model_plan_batches_with_full_context_and_exact_original_preservation(monkeypatch):
    project = long_plan_project()
    original = deepcopy(project)
    requests, contracts, progress = [], [], []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        request = json.loads(content)
        if request.get("stage") == editorial_contract.VERSION:
            contracts.append(request)
            assert kwargs["reasoning"] is False
            return source_contract_response(request), {"model": "test-contract", "usage": {"total_tokens": 13,
                                                   "prompt_tokens_details": {"cached_tokens": 3}}}
        requests.append(request)
        rows = [{"source_unit_id": unit["source_unit_id"], "visual": "画面条件"} for unit in request["source_units"]]
        return {"shots": rows}, {"model": "test-model", "usage": {"total_tokens": 23,
                                        "prompt_tokens_details": {"cached_tokens": 7}}}

    monkeypatch.setattr(planner, "chat", chat)
    shots, info = planner.plan(project, "model", False, progress=lambda done, total: progress.append((done, total)))
    assert len(requests) > 1 and all(len(request["script"]) <= 900 for request in requests)
    assert all(request["full_script_context"] == project["script"] for request in requests)
    assert [request["batch_index"] for request in requests] == list(range(1, len(requests) + 1))
    assert planner.normalized("".join(request["script"] for request in requests)) == planner.normalized(project["script"])
    assert planner.normalized("".join(shot["narration"] for shot in shots)) == planner.normalized(project["script"])
    assert project == original
    assert info["batch_count"] == len(requests)
    assert len(contracts) == (len(shots) + editorial_contract.BATCH_SIZE - 1) // editorial_contract.BATCH_SIZE
    assert all("".join(u["text"] for u in r["script_units"]) == project["script"]
               and r["requirements"] == project["requirements"]
               and len(r["targets"]) <= editorial_contract.BATCH_SIZE for r in contracts)
    assert [ref for r in contracts for ref in r["targets"]] == [f"s{i}" for i in range(1, len(shots) + 1)]
    assert [s["source_unit_id"] for s in shots] == [u["source_unit_id"] for r in requests for u in r["source_units"]]
    assert [editorial_contract.for_shot(s, project)["shot_id"] for s in shots] == [s["id"] for s in shots]
    assert info["plan_consistency"]["contract_count"] == len(shots)
    assert info["usage"]["total_tokens"] == 23 * len(requests) + 13 * len(contracts)
    assert info["usage"]["prompt_tokens_details"]["cached_tokens"] == 7 * len(requests) + 3 * len(contracts)
    assert progress == [(i, len(requests) + 1) for i in range(len(requests) + 2)]


def test_failed_later_batch_does_not_mix_partial_plan_or_mark_complete(monkeypatch):
    project = long_plan_project()
    original = deepcopy(project)
    calls, progress = [], []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request)
        rows = [{"source_unit_id": unit["source_unit_id"]} for unit in request["source_units"]]
        if len(calls) == 2:
            rows.pop()
        return {"shots": rows}, {}

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="第 2/.*整份计划未写入.*片段编号"):
        planner.plan(project, "model", False, progress=lambda done, total: progress.append((done, total)))
    assert len(calls) == 2 and project == original
    assert [done for done, _ in progress] == [0, 1]


def test_long_plan_can_be_cancelled_between_cloud_batches(monkeypatch):
    project = long_plan_project()
    original = deepcopy(project)
    calls, completed = [], []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request)
        return {"shots": [{"source_unit_id": unit["source_unit_id"]} for unit in request["source_units"]]}, {}

    def check():
        if completed and completed[-1] == 1:
            raise RuntimeError("cancelled at batch boundary")

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(RuntimeError, match="cancelled"):
        planner.plan(project, "model", False, check=check, progress=lambda done, _: completed.append(done))
    assert len(calls) == 1 and project == original


def test_batch_boundaries_preserve_long_unpunctuated_text_without_dropping_characters():
    script = "甲" * 901 + "，" + "乙" * 1100 + "。\n\n第三段。"
    batches = planner.script_batches(script)
    assert "".join(batches) == script
    assert all(0 < len(batch) <= 900 for batch in batches)


def test_model_cannot_change_narration_when_source_unit_ids_are_correct(monkeypatch):
    project = {"script": "梁靖崑比赛，比分10比8。随后来到10比10。", "style": "解说", "requirements": "保留原文"}
    original, stages = deepcopy(project), []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        data = json.loads(content)
        stages.append(data.get("stage", "plan"))
        if data.get("stage") == editorial_contract.VERSION:
            assert "".join(u["text"] for u in data["script_units"]) == project["script"]
            assert "篡改人名和比分" not in json.dumps(data, ensure_ascii=False)
            return source_contract_response(data, rows={"s1": {
                "facts": [["person", "s1", "梁靖崑"]],
            }}), {}
        return {"shots": [{"source_unit_id": unit["source_unit_id"], "narration": "篡改人名和比分！",
                           "visual": "实际画面"} for unit in data["source_units"]]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    shots, _ = planner.plan(project, "model", False)
    assert "".join(shot["narration"] for shot in shots) == project["script"]
    assert [shot["source_unit_id"] for shot in shots] == [f"u{i:04d}" for i in range(1, len(shots) + 1)]
    assert stages == ["plan", editorial_contract.VERSION] and project == original
    assert editorial_contract.for_shot(shots[0], project)["facts"][0]["quote"] == "梁靖崑"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "cross_batch"])
def test_source_unit_ids_must_be_complete_unique_ordered_and_local_to_current_batch(monkeypatch, corruption):
    project = long_plan_project()
    original = deepcopy(project)
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    previous = []

    def chat(system, content, **kwargs):
        data = json.loads(content)
        ids = [unit["source_unit_id"] for unit in data["source_units"]]
        if corruption == "missing":
            ids.pop()
        elif corruption == "duplicate":
            ids[-1] = ids[0]
        elif corruption == "reordered":
            ids.reverse()
        elif previous:
            ids[0] = previous[0]
        previous.extend(ids)
        return {"shots": [{"source_unit_id": source_id} for source_id in ids]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="片段编号"):
        planner.plan(project, "model", False)
    assert project == original


def test_rewrite_mode_keeps_explicit_model_narration_without_fixed_units(monkeypatch):
    project = {"script": "原来的稿件。", "style": "解说", "requirements": "改成口语"}
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    stages = []

    def chat(system, content, **kwargs):
        data = json.loads(content)
        stages.append(data.get("stage", "plan"))
        if data.get("stage") == "plan_consistency":
            return consistency_response(data), {}
        return {"shots": [{"narration": "整理后的口播。"}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    shots, info = planner.plan(project, "model", True)
    assert shots[0]["narration"] == "整理后的口播。" and info["rewritten"]
    assert stages == ["plan", "plan_consistency"]


def test_plan_accepts_twelve_search_strings_then_deduplicates_and_limits_active_queries(monkeypatch):
    project = {"script": "工程师检查轨道。", "style": "解说", "requirements": "保持原文"}
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    queries = ["轨道检查", " rail inspection ", "rail inspection", "", "track maintenance",
               "铁路", "railway", "rail", "fastener", "维修", "检修", "工程"]
    generated_queries = []
    extract = editorial_contract.extract

    def extract_after_initial_query_validation(project, shots, **kwargs):
        generated_queries.extend(s["search_queries"] for s in shots)
        return extract(project, shots, **kwargs)

    def chat(system, content, **kwargs):
        data = json.loads(content)
        if data.get("stage") == editorial_contract.VERSION:
            return source_contract_response(data, queries=("轨道部件检查", "工程师检查作业")), {}
        assert "仅rewrite=true" in system
        return {"shots": [{"source_unit_id": unit["source_unit_id"], "query": "轨道检查",
                           "search_queries": queries} for unit in data["source_units"]]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    monkeypatch.setattr(editorial_contract, "extract", extract_after_initial_query_validation)
    shots, _ = planner.plan(project, "model", False)
    assert generated_queries == [["轨道检查", "rail inspection", "track maintenance"]]
    assert shots[0]["search_queries"] == ["轨道部件检查", "工程师检查作业"]
    assert shots[0]["query"] == "轨道部件检查"
    assert editorial_contract.for_shot(shots[0], project)["must"] == []
    assert shots[0]["narration"] == project["script"]


@pytest.mark.parametrize("queries", ["轨道检查", {"query": "轨道"}, [{"query": "轨道"}],
                                    ["x" * 201], [f"query{i}" for i in range(13)]])
def test_plan_still_rejects_non_array_non_string_or_unbounded_search_queries(monkeypatch, queries):
    project = {"script": "工程师检查轨道。", "style": "解说", "requirements": "保持原文"}
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        data = json.loads(content)
        return {"shots": [{"source_unit_id": unit["source_unit_id"], "search_queries": queries}
                          for unit in data["source_units"]]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="检索词格式无效"):
        planner.plan(project, "model", False)
