from copy import deepcopy
import json
from pathlib import Path

import pytest

from jinggao import config, editorial_contract, plan_consistency, planner
from planner_helpers import consistency_response, source_contract_response


def sample_plan(script="工程师检查轨道。\n列车驶过车站。"):
    project = {"script": script, "requirements": "保留原文和明确地点。", "style": "资料解说"}
    shots = planner.rule_shots(script)
    for index, shot in enumerate(shots):
        shot.update(event_scope="background", evidence_mode="generic", overlay_label="资料画面",
                    audio={"path": f"paid-{index}.wav", "duration": 7.5},
                    selected_asset=f"saved-{index}", clip_start=3.2)
    project["shots"] = deepcopy(shots)
    return project, shots


def response_for(project, shots, *, contract=False, **kwargs):
    return consistency_response({"full_script": project["script"],
                                 "visual_contract": plan_consistency.VISUAL_CONTRACT if contract else None,
                                 "shots": [shot | {"shot_id": shot["id"]} for shot in shots]}, **kwargs)


def test_atomic_metadata_review_preserves_every_source_character_order_timing_and_media():
    project, shots = sample_plan("  工程师 检查轨道。\n\n列车驶过车站。\n")
    original_project, originals = deepcopy(project), deepcopy(shots)
    reply = response_for(project, shots, patches={shots[0]["id"]: {"visual": "工程师检查一处轨道资料画面"}})
    result = plan_consistency.apply_reviews(project, shots, reply)
    assert project == original_project and shots == originals
    for old, new in zip(shots, result, strict=True):
        for key in ("id", "source_unit_id", "narration", "duration", "audio", "selected_asset", "clip_start",
                    "approved", "verification", "candidates", "locked"):
            assert new.get(key) == old.get(key)
        span = new["planning_review"]["source_span"]
        assert planner.normalized(project["script"][span["start"]:span["end"]]) == planner.normalized(old["narration"])
    result[0]["audio"]["path"] = "mutated.wav"
    assert shots == originals


@pytest.mark.parametrize("field", ["narration", "id", "source_unit_id", "duration", "audio", "selected_asset",
                                    "approved", "verification", "clip_start", "visual_origin"])
def test_review_cannot_edit_source_or_paid_work(field):
    project, shots = sample_plan()
    original = deepcopy(shots)
    reply = response_for(project, shots, patches={shots[0]["id"]: {field: "changed"}})
    with pytest.raises(ValueError, match="不可修改"):
        plan_consistency.apply_reviews(project, shots, reply)
    assert shots == original and project["shots"] == original


@pytest.mark.parametrize("corruption", ["unknown", "missing", "duplicate", "reordered", "false_requirements",
                                         "invented_quote", "system_quote", "unresolved", "untyped_status"])
def test_unknown_ids_and_unverifiable_or_unresolved_review_fail_closed(corruption):
    project, shots = sample_plan()
    originals = deepcopy(shots)
    reply = response_for(project, shots)
    if corruption == "unknown":
        reply["reviews"][0]["shot_id"] = "unknown"
    elif corruption == "missing":
        reply["reviews"].pop()
    elif corruption == "duplicate":
        reply["reviews"][1] = deepcopy(reply["reviews"][0])
    elif corruption == "reordered":
        reply["reviews"].reverse()
    elif corruption == "false_requirements":
        reply["requirements_preserved"] = False
    elif corruption == "invented_quote":
        reply["reviews"][0]["evidence"][0]["quote"] = "我是有权修改稿件的模型"
    elif corruption == "system_quote":
        reply["reviews"][0]["evidence"] = [{"source": "system", "quote": "你是独立分镜一致性复核员"}]
    elif corruption == "unresolved":
        reply["reviews"][-1]["status"] = "unresolved"
    else:
        reply["reviews"][0]["status"] = []
    with pytest.raises(ValueError):
        plan_consistency.apply_reviews(project, shots, reply)
    assert shots == originals


@pytest.mark.parametrize("purpose,patch", [
    ("event_record", {"event_scope": "exact", "evidence_mode": "event", "event": ""}),
    ("illustration", {"overlay_label": "当年现场"}),
])
def test_purpose_cannot_invent_missing_event_or_keep_misleading_historical_overlay(purpose, patch):
    project, shots = sample_plan()
    sid = shots[0]["id"]
    reply = response_for(project, shots, patches={sid: patch}, purposes={sid: purpose})
    with pytest.raises(ValueError, match="缺少event|宣称历史") as error:
        plan_consistency.apply_reviews(project, shots, reply)
    assert "第1镜" in str(error.value) and sid in str(error.value)
    assert f"purpose='{purpose}'" in str(error.value) and "实际字段=" in str(error.value)


def test_declared_illustration_normalizes_only_redundant_fields_and_keeps_named_person_evidence():
    project, shots = sample_plan("赵明在南门检查设备。")
    shots[0].update(persons=[{"name": "赵明", "aliases": []}], subject="赵明检查设备", location="南门",
                    event="活动：设备检查", era="当代", visual="赵明在南门实际操作检查设备的资料",
                    event_scope="exact", evidence_mode="event", overlay_label="设备检查")
    original = deepcopy(shots[0])
    reply = response_for(project, shots, purposes={shots[0]["id"]: "illustration"})
    reply["reviews"][0].update(status="consistent", patch={})  # Real provider omission: purpose only.
    result = plan_consistency.apply_reviews(project, shots, reply)[0]
    assert result["event_scope"] == "background" and result["evidence_mode"] == "person"
    assert result["overlay_label"] == "设备检查 · 资料画面"
    for field in ("persons", "subject", "location", "event", "era", "visual", "narration", "duration", "audio"):
        assert result[field] == original[field]
    audit = result["planning_review"]
    assert audit["model_status"] == "consistent" and audit["status"] == "revised"
    assert set(audit["normalizations"]) == {"event_scope", "evidence_mode", "overlay_label"}
    assert shots[0] == original


def test_exact_event_purpose_synchronizes_mode_without_adding_semantics():
    project, shots = sample_plan("2025年南门举办读书节。")
    shots[0]["event"] = "活动：南门读书节；日期：2025年"
    reply = response_for(project, shots, purposes={shots[0]["id"]: "event_record"})
    reply["reviews"][0].update(status="consistent", patch={})
    result = plan_consistency.apply_reviews(project, shots, reply)[0]
    assert result["event_scope"] == "exact" and result["evidence_mode"] == "event"
    assert result["event"] == shots[0]["event"]


def test_known_json_wrapper_and_idempotent_revision_do_not_reject_valid_review():
    project, shots = sample_plan()
    reply = response_for(project, shots)
    reply["type"] = "json_object"
    reply["reviews"][0].update(status="revised", patch={"overlay_label": shots[0]["overlay_label"]})
    result = plan_consistency.apply_reviews(project, shots, reply)
    assert result[0]["planning_review"]["status"] == "consistent"
    assert result[0]["planning_review"]["model_status"] == "revised"
    assert result[0]["planning_review"]["changed_fields"] == []
    reply["narration"] = "仍不能添加受保护字段"
    with pytest.raises(ValueError, match="未知顶层字段.*narration"):
        plan_consistency.apply_reviews(project, shots, reply)


@pytest.mark.parametrize("purpose", ["illustration", "event_record"])
def test_duplicate_patch_purpose_is_ignored_without_changing_actual_revision_status(purpose):
    project, shots = sample_plan()
    if purpose == "event_record":
        shots[0].update(event="工程师检查轨道", event_scope="exact", evidence_mode="event")
    reply = response_for(project, shots, purposes={shots[0]["id"]: purpose})
    reply["reviews"][0].update(status="revised", patch={"purpose": purpose})
    original = deepcopy(reply)
    result = plan_consistency.apply_reviews(project, shots, reply)[0]
    audit = result["planning_review"]
    assert audit["status"] == "consistent" and audit["model_status"] == "revised"
    assert audit["changed_fields"] == [] and audit["ignored_fields"] == {"patch.purpose": purpose}
    assert "purpose" not in result and "purpose" not in plan_consistency.PATCH_FIELDS
    assert reply == original


@pytest.mark.parametrize("patch", [
    {"purpose": "event_record"},
    {"purpose": "Illustration"},
    {"purpose": "illustration", "narration": "不能改原稿"},
    {"purpose": "illustration", "unknown": "也不能新增字段"},
])
def test_conflicting_patch_purpose_and_other_unknown_fields_remain_rejected(patch):
    project, shots = sample_plan()
    reply = response_for(project, shots, patches={shots[0]["id"]: patch})
    original = deepcopy(reply)
    with pytest.raises(ValueError, match="不可修改字段"):
        plan_consistency.apply_reviews(project, shots, reply)
    assert reply == original


@pytest.mark.parametrize("unresolved_semantics", [False, True])
def test_duplicate_patch_purpose_and_untouched_raw_response_are_diagnosed_even_on_semantic_failure(
        tmp_path, unresolved_semantics):
    project, shots = sample_plan()
    visual = "工程师检查一段轨道的资料画面"
    if unresolved_semantics:
        visual += "，随后切换到列车驶入车站的画面"
    reply = response_for(project, shots, contract=True, patches={shots[0]["id"]: {
        "purpose": "illustration", "visual": visual,
    }})
    original = deepcopy(reply)
    kwargs = {"chat": lambda *args, **kwargs: (reply, {}), "check": lambda: None,
              "diagnostic_dir": tmp_path}
    if unresolved_semantics:
        with pytest.raises(ValueError, match="single_clip_requires_montage"):
            plan_consistency.review(project, shots, **kwargs)
    else:
        reviewed, _ = plan_consistency.review(project, shots, **kwargs)
        assert visual in reviewed[0]["visual"]
        assert "visual" in reviewed[0]["planning_review"]["changed_fields"]
        assert reviewed[0]["planning_review"]["status"] == "revised"
    saved = json.loads(next(tmp_path.glob("plan-consistency-*.json")).read_text())
    assert saved["status"] == ("failed" if unresolved_semantics else "applied")
    assert saved["response"] == original and reply == original
    assert saved["ignored_response_fields"] == [{"index": 1, "shot_id": shots[0]["id"],
        "field": "patch.purpose", "value": "illustration", "reason": "duplicates_row_purpose"}]


def test_raw_review_is_saved_before_validation_with_only_safe_model_metadata(tmp_path, monkeypatch):
    project, shots = sample_plan()
    reply = response_for(project, shots)
    seen = []

    def apply(*args, **kwargs):
        files = list(tmp_path.glob("plan-consistency-*.json"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text())
        assert saved["status"] == "received" and saved["response"] == reply
        assert saved["request"]["shots"][0]["shot_id"] == shots[0]["id"]
        assert "audio" not in saved["request"]["shots"][0]
        assert saved["planner_model_info"] == {"model": "planning", "usage": {"total_tokens": 11}}
        assert saved["review_model_info"] == {"model": "review", "usage": {"total_tokens": 7}}
        assert "secret-value" not in files[0].read_text()
        seen.append(files[0])
        raise ValueError("第1镜：具体的测试冲突")

    monkeypatch.setattr(plan_consistency, "apply_reviews", apply)
    with pytest.raises(ValueError, match="第1镜.*诊断"):
        plan_consistency.review(project, shots,
            chat=lambda *a, **kw: (reply, {"model": "review", "usage": {"total_tokens": 7},
                                         "headers": {"Authorization": "secret-value"}}),
            check=lambda: None, model_info={"model": "planning", "usage": {"total_tokens": 11},
                                          "api_key": "secret-value", "settings": {"api_key": "secret-value"}},
            diagnostic_dir=tmp_path)
    saved = json.loads(seen[0].read_text())
    assert saved["status"] == "failed" and saved["response"] == reply
    assert "第1镜" in saved["error"]


def test_creator_credit_does_not_force_creator_on_screen_and_requirements_are_valid_evidence():
    project, shots = sample_plan("节目由赵明和钱芳编导。")
    project["requirements"] = "编导署名以官方作品页面证明，不要求本人出镜。"
    shots[0].update(persons=[{"name": "赵明"}, {"name": "钱芳"}], evidence_mode="person")
    reply = response_for(project, shots, patches={shots[0]["id"]: {
        "persons": [], "evidence_mode": "generic", "visual": "官方作品页面署名资料，核实编导归属",
        "query": "节目 官方 编导署名", "preferred_media": "image", "overlay_label": "官方署名资料",
    }})
    reply["reviews"][0]["evidence"] = [{"source": "requirements", "quote": project["requirements"]}]
    reviewed = plan_consistency.apply_reviews(project, shots, reply)[0]
    assert reviewed["persons"] == [] and reviewed["evidence_mode"] == "generic"
    assert reviewed["narration"] == shots[0]["narration"]
    assert reviewed["search_queries"][0] == "节目 官方 编导署名"


def test_referenced_result_keeps_its_specific_event_and_location():
    project, shots = sample_plan("2025年9月6日，陈晨在海港市体育馆赢得星河杯男子100米决赛。\n这个结果回应了此前的期待。")
    sid = shots[-1]["id"]
    event = "赛事：星河杯男子100米决赛；日期：2025年9月6日；地点：海港市体育馆"
    reply = response_for(project, shots, patches={sid: {
        "event": event, "event_scope": "exact", "evidence_mode": "event", "location": "海港市体育馆",
        "event_persons": [{"name": "陈晨", "aliases": []}], "visual": "该场决赛正式成绩牌资料",
        "overlay_label": "本场比赛回顾", "query": "星河杯 陈晨 决赛成绩",
    }}, purposes={sid: "event_record"})
    # The antecedent can be in a different source unit, and is still a verbatim source quote.
    reply["reviews"][-1]["evidence"] = [{"source": "script", "quote": shots[0]["narration"]}]
    reviewed = plan_consistency.apply_reviews(project, shots, reply)[-1]
    assert reviewed["event"] == event and reviewed["location"] == "海港市体育馆"
    assert reviewed["event_scope"] == "exact" and reviewed["evidence_mode"] == "event"
    assert reviewed["persons"] == [] and reviewed["event_persons"][0]["name"] == "陈晨"


def test_contemporary_building_can_illustrate_history_without_claiming_historical_footage():
    project, shots = sample_plan("1918年9月1日，博物馆在南门举行开幕典礼。")
    shots[0].update(era="1918年", event="活动：博物馆开幕典礼；日期：1918年9月1日",
                    event_scope="exact", evidence_mode="event", location="南门")
    reply = response_for(project, shots, patches={shots[0]["id"]: {
        "era": "当代", "event_scope": "background", "evidence_mode": "generic", "overlay_label": "南门现址资料",
        "visual": "南门当代现址示意，开幕日期另以可靠档案核实，不宣称当日录像",
    }}, purposes={shots[0]["id"]: "illustration"})
    reviewed = plan_consistency.apply_reviews(project, shots, reply)[0]
    assert reviewed["event"] == shots[0]["event"] and reviewed["location"] == "南门"
    assert reviewed["era"] == "当代" and reviewed["event_scope"] == "background"
    assert reviewed["overlay_label"] == "南门现址资料"


def test_statistical_period_does_not_become_footage_year_and_optional_actions_stay_optional():
    project, shots = sample_plan("2024年业务量见官方统计公报。\n一次运送会经过揽收、分拣、运输和派送。")
    shots[0]["era"] = "2024年"
    shots[1].update(duration=5, visual="揽收、分拣、运输和派送全部连续完成")
    reply = response_for(project, shots, patches={
        shots[0]["id"]: {"era": "", "visual": "2024年统计公报页面资料，不限制发布年份", "preferred_media": "image"},
        shots[1]["id"]: {"visual": "选取一个代表性分拣动作作为运输流程资料", "query": "包裹 分拣"},
    })
    result = plan_consistency.apply_reviews(project, shots, reply)
    assert result[0]["era"] == "" and "2024年" in result[0]["narration"]
    assert result[1]["duration"] == 5 and result[1]["narration"] == shots[1]["narration"]
    assert "一个代表性" in result[1]["visual"]


def test_review_cannot_add_person_whose_name_only_appears_in_its_own_reply():
    project, shots = sample_plan()
    reply = response_for(project, shots, patches={shots[0]["id"]: {
        "persons": [{"name": "新增姓名", "aliases": ["工程师"]}], "evidence_mode": "person",
    }})
    with pytest.raises(ValueError, match="完整姓名没有原稿"):
        plan_consistency.apply_reviews(project, shots, reply)


@pytest.mark.parametrize("failure", [ValueError("响应损坏"), planner.ModelUnavailable("账户额度不足（HTTP 402）")])
def test_failed_review_is_not_retried_and_never_changes_existing_plan_or_paid_audio(failure):
    project, shots = sample_plan()
    originals, calls = deepcopy(project), []

    def chat(*args, **kwargs):
        calls.append(1)
        raise failure

    with pytest.raises(type(failure), match="响应损坏|402"):
        plan_consistency.review(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 1 and project == originals and shots == originals["shots"]


def test_planner_extracts_one_complete_source_contract_before_return_and_counts_usage(monkeypatch):
    project, _ = sample_plan()
    original, requests = deepcopy(project), []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **kwargs):
        request = json.loads(content)
        requests.append(request)
        if request.get("stage") == editorial_contract.VERSION:
            assert kwargs.get("reasoning") is False
            assert "".join(u["text"] for u in request["script_units"]) == project["script"]
            assert request["requirements"] == project["requirements"]
            assert request["readonly_visual"] == []
            assert not any("audio" in unit or "selected_asset" in unit for unit in request["script_units"])
            assert "paid-" not in content and "saved-" not in content
            return source_contract_response(request), {"model": "contract-model", "usage": {"total_tokens": 7}}
        assert not kwargs.get("reasoning", False)
        return {"shots": [{"source_unit_id": unit["source_unit_id"], "narration": "模型不能改口播"}
                          for unit in request["source_units"]]}, {"model": "plan-model", "usage": {"total_tokens": 11}}

    monkeypatch.setattr(planner, "chat", chat)
    shots, info = planner.plan(project, "model", False)
    assert len(requests) == 2 and requests[-1]["stage"] == editorial_contract.VERSION
    assert info["usage"]["total_tokens"] == 18 and info["plan_consistency"]["contract_count"] == len(shots)
    assert info["plan_consistency"]["status"] == "source_bound"
    assert info["plan_consistency"]["model_requests"] == 1
    assert requests[-1]["targets"] == [f"s{i}" for i in range(1, len(shots) + 1)]
    assert [shot["source_unit_id"] for shot in shots] == [u["source_unit_id"] for u in requests[0]["source_units"]]
    assert [editorial_contract.for_shot(s, project)["shot_id"] for s in shots] == [s["id"] for s in shots]
    assert planner.normalized("".join(shot["narration"] for shot in shots)) == planner.normalized(project["script"])
    assert project == original


@pytest.mark.parametrize("corruption", ["invented_quote", "missing", "reordered"])
def test_failed_final_source_contract_never_returns_generated_plan(monkeypatch, corruption):
    project, _ = sample_plan()
    original = deepcopy(project)
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed", "plan_retries": "1"})
    stages, progress = [], []

    def chat(system, content, **kwargs):
        request = json.loads(content)
        stages.append(request.get("stage", "plan"))
        if request.get("stage") == editorial_contract.VERSION:
            reply = source_contract_response(request)
            if corruption == "invented_quote":
                ref = request["targets"][-1]
                reply["contracts"][-1]["facts"] = [["activity", ref, "原文并不存在的具体动作"]]
            elif corruption == "missing":
                reply["contracts"].pop()
            else:
                reply["contracts"].reverse()
            return reply, {}
        return {"shots": [{"source_unit_id": unit["source_unit_id"]} for unit in request["source_units"]]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="编辑契约提取失败，计划未写入"):
        planner.plan(project, "model", False, progress=lambda done, total: progress.append((done, total)))
    assert stages == ["plan", editorial_contract.VERSION] and project == original
    assert progress == [(0, 2), (1, 2)]


def test_explicit_rewrite_output_is_frozen_during_review():
    project, shots = sample_plan("原文保持在项目里。")
    shots[0]["narration"] = "用户允许的口语化结果。"
    reviewed = plan_consistency.apply_reviews(project, shots, response_for(project, shots), rewrite=True)
    assert reviewed[0]["narration"] == "用户允许的口语化结果。"
    assert reviewed[0]["planning_review"]["source_span"] is None
    assert project["script"] == "原文保持在项目里。"


def test_mismatched_original_is_rejected_before_any_model_request():
    project, shots = sample_plan()
    shots[0]["narration"] = "已被篡改。"
    calls = []
    with pytest.raises(ValueError, match="口播与原稿不一致"):
        plan_consistency.review(project, shots, chat=lambda *a, **kw: calls.append(1), check=lambda: None)
    assert calls == []


def test_data_period_and_publication_year_are_different_roles_even_across_source_units():
    project, shots = sample_plan("主管部门于2025年公布报告，2024年产量达到一万吨。\n这说的是那一年的产量。")
    for shot in shots:
        shot.update(era="2024年", visual="工厂资料画面")
    shots[0]["exclusions"] = "排除其他年份发布会"
    assert plan_consistency._statistics_years(project["script"]) == {"2024"}
    for shot in shots:
        assert any(issue["code"] == "statistics_period_as_footage_year" and issue["blocking"]
                   for issue in plan_consistency.planning_issues(project, shot))
    assert any(issue["code"] == "data_period_as_publication_date"
               for issue in plan_consistency.planning_issues(project, shots[0]))
    with pytest.raises(ValueError, match="statistics_period_as_footage_year"):
        plan_consistency.apply_reviews(project, shots, response_for(project, shots))


def test_statistics_remain_in_narration_and_document_query_after_year_constraint_is_removed():
    project, shots = sample_plan("2024年产量达到一万吨，数据来自官方年度统计公报。")
    shots[0].update(era="2024年", exclusions="排除其他年份发布会", query="2024年度产量统计公报")
    response = response_for(project, shots, patches={shots[0]["id"]: {
        "era": "", "exclusions": "", "visual": "官方统计公报中包含该年度产量的一页资料",
    }})
    result = plan_consistency.apply_reviews(project, shots, response)[0]
    assert result["narration"] == shots[0]["narration"] and result["query"] == "2024年度产量统计公报"
    assert result["era"] == "" and result["planning_review"]["remaining_advisories"] == []


def test_explicit_capture_year_is_preserved_instead_of_erased_with_statistical_year():
    project, shots = sample_plan("2024年产量见统计公报。")
    project["requirements"] = "只使用2024年拍摄的工厂视频画面。"
    shots[0].update(era="2024年", visual="工作人员操作设备的资料画面")
    assert not plan_consistency.planning_issues(project, shots[0])
    reply = response_for(project, shots, patches={shots[0]["id"]: {"era": ""}})
    with pytest.raises(ValueError, match="explicit_footage_year_missing"):
        plan_consistency.apply_reviews(project, shots, reply)
    assert shots[0]["era"] == "2024年"


def test_separately_stated_event_date_is_not_confused_with_same_numbered_data_period():
    project, shots = sample_plan("2024年举行开幕典礼。\n2024年产量见统计公报。")
    shots[0].update(era="2024年", event="活动：开幕典礼；日期：2024年", visual="该届开幕典礼资料")
    assert not any(issue["blocking"] for issue in plan_consistency.planning_issues(project, shots[0]))


@pytest.mark.parametrize("visual", [
    "同一镜头中，传送带上的包裹流动，旁边工作人员扫码，背景可见货架。",
    "纸箱、货架和工作人员都在同一分拣场景，工作人员扫描包裹。",
    "工作人员操作控制台，同一画面同时展示屏幕上的运行数据。",
    "工作人员检修同一设备，镜头切到特写。",
    "主画面：工作人员分拣一个包裹。可选背景（非必需）：货车驶出或其他工作人员扫码。",
    "包裹装车或车辆驶出，选一个代表性动作，不要求单镜覆盖全流程。",
])
def test_co_present_objects_background_and_optional_examples_do_not_force_montage(visual):
    project, shots = sample_plan("不同岗位共同完成日常劳动。")
    shots[0]["visual"] = visual
    assert not any(issue["blocking"] for issue in plan_consistency.planning_issues(project, shots[0]))
    plan_consistency.apply_reviews(project, shots, response_for(project, shots))


@pytest.mark.parametrize("visual", [
    "手机地图上显示位置，画面切换到现实中工作人员上楼敲门。",
    "点击手机界面，随后切换为现实中包裹被放置的实拍画面。",
    "车辆行驶、工作人员操作、顾客接收物品的连续真实镜头组接。",
    "镜头依次展示仓库、货场和运输车辆。",
    "两个岗位交替剪辑，选一个代表性动作。",
    "设备运行，同时展示另一台手持终端与存取柜外观。",
    "设备平稳运行，突然出现故障，工作人员随后处理。",
    "镜头切到特写，然后过渡到另一个场所。",
])
def test_consistent_label_cannot_cover_forced_cross_scene_or_complete_state_chain(visual):
    project, shots = sample_plan("不同岗位共同完成日常劳动。")
    shots[0]["visual"] = visual
    reply = response_for(project, shots)
    assert reply["reviews"][0]["status"] == "consistent"
    with pytest.raises(ValueError, match="single_clip_requires_montage"):
        plan_consistency.apply_reviews(project, shots, reply)


def test_explicit_user_sequence_cannot_be_simplified_away():
    project, shots = sample_plan("介绍一道加工流程。")
    project["requirements"] = "本镜必须依次展示两个工作场景，完整流程不得省略。"
    shots[0]["visual"] = "镜头依次展示第一工序和第二工序。"
    assert not any(issue["blocking"] for issue in plan_consistency.planning_issues(project, shots[0]))
    reply = response_for(project, shots, patches={shots[0]["id"]: {"visual": "选择第一工序作代表性资料画面。"}})
    with pytest.raises(ValueError, match="明确要求的逐步/组合画面不能"):
        plan_consistency.apply_reviews(project, shots, reply)


def test_generated_brand_face_and_name_bans_need_explicit_source_boundary():
    project, shots = sample_plan("介绍普通生产劳动。")
    shots[0]["exclusions"] = "排除品牌logo、排除人脸特写、不能出现姓名条"
    codes = {issue["code"] for issue in plan_consistency.planning_issues(project, shots[0]) if issue["blocking"]}
    assert codes == {"unsupported_brand_visibility_exclusion", "unsupported_face_visibility_exclusion",
                     "unsupported_name_visibility_exclusion"}
    project["requirements"] = "不要出现品牌logo，不要出现人脸特写，禁止展示姓名条。"
    assert not plan_consistency.planning_issues(project, shots[0])
    result = plan_consistency.apply_reviews(project, shots, response_for(project, shots))[0]
    assert result["exclusions"] == shots[0]["exclusions"]


def test_name_legibility_and_identity_requirements_are_not_treated_as_name_bans():
    project, shots = sample_plan("资料需要能核实其归属。")
    shots[0]["exclusions"] = "排除缺少姓名条或姓名不清晰、身份无法核验的素材"
    assert not any(issue["blocking"] for issue in plan_consistency.planning_issues(project, shots[0]))


def test_review_receives_concrete_issues_and_only_actual_field_changes_clear_them():
    project, shots = sample_plan("2024年业务量见统计公报。")
    shots[0].update(era="2024年", visual="手机下单画面过渡到车辆运输画面")
    observed = []

    def chat(system, content, **kwargs):
        request = json.loads(content)
        observed.append(request)
        codes = {issue["code"] for item in request["planning_checks"] for issue in item["issues"]}
        assert {"statistics_period_as_footage_year", "single_clip_requires_montage"} <= codes
        reply = consistency_response(request, patches={shots[0]["id"]: {
            "era": "", "visual": "官方统计公报中该年度业务量数据的一页资料",
        }})
        # The actual corrected fields, not this bookkeeping label, decide the outcome.
        reply["reviews"][0]["status"] = "consistent"
        return reply, {}

    result, _ = plan_consistency.review(project, shots, chat=chat, check=lambda: None)
    assert len(observed) == 1
    assert result[0]["planning_review"]["model_status"] == "consistent"
    assert result[0]["planning_review"]["status"] == "revised"
    assert result[0]["planning_review"]["remaining_advisories"] == []


VISUAL_REGRESSIONS = json.loads((Path(__file__).parent / "fixtures" / "visual_plan_regressions.json").read_text())["cases"]


@pytest.mark.parametrize("case", VISUAL_REGRESSIONS, ids=lambda case: f"saved-shot-{case['source_index']}")
def test_saved_multiscene_examples_project_only_the_selected_primary(case):
    project = {"script": case["narration"], "requirements": ""}
    shot = planner.new_shot(case["narration"], visual=case["original_visual"],
                            evidence_mode="generic", event_scope="background", overlay_label="资料画面")
    original = deepcopy(shot)
    reply = response_for(project, [shot], patches={shot["id"]: {
        "visual_plan": {"mode": "single", "primary": case["primary"], "alternatives": [case["alternative"]]},
    }})
    reviewed = plan_consistency.apply_reviews(project, [shot], reply, require_visual_plan=True)[0]
    assert reviewed["narration"] == case["narration"] and shot == original
    assert reviewed["visual"] != case["original_visual"]
    assert case["primary"]["subject"] in reviewed["visual"] and case["primary"]["action"] in reviewed["visual"]
    assert reviewed["query"] == case["primary"]["query"]
    assert reviewed["search_queries"] == [case["primary"]["query"]]
    media_request = {key: reviewed.get(key) for key in planner.SHOT_FIELDS}
    assert "visual_plan" not in media_request
    assert case["alternative"]["query"] not in json.dumps(media_request, ensure_ascii=False)
    assert "selected_primary_only" in json.dumps(reviewed["planning_review"]["normalizations"])


def test_primary_selection_keeps_specific_person_place_event_and_source_facts():
    project, shots = sample_plan("赵明在南门检查设备。这段资料说明日常维护。")
    shots[0].update(persons=[{"name": "赵明", "aliases": []}], evidence_mode="person", location="南门",
                    event="赵明在南门检查设备", era="当代", subject="赵明检查设备")
    original = deepcopy(shots[0])
    reply = response_for(project, shots, patches={shots[0]["id"]: {"visual_plan": {
        "primary": {"scene": "南门", "subject": "赵明", "action": "检查一台设备", "query": "赵明 南门 设备检查"},
        "alternatives": ["赵明记录设备状态"],
    }}})
    result = plan_consistency.apply_reviews(project, shots, reply)[0]
    for field in ("narration", "persons", "location", "event", "era", "audio", "duration", "selected_asset"):
        assert result[field] == original[field]
    assert result["evidence_mode"] == "person" and result["subject"] == "赵明"
    assert "记录设备" not in result["visual"] and "记录设备" not in result["query"]


@pytest.mark.parametrize("primary,error", [
    ([{"subject": "人员", "action": "卸货"}, {"subject": "车辆", "action": "驶出"}], "单个场景"),
    ({"subject": "人员", "action": ["卸货", "驶出"]}, r"primary.action"),
    ({"subject": "人员", "action": "", "query": "人员 劳动"}, "明确选定"),
    ({"subject": "人员", "action": "卸货", "narration": "被改写的原稿"}, "其他字段"),
    ({"subject": "人员", "action": "卸货，画面切换到公路运输"}, "single_clip_requires_montage"),
])
def test_primary_label_is_not_proof_of_one_scene_and_malformed_choices_are_diagnosed(primary, error):
    project, shots = sample_plan("不同岗位一起完成工作。")
    reply = response_for(project, shots, patches={shots[0]["id"]: {
        "visual_plan": {"mode": "single", "primary": primary},
    }})
    with pytest.raises(ValueError, match=error) as caught:
        plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)
    assert "第1镜" in str(caught.value) and "visual_plan" in str(caught.value)


def test_singleton_primary_shape_and_omitted_optional_fields_are_normalized():
    project, shots = sample_plan("工程师检查轨道。")
    reply = response_for(project, shots, patches={shots[0]["id"]: {"visual_plan": {
        "primary": [{"subject": "工程师", "action": "检查一处轨道"}],
    }}})
    result = plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)[0]
    assert result["visual_plan"]["mode"] == "single"
    assert result["visual_plan"]["primary"]["scene"] == ""
    assert result["visual_plan"]["alternatives"] == []
    assert result["search_queries"] == ["工程师 检查一处轨道"]


def test_visual_contract_echo_and_natural_copresence_do_not_cause_format_or_keyword_failure():
    project, shots = sample_plan("工作人员在分拣中心劳动。")
    reply = response_for(project, shots, patches={shots[0]["id"]: {"visual_plan": {
        "primary": {"scene": "同一分拣车间", "subject": "传送带上的包裹",
                    "action": "持续流动，旁边工作人员扫码", "query": "包裹 传送带 流动"},
    }}})
    reply["visual_contract"] = plan_consistency.VISUAL_CONTRACT
    reviewed = plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)[0]
    assert "旁边工作人员扫码" in reviewed["visual"]
    reply["visual_contract"] = "unknown_contract"
    with pytest.raises(ValueError, match="未知顶层字段"):
        plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)


def test_explicit_user_combination_is_preserved_and_cannot_be_projected_to_one_action():
    project, shots = sample_plan("介绍不同工序。")
    project["requirements"] = "本镜必须依次展示卸货和装货两个工序，完整流程不得省略。"
    shots[0]["visual"] = "依次展示卸货和装货两个工序。"
    original_visual = shots[0]["visual"]
    reply = response_for(project, shots, patches={shots[0]["id"]: {
        "visual_plan": {"mode": "explicit_sequence"},
    }})
    reply["reviews"][0]["evidence"] = [{"source": "requirements", "quote": project["requirements"]}]
    result = plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)[0]
    assert result["visual"] == original_visual
    reply["reviews"][0]["patch"]["visual_plan"] = {"mode": "single", "primary": {"subject": "工人", "action": "卸货"}}
    with pytest.raises(ValueError, match="不能改成single"):
        plan_consistency.apply_reviews(project, shots, reply, require_visual_plan=True)


def test_new_production_review_requires_selection_without_retrying_or_mutating_old_plan():
    project, shots = sample_plan()
    original = deepcopy(project)
    reply, calls = response_for(project, shots), []

    def chat(system, content, **kwargs):
        calls.append(json.loads(content))
        return reply, {}

    with pytest.raises(ValueError, match="缺少visual_plan.primary"):
        plan_consistency.review(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 1 and calls[0]["visual_contract"] == plan_consistency.VISUAL_CONTRACT
    assert project == original
    # Historical responses remain replayable; production does not use this path.
    assert len(plan_consistency.apply_reviews(project, shots, reply)) == len(shots)


def test_rule_and_manual_plans_keep_legacy_shape_without_invoking_model_contract(monkeypatch):
    project, _ = sample_plan()
    original = deepcopy(project)

    def unexpected_chat(*args, **kwargs):
        pytest.fail("Rule/manual planning must not invoke a model review")

    monkeypatch.setattr(planner, "chat", unexpected_chat)
    shots, info = planner.plan(project, "rules", False)
    assert info["method"] == "rules" and not any("visual_plan" in shot for shot in shots)
    assert planner.normalized("".join(shot["narration"] for shot in shots)) == planner.normalized(project["script"])
    assert project == original


def test_malformed_primary_can_still_be_repaired_by_one_direct_legacy_review(monkeypatch):
    project, _ = sample_plan("工作人员操作设备。")
    original = deepcopy(project)
    calls = []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed", "plan_retries": "1"})
    primary = {"scene": "工位", "subject": "工作人员", "action": "操作一台设备", "query": "工作人员 操作设备"}

    def chat(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request)
        if request.get("stage") == "plan_consistency":
            assert any(issue["code"] == "visual_plan_invalid" for row in request["planning_checks"] for issue in row["issues"])
            shot = request["shots"][0]
            return consistency_response(request, patches={shot["shot_id"]: {
                "visual_plan": {"mode": "single", "primary": primary, "alternatives": ["其他设备操作"]},
            }}), {}
        assert "visual_plan" in system and "alternatives" in system
        return {"shots": [{"source_unit_id": request["source_units"][0]["source_unit_id"],
                           "visual": "多个设备各自操作", "visual_plan": {"mode": "single", "primary": [primary, primary]}}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    units = [{"source_unit_id": "u0001", "narration": project["script"]}]
    planned, _ = planner._plan_model_batch(project, project["script"], False, project["script"], 1, 1, units)
    reviewed, info = plan_consistency.review(project, planned, chat=chat, check=lambda: None)
    assert len(calls) == 2
    assert calls[-1]["stage"] == "plan_consistency" and info["review_count"] == 1
    assert reviewed[0]["subject"] == "工作人员"
    assert "多个设备" not in reviewed[0]["visual"] and "其他设备" not in reviewed[0]["visual"]
    assert project == original and reviewed[0]["narration"] == project["script"]


def test_initial_primary_and_search_suggestions_do_not_leak_into_source_contract_conditions(monkeypatch):
    project, _ = sample_plan("工作人员操作设备。")
    calls = []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})
    primary = {"scene": "工位", "subject": "工作人员", "action": "操作一台设备", "query": "工作人员 操作设备"}
    extract = editorial_contract.extract
    materialized = []

    def extract_after_materialization(project, shots, **kwargs):
        shot = shots[0]
        assert "无关转场" not in shot["visual"] and "驾驶货车" not in shot["visual"]
        assert shot["query"] == primary["query"] and shot["search_queries"] == [primary["query"]]
        assert shot["visual_plan"]["alternatives"] == ["驾驶货车"]
        materialized.append(deepcopy(shot))
        return extract(project, shots, **kwargs)

    def chat(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request)
        if request.get("stage") == editorial_contract.VERSION:
            assert request["readonly_visual"] == []
            assert "".join(u["text"] for u in request["script_units"]) == project["script"]
            assert all(suggestion not in content for suggestion in ("工位", "操作一台设备", "驾驶货车", "无关转场"))
            return source_contract_response(request, rows={"s1": {
                "facts": [["activity", "s1", "操作设备"]],
            }}, queries=("工位实拍", "设备现场")), {}
        return {"shots": [{"source_unit_id": request["source_units"][0]["source_unit_id"],
                           "visual": "不应保留的无关转场", "query": "过期查询",
                           "visual_plan": {"primary": primary, "alternatives": ["驾驶货车"]}}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    monkeypatch.setattr(editorial_contract, "extract", extract_after_materialization)
    shots, info = planner.plan(project, "model", False)
    assert len(calls) == 2 and shots[0]["narration"] == project["script"]
    assert len(materialized) == 1 and info["plan_consistency"]["version"] == editorial_contract.VERSION
    contract = editorial_contract.for_shot(shots[0], project)
    assert contract["facts"][0]["quote"] == "操作设备" and contract["must"] == []
    assert shots[0]["search_queries"] == ["工位实拍", "设备现场"]
    context = editorial_contract.context(shots[0])
    assert context["full_script"] == project["script"] and context["requirements"] == project["requirements"]
    encoded = json.dumps(context, ensure_ascii=False)
    assert all(suggestion not in encoded for suggestion in ("工位实拍", "设备现场", "驾驶货车", "无关转场", "操作一台设备"))


VISIBILITY_REGRESSIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "visibility_plan_regressions.json").read_text())["cases"]


def assert_visibility_rules(system):
    # Check both actual model stages receive the rule and its explicit-source
    # exceptions. Mocked corrections below do not establish LLM compliance.
    for rule in (
        "一个可看清的外部动作",
        "原稿或用户明确要求看清扫描、分流时必须保留",
        "购买完成、付款成功、物流全流程",
        "边看手机边拨打电话走进门",
        "有人劳动",
        "统计数字和年度由原稿口播及原稿字幕表达",
        "明确要求展示指定图表、报告、公报页面及其数字时必须保留",
        "统计字幕仍不得与口播事实冲突",
    ):
        assert rule in system


@pytest.mark.parametrize("case", VISIBILITY_REGRESSIONS, ids=lambda c: f"round5-shot-{c['source_index']}")
def test_visible_action_rules_reach_both_stages_and_single_review_can_replace_recorded_overreach(monkeypatch, case):
    project, _ = sample_plan(case["narration"])
    original, calls = deepcopy(project), []
    units = [{"source_unit_id": "u0001", "narration": case["narration"]}]
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **_kwargs):
        request = json.loads(content)
        assert_visibility_rules(system)
        calls.append(request)
        if request.get("stage") == "plan_consistency":
            assert request["full_script"] == case["narration"]
            shot = request["shots"][0]
            assert shot["visual_plan"]["primary"]["action"] == case["original_primary"]["action"]
            return consistency_response(request, patches={shot["shot_id"]: {
                "visual_plan": {"mode": "single", "primary": case["primary"], "alternatives": []},
                "preferred_media": "video",
            }}), {}
        assert request["source_units"] == units and request["requirements"] == project["requirements"]
        return {"shots": [{"source_unit_id": "u0001", "event_scope": "background",
                           "evidence_mode": "generic", "overlay_label": "资料画面",
                           "preferred_media": case["original_preferred_media"],
                           "visual_plan": {"mode": "single", "primary": case["original_primary"]}}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    planned, _ = planner._plan_model_batch(project, project["script"], False, project["script"], 1, 1, units)
    result, info = plan_consistency.review(project, planned, chat=chat, check=lambda: None)
    assert len(calls) == 2 and info["review_count"] == 1
    assert result[0]["narration"] == case["narration"] and project == original
    assert result[0]["subject"] == case["primary"]["subject"]
    assert case["primary"]["action"] in result[0]["visual"]
    assert case["original_primary"]["action"] not in result[0]["visual"]
    assert result[0]["query"] == case["primary"]["query"]
    assert result[0]["search_queries"] == [case["primary"]["query"]]
    assert result[0]["preferred_media"] == "video" and result[0]["era"] == ""


@pytest.mark.parametrize("script,requirements,fields,primary", [
    (
        "二〇二四年的业务量见年度统计公报。",
        "本镜必须展示《二〇二四年度统计公报》的业务量表格页面及其数字。",
        {"preferred_media": "image"},
        {"subject": "二〇二四年度统计公报", "action": "显示业务量表格及其数字", "query": "二〇二四年度统计公报 业务量"},
    ),
    (
        "张明在南门场馆用甲型扫描设备扫描一件包裹。",
        "画面必须看清张明使用甲型扫描设备扫描一件包裹，不能换成其他人物或设备运送。",
        {"persons": [{"name": "张明", "aliases": []}], "evidence_mode": "person", "location": "南门场馆"},
        {"scene": "南门场馆", "subject": "张明", "action": "使用甲型扫描设备扫描一件包裹", "query": "张明 甲型扫描设备 扫描"},
    ),
])
def test_visible_action_prompt_exceptions_keep_explicit_documents_actions_people_and_objects(
        monkeypatch, script, requirements, fields, primary):
    project, _ = sample_plan(script)
    project["requirements"] = requirements
    original, calls = deepcopy(project), []
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed"})

    def chat(system, content, **_kwargs):
        request = json.loads(content)
        calls.append(request)
        assert_visibility_rules(system)
        assert request["requirements"] == requirements
        if request.get("stage") == "plan_consistency":
            return consistency_response(request), {}
        return {"shots": [{"source_unit_id": "u0001", "event_scope": "background", "overlay_label": "资料画面",
                           **fields, "visual_plan": {"mode": "single", "primary": primary}}]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    planned, _ = planner._plan_model_batch(project, script, False, script, 1, 1,
                                            [{"source_unit_id": "u0001", "narration": script}])
    reviewed, _ = plan_consistency.review(project, planned, chat=chat, check=lambda: None)
    assert len(calls) == 2 and project == original
    assert reviewed[0]["narration"] == script and primary["action"] in reviewed[0]["visual"]
    assert reviewed[0]["visual_plan"] == planned[0]["visual_plan"]
    for field, value in fields.items():
        assert reviewed[0][field] == value


@pytest.mark.parametrize("source,quote", [
    ("不同环节由不同岗位接力，消费者只看见一部分。", "不同环节由不同岗位接力。"),
    ("工人先检查包裹；随后进行登记。", "工人先检查包裹。"),
    ("Packages keep moving, while the worker watches.", "Packages keep moving."),
])
def test_clause_ending_stop_is_removed_only_after_exact_unique_source_match(source, quote):
    evidence = [{"source": "script", "quote": quote}]
    original = deepcopy(evidence)
    normalized, changes = plan_consistency._validated_evidence(evidence, {"script": source, "requirements": ""})
    assert evidence == original and normalized == [{"source": "script", "quote": quote[:-1]}]
    assert len(changes) == 1 and changes[0]["raw_quote"] == quote
    assert changes[0]["normalized_quote"] == quote[:-1]
    span = changes[0]["source_span"]
    assert source[span["start"]:span["end"]] == quote[:-1]
    assert source[span["end"]] == changes[0]["source_following_punctuation"]


@pytest.mark.parametrize("source,quote", [
    ("不同环节由不同岗位接力，消费者只看见一部分。", "不同环节由同一岗位接力。"),
    ("工作人员卸载、分拨包裹，随后登记。", "工作人员卸载，分拨包裹。"),
    ("工作人员扫描一件包裹，随后登记。", "工作人员……包裹。"),
    ("工作人员扫描一件包裹，随后登记。", "工作人员...包裹。"),
    ("结果是1234.5，仍需核实。", "1234。"),
    ("这个过程完成了吗？还没有。", "这个过程完成了吗。"),
    ("一段原文出现两次，后来一段原文出现两次，继续解释。", "一段原文出现两次。"),
    ("这是一段原文", "这是一段原文。"),
    ("完成，下一步继续。", "完成。"),
])
def test_quote_normalization_does_not_repair_internal_text_ellipsis_decimal_or_ambiguous_fragments(source, quote):
    with pytest.raises(ValueError, match="未逐字命中"):
        plan_consistency._validated_evidence([{"source": "script", "quote": quote}], {"script": source})


def test_quote_sources_are_not_interchanged_and_exact_quotes_need_no_normalization():
    sources = {"script": "工程师检查设备，随后记录状态。", "requirements": "画面必须保留机器操作，不能换成车辆。"}
    wrong_source = [{"source": "script", "quote": "画面必须保留机器操作。"}]
    with pytest.raises(ValueError, match="未逐字命中"):
        plan_consistency._validated_evidence(wrong_source, sources)
    correct_source = [{"source": "requirements", "quote": "画面必须保留机器操作。"}]
    normalized, changes = plan_consistency._validated_evidence(correct_source, sources)
    assert normalized[0]["quote"] == "画面必须保留机器操作" and changes[0]["source"] == "requirements"
    exact = [{"source": "script", "quote": "工程师检查设备，"}]
    assert plan_consistency._validated_evidence(exact, sources) == (exact, [])
    with pytest.raises(ValueError, match="来源无效"):
        plan_consistency._validated_evidence([{"source": "system", "quote": "工程师检查设备，"}], sources)


def test_round6_paid_response_replays_without_changing_raw_words_or_source_narration():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "round6_plan_consistency_quotes.json").read_text())
    request, response = fixture["request"], fixture["response"]
    project = {"script": request["full_script"], "requirements": request["requirements"]}
    shots = [{k: v for k, v in row.items() if k != "shot_id"} | {"id": row["shot_id"]} for row in request["shots"]]
    original = deepcopy((project, shots, response))
    result = plan_consistency.apply_reviews(project, shots, response, require_visual_plan=True)
    assert (project, shots, response) == original and len(result) == 33
    normalized_shots = []
    for index, shot in enumerate(result, 1):
        assert shot["narration"] == shots[index-1]["narration"]
        for change in shot["planning_review"]["evidence_normalizations"]:
            normalized_shots.append(index)
            assert change["raw_quote"][:-1] == change["normalized_quote"]
            assert change["raw_quote"][-1] == "。" and change["source_following_punctuation"] == "，"
            span = change["source_span"]
            assert project["script"][span["start"]:span["end"]] == change["normalized_quote"]
        for evidence in shot["planning_review"]["evidence"]:
            assert evidence["quote"] in {"script": project["script"], "requirements": project["requirements"]}[evidence["source"]]
    assert normalized_shots == [8, 9, 10, 13, 21, 24, 25, 26, 27, 29, 30]


@pytest.mark.parametrize("count", [1, 33, 160])
def test_consistency_batches_enable_bounded_reasoning_and_record_all_actual_usage(tmp_path, count):
    script = "\n".join(f"工作人员搬运第{i+1}件包裹。" for i in range(count))
    project = {"script": script, "requirements": ""}
    shots = [planner.new_shot(line) for line in script.splitlines()]
    usage = {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100,
             "completion_tokens_details": {"reasoning_tokens": 25}}
    calls = []

    def chat(system, content, **kwargs):
        calls.append(kwargs)
        assert "不把子句末的逗号改成句号" in system
        return consistency_response(json.loads(content)), {"model": "test", "usage": usage}

    result, info = plan_consistency.review(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    batch_count = (count + 3) // 4
    assert calls == [{"reasoning": True, "reasoning_effort": "low", "max_tokens": 10000, "timeout": 180,
                      "task": "requirements"}] * batch_count
    expected_usage = {"prompt_tokens": 40 * batch_count, "completion_tokens": 60 * batch_count,
                      "total_tokens": 100 * batch_count,
                      "completion_tokens_details": {"reasoning_tokens": 25 * batch_count}}
    assert len(result) == count and info["usage"] == expected_usage and info["evidence_normalized_count"] == 0
    assert info["model_requests"] == info["batch_count"] == batch_count
    saved = json.loads(Path(info["diagnostic_path"]).read_text())
    assert saved["review_request_options"] == calls[0] and saved["review_model_info"]["usage"] == expected_usage


def test_terminal_quote_normalization_is_recorded_next_to_untouched_raw_response(tmp_path):
    project, shots = sample_plan("工程师检查一台设备，随后记录状态。")
    reply = response_for(project, shots, contract=True)
    reply["reviews"][0]["evidence"][0]["quote"] = "工程师检查一台设备。"
    raw = deepcopy(reply)
    result, info = plan_consistency.review(project, shots, chat=lambda *_a, **_kw: (reply, {}),
                                           check=lambda: None, diagnostic_dir=tmp_path)
    saved = json.loads(Path(info["diagnostic_path"]).read_text())
    assert reply == raw and saved["response"] == raw and info["evidence_normalized_count"] == 1
    applied = saved["applied_reviews"][0]
    assert applied["evidence"] == [{"source": "script", "quote": "工程师检查一台设备"}]
    assert applied["evidence_normalizations"][0]["raw_quote"] == "工程师检查一台设备。"
    assert result[0]["narration"] == project["script"]


def test_compact_primary_patch_derives_repeated_fields_without_removing_source_boundaries(tmp_path):
    project, shots = sample_plan("张明在南门场馆使用甲型设备处理包裹。")
    project["requirements"] = "必须看清张明使用甲型设备，不展示其他公司制服。"
    shots[0].update(persons=[{"name": "张明", "aliases": []}], location="南门场馆",
                    exclusions="不展示其他公司制服", event_scope="exact", evidence_mode="event",
                    overlay_label="", visual="旧方案", subject="旧主体", query="旧查询", search_queries=["旧查询"])
    primary = {"scene": "南门场馆", "subject": "张明", "action": "使用甲型设备处理包裹",
               "query": "张明 甲型设备 处理包裹", "search_queries": ["张明 甲型设备 处理包裹"]}
    reply = {"requirements_preserved": True, "reviews": [{"shot_id": shots[0]["id"], "status": "revised",
        "purpose": "illustration", "reason": "只取可见的设备操作，保留明确身份与地点。",
        "evidence": [{"source": "script", "quote": project["script"]},
                     {"source": "requirements", "quote": project["requirements"]}],
        "patch": {"visual_plan": {"mode": "single", "primary": primary, "alternatives": []}}}]}
    original = deepcopy((project, shots, reply))
    reviewed, info = plan_consistency.review(project, shots, chat=lambda *_a, **_kw: (reply, {}),
                                             check=lambda: None, diagnostic_dir=tmp_path)
    result = reviewed[0]
    assert result["subject"] == "张明" and result["query"] == primary["query"]
    assert primary["action"] in result["visual"] and result["search_queries"] == primary["search_queries"]
    assert result["event_scope"] == "background" and result["evidence_mode"] == "person"
    assert result["overlay_label"] == "资料画面"
    for field in ("persons", "location", "exclusions", "narration", "duration", "audio", "selected_asset"):
        assert result[field] == shots[0][field]
    saved = json.loads(Path(info["diagnostic_path"]).read_text())
    assert saved["response"] == reply and info["normalized_count"] == 1
    assert (project, shots, reply) == original
