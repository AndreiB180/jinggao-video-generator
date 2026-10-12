"""Source authority stays separate from an imagined primary or model verdict."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from jinggao import editorial_contract as contract


def example(lines, requirements=""):
    project = {"script": "  " + "\n\n".join(lines) + "\n", "requirements": requirements}
    shots = [{"id": f"shot-{i}", "narration": line, "duration": 7.0, "visual_origin": "generated",
              "audio": {"path": f"paid-{i}.wav"}, "locked": False}
             for i, line in enumerate(lines, 1)]
    return project, shots


def row(ref, *, use="illustration", refs=(), facts=(), must=(), queries=("说明资料",)):
    return {"ref": ref, "use": use, "refs": list(refs), "facts": list(facts),
            "must": list(must), "queries": list(queries)}


def compile_rows(project, shots, rows):
    return contract.compile_responses(project, shots, [{"contracts": rows}])


@pytest.mark.parametrize("hints, expected", [
    (["楼道 快递", "电梯 门禁 快递员", "雨天 快递派送"], ["楼道 快递", "电梯 门禁 快递员"]),
    (["搬运包裹", " 搬运包裹 ", None], ["搬运包裹"]),
    (None, []), ([], []), ({"bad": "object"}, []), ("搬运包裹", ["搬运包裹"]),
])
def test_optional_search_hint_shape_never_changes_source_requirements(hints, expected):
    project, shots = example(["工作人员搬运包裹。"])
    raw = row("s1", facts=[["activity", "s1", "搬运包裹"]])
    raw["queries"] = hints
    original = deepcopy(raw)
    bundle = compile_rows(project, shots, [raw])
    attached = contract.attach(project, shots, bundle)
    bound = contract.for_shot(attached[0], project)
    assert bound["query_hints"] == expected
    assert bound["facts"][0]["quote"] == "搬运包裹" and raw == original
    assert contract.retrieval_fields(attached[0])["query"] == (expected[0] if expected else "")
    assert "query_hints" not in contract.context(attached[0])["contract"]


def test_typical_full_script_is_one_request_and_source_appears_once_without_audio_or_primary():
    project, shots = example([f"第{i}件包裹由工作人员处理。" for i in range(1, 34)], "保留完整原文。")
    for shot in shots:
        shot["visual_plan"] = {"mode": "single", "primary": {"action": "模型额外要求扫描分流"}}
    original = deepcopy((project, shots))
    requests = contract.build_requests(project, shots)
    assert len(requests) == 1 and requests[0]["targets"] == [f"s{i}" for i in range(1, 34)]
    assert "".join(u["text"] for u in requests[0]["script_units"]) == project["script"]
    assert requests[0]["requirements"] == project["requirements"]
    encoded = json.dumps(requests, ensure_ascii=False)
    assert "paid-" not in encoded and "扫描分流" not in encoded and "primary" not in encoded
    assert (project, shots) == original


def test_round3_examples_keep_source_boundaries_without_turning_generated_actions_into_demands():
    project, shots = example([
        "中间那么长一段路，在手机上往往只剩几行字。",
        "把他们全部叫成一个模糊的送货人，会漏掉很多具体劳动。",
        "确实，自动化分拣设备、手持终端、智能快件箱，都已进入相关职业技能的内容。",
    ])
    for shot, action in zip(shots, ("滑动屏幕", "多岗位交接同框", "内部扫描并分流"), strict=True):
        shot["visual_plan"] = {"mode": "single", "primary": {"subject": "旧主体", "action": action}}
        shot["visual"] = action
    original = deepcopy((project, shots))
    rows = [row("s1", facts=[["object", "s1", "手机"]], queries=["查询快递", "手机 物流进度"]),
            row("s2", facts=[["activity", "s2", "具体劳动"]], queries=["快递工作人员 处理包裹"]),
            row("s3", facts=[["object", "s3", "自动化分拣设备"]], queries=["自动化分拣设备 包裹"])]
    bundle = compile_rows(project, shots, rows)
    search = contract.retrieval_payload(project, shots, bundle)
    audit = contract.audit_payload(project, shots, bundle)
    assert search["shots"][0]["query_hints"] == ["查询快递", "手机 物流进度"]
    assert all("query_hints" not in r and not r["must"] for r in audit["shots"])
    assert audit["shots"][2]["facts"][0]["quote"] == "自动化分拣设备"
    assert "".join(u["text"] for u in audit["script_units"]) == project["script"]
    encoded = json.dumps(audit, ensure_ascii=False)
    for invented in ("滑动屏幕", "多岗位交接同框", "内部扫描并分流"):
        assert invented not in encoded
    assert audit["approval"] == "pending_full_review" and (project, shots) == original
    # These source contracts do not assert that any candidate or query result
    # is acceptable. Observed lookup, human labour and the actual equipment
    # must still be judged from full windows before a primary is recorded.


def test_anaphora_specific_event_and_explicit_action_are_bound_to_real_source_positions():
    project, shots = example([
        "刘翔在雅典奥运会男子一百一十米栏比赛中夺冠。", "这个结果来自赛场上的真实表现。",
    ], "第二镜必须看清刘翔跨栏动作，不得使用其他运动员。")
    event = "雅典奥运会男子一百一十米栏比赛"
    rows = [row("s1", use="event_record", facts=[["event", "s1", event], ["person", "s1", "刘翔"]]),
            row("s2", use="event_record", refs=["s1"],
                facts=[["event", "s1", event], ["person", "s1", "刘翔"]],
                must=[["activity", "r", "必须看清刘翔跨栏动作"], ["exclusion", "r", "不得使用其他运动员"]])]
    bundle = compile_rows(project, shots, rows)
    view = contract.audit_payload(project, shots, bundle, [shots[1]["id"]])
    target = view["shots"][0]
    assert target["use"] == "event_record" and target["refs"] == ["s1"]
    assert target["must"][0]["quote"] == "必须看清刘翔跨栏动作"
    assert view["requirements"] == project["requirements"]
    for citation in target["facts"] + target["must"]:
        source = project["script"] if citation["source"] == "script" else project["requirements"]
        assert all(source[span["start"]:span["end"]] == citation["quote"] for span in citation["source_spans"])


def test_authorship_and_statistical_period_do_not_become_must_appear_or_capture_year():
    project, shots = example(["编导署名为陈琳、袁时。", "二〇二四年我国快递业务量完成一千七百五十点八亿件。"])
    bundle = compile_rows(project, shots, [
        row("s1", facts=[["attribution", "s1", "陈琳、袁时"]], queries=["作品 舞台演出"]),
        row("s2", facts=[["data_period", "s2", "二〇二四年"]], queries=["快递 包裹"]),
    ])
    view = contract.audit_payload(project, shots, bundle)
    assert all(not r["must"] and "persons" not in r and "era" not in r for r in view["shots"])
    assert view["shots"][1]["facts"][0]["kind"] == "data_period"
    assert "统计年份" in view["rules"] and view["approval"] == "pending_full_review"


@pytest.mark.parametrize("protected", [
    {"visual_origin": "manual"}, {"locked": True}, {"visual_origin": None},
])
def test_user_locked_and_legacy_plans_never_become_reselectable_suggestions(protected):
    project, shots = example(["包裹正在装车。", "消费者了解进度。"], "第一镜必须展示完整装车过程。")
    shots[0].update(protected, visual="先扫描条码，再搬上货车", persons=[{"name": "张明", "aliases": []}],
                    location="北门", event="指定活动", exclusions="不出现其他品牌",
                    visual_plan={"mode": "explicit_sequence"})
    original = deepcopy((project, shots))
    requests = contract.build_requests(project, shots)
    assert requests[0]["targets"] == ["s2"]
    assert requests[0]["readonly_visual"][0]["visual"] == shots[0]["visual"]
    bundle = compile_rows(project, shots, [row("s2")])
    saved = contract.audit_payload(project, shots, bundle)["shots"][0]
    assert saved["use"] == "locked" and saved["readonly_visual"]["persons"] == shots[0]["persons"]
    assert saved["readonly_visual"]["visual_plan"] == {"mode": "explicit_sequence"}
    assert (project, shots) == original


def test_explicit_multiscene_and_date_requirements_are_retained_not_replaced_for_recall():
    project, shots = example(["展示工作人员处理包裹。"], "画面必须依次展示扫码和装车；只用二〇二四年拍摄的资料。")
    bundle = compile_rows(project, shots, [row("s1", must=[
        ["sequence", "r", "画面必须依次展示扫码和装车"],
        ["capture_period", "r", "只用二〇二四年拍摄的资料"],
    ])])
    requirements = contract.audit_payload(project, shots, bundle)["shots"][0]["must"]
    assert [r["kind"] for r in requirements] == ["sequence", "capture_period"]
    assert len(requirements) == 2


@pytest.mark.parametrize("corruption", ["fake-quote", "wrong-source", "added-punctuation", "foreign-ref",
                                         "unknown-citation-ref", "missing-event", "narration", "primary"])
def test_fabricated_citations_or_attempts_to_edit_source_fail_atomically(corruption):
    project, shots = example(["张明处理包裹，随后离开。", "另一位工作人员记录数量。"])
    original = deepcopy((project, shots))
    rows = [row("s1", facts=[["person", "s1", "张明"]]), row("s2")]
    if corruption == "fake-quote":
        rows[0]["facts"][0][2] = "王强"
    elif corruption == "wrong-source":
        rows[0]["facts"][0][1] = "r"
    elif corruption == "added-punctuation":
        rows[0]["facts"][0][2] = "张明处理包裹。"
    elif corruption == "foreign-ref":
        rows[0]["refs"] = ["another-project"]
    elif corruption == "unknown-citation-ref":
        rows[0]["facts"] = [["activity", "s404", "记录数量"]]
    elif corruption == "missing-event":
        rows[0]["use"] = "event_record"
    else:
        rows[0][corruption] = "不能修改"
    with pytest.raises(ValueError):
        compile_rows(project, shots, rows)
    assert (project, shots) == original


def test_explicit_citation_restores_redundant_context_ref_without_changing_source():
    lines = ["二〇二四年我国快递业务量完成一千七百五十点八亿件。",
             "这说的是那一年的业务量，不是人口数。"]
    project, shots = example(lines)
    rows = [row("s1"), row("s2", facts=[["data_period", "s1", lines[0][:-1]]])]
    original = deepcopy(rows)
    bundle = compile_rows(project, shots, rows)
    second = bundle["contracts"][1]
    assert second["refs"] == ["s1"] and second["facts"][0]["quote"] == lines[0][:-1]
    assert bundle["raw_responses"] == [{"contracts": original}] and rows == original
    assert bundle["normalizations"] == [{"batch_index": 1, "ref": "s2", "field": "refs",
        "operation": "explicit_citation_refs", "from": [], "to": ["s1"]}]
    attached = contract.attach(project, shots, bundle)
    view = contract.context(attached[1])
    assert [s["ref"] for s in view["source_refs"]] == ["s2", "s1"]
    assert second["state"] == "source_bound_only" and bundle["approval"] == "pending_full_review"
    rows[1]["facts"][0][2] = "二〇二四年我国快递业务量完成一千七百六十点八亿件"
    with pytest.raises(ValueError, match="逐字"):
        compile_rows(project, shots, rows)


def test_wrong_batch_or_incomplete_reply_is_not_merged_into_a_partial_contract():
    project, shots = example([f"第{i}镜展示劳动。" for i in range(1, 6)])
    requests = contract.build_requests(project, shots, batch_size=3)
    assert [r["targets"] for r in requests] == [["s1", "s2", "s3"], ["s4", "s5"]]
    replies = [{"contracts": [row(ref) for ref in r["targets"]]} for r in requests]
    replies[1]["contracts"][0]["ref"] = "s1"
    with pytest.raises(ValueError, match="第2批.*ID"):
        contract.compile_responses(project, shots, replies, batch_size=3)
    with pytest.raises(ValueError, match="批次缺失"):
        contract.compile_responses(project, shots, replies[:1], batch_size=3)


@pytest.mark.parametrize("change", ["script", "requirements", "manual-visual", "canonical", "raw-only"])
def test_source_edits_or_tampered_compiled_contracts_invalidate_projection(change):
    project, shots = example(["工作人员搬运包裹。"])
    bundle = compile_rows(project, shots, [row("s1")])
    if change == "script":
        project["script"] += " 不得替换物件。"
    elif change == "requirements":
        project["requirements"] = "必须展示北门。"
    elif change == "manual-visual":
        shots[0].update(visual="手动要求新动作", visual_origin="manual")
    elif change == "canonical":
        bundle["contracts"][0]["use"] = "event_record"
    else:
        bundle["raw_responses"][0]["contracts"][0]["queries"] = ["另一查询"]
    with pytest.raises(ValueError):
        contract.audit_payload(project, shots, bundle)


def test_literal_binding_is_not_a_semantic_or_media_approval():
    project, shots = example(["二〇二四年业务量增长。"], "不得显示其他年份的数据字幕。")
    # Even an incomplete model extraction cannot erase the complete sources
    # from the actual judge input or produce an approval flag.
    bundle = compile_rows(project, shots, [row("s1")])
    view = contract.audit_payload(project, shots, bundle)
    assert view["requirements"] == project["requirements"]
    assert "".join(r["text"] for r in view["script_units"]) == project["script"]
    assert view["approval"] == "pending_full_review" and view["shots"][0]["state"] == "source_bound_only"
    assert "原稿事实即使未被facts列出仍然有效" in view["rules"]


def test_attached_contract_survives_voice_media_and_generated_framing_changes_but_not_user_edits():
    project, shots = example(["快递员再次联系收件人。"])
    shots[0].update(persons=[], event="", event_scope="background", evidence_mode="generic")
    bundle = compile_rows(project, shots, [row("s1", facts=[["activity", "s1", "再次联系收件人"]],
                                            queries=["快递员 联系"] )])
    attached = contract.attach(project, shots, bundle)
    current = attached[0]
    current.update(duration=11.2, audio={"path": "new-paid.wav"}, approved=True, selected_asset="new-video",
                   visual="实际观察后选定的动作", visual_plan={"primary": {"action": "打电话"}})
    assert contract.for_shot(current, project) and contract.valid(project, current)
    review = contract.context(current)
    assert review["full_script"] == project["script"]
    assert review["required_checks"][:2] == ["source_alignment", "caption_context"]
    assert "原文明示主体与真实动作" in review["source_alignment"]
    assert review["contract"]["facts"][0]["quote"] == "再次联系收件人"
    assert "query_hints" not in review["contract"]
    recall = contract.retrieval_fields(current)
    assert recall["query"] == "快递员 联系" and recall["persons"] == []
    assert "visual_plan" not in recall and "visual" not in recall and "subject" not in recall
    assert "editorial_contract" not in shots[0] and shots[0]["audio"]["path"] == "paid-1.wav"
    assert not contract.for_shot(current, project | {"requirements": "必须看清拨号动作"})
    current["visual_origin"] = "manual"
    assert not contract.for_shot(current) and not contract.context(current) and not contract.retrieval_fields(current)


def test_explicit_action_remains_mandatory_and_same_object_does_not_excuse_wrong_action():
    project, shots = example([
        "对你来说只是一个疏忽，到了送件的人那里，就多了确认地址、再次联系、调整安排这些动作。"
    ], "画面必须展示再次联系，不能用拍照代替。")
    bundle = compile_rows(project, shots, [row("s1",
        facts=[["activity", "s1", "确认地址、再次联系、调整安排这些动作"]],
        must=[["activity", "r", "必须展示再次联系"], ["exclusion", "r", "不能用拍照代替"]],
        queries=["快递员 手机"] )])
    shot = contract.attach(project, shots, bundle)[0]
    audit = contract.context(shot)
    assert audit["required_checks"] == ["source_alignment", "caption_context", "must_1", "must_2"]
    assert audit["contract"]["must"][0]["quote"] == "必须展示再次联系"
    assert audit["requirements"] == project["requirements"]
    assert "快递员 手机" not in json.dumps(audit, ensure_ascii=False)
    # The source alignment obligation remains even when the model omitted must.
    incomplete = compile_rows(project, shots, [row("s1")])
    assert contract.context(contract.attach(project, shots, incomplete)[0])["required_checks"] == [
        "source_alignment", "caption_context"]


def test_extract_one_call_saves_raw_usage_and_preserves_project_and_old_suggestions(tmp_path):
    project, shots = example([f"第{i}件包裹由工作人员处理。" for i in range(1, 34)])
    for shot in shots:
        shot["visual_plan"] = {"mode": "single", "primary": {"action": "原先想象的动作"}}
    original = deepcopy((project, shots))
    calls, progress = [], []

    def chat(_system, content, **options):
        request = json.loads(content)
        calls.append(request)
        assert options == {"reasoning": False, "max_tokens": 8000, "timeout": 45, "task": "requirements"}
        return {"contracts": [row(ref) for ref in request["targets"]]}, {
            "model": "test", "usage": {"completion_tokens": 700, "total_tokens": 1400}, "api_key": "must-not-leak"}

    result, info = contract.extract(project, shots, chat=chat, check=lambda: None,
                                    progress=lambda n, total: progress.append((n, total)), diagnostic_dir=tmp_path)
    assert len(calls) == 1 and info["model_requests"] == 1 and info["contract_count"] == 33
    assert progress == [(0, 1), (1, 1)]
    assert all(contract.for_shot(s, project) for s in result) and (project, shots) == original
    assert all(s["visual_plan"]["primary"]["action"] == "原先想象的动作" for s in result)
    saved = json.loads(next(tmp_path.glob("editorial-contract-*.json")).read_text())
    assert saved["status"] == "attached" and len(saved["batches"][0]["response"]["contracts"]) == 33
    assert saved["usage"] == info["usage"] == {"completion_tokens": 700, "total_tokens": 1400}
    assert "must-not-leak" not in json.dumps(saved)


def test_failed_second_paid_extract_saves_both_usage_and_never_attaches(tmp_path, monkeypatch):
    from jinggao import planner

    project, shots = example([f"第{i}件包裹由工作人员处理。" for i in range(1, 42)])
    original, calls = deepcopy((project, shots)), []

    def chat(_system, content, **_options):
        request = json.loads(content)
        calls.append(request)
        if len(calls) == 2:
            raise planner.ModelResponseError("原始响应截断", diagnostic_path=tmp_path / "raw-provider.json",
                model_info={"model": "test", "usage": {"total_tokens": 8000}, "headers": {"secret": "hidden"}})
        return {"contracts": [row(ref) for ref in request["targets"]]}, {"model": "test", "usage": {"total_tokens": 1000}}

    monkeypatch.setattr(contract, "attach", lambda *_: pytest.fail("partial contract attached"))
    with pytest.raises(ValueError, match="编辑契约提取失败"):
        contract.extract(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    saved = json.loads(next(tmp_path.glob("editorial-contract-*.json")).read_text())
    assert len(calls) == 2 and saved["status"] == "failed" and (project, shots) == original
    assert saved["usage"] == {"total_tokens": 9000} and len(saved["batches"][0]["response"]["contracts"]) == 40
    assert saved["batches"][1]["model_response_diagnostic_path"].endswith("raw-provider.json")
    assert "hidden" not in json.dumps(saved)


@pytest.mark.parametrize("repair", ["valid", "drop_fact", "still_wrong"])
def test_wrong_unit_pointer_has_one_bounded_repair_without_changing_quoted_facts(tmp_path, repair):
    project, shots = example(["刘翔参加男子一百一十米栏决赛。", "刘翔随后接受采访。", "那个夜晚，他夺得金牌。"])
    before = deepcopy((project, shots))
    calls = []
    bad = row("s3", facts=[["person", "s3", "刘翔"]])
    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        if len(calls) == 1:
            return {"contracts": [row("s1"), row("s2"), deepcopy(bad)]}, {"usage": {"total_tokens": 100}}
        assert len(calls) == 2 and request["targets"] == ["s3"]
        assert request["failed_rows"][0]["readonly_citations"]["facts"] == bad["facts"]
        assert request["failed_rows"][0]["pointers"][0]["candidates"] == ["s1", "s2"]
        pointers = [{"id": "facts:0", "source_ref": "s3"}]
        if repair == "valid":
            pointers[0]["source_ref"] = "s1"
        elif repair == "drop_fact":
            pointers = []
        return {"repairs": [{"ref": "s3", "pointers": pointers, "event": []}]}, {"usage": {"total_tokens": 50}}
    if repair == "valid":
        result, info = contract.extract(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
        fact = contract.for_shot(result[2], project)["facts"][0]
        assert (fact["kind"], fact["ref"], fact["quote"]) == ("person", "s1", "刘翔")
        assert info["model_requests"] == 2 and info["usage"]["total_tokens"] == 150
    else:
        with pytest.raises(ValueError, match="编辑契约提取失败"):
            contract.extract(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    assert len(calls) == 2 and (project, shots) == before
    diagnostic = json.loads(next(tmp_path.glob("editorial-contract-*.json")).read_text())
    assert diagnostic["batches"][0]["response"]["contracts"][2] == bad
    assert diagnostic["batches"][1]["correction"]


def test_fabricated_quotation_cannot_enter_pointer_repair(tmp_path):
    project, shots = example(["比赛结束。"])
    calls = []
    def chat(*args, **kwargs):
        calls.append(1)
        return {"contracts": [row("s1", facts=[["person", "s1", "未在原文出现的人"]])]}, {}
    with pytest.raises(ValueError, match="编辑契约提取失败"):
        contract.extract(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("existing", [{"selected_asset": "old-approved-video"}, {"approved": True}])
def test_extraction_preserves_existing_media_but_does_not_make_runtime_selection_invalidate_contract(existing):
    project, shots = example(["工作人员搬运包裹。", "消费者查看物流信息。"])
    shots[0].update(existing, visual="已有取景不能被新提取改写")
    requests = contract.build_requests(project, shots)
    assert requests[0]["targets"] == ["s2"]
    bundle = compile_rows(project, shots, [row("s2")])
    result = contract.attach(project, shots, bundle)
    assert result[0] == shots[0] and not contract.for_shot(result[0])
    result[1].update(selected_asset="new-video", approved=True)
    assert contract.for_shot(result[1], project)


def test_statistics_must_error_is_not_silently_repaired_and_explicit_document_command_is_preserved():
    project, shots = example(["二〇二四年我国快递业务量完成一千七百五十点八亿件。"])
    bad = row("s1", facts=[["data_period", "s1", "二〇二四年"]],
              must=[["data_period", "s1", "二〇二四年"]])
    original = deepcopy(bad)
    with pytest.raises(ValueError, match="must类型"):
        compile_rows(project, shots, [bad])
    assert bad == original and "data_period" not in contract.MUST_KINDS
    assert "data_period/event_date/artifact_period/attribution只放facts" in contract.SYSTEM
    assert "没有明确命令就填[]" in contract.SYSTEM
    project["requirements"] = "画面必须展示二〇二四年度统计公报页面。"
    good = row("s1", facts=[["data_period", "s1", "二〇二四年"]],
               must=[["document", "r", "画面必须展示二〇二四年度统计公报页面"]])
    view = contract.context(contract.attach(project, shots, compile_rows(project, shots, [good]))[0])
    assert view["contract"]["must"][0]["kind"] == "document"
    assert "must_1" in view["required_checks"] and view["requirements"] == project["requirements"]


def test_historical_site_illustration_keeps_exact_facts_and_anaphora_without_demanding_event_footage():
    project, shots = example(["一九二五年十月十日，故宫博物院在乾清门举行开幕典礼。", "那场开幕典礼有重要意义。"])
    facts = [["event_date", "s1", "一九二五年十月十日"], ["location", "s1", "乾清门"],
             ["event", "s1", "故宫博物院在乾清门举行开幕典礼"]]
    rows = [row("s1", facts=facts, queries=["乾清门 现址"]),
            row("s2", refs=["s1"], facts=facts, queries=["乾清门 现址"])]
    attached = contract.attach(project, shots, compile_rows(project, shots, rows))
    view = contract.context(attached[1])
    assert view["contract"]["use"] == "illustration" and view["contract"]["must"] == []
    assert view["contract"]["refs"] == ["s1"]
    assert {r["quote"] for r in view["contract"]["facts"]} == {r[2] for r in facts}
    assert "当代历史现址应明确标注" in view["rules"]
    assert "不等于命令使用当日现场录像" in contract.SYSTEM and "具体比赛动作" in contract.SYSTEM


V2_RESPONSES = json.loads((Path(__file__).parent / "fixtures" / "editorial_contract_v2_shapes.json").read_text())


@pytest.mark.parametrize("record", V2_RESPONSES, ids=lambda r: r["label"])
def test_actual_v2_objects_and_omitted_arrays_compile_losslessly_without_changing_use_or_conditions(record):
    project = {"script": "".join(u["text"] for u in record["script_units"]), "requirements": record["requirements"]}
    shots = [{"id": u["ref"], "source_unit_id": u["ref"], "narration": u["text"].strip(),
              "visual_origin": "generated"} for u in record["script_units"]]
    raw = deepcopy(record["response"])
    bundle = contract.compile_responses(project, shots, [raw])
    attached = contract.attach(project, shots, bundle)
    assert all(contract.for_shot(s, project) for s in attached)
    assert raw == record["response"] == bundle["raw_responses"][0]
    for supplied, bound in zip(raw["contracts"], bundle["contracts"], strict=True):
        assert bound["use"] == supplied["use"] and bound["refs"] == supplied.get("refs", [])
        for field in ("facts", "must"):
            expected = [c if isinstance(c, list) else [c["kind"], c["source_ref"], c["quote"]]
                        for c in supplied.get(field, [])]
            assert [[c["kind"], c["ref"], c["quote"]] for c in bound[field]] == expected
    changes = bundle["normalizations"]
    assert len(changes) == (12 if record["label"] == "03" else 29)
    omitted = [c["ref"] for c in changes if c["operation"] == "omitted_empty_array"]
    assert omitted == (["s22", "s29", "s30", "s31", "s32", "s33"] if record["label"] == "03" else [])
    assert all(contract.context(s)["required_checks"][:2] == ["source_alignment", "caption_context"] for s in attached)


def test_omitted_empty_arrays_are_explicitly_recorded_but_not_inferred_semantics():
    project, shots = example(["工作人员搬运包裹。"], "必须展示搬运动作。")
    raw = {"contracts": [{"ref": "s1", "use": "illustration", "queries": ["搬运包裹"]}]}
    original = deepcopy(raw)
    bundle = contract.compile_responses(project, shots, [raw])
    assert [c["field"] for c in bundle["normalizations"]] == ["refs", "facts", "must"]
    view = contract.context(contract.attach(project, shots, bundle)[0])
    assert view["requirements"] == project["requirements"] and "source_alignment" in view["required_checks"]
    assert raw == original and "must" not in raw["contracts"][0]
    raw["contracts"][0]["use"] = "event_record"
    with pytest.raises(ValueError, match="具体事件"):
        contract.compile_responses(project, shots, [raw])


@pytest.mark.parametrize("corruption", ["extra-key", "missing-key", "fake-quote", "wrong-source-key", "wrong-must-kind", "null-array"])
def test_equivalent_object_support_does_not_accept_extra_keys_bad_quotes_or_new_must_types(corruption):
    project, shots = example(["二〇二四年业务量增长。"])
    citation = {"kind": "data_period", "source_ref": "s1", "quote": "二〇二四年"}
    value = row("s1", facts=[citation])
    if corruption == "extra-key":
        citation["optional"] = True
    elif corruption == "missing-key":
        citation.pop("quote")
    elif corruption == "fake-quote":
        citation["quote"] = "二〇二五年"
    elif corruption == "wrong-source-key":
        citation["ref"] = citation.pop("source_ref")
    elif corruption == "wrong-must-kind":
        value["must"] = [deepcopy(citation)]
    else:
        value["must"] = None
    original = deepcopy(value)
    with pytest.raises(ValueError):
        compile_rows(project, shots, [value])
    assert value == original


def test_extract_saves_raw_objects_and_normalization_trace_without_another_call(tmp_path):
    project, shots = example(["工作人员搬运包裹。"])
    raw = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [],
        "facts": [{"kind": "activity", "source_ref": "s1", "quote": "搬运包裹"}], "queries": ["搬运包裹"]}]}
    original, calls = deepcopy(raw), []

    def chat(*args, **kwargs):
        calls.append((args, kwargs))
        return raw, {"usage": {"total_tokens": 100}}

    result, info = contract.extract(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    saved = json.loads(Path(info["diagnostic_path"]).read_text())
    assert len(calls) == 1 and raw == original == saved["batches"][0]["response"]
    assert saved["normalizations"] == info["normalizations"] and len(info["normalizations"]) == 2
    assert contract.for_shot(result[0], project)["facts"][0]["quote"] == "搬运包裹"
