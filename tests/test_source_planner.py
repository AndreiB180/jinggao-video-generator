"""Offline provenance/transport tests; no model, TTS or production project IO."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from threading import Event

import pytest

from jinggao import config, editorial_contract, frame_contract, paths, planner, source_planner


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    cfg = {"model_base_url": "https://model.invalid/v1", "model_name": "mock-contract",
           "requirements_model": "mock-contract"}
    monkeypatch.setattr(config, "read", lambda: dict(cfg))
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(planner, "chat", lambda *a, **k: pytest.fail("No real model is allowed"))
    return cfg


def project(lines, requirements="", **fields):
    return {"script": "  " + "\n\n".join(lines) + "\n", "requirements": requirements,
            "style": "清晰叙述", **fields}


def row(ref, *, facts=(), must=(), refs=(), use="illustration", queries=("相关资料",)):
    return {"ref": ref, "use": use, "refs": list(refs), "facts": list(facts),
            "must": list(must), "queries": list(queries)}


def reply(request):
    units = {u["ref"]: u["text"] for u in request["script_units"]}
    return {"contracts": [row(ref) for ref in request["targets"]],
            "presentation": [{"ref": ref, "persons": [], "event_persons": [], "event": [],
                              "sources": [[ref, units[ref].strip()[:160]]]}
                             for ref in request["targets"]]}


def response_for(p):
    shots = source_planner.source_shots(p)
    requests = source_planner.build_requests(p, shots)
    assert len(requests) == 1
    return shots, reply(requests[0])


def person(name, *refs, aliases=()):
    return {"name": name, "aliases": list(aliases), "refs": list(refs)}


def compile_one(p, shots, response):
    original = deepcopy((p, shots, response))
    result = source_planner.compile_responses(p, shots, [response])
    assert (p, shots, response) == original
    return result


def test_build_retains_entire_source_and_requirements_without_primary_or_repeated_script():
    p = project([f"第{i}件包裹由工作人员处理。" for i in range(34)], "请保留原稿。")
    shots = source_planner.source_shots(p)
    requests = source_planner.build_requests(p, shots)
    assert len(shots) == 34 and len(requests) == 1
    assert "".join(s["narration"] for s in shots) == p["script"]
    assert "".join(u["text"] for u in requests[0]["script_units"]) == p["script"]
    assert requests[0]["requirements"] == p["requirements"]
    assert requests[0]["source_unit_ids"] == [
        {"ref": f"s{i}", "source_unit_id": f"u{i:04d}"} for i in range(1, 35)]
    assert "primary" not in json.dumps(requests, ensure_ascii=False)
    assert "只返回JSON {\"contracts\"" in source_planner.SYSTEM
    assert "姓名、型号" not in json.dumps(requests, ensure_ascii=False)


def test_generic_and_attribution_mentions_do_not_require_all_people_on_screen():
    p = project(["编导署名为陈琳、袁时。", "雷军介绍汽车后，消费者体验产品。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["attribution", "s1", "陈琳、袁时"]]
    response["contracts"][1]["facts"] = [["person", "s2", "雷军"]]
    result = compile_one(p, shots, response)
    assert all(s["persons"] == [] and s["evidence_mode"] == "generic" for s in result)
    assert all(not s["approved"] and s["verification"] is None for s in result)
    assert all("visual_plan" not in s for s in result)
    assert all(editorial_contract.for_shot(s, p) for s in result)


def test_named_illustration_keeps_identity_gate_and_only_source_backed_alias():
    p = project(["刘翔（Liu Xiang）参加日常训练。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["person", "s1", "刘翔（Liu Xiang）"]]
    response["presentation"][0]["persons"] = [person("刘翔", "f1", aliases=["Liu Xiang"])]
    result = compile_one(p, shots, response)[0]
    assert result["persons"] == [{"name": "刘翔", "aliases": ["Liu Xiang"]}]
    assert result["evidence_mode"] == "person" and result["event_scope"] == "background"
    assert result["overlay_label"] == "资料画面"
    assert result["source_presentation"]["source_checksum"] == result["editorial_contract"]["checksum"]


def test_exact_event_anaphora_retains_date_round_opponents_and_original_spans():
    date = "二〇二四年五月一日"
    event = "甲赛事男子单打半决赛，马龙对阵王楚钦"
    p = project([date + "，" + event + "。", "他在这场比赛中完成击球。"], "第二镜必须呈现马龙。")
    shots, response = response_for(p)
    facts = [["person", "s1", "马龙"], ["person", "s1", "王楚钦"],
             ["event", "s1", event], ["event_date", "s1", date]]
    for i in range(2):
        response["contracts"][i].update(use="event_record", facts=deepcopy(facts), refs=["s1"] if i else [])
        response["presentation"][i].update(persons=[person("马龙", "f1")],
                                          event_persons=[person("王楚钦", "f2")], event=["f3", "f4"])
    response["contracts"][1]["must"] = [["person", "r", "第二镜必须呈现马龙"]]
    response["presentation"][1]["persons"][0]["refs"].append("m1")
    result = compile_one(p, shots, response)
    for shot in result:
        assert shot["event"] == event + "；" + date
        assert shot["event_persons"] == [{"name": "王楚钦", "aliases": []}]
        assert shot["evidence_mode"] == "event" and shot["event_scope"] == "exact"
        assert shot["overlay_label"] == "事件资料"
        bound = editorial_contract.for_shot(shot, p)
        for citation in bound["facts"]:
            assert all(p["script"][span["start"]:span["end"]] == citation["quote"]
                       for span in citation["source_spans"])
    assert "must_1" in frame_contract.requirements(result[1])


@pytest.mark.parametrize("change,match", [
    ("unknown_person", "姓名"), ("unreferenced_person", "引用"), ("wrong_person_kind", "类型"),
    ("invented_alias", "别名"), ("other_person_alias", "别名"),
    ("duplicate_person", "重复"), ("person_none", "逐项"), ("missing_persons", "字段"),
])
def test_unbound_or_ambiguous_people_are_rejected(change, match):
    p = project(["刘翔和苏炳添参加训练。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["person", "s1", "刘翔和苏炳添"], ["activity", "s1", "训练"]]
    target = response["presentation"][0]
    target["persons"] = [person("刘翔", "f1")]
    if change == "unknown_person":
        target["persons"][0]["name"] = "姚明"
    elif change == "unreferenced_person":
        target["persons"][0]["refs"] = []
    elif change == "wrong_person_kind":
        target["persons"][0]["refs"] = ["f2"]
    elif change in {"invented_alias", "other_person_alias"}:
        target["persons"][0]["aliases"] = ["Liu Xiang" if change == "invented_alias" else "苏炳添"]
    elif change == "duplicate_person":
        target["persons"].append(deepcopy(target["persons"][0]))
    elif change == "person_none":
        target["persons"] = None
    else:
        target.pop("persons")
    with pytest.raises(ValueError, match=match):
        compile_one(p, shots, response)


def test_explicit_multiple_people_cannot_be_dropped_or_only_be_event_participants():
    p = project(["刘翔和苏炳添参加训练。"], "必须展示刘翔和苏炳添。")
    shots, response = response_for(p)
    response["contracts"][0].update(facts=[["person", "s1", "刘翔"], ["person", "s1", "苏炳添"]],
                                     must=[["person", "r", "必须展示刘翔和苏炳添"]])
    with pytest.raises(ValueError, match="明确要求呈现的人物"):
        compile_one(p, shots, response)
    response["presentation"][0]["persons"] = [person("刘翔", "f1", "m1")]
    with pytest.raises(ValueError, match="多人展示"):
        compile_one(p, shots, response)
    response["presentation"][0]["persons"].append(person("苏炳添", "f2", "m1"))
    assert len(compile_one(p, shots, response)[0]["persons"]) == 2


def test_sequence_document_and_literal_overlay_remain_strict_source_conditions():
    requirements = "画面必须依次展示扫码和装车；必须展示年度公报；请在屏幕上显示“1750.8亿件”。"
    p = project(["工作人员处理快件。"], requirements)
    shots, response = response_for(p)
    response["contracts"][0]["must"] = [
        ["sequence", "r", "画面必须依次展示扫码和装车"],
        ["document", "r", "必须展示年度公报"],
        ["onscreen_text", "r", "请在屏幕上显示“1750.8亿件”"],
    ]
    shot = compile_one(p, shots, response)[0]
    assert shot["preferred_media"] == "image" and shot["overlay_label"] == "1750.8亿件"
    assert set(frame_contract.requirements(shot)) == {"source_alignment", "caption_context", "must_1", "must_2"}
    assert [c["kind"] for c in editorial_contract.for_shot(shot, p)["must"]] == [
        "sequence", "document", "onscreen_text"]
    assert shot["visual_plan"] == {"mode": "explicit_sequence"}
    assert shot["visual"] == "画面必须依次展示扫码和装车"


def test_explicit_sequence_cannot_enter_the_real_generic_montage_matcher(monkeypatch):
    from jinggao import library, montage_match

    p = project(["工作人员处理快件。"], "必须依次展示扫码和装车。")
    shots, response = response_for(p)
    response["contracts"][0]["must"] = [["sequence", "r", "必须依次展示扫码和装车"]]
    result = compile_one(p, shots, response)
    result[0]["duration"] = 8
    assert montage_match.piece_lengths(result[0]["duration"])
    assert result[0]["evidence_mode"] == "generic" and not result[0]["locked"]
    monkeypatch.setattr(library, "search", lambda *a, **k: pytest.fail("Explicit sequence reached generic retrieval"))
    passed, info = montage_match.match(p | {"shots": result}, result, lambda: None, lambda *a: None,
                                      review_batch=lambda *a: pytest.fail("No montage review is allowed"))
    assert passed == set() and info["model_requests"] == 0


def test_date_roles_remain_distinct_and_cannot_be_used_as_event_dates():
    p = project(["二〇二四年统计收录了一件唐代文物，素材拍摄于二〇二五年。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [
        ["data_period", "s1", "二〇二四年"], ["artifact_period", "s1", "唐代"],
        ["capture_period", "s1", "二〇二五年"],
    ]
    shot = compile_one(p, shots, response)[0]
    assert shot["era"] == "" and shot["event"] == ""
    assert [c["kind"] for c in editorial_contract.for_shot(shot, p)["facts"]] == [
        "data_period", "artifact_period", "capture_period"]
    response["presentation"][0]["event"] = ["f1"]
    with pytest.raises(ValueError, match="类型"):
        compile_one(p, shots, response)


@pytest.mark.parametrize("two_events", [False, True])
def test_real_c919_first_flight_and_commercial_flight_require_split_not_false_single_event(two_events):
    text = "二〇一七年五月五日，C919成功首飞。二〇二三年五月二十八日，它完成首次商业飞行。"
    p = project([text])
    shots, response = response_for(p)
    assert len(shots) == 1  # Real rule_shots behavior that caused the ambiguity.
    facts = [["event", "s1", "C919成功首飞"], ["event_date", "s1", "二〇一七年五月五日"],
             ["event_date", "s1", "二〇二三年五月二十八日"]]
    if two_events:
        facts.append(["event", "s1", "它完成首次商业飞行"])
    response["contracts"][0].update(use="event_record", facts=facts)
    response["presentation"][0]["event"] = [f"f{i}" for i in range(1, len(facts) + 1)]
    with pytest.raises(ValueError, match="需先拆分"):
        compile_one(p, shots, response)
    response["presentation"][0]["event"] = ["f1", "f2"]
    with pytest.raises(ValueError, match="全部事件"):
        compile_one(p, shots, response)


@pytest.mark.parametrize("change,match", [
    ("missing_event", "缺少拟呈现事件"), ("date_only", "全部事件"),
    ("missing_date", "全部事件"), ("missing_opponent", "参与者"), ("must_event_illustration", "降为"),
])
def test_specific_event_cannot_shed_its_identity_fields(change, match):
    event = "甲赛事决赛，马龙对阵王楚钦"
    p = project(["二〇二四年五月一日，" + event + "。"], "必须展示甲赛事决赛。")
    shots, response = response_for(p)
    response["contracts"][0].update(use="event_record", facts=[
        ["event", "s1", event], ["event_date", "s1", "二〇二四年五月一日"],
        ["person", "s1", "马龙"], ["person", "s1", "王楚钦"],
    ])
    target = response["presentation"][0]
    target.update(persons=[person("马龙", "f3")], event_persons=[person("王楚钦", "f4")], event=["f1", "f2"])
    if change == "missing_event":
        target["event"] = []
    elif change == "date_only":
        target["event"] = ["f2"]
    elif change == "missing_date":
        target["event"] = ["f1"]
    elif change == "missing_opponent":
        target["event_persons"] = []
    else:
        response["contracts"][0].update(use="illustration", must=[["event", "r", "必须展示甲赛事决赛"]])
        target["event"].append("m1")
    with pytest.raises(ValueError, match=match):
        compile_one(p, shots, response)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "foreign", "source", "quote", "primary",
                                         "missing_sources", "empty_sources"])
def test_bad_presentation_rows_fail_atomically(corruption):
    p = project(["工作人员搬运包裹。", "消费者查看进度。"])
    shots, response = response_for(p)
    if corruption == "missing":
        response["presentation"].pop()
    elif corruption == "duplicate":
        response["presentation"][1] = deepcopy(response["presentation"][0])
    elif corruption == "reordered":
        response["presentation"].reverse()
    elif corruption == "foreign":
        response["presentation"][0]["ref"] = "other-project"
    elif corruption == "source":
        response["presentation"][0]["sources"] = [["s2", "消费者"]]
    elif corruption == "quote":
        response["presentation"][0]["sources"] = [["s1", "包裹已送达"]]
    elif corruption == "primary":
        response["presentation"][0]["primary"] = {"action": "扫描"}
    elif corruption == "missing_sources":
        response["presentation"][0].pop("sources")
    else:
        response["presentation"][0]["sources"] = []
    original = deepcopy((p, shots, response))
    with pytest.raises(ValueError):
        source_planner.compile_responses(p, shots, [response])
    assert (p, shots, response) == original


@pytest.mark.parametrize("selected", [{"selected_asset": {"id": "already-picked"}}, {"approved": True}])
def test_protected_manual_shot_is_readonly_and_selected_generated_shot_is_not_replanned(selected):
    p = project(["工作人员搬运包裹。", "消费者查看进度。"])
    shots = source_planner.source_shots(p)
    shots[0].update(visual_origin="manual", locked=True, visual="固定保留完整组合", persons=[{"name": "张明", "aliases": []}])
    requests = source_planner.build_requests(p, shots)
    assert requests[0]["targets"] == ["s2"]
    result = compile_one(p, shots, reply(requests[0]))
    assert result[0] == shots[0]
    shots[1].update(selected)
    with pytest.raises(ValueError, match="已选择"):
        source_planner.build_requests(p, shots)


def test_mock_cold_request_then_whole_cache_recompile_counts_zero_paid_requests_on_hit():
    p = project([f"第{i}件快递由工作人员处理。" for i in range(34)], id="offline-project")
    original, calls, progress = deepcopy(p), [], []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        assert system == source_planner.SYSTEM and options == source_planner.OPTIONS
        assert "".join(u["text"] for u in request["script_units"]) == p["script"]
        return reply(request), {"model": "mock-contract", "request_count": 1,
                                "usage": {"prompt_tokens": 123, "completion_tokens": 45}}

    first, info = source_planner.plan(p, chat=chat, progress=lambda a, b: progress.append((a, b)))
    second, cached = source_planner.plan(p, chat=lambda *a, **k: pytest.fail("cache should be complete"))
    assert len(calls) == 1 and info["model_requests"] == info["request_count"] == 1
    assert info["usage"] == {"prompt_tokens": 123, "completion_tokens": 45}
    assert cached["cache_hit"] and cached["cached_batches"] == 1
    assert cached["model_requests"] == cached["request_count"] == 0 and cached["usage"] == {}
    assert first[0]["id"] != second[0]["id"] and all(editorial_contract.for_shot(s, p) for s in second)
    assert info["contract_count"] == 34 and progress == [(0, 1), (1, 1)]
    assert info["chat_calls"] == 1 and info["request_count_complete"]
    assert p == original
    assert len(list((paths.DATA / "cache" / "source-planning").glob("*.json"))) == 1
    assert not (paths.DATA / "cache" / "editorial-contracts").exists()


@pytest.mark.parametrize("changed", ["script", "requirements", "style", "endpoint", "model", "system", "options", "transport",
                                     "quote_repair_version"])
def test_cache_binds_complete_source_route_and_protocol(changed, isolated, monkeypatch):
    p = project(["工作人员处理包裹。"], id="offline-project")
    calls = []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        return reply(request), {"model": isolated["requirements_model"], "request_count": 1, "usage": {}}

    source_planner.plan(p, chat=chat)
    if changed in {"script", "requirements", "style"}:
        p[changed] += " 完整追加条件。"
    elif changed == "endpoint":
        isolated["model_base_url"] = "https://new.invalid/v1"
    elif changed == "model":
        isolated["requirements_model"] = "another-model"
    elif changed == "system":
        monkeypatch.setattr(source_planner, "SYSTEM", source_planner.SYSTEM + "\n新增协议")
    elif changed == "options":
        monkeypatch.setattr(source_planner, "OPTIONS", source_planner.OPTIONS | {"max_tokens": 7000})
    elif changed == "quote_repair_version":
        monkeypatch.setattr(source_planner, "QUOTE_REPAIR_VERSION", "different-local-repair-policy")
    else:
        from jinggao import model_protocol
        apply = model_protocol.apply

        def updated(payload, cfg, **kwargs):
            apply(payload, cfg, **kwargs)
            payload["temperature"] = .123

        monkeypatch.setattr(model_protocol, "apply", updated)
    _, info = source_planner.plan(p, chat=chat)
    assert len(calls) == 2 and not info["cache_hit"]


def test_cache_hit_revalidates_presentation_and_makes_one_fresh_call_for_corrupt_cache():
    p = project(["工作人员处理包裹。"], id="offline-project")
    calls = []

    def chat(system, content, **options):
        calls.append(content)
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    source_planner.plan(p, chat=chat)
    path, = (paths.DATA / "cache" / "source-planning").glob("*.json")
    data = json.loads(path.read_text())
    data["response"][0]["presentation"][0]["persons"] = [person("刘翔", "f1")]
    path.write_text(json.dumps(data))
    _, info = source_planner.plan(p, chat=chat)
    assert len(calls) == 2 and info["model_requests"] == 1 and not info["cache_hit"]


@pytest.mark.parametrize("failure", ["invalid", "network", "cancel"])
def test_failure_has_no_paid_retry_old_path_fallback_or_cache(failure, monkeypatch):
    p = project(["工作人员处理包裹。"], id="offline-project")
    calls = []
    monkeypatch.setattr(planner, "_plan_model_batch", lambda *a, **k: pytest.fail("No old planning fallback"))

    def chat(system, content, **options):
        calls.append(content)
        if failure == "network":
            raise RuntimeError("offline network failure")
        response = reply(json.loads(content))
        if failure == "invalid":
            response["presentation"] = []
        return response, {"model": "mock-contract", "request_count": 1, "usage": {}}

    def check():
        if calls and failure == "cancel":
            raise RuntimeError("cancelled")

    with pytest.raises((ValueError, RuntimeError)):
        source_planner.plan(p, chat=chat, check=check)
    assert len(calls) == 1 and not list((paths.DATA / "cache" / "source-planning").glob("*.json"))


def test_changed_route_during_call_does_not_publish_under_original_key(isolated):
    p = project(["工作人员处理包裹。"], id="offline-project")

    def chat(system, content, **options):
        isolated["model_base_url"] = "https://moved.invalid/v1"
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    _, info = source_planner.plan(p, chat=chat)
    assert info["model_requests"] == 1
    assert not list((paths.DATA / "cache" / "source-planning").glob("*.json"))


def test_identical_concurrent_cold_calls_share_one_fill():
    p = project(["工作人员处理包裹。"], id="offline-project")
    entered, release = Event(), Event()
    calls = []

    def chat(system, content, **options):
        calls.append(content)
        entered.set()
        assert release.wait(3)
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(source_planner.plan, p, chat=chat) for _ in range(4)]
        assert entered.wait(3)
        release.set()
        result = [future.result(timeout=3) for future in futures]
    assert len(calls) == 1 and sum(info["model_requests"] for _, info in result) == 1
    assert sum(info["cache_hit"] for _, info in result) == 3


def test_more_than_forty_units_remain_sequential_complete_batches():
    p = project([f"第{i}件快递由工作人员处理。" for i in range(41)], id="offline-project")
    calls = []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        return reply(request), {"model": "mock-contract", "request_count": 1, "usage": {"prompt_tokens": 10}}

    result, info = source_planner.plan(p, chat=chat)
    assert [len(r["targets"]) for r in calls] == [40, 1]
    assert len(result) == 41 and info["model_requests"] == 2
    assert info["usage"] == {"prompt_tokens": 20}


@pytest.mark.parametrize("reported,expected", [(3, 3), (0, 0), (None, None), (True, None)])
def test_actual_transport_counts_are_not_guessed_from_logical_calls(reported, expected):
    p = project(["工作人员处理包裹。"])

    def chat(system, content, **options):
        info = {"model": "mock-contract", "usage": {}}
        if reported is not None:
            info["request_count"] = reported
        return reply(json.loads(content)), info

    _, info = source_planner.plan(p, chat=chat)
    assert info["chat_calls"] == 1
    assert info["request_count"] == info["model_requests"] == expected
    assert info["request_count_complete"] is (expected is not None)


def test_failed_transport_reports_prior_and_current_actual_counts_without_retry():
    p = project([f"第{i}件包裹由工作人员处理。" for i in range(41)])
    calls = []

    def chat(system, content, **options):
        calls.append(content)
        if len(calls) == 2:
            exc = RuntimeError("bounded transport failed")
            exc.request_count = 2
            exc.model_requests = 2
            raise exc
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    with pytest.raises(RuntimeError) as error:
        source_planner.plan(p, chat=chat)
    assert len(calls) == 2 and error.value.request_count == 3
    assert error.value.model_requests == 3
    assert error.value.model_info["model_requests"] == error.value.model_info["request_count"] == 3
    assert error.value.model_info["chat_calls"] == 2


def test_cancellation_while_waiting_on_same_key_has_zero_transport_calls():
    p = project(["工作人员处理包裹。"], id="offline-project")
    entered, release, cancel = Event(), Event(), Event()

    def chat(system, content, **options):
        entered.set()
        assert release.wait(3)
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    def check():
        if cancel.is_set():
            raise RuntimeError("cancelled while waiting")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(source_planner.plan, p, chat=chat)
        assert entered.wait(3)
        waiter = pool.submit(source_planner.plan, p, chat=chat, check=check)
        cancel.set()
        try:
            with pytest.raises(RuntimeError) as error:
                waiter.result(timeout=2)
            assert error.value.request_count == 0
        finally:
            release.set()
        first.result(timeout=3)


def test_c919_fixture_has_34_unchanged_units_and_one_mock_request():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "editorial_contract_c919_quote_repair.json").read_text())
    p = {"script": "".join(u["text"] for u in fixture["script_units"]),
         "requirements": fixture["requirements"], "style": "source replay"}
    shots = source_planner.source_shots(p)
    requests = source_planner.build_requests(p, shots)
    assert len(shots) == 34 and len(requests) == 1
    assert "".join(s["narration"] for s in shots) == p["script"]
    # Transport/source preservation only: empty mock facts are not a semantic
    # C919 planning evaluation and may never be mistaken for media approval.
    result = source_planner.compile_responses(p, shots, [reply(requests[0])])
    assert all(not s["approved"] for s in result)


@pytest.mark.parametrize("text", [
    "二〇一七年五月五日，C919成功首飞。二〇二三年五月二十八日，它完成首次商业飞行。",
    "二〇一七年五月五日C919成功首飞，二〇二三年五月二十八日完成首次商业飞行。",
])
def test_a_single_event_fact_cannot_hide_multiple_event_anchors(text):
    p = project([text])
    shots, response = response_for(p)
    assert len(shots) == 1
    response["contracts"][0].update(use="event_record", facts=[["event", "s1", text]])
    response["presentation"][0]["event"] = ["f1"]
    original = deepcopy((p, shots, response))
    with pytest.raises(ValueError, match="原子|时间锚点"):
        source_planner.compile_responses(p, shots, [response])
    assert (p, shots, response) == original


@pytest.mark.parametrize("text", ["从2017到2023持续测试。", "从2017年到2023年持续测试。",
                                     "二〇一七年至二〇二三年持续测试。"])
def test_one_continuous_time_range_is_not_counted_as_two_events(text):
    p = project([text])
    shots, response = response_for(p)
    response["contracts"][0].update(use="event_record", facts=[["event", "s1", text]])
    response["presentation"][0]["event"] = ["f1"]
    result = compile_one(p, shots, response)
    assert result[0]["event"] == text and not result[0]["approved"]


@pytest.mark.parametrize("with_facts", [True, False])
def test_combined_person_quote_does_not_allow_only_one_of_two_required_people(with_facts):
    p = project(["刘翔和苏炳添参加训练。"], "必须展示刘翔和苏炳添。")
    shots, response = response_for(p)
    response["contracts"][0].update(facts=[["person", "s1", "刘翔和苏炳添"]] if with_facts else [],
                                     must=[["person", "r", "必须展示刘翔和苏炳添"]])
    refs = ["f1", "m1"] if with_facts else ["m1"]
    response["presentation"][0]["persons"] = [person("刘翔", *refs)]
    with pytest.raises(ValueError, match="独立姓名|多人展示"):
        compile_one(p, shots, response)


def test_same_wave_failure_is_broadcast_without_per_waiter_transport_or_durable_negative_cache():
    p = project(["工作人员处理包裹。"], id="offline-project")
    entered, release, waiting = Event(), Event(), Event()
    calls, checks = [], []

    def chat(system, content, **options):
        calls.append(content)
        entered.set()
        assert release.wait(3)
        exc = RuntimeError("same wave mock failure")
        exc.request_count = exc.model_requests = 1
        raise exc

    def check_waiter():
        checks.append(1)
        if len(checks) >= 3:
            waiting.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(source_planner.plan, p, chat=chat)
        assert entered.wait(3)
        waiter = pool.submit(source_planner.plan, p, chat=chat, check=check_waiter)
        assert waiting.wait(2)
        release.set()
        errors = []
        for future in (leader, waiter):
            with pytest.raises(RuntimeError) as caught:
                future.result(timeout=3)
            errors.append(caught.value)
    assert len(calls) == 1 and errors[0] is not errors[1]
    assert errors[0].request_count == errors[0].model_requests == 1
    assert errors[1].request_count == errors[1].model_requests == 0
    assert errors[1].model_info["shared_failure"] and errors[1].model_info["shared_request_count"] == 1
    assert not list((paths.DATA / "cache" / "source-planning").glob("*.json"))
    with pytest.raises(RuntimeError):
        source_planner.plan(p, chat=chat)
    assert len(calls) == 2  # A new request after the failed wave may try again.
    assert not source_planner._waves


def test_unknown_prior_request_count_keeps_every_exception_alias_unknown():
    p = project([f"第{i}件包裹由工作人员处理。" for i in range(41)])
    calls = []

    def chat(system, content, **options):
        calls.append(content)
        if len(calls) == 2:
            exc = RuntimeError("known second failure, unknown first transport")
            exc.request_count = exc.model_requests = 2
            raise exc
        return reply(json.loads(content)), {"model": "mock-contract", "usage": {}}

    with pytest.raises(RuntimeError) as error:
        source_planner.plan(p, chat=chat)
    exc = error.value
    assert exc.request_count is exc.model_requests is None
    assert exc.model_info["request_count"] is exc.model_info["model_requests"] is None
    assert not exc.model_info["request_count_complete"] and exc.model_info["known_request_count"] == 2


def test_leader_cancellation_does_not_poison_other_registered_callers():
    p = project(["工作人员处理包裹。"], id="offline-project")
    entered, release, cancel_leader, waiting = Event(), Event(), Event(), Event()
    calls, waiter_checks = [], []

    def chat(system, content, **options):
        calls.append(content)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    def leader_check():
        if cancel_leader.is_set():
            raise RuntimeError("leader cancelled")

    def waiter_check():
        waiter_checks.append(1)
        if len(waiter_checks) >= 3:
            waiting.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(source_planner.plan, p, chat=chat, check=leader_check)
        assert entered.wait(3)
        waiter = pool.submit(source_planner.plan, p, chat=chat, check=waiter_check)
        assert waiting.wait(2)
        cancel_leader.set()
        release.set()
        with pytest.raises(RuntimeError, match="leader cancelled") as error:
            leader.result(timeout=3)
        assert error.value.request_count == error.value.model_requests == 1
        result, info = waiter.result(timeout=3)
    assert len(calls) == 2 and info["request_count"] == 1 and result
    assert not source_planner._waves


def test_full_names_containing_a_coordinator_and_complete_must_only_lists_still_work():
    p = project(["王和平与刘翔参加训练。"], "必须展示王和平与刘翔。")
    shots, response = response_for(p)
    response["contracts"][0]["must"] = [["person", "r", "必须展示王和平与刘翔"]]
    response["presentation"][0]["persons"] = [person("王和平", "m1"), person("刘翔", "m1")]
    result = compile_one(p, shots, response)
    assert [p["name"] for p in result[0]["persons"]] == ["王和平", "刘翔"]


@pytest.mark.parametrize("command", ["必须展示刘翔，随后展示苏炳添。",
                                      "必须展示刘翔随后展示苏炳添。",
                                      "必须展示刘翔；展示苏炳添。"])
def test_must_only_separate_person_commands_need_each_person(command):
    p = project(["刘翔和苏炳添参加训练。"], command)
    shots, response = response_for(p)
    response["contracts"][0].update(facts=[], must=[["person", "r", command]])
    response["presentation"][0]["persons"] = [person("刘翔", "m1")]
    original = deepcopy((p, shots, response))
    with pytest.raises(ValueError, match="多人展示"):
        compile_one(p, shots, response)
    assert (p, shots, response) == original

    response["presentation"][0]["persons"].append(person("苏炳添", "m1"))
    result = compile_one(p, shots, response)
    assert [p["name"] for p in result[0]["persons"]] == ["刘翔", "苏炳添"]
    assert result[0]["narration"] == p["script"]
    assert editorial_contract.for_shot(result[0], p)["must"][0]["quote"] == command
    assert not result[0]["approved"]


def tea_quote_response():
    # These are the three actual invalid quotes in the first v21 paid response.
    p = project([
        "接下来是杀青。这个名字听着有些陌生，作用却可以说清楚。加工时用高温降低鲜叶中酶的活性，抑制多酚类物质发生酶促氧化，帮助控制叶子红变。",
        "设备使用前要清洁，不合格的茶叶要单独处理，加工批次也需要记录。这些工作不如炒茶动作显眼，却直接关系到产品能否稳定地做好。",
    ])
    shots, response = response_for(p)
    assert len(shots) == 2
    response["contracts"][0]["facts"] = [["activity", "s1", "杀青...高温降低鲜叶中酶的活性"]]
    response["contracts"][1]["facts"] = [["activity", "s2", "设备使用前要清洁...加工批次也需要记录"]]
    response["presentation"][1]["sources"] = [["s2", "设备使用前要清洁...加工批次也需要记录"]]
    return p, shots, response


def test_actual_three_tea_omissions_expand_only_the_unique_continuous_source_span():
    p, shots, response = tea_quote_response()
    result = compile_one(p, shots, response)
    repairs = source_planner._compiled_repairs(result)
    assert len(repairs) == 3
    assert [r["field"] for r in repairs] == ["facts", "facts", "presentation.sources"]
    assert [len(r["after"]) for r in repairs] == [38, 30, 30]
    for repair in repairs:
        span = repair["source_span"]
        assert p["script"][span["start"]:span["end"]] == repair["after"]
        assert repair["version"] == source_planner.QUOTE_REPAIR_VERSION
    assert "不合格的茶叶要单独处理" in repairs[1]["after"] == repairs[2]["after"]
    assert "这个名字听着有些陌生，作用却可以说清楚。加工时用" in repairs[0]["after"]
    assert "".join(s["narration"] for s in result) == p["script"]
    for shot, raw in zip(result, response["presentation"], strict=True):
        assert shot["source_presentation"]["raw"] == raw
        assert not shot["approved"] and shot["persons"] == [] and shot["event"] == ""
    for shot, raw in zip(result, response["contracts"], strict=True):
        bound = editorial_contract.for_shot(shot, p)
        assert bound["use"] == raw["use"] and bound["query_hints"] == raw["queries"]
        assert bound["facts"][0]["quote"] == next(r["after"] for r in repairs
                                                   if r["ref"] == raw["ref"] and r["field"] == "facts")


@pytest.mark.parametrize("text,quote", [
    ("A甲B乙B", "A...B"),  # One prefix, two possible suffixes.
    ("A甲A乙B", "A...B"),  # Two prefixes, one suffix.
    ("A甲B，A甲B", "A...B"),
    ("aaaXb", "aa...b"),  # Overlapping endpoint occurrences still count.
    ("aXbbb", "a...bb"),
    ("A甲B" + "中" * 161 + "B", "A...B"),  # A long alternative is still ambiguity.
    ("A" + "中" * 159 + "B", "A...B"),
    ("尾在前，首在后", "首...尾"),
    ("首和另一端", "首...尾"),
    ("首中尾", "...尾"),
    ("首中尾", "首..."),
    ("首中尾", "首......尾"),
    ("首中尾", "首……尾"),
    ("首中尾", "首....尾"),
    ("首中尾", "首...中…尾"),
    ("鲜叶接受高温处理", "高温令叶片软化"),  # Ordinary paraphrases are never repaired.
])
def test_ambiguous_missing_reversed_long_or_nonliteral_quotes_still_fail(text, quote):
    p = project(["这里说明制茶过程。"], text)
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["activity", "r", quote]]
    original = deepcopy(response)
    with pytest.raises(ValueError, match="未逐字命中") as error:
        compile_one(p, shots, response)
    assert error.value.quote_repairs == [] and response == original


@pytest.mark.parametrize("quote", ["鲜叶...杀青", "杀青...处理"])
def test_quote_repair_cannot_join_sources_or_move_the_declared_reference(quote):
    p = project(["鲜叶需要摊放。", "杀青开始处理。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["activity", "s1", quote]]
    with pytest.raises(ValueError, match="未逐字命中") as error:
        compile_one(p, shots, response)
    assert error.value.quote_repairs == []


def test_quote_expansion_does_not_grant_an_undeclared_context_ref():
    p = project(["鲜叶需要摊放。", "杀青开始处理。"])
    shots, response = response_for(p)
    response["presentation"][0]["sources"].append(["s2", "杀青...处理"])
    with pytest.raises(ValueError, match="来源或上下文引用无效"):
        compile_one(p, shots, response)


def test_quote_repair_also_preserves_the_existing_exact_object_citation_shape():
    p = project(["鲜叶先摊放然后杀青。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [{"kind": "activity", "source_ref": "s1", "quote": "鲜叶…杀青"}]
    result = compile_one(p, shots, response)
    assert editorial_contract.for_shot(result[0], p)["facts"][0]["quote"] == "鲜叶先摊放然后杀青"
    assert source_planner._compiled_repairs(result)[0]["before"] == "鲜叶…杀青"
    assert response["contracts"][0]["facts"][0]["quote"] == "鲜叶…杀青"


@pytest.mark.parametrize("quote", ["叶子...摊开", "叶子…摊开", "叶子……摊开"])
def test_literal_source_ellipsis_is_not_interpreted_as_an_omission(quote):
    p = project([quote + "。"])
    shots, response = response_for(p)
    response["contracts"][0]["facts"] = [["activity", "s1", quote]]
    response["presentation"][0]["sources"] = [["s1", quote]]
    result = compile_one(p, shots, response)
    assert source_planner._compiled_repairs(result) == []
    assert editorial_contract.for_shot(result[0], p)["facts"][0]["quote"] == quote


def test_unicode_omission_in_must_preserves_intermediate_people_and_the_identity_gate():
    command = "必须展示刘翔，随后展示苏炳添。"
    p = project(["两位运动员参加训练。"], command)
    shots, response = response_for(p)
    response["contracts"][0]["must"] = [["person", "r", "必须展示刘翔…苏炳添"]]
    response["presentation"][0]["persons"] = [person("刘翔", "m1")]
    with pytest.raises(ValueError, match="多人展示") as error:
        compile_one(p, shots, response)
    assert error.value.quote_repairs[0]["after"] == command[:-1]
    response["presentation"][0]["persons"].append(person("苏炳添", "m1"))
    result = compile_one(p, shots, response)
    assert result[0]["evidence_mode"] == "person" and len(result[0]["persons"]) == 2
    assert source_planner._compiled_repairs(result)[0]["source"] == "requirements"
    assert not result[0]["approved"]


def test_expanded_event_still_cannot_hide_two_historical_events():
    text = "2017年C919成功首飞。2023年C919首次商业飞行。"
    p = project([text])
    shots, response = response_for(p)
    response["contracts"][0].update(use="event_record", facts=[["event", "s1", "2017年…首次商业飞行"]])
    response["presentation"][0]["event"] = ["f1"]
    with pytest.raises(ValueError, match="原子锚点|单一事件"):
        compile_one(p, shots, response)


def test_repair_log_and_raw_response_survive_diagnostics_and_cache_recompilation(tmp_path):
    p, _, raw = tea_quote_response()
    p.update(id="offline-project", _plan_diagnostics_dir=str(tmp_path / "repair-diagnostics"))
    calls = []

    def chat(*args, **kwargs):
        calls.append(1)
        return deepcopy(raw), {"model": "mock-contract", "request_count": 1, "usage": {}}

    first, info = source_planner.plan(p, chat=chat)
    second, cached = source_planner.plan(p, chat=lambda *a, **k: pytest.fail("Repair never calls another model"))
    assert len(calls) == 1 and info["request_count"] == 1 and cached["request_count"] == 0
    assert info["quote_repair_count"] == cached["quote_repair_count"] == 3
    assert info["quote_repairs"] == cached["quote_repairs"] == source_planner._compiled_repairs(first)
    assert source_planner._compiled_repairs(first) == source_planner._compiled_repairs(second)
    cache_path, = (paths.DATA / "cache" / "source-planning").glob("*.json")
    assert json.loads(cache_path.read_text())["response"] == [raw]
    for metadata in (info, cached):
        diagnostic = json.loads(Path(metadata["diagnostic_path"]).read_text())
        assert diagnostic["quote_repairs"] == metadata["quote_repairs"]
        assert diagnostic["quote_repair_count"] == 3 and diagnostic["status"] == "compiled"
        assert diagnostic["batches"][0]["response"] == raw


def test_repairs_remain_diagnostic_when_another_strict_validator_rejects(tmp_path):
    p, _, raw = tea_quote_response()
    p["_plan_diagnostics_dir"] = str(tmp_path / "failed-repairs")
    raw["presentation"][0]["persons"] = [person("无据姓名", "f1")]

    def chat(*args, **kwargs):
        return deepcopy(raw), {"model": "mock-contract", "request_count": 1, "usage": {}}

    with pytest.raises(ValueError, match="类型无效") as error:
        source_planner.plan(p, chat=chat)
    diagnostic = json.loads(Path(error.value.diagnostic_path).read_text())
    assert diagnostic["status"] == "failed" and diagnostic["quote_repair_count"] == 3
    assert diagnostic["batches"][0]["response"] == raw
    assert error.value.model_info["quote_repair_count"] == 3 and error.value.request_count == 1


def test_paid_raw_response_is_saved_before_compile_rejects_it_with_safe_metadata(tmp_path, monkeypatch):
    directory = tmp_path / "explicit-diagnostics"
    p = project(["工作人员处理包裹。"], _plan_diagnostics_dir=str(directory))
    response = {"contracts": [], "presentation": []}

    def chat(system, content, **options):
        return response, {"model": "mock-contract", "request_count": 1, "usage": {"prompt_tokens": 9},
                          "api_key": "fake-secret-not-for-diagnostics", "headers": {"Authorization": "fake-secret"}}

    def rejected(*args, **kwargs):
        path, = directory.glob("source-planner-*.json")
        recorded = json.loads(path.read_text())
        assert recorded["batches"][0]["response"] == response
        assert recorded["batches"][0]["status"] == "received"
        raise ValueError("intentional offline compile rejection")

    monkeypatch.setattr(source_planner, "compile_responses", rejected)
    with pytest.raises(ValueError) as error:
        source_planner.plan(p, chat=chat)
    path, = directory.glob("source-planner-*.json")
    text = path.read_text()
    recorded = json.loads(text)
    assert "fake-secret" not in text
    assert recorded["status"] == "failed" and recorded["request_count"] == 1
    assert recorded["system"] == source_planner.SYSTEM
    assert recorded["batches"][0]["response"] == response
    assert "".join(u["text"] for u in recorded["batches"][0]["request"]["script_units"]) == p["script"]
    assert error.value.diagnostic_path == str(path)
    assert not list((paths.DATA / "cache" / "source-planning").glob("*.json"))


@pytest.mark.parametrize("pid,expected", [("a" * 32, True), ("offline-project", False), (None, False)])
def test_diagnostics_default_only_to_a_real_project_id(pid, expected):
    p = project(["工作人员处理包裹。"], **({"id": pid} if pid else {}))

    def chat(system, content, **options):
        return reply(json.loads(content)), {"model": "mock-contract", "request_count": 1, "usage": {}}

    _, info = source_planner.plan(p, chat=chat)
    files = list((paths.DATA / "diagnostics").glob("*/source-planner-*.json"))
    assert bool(files) is expected
    assert bool(info["diagnostic_path"]) is expected
    if expected:
        assert json.loads(files[0].read_text())["status"] == "compiled"
