from copy import deepcopy
import json
from pathlib import Path

import pytest

from jinggao import config, editorial_contract as contract, paths


C919 = json.loads((Path(__file__).parent / "fixtures" / "editorial_contract_c919_quote_repair.json").read_text())


@pytest.fixture
def c919(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {"requirements_model": "test"})
    project = {"id": "c" * 32, "script": "".join(u["text"] for u in C919["script_units"]),
               "requirements": C919["requirements"]}
    shots = [{"id": u["ref"], "source_unit_id": u["ref"], "narration": u["text"].strip(),
              "visual_origin": "generated", "persons": [], "event": "C919商业首航", "approved": False}
             for u in C919["script_units"]]
    return project, shots


def c919_event_patch(request):
    # Hand-labelled repair fixture. These are literal event descriptions, not
    # dates locally reclassified as an event by production code.
    quotes = {"s3": ["s3", "C919成功首飞"],
              "s5": ["s5", "东方航空MU9191，从上海虹桥飞往北京首都。"],
              "s7": ["s5", "东方航空MU9191，从上海虹桥飞往北京首都。"],
              "s14": ["s14", "C919在二〇二二年九月二十九日取得中国民航局颁发的型号合格证"],
              "s32": ["s32", "当MU9191从上海飞到北京"]}
    return {"repairs": [{"ref": ref, "pointers": [], "event": quotes[ref]} for ref in request["targets"]]}


def test_real_c919_response_repairs_only_source_pointers_and_missing_event_citations(c919):
    project, shots = c919
    before, initial, calls = deepcopy((project, shots)), deepcopy(C919["initial_response"]), []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        if len(calls) == 1:
            return deepcopy(initial), {"model": "test", "usage": {"total_tokens": 100}}
        assert system == contract.QUOTE_REPAIR_SYSTEM and system != contract.SYSTEM
        assert request["targets"] == ["s3", "s5", "s7", "s14", "s32"]
        assert all(row["pointers"] == [] and row["event_required"] for row in request["failed_rows"])
        return c919_event_patch(request), {"model": "test", "usage": {"total_tokens": 30}}

    result, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert info["model_requests"] == 2 and info["usage"] == {"total_tokens": 130}
    assert len(result) == 34 and (project, shots) == before
    for original, shot in zip(initial["contracts"], result, strict=True):
        normalized, _ = contract._normalize_row(original)
        bound = contract.for_shot(shot, project)
        assert bound["use"] == original["use"] and bound["query_hints"] == normalized["queries"]
        assert all(ref in bound["refs"] for ref in original.get("refs", []))
        for field in ("facts", "must"):
            old = [(c[0], c[2]) for c in normalized[field]]
            assert [(c["kind"], c["quote"]) for c in bound[field]][:len(old)] == old
        assert shot["event"] == "C919商业首航" and shot["approved"] is False
    diagnostic = json.loads(Path(info["diagnostic_path"]).read_text())
    assert len(diagnostic["local_quote_repairs"]) == 6
    assert diagnostic["batches"][0]["response"] == initial == C919["initial_response"]
    assert all(item["from"] == "r" and item["to"] == item["ref"] for item in diagnostic["local_quote_repairs"])
    again, warm = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert again == result and warm["model_requests"] == 0 and len(calls) == 2


def test_actual_old_full_row_repair_cannot_replace_or_downgrade_the_original_rows(c919):
    project, shots = c919
    replies = [C919["initial_response"], C919["rejected_full_row_repair"]]

    def chat(*args, **kwargs):
        return deepcopy(replies.pop(0)), {"model": "test", "usage": {}}

    with pytest.raises(ValueError, match="引用修正未按原错误镜头返回"):
        contract.extract(project, shots, chat=chat, check=lambda: None)
    assert not replies and not list((paths.DATA / "cache" / "editorial-contracts").glob("*.json"))


@pytest.mark.parametrize("damage", ["use", "queries", "facts", "refs", "event_quote", "event_missing",
                                    "unrelated_event", "extra_pointer", "foreign_ref"])
def test_narrow_event_patch_cannot_modify_readonly_semantics_or_invent_citations(c919, damage):
    project, shots = c919
    calls = []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        if len(calls) == 1:
            return deepcopy(C919["initial_response"]), {"model": "test"}
        patch = c919_event_patch(request)
        row = patch["repairs"][0]
        if damage in ("use", "queries", "facts", "refs"):
            row[damage] = "illustration" if damage == "use" else []
        elif damage == "event_quote":
            row["event"][1] = "从未发生也不在原文中的另一场飞行"
        elif damage == "event_missing":
            row["event"] = []
        elif damage == "unrelated_event":
            row["event"] = ["s14", "C919在二〇二二年九月二十九日取得中国民航局颁发的型号合格证"]
        elif damage == "extra_pointer":
            row["pointers"] = [{"id": "facts:0", "source_ref": "s5"}]
        else:
            row["ref"] = "s999"
        return patch, {"model": "test"}

    with pytest.raises(ValueError, match="编辑契约提取失败"):
        contract.extract(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 2 and not list((paths.DATA / "cache" / "editorial-contracts").glob("*.json"))


@pytest.mark.parametrize("quotation", ["刘翔", "必须展示跨栏动作"])
def test_unique_exact_pointer_repair_needs_no_second_request_and_preserves_existing_refs(quotation):
    field, kind = ("facts", "person") if quotation == "刘翔" else ("must", "activity")
    units = ["刘翔参加决赛，必须展示跨栏动作。", "这是另一句原文。", "他随后接受采访。"]
    project = {"script": "".join(units), "requirements": ""}
    shots = [{"id": f"shot-{i}", "narration": text, "visual_origin": "generated"} for i, text in enumerate(units)]
    rows = [{"ref": f"s{i}", "use": "illustration", "refs": [], "facts": [], "must": [], "queries": ["原方向"]}
            for i in range(1, 4)]
    rows[2].update(refs=["s2"])
    rows[2][field] = [[kind, "r", quotation]]
    calls = []

    def chat(*args, **kwargs):
        calls.append(1)
        return {"contracts": deepcopy(rows)}, {}

    result, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    repaired = contract.for_shot(result[2], project)
    assert len(calls) == info["model_requests"] == 1
    assert repaired["refs"] == ["s2", "s1"] and repaired["query_hints"] == ["原方向"]
    assert [(c["kind"], c["ref"], c["quote"]) for c in repaired[field]] == [(kind, "s1", quotation)]


def test_repeated_occurrences_even_in_one_unit_are_not_locally_guessed():
    row = {"ref": "s2", "use": "illustration", "refs": [],
           "facts": [["person", "s2", "刘翔"]], "must": [], "queries": ["原检索方向"]}
    sources = {"s1": "刘翔参加比赛，刘翔随后接受采访。", "s2": "他回到赛场。", "r": ""}
    revised, task, local = contract._prepare_quote_repair(row, sources)
    assert revised == row and not local
    assert task["pointers"][0]["candidates"] == ["s1"]


def test_event_repair_may_use_context_added_only_by_an_existing_literal_citation():
    row = {"ref": "s2", "use": "event_record", "refs": [],
           "facts": [["person", "r", "刘翔"]], "must": [], "queries": ["原检索方向"]}
    sources = {"s1": "刘翔参加跨栏决赛。", "s2": "他夺得金牌。", "s3": "另一人参加游泳决赛。", "r": ""}
    revised, task, local = contract._prepare_quote_repair(row, sources)
    assert len(local) == 1 and task["event_sources"] == ["s2", "s1", "r"]
    patch = {"ref": "s2", "pointers": [], "event": ["s1", "刘翔参加跨栏决赛"]}
    merged = contract._merge_quote_repair(revised, task, patch, sources)
    assert merged["use"] == "event_record" and merged["queries"] == row["queries"]
    assert merged["facts"] == [["person", "s1", "刘翔"], ["event", "s1", "刘翔参加跨栏决赛"]]
