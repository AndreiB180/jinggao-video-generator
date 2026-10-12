from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
from threading import Barrier
import time

import pytest

from jinggao import config, editorial_contract as contract, paths, planner


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"requirements_model": "contract-v1", "planning_model": "planner-v1"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    lines = ["刘翔参加雅典奥运会男子一百一十米栏决赛。", "这个结果来自赛场上的表现。"]
    project = {"id": "a" * 32, "script": "\n\n".join(lines), "requirements": "保留身份和赛事。"}
    shots = [planner.new_shot(line, persons=[{"name": "刘翔", "aliases": []}],
                              event="雅典奥运会男子一百一十米栏决赛", evidence_mode="event")
             for line in lines]
    return project, shots, cfg


def response(request):
    return {"contracts": [{"ref": ref, "use": "illustration", "refs": [] if ref == "s1" else ["s1"],
                           "facts": [["person", "s1", "刘翔"]], "must": [], "queries": ["刘翔 跨栏"]}
                          for ref in request["targets"]]}


def cache_files(tmp_path):
    return list((tmp_path / "cache" / "editorial-contracts").glob("*.json"))


def test_complete_contract_cache_rebinds_new_shot_ids_and_counts_only_fresh_usage(setup, tmp_path):
    project, shots, _ = setup
    original = deepcopy((project, shots))
    calls = []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        assert options["task"] == "requirements"
        assert len(content) < len(json.dumps(request, ensure_ascii=False))
        return response(request), {"model": "contract-v1", "usage": {"prompt_tokens": 100, "completion_tokens": 40}}

    first, initial = contract.extract(project, shots, chat=chat, check=lambda: None)
    renamed = [s | {"id": f"new-shot-{i}"} for i, s in enumerate(shots)]
    progress = []
    second, resumed = contract.extract(project, renamed, chat=chat, check=lambda: None,
                                      progress=lambda n, total: progress.append((n, total)))
    assert len(calls) == len(cache_files(tmp_path)) == 1
    assert initial["model_requests"] == 1 and initial["cached_batches"] == 0
    assert resumed["model_requests"] == 0 and resumed["cached_batches"] == 1
    assert initial["usage"] == {"prompt_tokens": 100, "completion_tokens": 40}
    assert resumed["usage"] == {} and resumed["usage_unavailable_batches"] == []
    assert progress == [(0, 1), (1, 1)]
    assert [s["id"] for s in second] == [s["id"] for s in renamed]
    for old, new in zip(first, second, strict=True):
        assert new["persons"] == old["persons"] == shots[0]["persons"]
        assert new["event"] == old["event"] == shots[0]["event"]
        assert new["evidence_mode"] == "event" and new["approved"] is False
        bound = contract.for_shot(new, project)
        assert bound["shot_id"] == new["id"] and bound["facts"][0]["quote"] == "刘翔"
    assert (project, shots) == original


@pytest.mark.parametrize("change", ["script", "requirements", "protected", "model", "endpoint", "system", "options", "version"])
def test_contract_cache_invalidates_every_semantic_or_policy_input(setup, tmp_path, monkeypatch, change):
    project, shots, cfg = setup
    calls = []

    def chat(system, content, **options):
        calls.append(json.loads(content))
        return response(calls[-1]), {"model": cfg["requirements_model"], "usage": {}}

    contract.extract(project, shots, chat=chat, check=lambda: None)
    project, shots = deepcopy(project), deepcopy(shots)
    if change == "script":
        shots[1]["narration"] += "仍须保留对应年份。"
        project["script"] = "\n\n".join(s["narration"] for s in shots)
    elif change == "requirements":
        project["requirements"] += "必须看清跨栏动作。"
    elif change == "protected":
        shots[0].update(visual_origin="manual", visual="完整跨栏动作，不可用采访")
    elif change in {"model", "endpoint"}:
        cfg["requirements_model" if change == "model" else "model_base_url"] = "changed"
    elif change == "system":
        monkeypatch.setattr(contract, "SYSTEM", contract.SYSTEM + "\n新增原文约束。")
    elif change == "version":
        monkeypatch.setattr(contract, "VERSION", contract.VERSION + "_next")
    else:
        monkeypatch.setattr(contract, "OPTIONS", contract.OPTIONS | {"max_tokens": 7999})
    _, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 2 and info["model_requests"] == 1 and info["cached_batches"] == 0
    assert len(cache_files(tmp_path)) == 2


@pytest.mark.parametrize("corruption", ["missing_batch", "missing_row", "fabricated_quote", "wrong_shape", "expired", "future"])
def test_invalid_or_stale_cached_contract_is_recompiled_and_replaced(setup, tmp_path, corruption):
    project, shots, _ = setup
    calls = []

    def chat(system, content, **options):
        calls.append(1)
        return response(json.loads(content)), {"model": "contract-v1", "usage": {}}

    contract.extract(project, shots, chat=chat, check=lambda: None)
    path = cache_files(tmp_path)[0]
    saved = json.loads(path.read_text())
    if corruption == "missing_batch":
        saved["response"] = []
    elif corruption == "missing_row":
        saved["response"][0]["contracts"].pop()
    elif corruption == "fabricated_quote":
        saved["response"][0]["contracts"][0]["facts"][0][2] = "另一个人"
    elif corruption == "wrong_shape":
        saved["response"] = {"contracts": []}
    elif corruption == "expired":
        saved["created"] = time.time() - 8 * 86400
    else:
        saved["created"] = time.time() + 60
    path.write_text(json.dumps(saved))
    _, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 2 and info["cached_batches"] == 0 and info["model_requests"] == 1
    assert json.loads(path.read_text())["response"] == [response(calls_request(project, shots))]


def calls_request(project, shots):
    return contract.build_requests(project, shots)[0]


def test_partial_failure_never_saves_a_contract_response_set(setup, tmp_path):
    project, shots, _ = setup
    shots = [shots[0] | {"id": f"shot-{i}"} for i in range(41)]
    project = project | {"script": "\n".join(s["narration"] for s in shots)}
    count = 0

    def chat(system, content, **options):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("incomplete last batch")
        return response(json.loads(content)), {"usage": {"total_tokens": 100}}

    with pytest.raises(ValueError, match="incomplete last batch"):
        contract.extract(project, shots, chat=chat, check=lambda: None)
    assert count == 2 and cache_files(tmp_path) == []
    assert all("editorial_contract" not in s for s in shots)


def test_cancellation_after_response_never_saves_contract_cache(setup, tmp_path):
    project, shots, _ = setup
    returned = False

    def chat(system, content, **options):
        nonlocal returned
        returned = True
        return response(json.loads(content)), {"usage": {"total_tokens": 100}}

    def check():
        if returned:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        contract.extract(project, shots, chat=chat, check=check)
    assert cache_files(tmp_path) == []


def test_only_locally_corrected_and_complete_contract_responses_enter_cache(setup, tmp_path):
    project, shots, _ = setup
    calls = []

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(request)
        result = response(request)
        result["contracts"][1]["facts"][0][1] = "s2"
        result["contracts"][1]["refs"] = []
        return result, {"model": "contract-v1", "usage": {"total_tokens": 100}}

    first, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    second, reused = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 1 and info["model_requests"] == 1 and reused["model_requests"] == 0
    assert reused["cached_batches"] == 1 and second == first
    saved = json.loads(cache_files(tmp_path)[0].read_text())["response"]
    assert saved[0]["contracts"][1]["facts"][0] == ["person", "s1", "刘翔"]


def test_concurrent_identical_contracts_share_one_complete_fill(setup, tmp_path):
    project, shots, _ = setup
    barrier = Barrier(6)
    calls = []

    def chat(system, content, **options):
        calls.append(1)
        time.sleep(.03)
        return response(json.loads(content)), {"model": "contract-v1", "usage": {"total_tokens": 100}}

    def invoke(index):
        barrier.wait(timeout=3)
        own_shots = [s | {"id": f"caller-{index}-shot-{i}"} for i, s in enumerate(shots)]
        return contract.extract(project, own_shots, chat=chat, check=lambda: None)

    with ThreadPoolExecutor(max_workers=6) as pool:
        result = list(pool.map(invoke, range(6)))
    assert len(calls) == len(cache_files(tmp_path)) == 1
    assert sum(info["model_requests"] for _, info in result) == 1
    assert sum(info["cached_batches"] for _, info in result) == 5
    assert sum(info["usage"].get("total_tokens", 0) for _, info in result) == 100
    for index, (rows, _) in enumerate(result):
        assert rows[0]["id"] == f"caller-{index}-shot-0"
        assert contract.for_shot(rows[0], project)["facts"][0]["quote"] == "刘翔"


@pytest.mark.parametrize("reported_model", [None, "different-routed-model"])
def test_unknown_or_changed_response_model_cannot_fill_original_model_key(setup, tmp_path, reported_model):
    project, shots, _ = setup

    def chat(system, content, **options):
        return response(json.loads(content)), {"model": reported_model, "usage": {"total_tokens": 100}}

    _, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert info["model_requests"] == 1 and cache_files(tmp_path) == []


@pytest.mark.parametrize("field", ["requirements_model", "model_base_url"])
def test_route_change_during_request_cannot_publish_under_original_key(setup, tmp_path, field):
    project, shots, cfg = setup

    def chat(system, content, **options):
        original_model = cfg["requirements_model"]
        cfg[field] = "changed-route"
        return response(json.loads(content)), {"model": original_model, "usage": {}}

    _, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert info["model_requests"] == 1 and cache_files(tmp_path) == []


@pytest.mark.parametrize("field", ["requirements_model", "model_base_url"])
def test_route_change_while_waiting_does_not_read_previous_route_cache(setup, tmp_path, monkeypatch, field):
    from jinggao import execution

    project, shots, cfg = setup
    calls = []

    def chat(system, content, **options):
        calls.append(1)
        return response(json.loads(content)), {"model": cfg["requirements_model"], "usage": {}}

    contract.extract(project, shots, chat=chat, check=lambda: None)

    @contextmanager
    def change_route_before_lock(key, *, check=None):
        cfg[field] = "changed-route"
        yield

    monkeypatch.setattr(execution, "keyed_lock", change_route_before_lock)
    _, info = contract.extract(project, shots, chat=chat, check=lambda: None)
    assert len(calls) == 2 and info["cached_batches"] == 0
    assert len(cache_files(tmp_path)) == 1


def test_compact_planning_fields_restore_exact_metadata_and_keep_identity(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS.copy())
    narration = "刘翔参加雅典奥运会男子一百一十米栏决赛。"
    project = {"script": narration, "requirements": "必须看清刘翔跨栏动作。", "style": "真实资料"}
    units = [{"source_unit_id": "u0001", "narration": narration}]
    primary = {"subject": "刘翔", "action": "跨栏", "scene": "赛场", "query": "刘翔 雅典 跨栏",
               "search_queries": ["刘翔 跨栏 决赛"]}
    common = {"source_unit_id": "u0001", "persons": [{"name": "刘翔", "aliases": ["Liu Xiang"]}],
              "event": "雅典奥运会男子一百一十米栏决赛", "event_scope": "exact", "evidence_mode": "event",
              "preferred_media": "video", "location": "雅典", "era": "2004年"}
    rich = common | {"narration": narration, "visual": "赛场；刘翔；跨栏", "subject": "刘翔",
                     "query": primary["query"], "search_queries": [primary["query"], *primary["search_queries"]],
                     "exclusions": "", "overlay_label": "",
                     "visual_plan": {"mode": "single", "primary": primary, "alternatives": []}}
    compact = common | {"visual_plan": {"mode": "single", "primary": primary}}
    pending = [rich, compact]

    def chat(system, content, **options):
        assert "不能为节省输出而省掉姓名" in system
        request = json.loads(content)
        assert request["source_units"] == units and request["requirements"] == project["requirements"]
        return {"shots": [pending.pop(0)]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    args = (project, narration, False, narration, 1, 1, units)
    first, _ = planner._plan_model_batch(*args)
    second, _ = planner._plan_model_batch(*args)
    assert {k: v for k, v in first[0].items() if k != "id"} == {k: v for k, v in second[0].items() if k != "id"}
    assert second[0]["persons"] == common["persons"] and second[0]["event"] == common["event"]
    assert second[0]["event_scope"] == "exact" and second[0]["evidence_mode"] == "event"
    assert len(json.dumps(compact, ensure_ascii=False)) < len(json.dumps(rich, ensure_ascii=False))


def test_compact_planning_retains_complete_explicit_sequence_and_nondefault_fields(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS.copy())
    narration = "刘翔在雅典奥运会决赛中起跑，随后完成跨栏。"
    project = {"script": narration, "requirements": "画面完整展示起跑和跨栏的过程。", "style": "真实资料"}
    units = [{"source_unit_id": "u0001", "narration": narration}]
    row = {"source_unit_id": "u0001", "visual_plan": {"mode": "explicit_sequence"},
           "visual": "完整展示刘翔起跑及随后跨栏的连续过程。", "subject": "刘翔",
           "query": "刘翔 雅典 决赛", "search_queries": ["刘翔 雅典 决赛", "刘翔 起跑 跨栏"],
           "persons": [{"name": "刘翔", "aliases": ["Liu Xiang"]}],
           "event": "雅典奥运会男子一百一十米栏决赛", "event_scope": "exact", "evidence_mode": "event",
           "preferred_media": "video", "location": "雅典", "era": "2004年",
           "exclusions": "不能用其他运动员", "overlay_label": "本场比赛回顾"}

    def chat(system, content, **options):
        assert "explicit_sequence保留完整visual/query/search_queries和组合要求" in system
        return {"shots": [deepcopy(row)]}, {}

    monkeypatch.setattr(planner, "chat", chat)
    shots, _ = planner._plan_model_batch(project, narration, False, narration, 1, 1, units)
    assert all(shots[0][key] == value for key, value in row.items())


def test_complete_planning_resume_reuses_both_stages_without_losing_named_event(setup, monkeypatch):
    project, _, cfg = setup
    project = project | {"style": "真实资料"}
    calls = []
    event = "雅典奥运会男子一百一十米栏决赛"

    def chat(system, content, **options):
        request = json.loads(content)
        calls.append(options["task"])
        if options["task"] == "planning":
            data = {"shots": [{"source_unit_id": u["source_unit_id"],
                              "persons": [{"name": "刘翔", "aliases": []}], "event": event,
                              "evidence_mode": "event", "event_scope": "exact", "preferred_media": "video"}
                             for u in request["source_units"]]}
            return data, {"model": cfg["planning_model"], "usage": {"total_tokens": 100}}
        data = response(request)
        for row in data["contracts"]:
            row["use"] = "event_record"
            row["facts"].append(["event", "s1", event])
        return data, {"model": cfg["requirements_model"], "usage": {"total_tokens": 50}}

    monkeypatch.setattr(planner, "chat", chat)
    first, cold = planner.plan(project, "model", False)
    second, warm = planner.plan(project, "model", False)
    assert calls == ["planning", "requirements"]
    assert cold["planning_model_requests"] == 1 and cold["plan_consistency"]["model_requests"] == 1
    assert warm["planning_model_requests"] == warm["plan_consistency"]["model_requests"] == 0
    assert warm["planning_cached_batches"] == warm["plan_consistency"]["cached_batches"] == 1
    assert cold["usage"] == {"total_tokens": 150} and warm["usage"] == {}
    assert first[0]["id"] != second[0]["id"]
    for shot in second:
        assert shot["persons"] == [{"name": "刘翔", "aliases": []}] and shot["event"] == event
        assert shot["event_scope"] == "exact" and shot["evidence_mode"] == "event"
        assert contract.for_shot(shot, project) and not shot["approved"]
    # A corrupt local result may enter the format-repair path without having
    # sent a request. Count the single actual replacement, not both attempts.
    file = next((paths.DATA / "cache" / "planning").glob("*.json"))
    damaged = json.loads(file.read_text())
    damaged["response"]["shots"] = []
    file.write_text(json.dumps(damaged))
    cfg["plan_retries"] = "1"
    _, repaired = planner.plan(project, "model", False)
    assert calls == ["planning", "requirements", "planning"]
    assert repaired["planning_model_requests"] == 1 and repaired["format_retries"] == 1
