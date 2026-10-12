from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from jinggao import media, plan_consistency, planner
from planner_helpers import consistency_response


def plan(count=33):
    script = "\n".join(f"工作人员搬运第{i+1}件包裹。" for i in range(count))
    shots = [planner.new_shot(line, audio={"path": "paid-audio.wav"}) for line in script.splitlines()]
    for shot in shots:
        shot.update(audio={"path": "paid-audio.wav"}, selected_asset="saved-asset")
    return {"script": script, "requirements": "保持原文和明确物件", "shots": deepcopy(shots)}, shots


def manifest(folder):
    return json.loads(next(folder.glob("plan-consistency-*.json")).read_text())


def test_batches_share_full_readonly_context_finish_out_of_order_and_apply_once(tmp_path, monkeypatch):
    project, shots = plan()
    original = deepcopy((project, shots))
    barrier, second_done, lock = Barrier(2, timeout=5), Event(), Lock()
    requests, finishes, applications, levels = [], [], [], []
    active = 0

    def chat(system, content, **kwargs):
        nonlocal active
        request = json.loads(content)
        with lock:
            active += 1
            levels.append(active)
            requests.append(request)
        assert "只复核shots中的至多4镜" in system
        assert request["full_script"] == project["script"] and request["requirements"] == project["requirements"]
        assert len(request["readonly_global_plan"]) == 33 and 1 <= len(request["shots"]) <= 4
        for summary, shot in zip(request["readonly_global_plan"], shots, strict=True):
            span = summary["source_span"]
            assert request["full_script"][span["start"]:span["end"]] == shot["narration"]
            assert "narration" not in summary and "visual" not in summary
        assert "paid-audio.wav" not in content and "saved-asset" not in content
        assert kwargs == {"reasoning": True, "reasoning_effort": "low", "max_tokens": 10000, "timeout": 180,
                          "task": "requirements"}
        index = request["batch_index"]
        if index <= 2:
            barrier.wait()
            if index == 1:
                assert second_done.wait(5)
        with lock:
            finishes.append(index)
            active -= 1
        if index == 2:
            second_done.set()
        return consistency_response(request), {"model": "test", "usage": {"total_tokens": index},
                                               "headers": {"Authorization": "secret-value"}}

    apply = plan_consistency.apply_reviews

    def atomic_apply(p, original_shots, response, **kwargs):
        applications.append([r["shot_id"] for r in response["reviews"]])
        files = list(tmp_path.rglob("batch-*.json"))
        assert len(files) == 9 and all(json.loads(f.read_text())["status"] == "validated" for f in files)
        assert (project, shots) == original
        return apply(p, original_shots, response, **kwargs)

    monkeypatch.setattr(plan_consistency, "apply_reviews", atomic_apply)
    result, info = plan_consistency.review(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    expected = [s["id"] for s in shots]
    assert applications == [expected] and [s["id"] for s in result] == expected
    assert len(requests) == 9 and max(levels) == 2 and finishes.index(2) < finishes.index(1)
    ordered = sorted(requests, key=lambda r: r["batch_index"])
    assert [s["shot_id"] for r in ordered for s in r["shots"]] == expected
    assert all(r["readonly_global_plan"] == ordered[0]["readonly_global_plan"] for r in ordered)
    assert info["batch_count"] == info["model_requests"] == 9 and info["usage"] == {"total_tokens": 45}
    assert info["usage_unavailable_batches"] == [] and (project, shots) == original
    saved = manifest(tmp_path)
    assert saved["status"] == "applied" and len(saved["response"]["reviews"]) == 33
    assert all("secret-value" not in f.read_text() for f in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("failed_usage", [{"total_tokens": 41}, {}])
def test_failed_response_keeps_diagnostic_usage_and_other_paid_success_stops_new_calls(
        tmp_path, monkeypatch, failed_usage):
    project, shots = plan()
    original = deepcopy((project, shots))
    barrier, failure_saved = Barrier(2, timeout=5), Event()
    calls, successful_response = [], {}
    model_diagnostic = tmp_path / "bad-model-output.json"
    model_diagnostic.write_text('{"content":"{broken","finish_reason":"unknown"}')
    write = plan_consistency._write_diagnostic

    def persist(path, value):
        write(path, value)
        if value.get("batch_index") == 2 and value.get("status") == "failed":
            failure_saved.set()

    def chat(_system, content, **_kwargs):
        request = json.loads(content)
        index = request["batch_index"]
        calls.append(index)
        assert index <= 2
        barrier.wait()
        if index == 2:
            raise planner.ModelResponseError("模型返回无效JSON", diagnostic_path=model_diagnostic,
                model_info={"model": "test", "usage": failed_usage, "api_key": "must-not-leak"})
        assert failure_saved.wait(5)
        response = consistency_response(request)
        successful_response.update(response)
        return response, {"model": "test", "usage": {"total_tokens": 7}}

    monkeypatch.setattr(plan_consistency, "_write_diagnostic", persist)
    monkeypatch.setattr(plan_consistency, "apply_reviews", lambda *_a, **_kw: pytest.fail("partial apply"))
    with pytest.raises(ValueError, match="第2/9批.*响应诊断"):
        plan_consistency.review(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    saved = manifest(tmp_path)
    assert sorted(calls) == [1, 2] and saved["model_requests"] == 2 and (project, shots) == original
    assert saved["status"] == "failed" and saved["review_model_info"]["usage"] == {"total_tokens": 7 + failed_usage.get("total_tokens", 0)}
    assert saved["usage_unavailable_batches"] == ([] if failed_usage else [2])
    assert [b["status"] for b in saved["batches"]] == ["validated", "failed"] + ["not_requested"] * 7
    first, second = [json.loads(Path(b["diagnostic_path"]).read_text()) for b in saved["batches"][:2]]
    assert first["response"] == successful_response
    assert second["model_response_diagnostic_path"] == str(model_diagnostic)
    assert second["review_model_info"] == {"model": "test", "usage": failed_usage}
    assert all("must-not-leak" not in f.read_text() for f in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("corruption", ["other-batch", "duplicate", "missing", "reordered", "unknown-field", "fake-quote"])
def test_batch_reply_cannot_mutate_readonly_shots_or_silently_repair_ownership(tmp_path, monkeypatch, corruption):
    project, shots = plan(17)
    original, calls = deepcopy((project, shots)), []
    failure_saved = Event()
    write = plan_consistency._write_diagnostic

    def persist(path, value):
        write(path, value)
        if value.get("batch_index") == 1 and value.get("status") == "failed":
            failure_saved.set()

    monkeypatch.setattr(plan_consistency, "_write_diagnostic", persist)

    def chat(_system, content, **_kwargs):
        request = json.loads(content)
        calls.append(request["batch_index"])
        if request["batch_index"] == 2:
            # Control which response becomes known first. If batch 2 completed
            # before batch 1's validation, starting batch 3 would be legitimate.
            assert failure_saved.wait(5)
        response = consistency_response(request)
        if request["batch_index"] == 1:
            rows = response["reviews"]
            if corruption == "other-batch":
                rows[0]["shot_id"] = shots[4]["id"]
            elif corruption == "duplicate":
                rows[1] = deepcopy(rows[0])
            elif corruption == "missing":
                rows.pop()
            elif corruption == "reordered":
                rows.reverse()
            elif corruption == "unknown-field":
                rows[0]["patch"]["narration"] = "不能改原文"
            else:
                rows[0]["evidence"][0]["quote"] = "没有出现在原文中的假引用。"
        return response, {}

    monkeypatch.setattr(plan_consistency, "apply_reviews", lambda *_a, **_kw: pytest.fail("partial apply"))
    with pytest.raises(ValueError, match="第1/5批"):
        plan_consistency.review(project, shots, chat=chat, check=lambda: None, diagnostic_dir=tmp_path)
    assert set(calls) <= {1, 2} and len(calls) <= 2 and (project, shots) == original
    assert manifest(tmp_path)["status"] == "failed"


def test_cancellation_after_paid_responses_retains_both_and_never_applies(tmp_path, monkeypatch):
    project, shots = plan(17)
    original, cancelled, barrier = deepcopy((project, shots)), Event(), Barrier(2, timeout=5)

    def chat(_system, content, **_kwargs):
        request = json.loads(content)
        barrier.wait()
        if request["batch_index"] == 2:
            cancelled.set()
        else:
            assert cancelled.wait(5)
        return consistency_response(request), {"model": "test", "usage": {"total_tokens": 5}}

    def check():
        if cancelled.is_set():
            raise media.Cancelled()

    monkeypatch.setattr(plan_consistency, "apply_reviews", lambda *_a, **_kw: pytest.fail("cancelled apply"))
    with pytest.raises(media.Cancelled):
        plan_consistency.review(project, shots, chat=chat, check=check, diagnostic_dir=tmp_path)
    saved = manifest(tmp_path)
    assert saved["model_requests"] == 2 and saved["review_model_info"]["usage"] == {"total_tokens": 10}
    assert all("response" in json.loads(Path(b["diagnostic_path"]).read_text()) for b in saved["batches"][:2])
    assert (project, shots) == original and saved["status"] == "failed"


def test_round6_saved_rows_can_be_partitioned_offline_with_same_atomic_result(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "round6_plan_consistency_quotes.json").read_text())
    request = fixture["request"]
    project = {"script": request["full_script"], "requirements": request["requirements"]}
    shots = [{k: v for k, v in s.items() if k != "shot_id"} | {"id": s["shot_id"]} for s in request["shots"]]
    replies = {r["shot_id"]: r for r in fixture["response"]["reviews"]}
    calls = []

    def offline_response(_system, content, **_kwargs):
        current = json.loads(content)
        calls.append(current["batch_index"])
        return {"requirements_preserved": True,
                "reviews": [deepcopy(replies[s["shot_id"]]) for s in current["shots"]]}, {}

    expected = plan_consistency.apply_reviews(project, shots, fixture["response"], require_visual_plan=True)
    reviewed, info = plan_consistency.review(project, shots, chat=offline_response, check=lambda: None,
                                            diagnostic_dir=tmp_path)
    assert reviewed == expected and sorted(calls) == list(range(1, 10))
    assert info["evidence_normalized_count"] == 11 and info["usage_unavailable_batches"] == list(range(1, 10))


@pytest.mark.parametrize("rewrite", [False, True])
def test_compact_global_context_keeps_source_and_fact_boundaries_for_both_modes(tmp_path, rewrite):
    source = "张明在南门场馆使用甲型设备处理包裹。\n列车驶过车站。"
    project = {"script": source, "requirements": "必须保留人物、地点和甲型设备。"}
    shots = [planner.new_shot(line) for line in source.splitlines()]
    facts = {"persons": [{"name": "张明", "aliases": []}], "event_persons": [{"name": "张明", "aliases": []}],
             "event": source.splitlines()[0], "location": "南门场馆", "event_scope": "exact", "evidence_mode": "event"}
    shots[0].update(facts)
    if rewrite:
        shots[0]["narration"] = "在南门场馆，张明使用甲型设备处理包裹。"
    original = deepcopy((project, shots))

    def chat(_system, content, **_kwargs):
        request = json.loads(content)
        assert request["full_script"] == source and request["requirements"] == project["requirements"]
        context = request["readonly_global_plan"][0]
        assert all(context[k] == v for k, v in facts.items())
        assert "visual" not in context and "visual_plan" not in context
        if rewrite:
            assert context["narration"] == shots[0]["narration"] and "source_span" not in context
        else:
            span = context["source_span"]
            assert source[span["start"]:span["end"]] == shots[0]["narration"] and "narration" not in context
        return consistency_response(request), {}

    result, _ = plan_consistency.review(project, shots, chat=chat, check=lambda: None,
                                        rewrite=rewrite, diagnostic_dir=tmp_path)
    assert all(result[0][k] == v for k, v in facts.items()) and (project, shots) == original
