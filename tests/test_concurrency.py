import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from jinggao import (baidu_web, config, editorial_contract, execution, identity, media, paths, planner,
                     sources, store, video_sources, worker)
from planner_helpers import source_contract_response


def test_lookahead_is_bounded_but_preserves_order_and_none():
    barrier = threading.Barrier(3)
    calls = []
    lock = threading.Lock()

    def prepare(item):
        with lock:
            calls.append(item)
            position = len(calls)
        if position <= 3:
            barrier.wait(timeout=3)
        return item

    results = execution.ordered_map(prepare, [None, 1, 2, 3, 4, 5], workers=3)
    assert next(results) is None
    # Three initial jobs plus at most one replenished item; no eager full queue.
    assert len(calls) <= 4
    assert list(results) == [1, 2, 3, 4, 5]


def test_parallel_plan_keeps_source_order_and_retries_only_invalid_batch(monkeypatch):
    monkeypatch.setattr(config, "read", lambda: {"search_provider": "mixed", "plan_workers": "3", "plan_retries": "1"})
    # Enough source text for all three configured planner workers to overlap.
    project = {"script": "\n".join(f"第{i}段。" + "保持原文的具体人物和动作。"*3 for i in range(60)),
               "style": "真实资料", "requirements": "", "shots": []}
    original = deepcopy(project)
    calls, contracts, lock = {}, [], threading.Lock()
    first_wave = threading.Barrier(3)
    active, peak = 0, 0

    def chat(system, content, **kwargs):
        nonlocal active, peak
        request = json.loads(content)
        if request.get("stage") == editorial_contract.VERSION:
            assert active == 0
            contracts.append(request)
            return source_contract_response(request), {"model": "test"}
        index = request["batch_index"]
        with lock:
            calls[index] = calls.get(index, 0)+1
            attempt = calls[index]
            active += 1
            peak = max(peak, active)
        try:
            if index <= 3 and attempt == 1:
                first_wave.wait(timeout=3)
            if index == 1:
                time.sleep(.04)
            rows = [{"source_unit_id": unit["source_unit_id"]} for unit in request["source_units"]]
            if index == 2 and attempt == 1:
                rows[0]["event_scope"] = "not-a-valid-scope"
            return {"shots": rows}, {"model": "test"}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(planner, "chat", chat)
    shots, info = planner.plan(project, "model", False)
    assert planner.normalized("".join(s["narration"] for s in shots)) == planner.normalized(project["script"])
    assert peak == 3
    assert [s["source_unit_id"] for s in shots] == [
        u["source_unit_id"] for batch in planner.source_unit_batches(project["script"]) for u in batch]
    assert calls[2] == 2 and all(n == 1 for i, n in calls.items() if i != 2)
    assert info["format_retries"] == 1 and project == original and project["shots"] == []
    assert len(contracts) == (len(shots) + editorial_contract.BATCH_SIZE - 1) // editorial_contract.BATCH_SIZE
    assert all(1 <= len(r["targets"]) <= editorial_contract.BATCH_SIZE for r in contracts)
    assert all("".join(u["text"] for u in r["script_units"]) == project["script"] for r in contracts)
    assert [ref for r in contracts for ref in r["targets"]] == [f"s{i}" for i in range(1, len(shots) + 1)]
    assert [editorial_contract.for_shot(s, project)["shot_id"] for s in shots] == [s["id"] for s in shots]


def test_same_video_search_only_fetches_once_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        time.sleep(.03)
        return [{"id": "result"}]

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: video_sources._cached("test", "same query", fetch), range(4)))
    assert len(calls) == 1 and all(row == [{"id": "result"}] for row in rows)
    assert not list(tmp_path.rglob("*.tmp"))


def test_shared_inspection_never_transfers_other_shots_preview_or_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = []

    def retrieve(candidate, check):
        calls.append(1)
        return candidate | {"media_streams": [{"url": "https://media.example/stream", "width": 1280, "height": 720}],
                            "source_evidence": {"readable": True}, "duration": 60}

    monkeypatch.setattr(video_sources, "_generic", retrieve)
    candidate = {"id": "one", "title": "Interview", "source_url": "https://example.com/interview",
                 "preview": {"verification": {"verdict": "match"}}, "min_video_height": 720}
    first = video_sources.inspect_video_candidate(candidate)
    second = video_sources.inspect_video_candidate({k:v for k,v in candidate.items() if k != "preview"} |
                                                   {"id": "two", "min_video_height": 1080})
    assert len(calls) == 1 and first["download_status"] == "available"
    assert second["download_status"] == "preview_only" and second["id"] == "two" and "preview" not in second


def test_round_aliases_do_not_create_conflict_but_different_round_still_does():
    assert identity.event_rounds("男單1/16決賽") == identity.event_rounds("轮次：男单32强") == {("size",32)}
    same = identity.event_checks("赛事：2026WTT中国大满贯；轮次：R32", "2026WTT中國大滿貫男單1/16決賽", [])
    assert all(q["status"] == "supported" for q in same)
    wrong = identity.event_checks("轮次：R32", "男單1/32決賽", [])
    assert wrong[0]["status"] == "conflict"


def test_quota_error_is_explicit_and_never_retried_as_page_loading(monkeypatch, tmp_path):
    message = baidu_web.response_error({"errno": 1000000, "errmsg": "您今日的调用次数已达上限，请明天再试"})
    assert "上限" in message
    calls = []
    def generate(*args):
        calls.append(1)
        raise ValueError("一镜网页提示："+message)
    monkeypatch.setattr(baidu_web, "_generate", generate)
    with pytest.raises(ValueError, match="上限"):
        baidu_web.generate("测试", tmp_path/"audio.wav", lambda: None)
    assert len(calls) == 1


def test_only_pre_submission_controls_are_retried(monkeypatch, tmp_path):
    calls = []
    def generate(*args):
        calls.append(1)
        if len(calls) == 1:
            raise baidu_web.PageControlTimeout("loading")
    monkeypatch.setattr(baidu_web, "_generate", generate)
    baidu_web.generate("测试", tmp_path/"audio.wav", lambda: None)
    assert len(calls) == 2


def test_concurrent_reservations_never_claim_overlapping_source_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("interval test")
    shots = [planner.new_shot("Test footage") for _ in range(3)]
    for shot in shots:
        shot["duration"] = 5
    store.mutate(p["id"], lambda body: body.update(shots=shots))
    session = worker.ClipReservations(p["id"])
    barrier = threading.Barrier(3)
    def claim(shot):
        starts = session.claim(shot, {"id": "source", "source_url": "https://example.org/film", "duration": 90})
        barrier.wait(timeout=3)
        return starts
    with ThreadPoolExecutor(max_workers=3) as pool:
        ranges = list(pool.map(claim, shots))
    assert all(ranges)
    flat = sorted(start for group in ranges for start in group)
    assert all(b >= a+5 for a,b in zip(flat, flat[1:], strict=False))
    for shot in shots:
        session.release(shot["id"])
    assert not session.pending


def test_parallel_image_selection_rechecks_budget_transactionally(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"model_api_key": "test", "match_workers": "2", "pipeline": "classic"})
    store.init()
    p = store.new_project("budget test", min_video_ratio=.5)
    shots = [planner.new_shot("Test footage") for _ in range(2)]
    for shot in shots:
        shot["duration"] = 10
    store.mutate(p["id"], lambda body: body.update(shots=shots))
    barrier = threading.Barrier(2)
    monkeypatch.setattr(sources, "search_shot", lambda *a: ([{"id": "image", "kind": "image", "title": "Photo"}], "found"))
    monkeypatch.setattr(planner, "rank_pool", lambda *a: [{"candidate_id": "image"}])
    def materialize(pid, candidate, check, **kwargs):
        asset = {"id": store.uid(), "kind": "image", "width": 1920, "height": 1080, "title": "Photo"}
        store.mutate(pid, lambda body: body["assets"].append(asset), internal=True)
        return asset
    monkeypatch.setattr(worker, "materialize", materialize)
    def verify(*args, **kwargs):
        barrier.wait(timeout=3)
        return {"verdict": "match", "reason": "Matches generic conditions", "selection_blocked": False}
    monkeypatch.setattr(planner, "verify", verify)
    store.enqueue(p["id"], "match")
    with pytest.raises(ValueError, match="图片预算"):
        worker.process(store.claim())
    result = store.get_project(p["id"])
    assert sum(bool(s["selected_asset"]) for s in result["shots"]) == 1


def test_full_voice_is_one_call_preserves_script_and_waits_for_acoustic_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"tts_provider": "baidu_web"})
    store.init()
    p=store.new_project("整段配音", "第一句完整的原稿。\n第二句同样保留。")
    shots=planner.rule_shots(p["script"])
    for s in shots:
        s.update(duration=3,approved=True,verification={"verdict":"match"})
    store.mutate(p["id"],lambda body:body.update(shots=shots))
    calls=[]
    monkeypatch.setattr(media,"synthesize",lambda text,*args:calls.append(text))
    monkeypatch.setattr(media,"probe",lambda *args:{"format":{"duration":"11.123"}})
    store.enqueue(p["id"],"voice_full")
    worker.process(store.claim())
    result=store.get_project(p["id"])
    assert calls==[p["script"]]
    assert result["full_audio"]["alignment"]=="pending"
    assert [s["duration"] for s in result["shots"]] == [3]*len(shots)
    assert all(not s["approved"] and s["verification"] is None for s in result["shots"])


def test_full_voice_rejects_locked_or_existing_audio_before_calling_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(paths,"DATA",tmp_path)
    store.init()
    p=store.new_project("现有配音","保留已完成的逐镜音轨。")
    shot=planner.new_shot(p["script"])
    shot["audio"]={"path":"audio/original.wav","duration":5}
    store.mutate(p["id"],lambda body:body.update(shots=[shot]))
    monkeypatch.setattr(media,"synthesize",lambda *a:pytest.fail("must not consume a call"))
    store.enqueue(p["id"],"voice_full")
    with pytest.raises(ValueError,match="确认替换"):
        worker.process(store.claim())
