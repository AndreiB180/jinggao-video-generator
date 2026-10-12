"""Lossless requests and paid-response reuse must retain strict evidence gates."""
from copy import deepcopy
import hashlib
import json
from threading import Barrier
import time

import httpx
import pytest

from jinggao import config, costs, execution, network, paths, planner, review_evidence, review_packet


def records():
    script = "李明在2024年参加比赛，不能使用2019年的画面。标题提及他不等于他是当前运动员。" * 8
    result = []
    for sid, action in [("abcdef0123456789abcdef0123456789", "挥拍击球"),
                        ("fedcba9876543210fedcba9876543210", "坐在椅子上休息")]:
        frames = [{"frame_index": 0, "objects": "场边的运动员", "action": action,
                   "text": ["当前选手：李明", "2019年比赛资料"]}]
        refs, _ = review_evidence.references(sid, frames)
        result.append({"shot_id": sid, "source_contract": {"full_script": script}, "requirements": script,
                       "narration": "李明在2024年参加比赛。", "selected_visual_requirements": {"action": "击球"},
                       "frame_observations": frames, "observation_references": refs,
                       "evidence_binding": {"frame_sha256": [hashlib.sha256(sid.encode()).hexdigest()],
                                            "clip_start": 3, "duration": 5}})
    return result


def packet(rows=None):
    return review_packet.Packet([{"type": "text", "text": json.dumps(r, ensure_ascii=False)}
                                 for r in (rows if rows is not None else records())])


def answer():
    return {"reviews": [{"shot_id": f"r{i}", "status": "conflict", "reason": "年份不符", "labels": [],
                         "evidence": [{"ref": f"r{i}:f0:text:1"}], "primary_checks": [
                             {"key": "action", "status": "unknown", "reason": "此文字不能证明动作",
                              "evidence": []}]} for i in range(2)]}


def strict_answer(result, *, typed=False):
    """Mock the provider's requested wire format, not its semantic judgment."""
    wire = {"reviews": {row["shot_id"]: {key: deepcopy(value) for key, value in row.items()
            if key not in {"shot_id", "primary_checks"}} | {"primary_checks": {
                check["key"]: {key: deepcopy(value) for key, value in check.items() if key != "key"}
                for check in row["primary_checks"]}}
            for row in result["reviews"]}}
    if typed:
        for row in wire["reviews"].values():
            row["source_use"] = "核对本句指定比赛的实际动作。"
            for check in row["primary_checks"].values():
                check["reason"] = {"observed": "此合成测试未确认所需动作。",
                    "relation": "unclear", "why": check["reason"]}
    return wire


def test_packet_losslessly_preserves_complete_source_facts_and_scoped_ids():
    source = records()
    original = deepcopy(source)
    packed = packet(source)
    assert packed.unpack() == original == source
    assert packed.stats["shared_texts"] == 1
    assert packed.stats["packed_bytes"] < packed.stats["original_bytes"] * .6
    restored = packed.restore(answer())
    assert restored["reviews"][0]["shot_id"] == source[0]["shot_id"]
    assert restored["reviews"][1]["evidence"][0]["ref"].startswith(source[1]["shot_id"]+":")
    assert "2019" in packed.body["texts"]["T0"] and "不能" in packed.body["texts"]["T0"]


def test_repeated_short_chinese_observations_are_shared_without_losing_negations():
    source = records()
    observation = "工人在纸箱旁站着等待，没有搬运包裹或查看标签。"
    assert len(observation) < 80
    for row in source:
        row["frame_observations"][0]["action"] = observation
        row["scene_description"] = observation
    packed = packet(source)
    assert packed.unpack() == source
    assert observation in packed.body["texts"].values()
    assert packed.stats["packed_bytes"] < packed.stats["original_bytes"]
    assert "supported" not in packed.body["texts"].values()


@pytest.mark.parametrize("corruption", ["cross_shot", "missing_check", "missing_shot", "invented_ref", "changed_quote"])
def test_cache_rejects_ungrounded_or_incomplete_response(corruption, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    raw = answer()
    row = raw["reviews"][0]
    if corruption == "cross_shot":
        row["evidence"] = [{"ref": "r1:f0:text:1"}]
    elif corruption == "missing_check":
        row["primary_checks"] = []
    elif corruption == "missing_shot":
        raw["reviews"].pop()
    elif corruption == "invented_ref":
        row["evidence"] = [{"ref": "r0:f8:action"}]
    else:
        row["evidence"] = [{"ref": "r0:f0:text:1", "quote": "2024年比赛"}]
    cache = review_packet.Cache(packet(), {"test": True}, "https://example.invalid")
    cache.save(raw)
    assert cache.read() is None and not cache.path.exists()


@pytest.mark.parametrize("where", ["review", "check"])
@pytest.mark.parametrize("invalid_status", [[], {}])
def test_malformed_provider_status_does_not_crash_cache_validation(where, invalid_status, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    raw = answer()
    row = raw["reviews"][0]
    (row if where == "review" else row["primary_checks"][0])["status"] = invalid_status
    cache = review_packet.Cache(packet(), {}, "endpoint")
    cache.save(raw)
    assert not cache.path.exists() and cache.read() is None


def test_cache_keys_bind_source_pixels_timing_model_and_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    def key(rows, model="model", protocol="all facts"):
        p = packet(rows)
        return review_packet.Cache(p, {"model": model, "messages": [protocol, p.text]}, "endpoint").path
    rows = records()
    original = key(rows)
    changed = deepcopy(rows)
    changed[0]["evidence_binding"]["frame_sha256"] = ["changed pixels"]
    assert key(changed) != original
    changed = deepcopy(rows)
    changed[0]["evidence_binding"]["clip_start"] += 1
    assert key(changed) != original
    changed = deepcopy(rows)
    changed[0]["source_contract"]["full_script"] += "附加要求"
    assert key(changed) != original
    assert key(rows, model="different") != original
    assert key(rows, protocol="changed rule") != original


def test_missing_pixel_binding_cannot_become_a_reusable_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    rows = records()
    rows[0]["evidence_binding"]["frame_sha256"] = []
    cache = review_packet.Cache(packet(rows), {}, "endpoint")
    cache.save(answer())
    assert not cache.path.exists() and cache.read() is None


@pytest.mark.parametrize("parallel", [1, 6])
def test_chat_reuses_raw_judgment_without_a_second_paid_request(tmp_path, monkeypatch, parallel):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"model_api_key": "test-key", "model_name": "qwen3.7-flash",
                             "review_model": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                             "_strict_text_review_trial": True}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    sent = []
    def respond(url, **kwargs):
        sent.append(kwargs["json"])
        time.sleep(.02)  # Other callers reach the same in-flight cache fill.
        assert json.loads(sent[-1]["messages"][1]["content"])["schema"] == "review_records_v1"
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(strict_answer(answer(), typed=True))}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}})
    monkeypatch.setattr(network, "post", respond)
    content = [{"type": "text", "text": json.dumps(r, ensure_ascii=False)} for r in records()]
    start = Barrier(parallel)
    def invoke():
        start.wait(timeout=3)
        return planner.chat("review", content, task="evidence_review", max_tokens=1000)
    with costs.scope("cache-test", "job"):
        with execution.ThreadPoolExecutor(max_workers=parallel) as pool:
            replies = list(pool.map(lambda _: invoke(), range(parallel)))
        first, info = next((r, info) for r, info in replies if not info.get("cache_hit"))
        assert all(r == first for r, _ in replies)
        assert sum(bool(info.get("cache_hit")) for _, info in replies) == parallel-1
        # A cache read remains free even when the user pauses further spending.
        cfg["model_project_budget_cny"] = "0"
        second, reused = planner.chat("review", content, task="evidence_review", max_tokens=1000)
    assert first == second and len(sent) == 1
    assert reused["cache_hit"] and not info.get("cache_hit")
    assert costs.summary("cache-test")["project"]["requests"] == 1
    assert first["reviews"][0]["status"] == "conflict"


@pytest.mark.usefixtures("isolated_pixel_quality")
def test_audit_checks_cached_response_again_before_approving(monkeypatch):
    from jinggao import neutral_observer
    shot = planner.new_shot(narration="工人正在搬运纸箱。", query="搬运纸箱")
    shot["visual_plan"] = {"mode": "single", "primary": {"action": "搬运纸箱"}}
    frames = [{"frame_index": 0, "objects": "人和纸箱", "action": "站着等待", "text": []}]
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "站着等待", "frames": frames}}, 0))
    # Simulates a cached raw response with a contradictory required check.
    # It is not an approval and must still fail the normal downstream gate.
    def cached(system, content, **kwargs):
        return {"reviews": [{"shot_id": shot["id"], "status": "supported", "reason": "行业相关",
            "labels": [], "evidence": [{"ref": shot["id"]+":f0:objects"}], "primary_checks": [
                {"key": "action", "status": "conflict", "reason": "没有搬运", "evidence": [
                    {"ref": shot["id"]+":f0:action"}]}]}]}, {"cache_hit": True}
    monkeypatch.setattr(planner, "chat", cached)
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"}, "frame_paths": []}],
                                          lambda: None)[shot["id"]]
    assert result["status"] != "supported" and result["request_count"] == 0
