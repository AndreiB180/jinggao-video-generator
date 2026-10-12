from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import time

import pytest

from jinggao import config, library, library_match, paths, planner, rank_cache


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    cfg = config.read() | {"model_name": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    file = library.file_path("blobs/clip.mp4")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"exact source pixels")
    clip = {"id": "clip-a", "path": "blobs/clip.mp4", "blob_id": hashlib.sha256(file.read_bytes()).hexdigest(),
            "start": 10, "offset": 0, "duration": 8, "observation_method": "blind_actual_frames_v1",
            "description": "工人在仓库分拣包裹", "visible_text": "分拣区域", "source": {
                "title": "实际作业", "source_url": "https://v.qq.com/x/page/work.html"}}
    shots = [{"id": "shot-a", "narration": "工人在分拣包裹", "duration": 6, "persons": [], "event_scope": "background"}]
    return shots, {"shot-a": [clip]}, cfg


def install_chat(monkeypatch, *, empty=False, invalid=False):
    calls = []
    def chat(system, content, **options):
        data = json.loads(content)
        calls.append(data)
        time.sleep(.025)
        return {"rankings": [{"shot_id": r["shot_id"], "candidate_refs": [] if empty else [
            "wrong-ref" if invalid else r["candidates"][0]["candidate_ref"]]} for r in data["shots"]]}, {"model": config.read()["model_name"]}
    monkeypatch.setattr(planner, "chat", chat)
    return calls


@pytest.mark.parametrize("empty", [False, True])
def test_complete_rankings_and_empty_shortlists_replay_without_paid_call(inputs, monkeypatch, empty):
    shots, candidates, _ = inputs
    calls = install_chat(monkeypatch, empty=empty)
    starts, hits = [], []
    first = deepcopy(candidates)
    library_match.rank_windows(shots, first, lambda: None, request_started=lambda: starts.append(1))
    second = deepcopy(candidates)
    second["shot-a"][0]["id"] = "new-clip-alias"
    library_match.rank_windows(shots, second, lambda: None, request_started=lambda: starts.append(1),
                               cache_hit=lambda: hits.append(1))
    assert len(calls) == len(starts) == len(hits) == 1
    assert second["shot-a"] == ([] if empty else [candidates["shot-a"][0] | {"id": "new-clip-alias"}])
    assert all("approved" not in r for rows in second.values() for r in rows)


def test_concurrent_fill_has_one_remote_request(inputs, monkeypatch):
    shots, candidates, _ = inputs
    calls = install_chat(monkeypatch)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: library_match.rank_windows(shots, deepcopy(candidates), lambda: None), range(6)))
    assert all(results) and len(calls) == 1


def test_invalid_rankings_are_not_negative_cached(inputs, monkeypatch):
    shots, candidates, _ = inputs
    calls = install_chat(monkeypatch, invalid=True)
    for _ in range(2):
        output = deepcopy(candidates)
        library_match.rank_windows(shots, output, lambda: None)
        assert output == {"shot-a": []}
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["narration", "persons", "source", "start", "duration", "description",
                                  "visible_text", "observation_method", "pixels", "model", "endpoint", "protocol"])
def test_changed_ranking_inputs_do_not_replay(inputs, monkeypatch, change):
    shots, candidates, cfg = inputs
    calls = install_chat(monkeypatch)
    library_match.rank_windows(shots, deepcopy(candidates), lambda: None)
    if change in ("narration", "persons"):
        shots[0][change] = "另一个条件" if change == "narration" else ["指定人物"]
    elif change == "source":
        candidates["shot-a"][0]["source"]["source_url"] = "https://v.qq.com/x/page/another.html"
    elif change in ("start", "duration"):
        candidates["shot-a"][0][change] += 1
    elif change in ("description", "visible_text", "observation_method"):
        candidates["shot-a"][0][change] = "changed observation"
    elif change == "pixels":
        library.file_path("blobs/clip.mp4").write_bytes(b"changed pixels")
    elif change == "model":
        cfg["model_name"] = "new-model"
    elif change == "endpoint":
        cfg["model_base_url"] = "https://other.test/v1"
    elif change == "protocol":
        monkeypatch.setattr(library_match, "_RANK_PROTOCOL", library_match._RANK_PROTOCOL + " new policy")
    library_match.rank_windows(shots, deepcopy(candidates), lambda: None)
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["expired", "malformed", "foreign_ref"])
def test_corrupt_stale_or_invalid_cached_response_is_rejected(inputs, monkeypatch, damage):
    shots, candidates, _ = inputs
    calls = install_chat(monkeypatch)
    library_match.rank_windows(shots, deepcopy(candidates), lambda: None)
    file = next((paths.DATA / "cache" / "library-rank-responses").glob("*.json"))
    body = json.loads(file.read_text())
    if damage == "malformed":
        file.write_text("{")
    else:
        if damage == "expired":
            body["created"] -= 8*86400
        else:
            body["response"]["rankings"][0]["candidate_refs"] = ["other-shot-ref"]
        file.write_text(json.dumps(body))
    library_match.rank_windows(shots, deepcopy(candidates), lambda: None)
    assert len(calls) == 2


def test_waiting_cache_rechecks_changed_bytes_on_read(inputs):
    shots, candidates, _ = inputs
    cache = rank_cache.Ranking("policy", {"shots": shots}, {"shot-a": {"candidate": candidates["shot-a"][0]}}, {})
    cache.save({"rankings": []}, lambda r: True)
    assert cache.read(lambda r: True)
    library.file_path("blobs/clip.mp4").write_bytes(b"changed during lock wait")
    assert cache.read(lambda r: True) is None


def test_different_response_model_is_not_written_under_request_model(inputs, monkeypatch):
    shots, candidates, _ = inputs
    original = install_chat(monkeypatch)
    chat = planner.chat
    def different(*args, **kwargs):
        result, _ = chat(*args, **kwargs)
        return result, {"model": "other-model"}
    monkeypatch.setattr(planner, "chat", different)
    for _ in range(2):
        library_match.rank_windows(shots, deepcopy(candidates), lambda: None)
    assert len(original) == 2
