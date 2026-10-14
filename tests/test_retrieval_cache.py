from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time

import pytest

from jinggao import config, paths, planner, pooled_match, retrieval_cache


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.read()
    monkeypatch.setattr(config, "read", lambda: cfg)
    shots = [{"id": str(i), "narration": f"第{i}镜作业", "persons": [], "query": "工人 生产"} for i in range(2)]
    project = {"id": "test", "script": "完整原文", "requirements": "保留要求", "shots": shots}
    calls = []
    response = {"groups": [{"label": "生产过程", "indices": [0, 1], "queries": ["工人 生产", "车间 实拍"]}]}
    def chat(*args, **kw):
        calls.append(args)
        time.sleep(.02)
        return deepcopy(response), {"model": cfg["model_name"]}
    monkeypatch.setattr(planner, "chat", chat)
    return project, shots, calls, cfg, response


def test_resume_groups_once_with_no_inherited_approvals(example):
    project, shots, calls, _, _ = example
    first, info1 = pooled_match.retrieval_groups(project, shots)
    second, info2 = pooled_match.retrieval_groups(project, shots)
    assert first == second and len(calls) == 1 and info2["cache_hit"] and not info1.get("cache_hit")
    assert all("approved" not in s for s in shots)


def test_parallel_grouping_is_one_request(example):
    project, shots, calls, _, _ = example
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: pooled_match.retrieval_groups(project, shots), range(6)))
    assert len(calls) == 1 and sum(r[1].get("cache_hit", False) for r in results) == 5


@pytest.mark.parametrize("field", ["script", "requirements", "persons", "event", "model"])
def test_changed_source_identity_or_model_invalidates(example, field):
    project, shots, calls, cfg, _ = example
    pooled_match.retrieval_groups(project, shots)
    if field in ("script", "requirements"):
        project[field] += "新条件"
    elif field == "model":
        cfg["model_name"] = "changed-model"
    else:
        shots[0][field] = "指定事件" if field == "event" else [{"name": "指定人物"}]
    pooled_match.retrieval_groups(project, shots)
    assert len(calls) == 2


@pytest.mark.parametrize(("indices", "requests"), [([0], 1), ([0, 0, 1], 1), ([True, 1], 2), ([0, 1, 2], 2)])
def test_invalid_grouping_is_never_cached_but_normalized_coverage_is(example, indices, requests):
    # retrieval_cache.normalize completes a missing/duplicate index deterministically
    # (search opportunities only) and caches that complete result; malformed indices raise.
    project, shots, calls, _, response = example
    response["groups"][0]["indices"] = indices
    assert not retrieval_cache.valid(response, 2)
    results = []
    for _ in range(2):
        try:
            results.append(pooled_match.retrieval_groups(project, shots)[0])
        except ValueError:
            pass
    assert len(calls) == requests
    for groups in results:
        assert sorted(i for g in groups for i in g["indices"]) == [0, 1]
