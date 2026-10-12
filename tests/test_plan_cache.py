from copy import deepcopy
import json

import pytest

from jinggao import config, paths, plan_consistency, planner
from planner_helpers import consistency_response


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.DEFAULTS | {"plan_workers": "1"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    script = "\n".join(f"工作人员搬运第{i+1}件包裹。" for i in range(8))
    project = {"id": "a"*32, "script": script, "requirements": ""}
    shots = [planner.new_shot(line) for line in script.splitlines()]
    return project, shots, cfg


def test_partial_resume_reuses_only_validated_batches_with_new_ids_and_no_usage_double_count(setup, tmp_path):
    project, shots, cfg = setup
    calls = []
    def first(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request["batch_index"])
        if request["batch_index"] == 2:
            raise ValueError("incomplete paid response")
        return consistency_response(request), {"model": "test", "usage": {"prompt_tokens": 40, "completion_tokens": 60}}
    with pytest.raises(ValueError, match="incomplete paid response"):
        plan_consistency.review(project, shots, chat=first, check=lambda: None, workers=1)
    assert calls == [1, 2]
    files = list((tmp_path / "cache" / "plan-review").glob("*.json"))
    assert len(files) == 1
    new_shots = [shot | {"id": str(i)*32} for i, shot in enumerate(shots)]
    def resume(system, content, **kwargs):
        request = json.loads(content)
        calls.append(request["batch_index"])
        return consistency_response(request), {"model": "test", "usage": {"prompt_tokens": 40, "completion_tokens": 60}}
    result, info = plan_consistency.review(project, new_shots, chat=resume, check=lambda: None, workers=1)
    assert calls == [1, 2, 2]
    assert info["cached_batches"] == 1 and info["model_requests"] == 1
    assert info["usage"] == {"prompt_tokens": 40, "completion_tokens": 60}
    assert [s["id"] for s in result] == [s["id"] for s in new_shots]
    assert [s["narration"] for s in result] == [s["narration"] for s in shots]


@pytest.mark.parametrize("change", ["requirements", "visual", "model", "endpoint", "protocol", "invalid_cache"])
def test_content_model_protocol_changes_or_invalid_cached_evidence_require_fresh_review(setup, monkeypatch, tmp_path, change):
    project, shots, cfg = setup
    def chat(system, content, **kwargs):
        return consistency_response(json.loads(content)), {"model": "test", "usage": {}}
    plan_consistency.review(project, shots, chat=chat, check=lambda: None)
    assert len(list((tmp_path/"cache"/"plan-review").glob("*.json"))) == 2
    project, shots = deepcopy(project), deepcopy(shots)
    if change == "requirements":
        project["requirements"] = "画面展示劳动"
    elif change == "visual":
        shots[0]["visual"] = "工人搬运包裹的近景"
    elif change in {"model", "endpoint"}:
        cfg["model_name" if change == "model" else "model_base_url"] = "changed"
    elif change == "protocol":
        monkeypatch.setattr(plan_consistency, "SYSTEM", plan_consistency.SYSTEM+"\n补充当前规则")
    else:
        for file in (tmp_path/"cache"/"plan-review").glob("*.json"):
            data = json.loads(file.read_text())
            data["response"]["reviews"] = []
            file.write_text(json.dumps(data))
    _, info = plan_consistency.review(project, shots, chat=chat, check=lambda: None)
    # A local visual edit only changes that batch's request. Shared context
    # changes invalidate every batch; unchanged batches still revalidate locally.
    expected = 1 if change == "visual" else 2
    assert info["model_requests"] == expected and info["cached_batches"] == 2-expected
