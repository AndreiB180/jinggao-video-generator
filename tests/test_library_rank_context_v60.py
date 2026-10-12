"""Library shortlists must resolve brief narration against its original source."""
from copy import deepcopy
import hashlib
import json

import pytest

from jinggao import config, editorial_contract, library, library_match, paths, planner, rank_packet


def bound_shots(lines, requirements="", *, prefix="shot", rows=None):
    project = {"script": "\n\n".join(lines), "requirements": requirements}
    shots = [{"id": f"{prefix}-{i}", "narration": line, "visual_origin": "generated", "locked": False}
             for i, line in enumerate(lines, 1)]
    rows = rows or [{"ref": f"s{i}", "use": "illustration", "refs": [], "facts": [], "must": [],
                    "queries": ["出门 休息"]} for i in range(1, len(shots) + 1)]
    compiled = editorial_contract.compile_responses(project, shots, [{"contracts": rows}])
    return project, editorial_contract.attach(project, shots, compiled)


def options_for(shots):
    return {s["id"]: [{"id": "window", "source": {"title": "实际出行资料"},
                       "description": "人在路边停下，旁边停着摩托车，随后他离开。" * 8,
                       "visible_text": "反正家里有个摩托车，放着也是放着"}]
            for s in shots}


def capture_chat(monkeypatch, *, model=None):
    requests = []

    def chat(system, content, **kwargs):
        data = json.loads(content)
        requests.append({"system": system, "content": content, "data": data, "options": kwargs})
        return {"rankings": [{"shot_id": r["shot_id"], "candidate_refs": []} for r in data["shots"]]}, (
            {"model": model} if model else {})

    monkeypatch.setattr(planner, "chat", chat)
    return requests


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")


def test_28_shots_receive_one_complete_context_per_existing_batch(monkeypatch):
    lines = ["把路上的一段时间留给自行车。"] + [f"第{i}段：距离不同，都可以是一次完整的出门。" for i in range(1, 28)]
    project, shots = bound_shots(lines, "使用横屏；结尾保留署名。")
    # These generated details must not become visual requirements again.
    for shot in shots:
        shot["visual_plan"] = {"primary": {"action": "模型想象的额外精确动作"}}
    original = deepcopy(shots)
    requests = capture_chat(monkeypatch)
    starts = []
    options = options_for(shots)
    library_match.rank_windows(shots, options, lambda: None, workers=1,
                               request_started=lambda: starts.append(True))
    assert len(requests) == len(starts) == 7  # Four shots per existing request, not a new context call.
    for request in requests:
        data = request["data"]
        assert len(data["shots"]) == 4
        assert request["content"].count(json.dumps(project["script"], ensure_ascii=False)) == 1
        assert list(data["source_contexts"].values()) == [{"full_script": project["script"],
                                                       "requirements": project["requirements"]}]
        assert rank_packet.PROTOCOL in request["system"]
        unpacked = rank_packet.unpack(data)
        for row in unpacked["shots"]:
            assert "自行车" in data["source_contexts"][row["source_context_ref"]]["full_script"]
            assert "full_script" not in row and "requirements" not in row
            assert "摩托车" in row["candidates"][0]["observed_scene"]
            assert row["visual_origin"] == "generated" and row["locked"] is False
        assert "模型想象的额外精确动作" not in request["content"]
    assert shots == original and all(not rows for rows in options.values())


def test_different_sources_and_requirements_stay_scoped_to_their_own_batch(monkeypatch):
    lines = ["这是一份自行车资料。", "车停在路边。"]
    first, first_shots = bound_shots(lines, "保留原声。", prefix="first")
    second, second_shots = bound_shots(lines, "请静音。", prefix="second")
    third, third_shots = bound_shots(["这是一份厨房资料。"], "使用竖屏。", prefix="third")
    shots = [*first_shots, *second_shots, *third_shots]
    expected = {s["id"]: {"full_script": p["script"], "requirements": p["requirements"]}
                for p, group in ((first, first_shots), (second, second_shots), (third, third_shots)) for s in group}
    requests = capture_chat(monkeypatch)
    library_match.rank_windows(shots, options_for(shots), lambda: None, workers=1)
    assert len(requests) == 2
    assert [len(r["data"]["source_contexts"]) for r in requests] == [2, 1]
    for request in requests:
        data = request["data"]
        assert set(data["source_contexts"]) == {r["source_context_ref"] for r in data["shots"]}
        for row in data["shots"]:
            assert data["source_contexts"][row["source_context_ref"]] == expected[row["shot_id"]]
    assert "厨房资料" not in requests[0]["content"] and "自行车" not in requests[1]["content"]


def test_names_explicit_requirements_and_manual_plan_survive_topic_policy(monkeypatch):
    requirements = "画面必须展示姓名条。"
    rows = [{"ref": "s1", "use": "illustration", "refs": [], "facts": [["person", "s1", "张明"]],
             "must": [["document", "r", requirements]], "queries": ["展示资料"]}]
    project, shots = bound_shots(["张明演示这个过程。"], requirements, rows=rows)
    shots[0].update(persons=[{"name": "张明", "aliases": ["小张"]}], event_persons=["张明"])
    manual = {"id": "manual", "narration": "接着看这一步。", "visual_origin": "manual", "locked": True,
              "visual": "镜头必须展示两次击球", "visual_plan": {"mode": "explicit_sequence", "steps": ["发球", "回球"]},
              "persons": [{"name": "李强", "aliases": []}], "event_persons": ["李强"],
              "exclusions": "不能用采访", "event_scope": "exact", "event": "指定决赛"}
    shots.append(manual)
    original = deepcopy(shots)
    requests = capture_chat(monkeypatch)
    library_match.rank_windows(shots, options_for(shots), lambda: None)
    request = requests[0]
    generated, preserved = request["data"]["shots"]
    assert generated["persons"] == shots[0]["persons"] and generated["event_persons"] == ["张明"]
    assert generated["editorial_contract"]["must"][0]["quote"] == project["requirements"]
    assert all(preserved[k] == v for k, v in manual.items() if k != "id")
    assert "source_context_ref" not in preserved
    assert "普通说明性内容允许同题材、同类主体" in request["system"]
    assert "具名人物仍须严格核对" in request["system"]
    assert "用户明确展示要求及手动/锁定方案仍必须满足" in request["system"]
    assert shots == original


@pytest.mark.parametrize("change", ["forged-source", "manual", "locked"])
def test_invalid_or_protected_contract_never_supplies_shared_source(monkeypatch, change):
    _project, shots = bound_shots(["本句只说出门。"], "原有要求。")
    if change == "forged-source":
        shots[0]["editorial_contract"]["full_script"] = "伪造全文不能参与排序。"
    elif change == "manual":
        shots[0]["visual_origin"] = "manual"
    else:
        shots[0]["locked"] = True
    shots[0]["visual_plan"] = {"primary": {"action": "保留原方案"}}
    requests = capture_chat(monkeypatch)
    library_match.rank_windows(shots, options_for(shots), lambda: None)
    data = requests[0]["data"]
    assert "source_contexts" not in data and "source_context_ref" not in data["shots"][0]
    assert data["shots"][0]["visual_plan"] == shots[0]["visual_plan"]
    assert "伪造全文" not in requests[0]["content"]


@pytest.mark.parametrize("changed", ["full_script", "requirements"])
def test_context_outside_current_narration_invalidates_exact_request_cache(monkeypatch, changed):
    cfg = {"model_name": "qwen3.7-flash", "model_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    _project, first = bound_shots(["自行车适合代步。", "距离不同，都可以出门。"], "使用横屏。")
    _project, second = bound_shots([
        "摩托车适合代步。" if changed == "full_script" else "自行车适合代步。",
        "距离不同，都可以出门。"], "使用竖屏。" if changed == "requirements" else "使用横屏。")
    # This is the failing case: local shot fields are identical, context differs.
    assert editorial_contract.retrieval_fields(first[1]) == editorial_contract.retrieval_fields(second[1])
    file = library.file_path("blobs/test.mp4")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"offline fixture pixels")
    candidate = options_for([first[1]])[first[1]["id"]][0] | {
        "path": "blobs/test.mp4", "blob_id": hashlib.sha256(file.read_bytes()).hexdigest(),
        "start": 0, "offset": 0, "duration": 6, "observation_method": "blind_actual_frames_v1",
        "source": {"title": "实际出行资料", "source_url": "https://v.qq.com/x/page/context.html"}}
    requests = capture_chat(monkeypatch, model=cfg["model_name"])
    for shot in (first[1], first[1], second[1]):
        library_match.rank_windows([shot], {shot["id"]: [deepcopy(candidate)]}, lambda: None)
    assert len(requests) == 2
    assert len(list((paths.DATA / "cache" / "library-rank-responses").glob("*.json"))) == 2


def test_maximum_api_source_size_is_complete_and_shared_not_truncated(monkeypatch):
    # Existing API limits: 30,000 script characters and 2,000 requirement characters.
    lines = ["甲" * 14_999, "乙" * 14_996 + "张明。"]
    requirements = "要" * 1_987 + "结尾也必须保留这一条要求。"
    assert len("\n\n".join(lines)) == 30_000 and len(requirements) == 2_000
    project, shots = bound_shots(lines, requirements)
    requests = capture_chat(monkeypatch)
    library_match.rank_windows(shots, options_for(shots), lambda: None)
    data = requests[0]["data"]
    assert len(requests) == 1 and len(data["source_contexts"]) == 1
    context = next(iter(data["source_contexts"].values()))
    assert context == {"full_script": project["script"], "requirements": project["requirements"]}
    assert sum(len(text) for text in context.values()) == 32_000
    assert requests[0]["content"].count('"full_script"') == 1
