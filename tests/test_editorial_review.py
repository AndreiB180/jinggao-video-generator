"""Source-first matching must retain actions, caption context and edit authority."""
from copy import deepcopy
import json
import socket

import pytest
from fastapi.testclient import TestClient

from jinggao import config, editorial_contract, frame_contract, neutral_observer, paths, planner, store
from jinggao.api import create_app

pytestmark = pytest.mark.usefixtures("isolated_pixel_quality")


@pytest.fixture(autouse=True)
def offline_review_config(monkeypatch, request):
    mode = getattr(request, "param", "strict")
    model = "qwen3.7-flash" if mode == "strict" else "deepseek-flash"
    cfg = config.DEFAULTS | {"model_api_key": "test-only-no-network", "model_name": model,
        "review_model": model, "pixel_observer_mode": "separate", "_strict_text_review_trial": mode == "strict",
        "model_base_url": ("https://dashscope.aliyuncs.com/compatible-mode/v1"
                           if mode == "strict" else "https://api.deepseek.com")}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    def forbidden(*args, **kwargs):
        pytest.fail("editorial review unit test attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    return mode


def bound(script="确认地址、再次联系、调整安排这些动作，需要人来完成。", requirements=""):
    project = {"script": script, "requirements": requirements}
    shot = planner.new_shot(narration=script, visual="拿手机低头操作", query="快递 手机")
    shot["source_unit_id"] = "u1"
    shot["visual_plan"] = {"mode": "single", "primary": {"subject": "拿手机的人", "action": "低头操作手机"}}
    reply = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [],
                            "facts": [["activity", "s1", script]], "must": [], "queries": ["快递员 联系"]}]}
    bundle = editorial_contract.compile_responses(project, [shot], [reply])
    project["shots"] = editorial_contract.attach(project, [shot], bundle)
    return project, project["shots"][0]


@pytest.mark.parametrize("failed", ["source_alignment", "caption_context"])
def test_old_primary_match_cannot_hide_wrong_original_action_or_unrelated_story(monkeypatch, failed):
    project, shot = bound()
    shot["overlay_label"] = "劳动资料画面"
    observation = {"scene_description": "男子对门口纸箱拍照，字幕描述家庭故事。", "frames": [
        {"frame_index": 0, "objects": "门口纸箱和手机", "action": "男子对纸箱拍照",
         "text": ["告诉女子的老公自己是来送快递的"]}]}
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check: ({shot["id"]: observation}, 0))
    evidence = [{"frame_index": 0, "field": "action", "quote": "男子对纸箱拍照"}]

    def judge(system, content, **options):
        payload = json.loads(content[0]["text"])
        assert payload["source_context"]["full_script"] == project["script"]
        assert payload["source_context"]["requirements"] == project["requirements"]
        assert payload["overlay_label"] == "劳动资料画面"
        assert [c["text"] for c in payload["context_label_context"]["captions"]] == ["告诉女子的老公自己是来送快递的"]
        assert not payload["visual"] and not payload["subject"]
        assert "低头操作手机" not in json.dumps(payload, ensure_ascii=False)
        checks = [{"key": key, "status": "conflict" if key == failed else "supported",
                   "reason": "不同动作或无关故事", "evidence": ([{"frame_index": 0, "field": "text",
                       "quote": observation["frames"][0]["text"][0]}] if key.startswith("caption_c") else evidence)}
                  for key in payload["selected_visual_requirements"]]
        return {"reviews": [{"shot_id": shot["id"], "status": "supported", "reason": "故意错误的总判定",
                             "labels": [], "evidence": evidence, "primary_checks": checks}]}, {}

    monkeypatch.setattr(planner, "chat", judge)
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"},
                                              "project_context": project, "frame_paths": []}], lambda: None)[shot["id"]]
    assert result["status"] == "conflict"
    assert next(r for r in result["primary_checks"] if r["key"] == failed)["status"] == "conflict"


def test_missing_caption_check_cannot_be_approved_when_original_has_no_explicit_must():
    _, shot = bound()
    assert editorial_contract.for_shot(shot)["must"] == []
    frames = [{"frame_index": 0, "objects": "快递员", "action": "打电话", "text": []}]
    result = frame_contract.validate(shot, [{"key": "source_alignment", "status": "supported", "reason": "通话",
        "evidence": [{"frame_index": 0, "field": "action", "quote": "打电话"}]}], frames)
    assert result[0] == "unknown"


@pytest.mark.parametrize("change, expected", [("none", "supported"), ("wrong-label", "conflict"),
                                             ("missing-action", "unknown"), ("name-overlay", "unknown")])
@pytest.mark.parametrize("offline_review_config", ["strict", "legacy"], indirect=True)
def test_screen_command_reaches_real_audit_without_becoming_pixel_or_identity_evidence(
        monkeypatch, change, expected, offline_review_config):
    project, shot = bound("工作人员搬运包裹。", "请在屏幕上显示“1750.8亿件”。必须看见搬运包裹。")
    command = "请在屏幕上显示“1750.8亿件”"
    reply = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [], "facts": [],
        "must": [["onscreen_text", "r", command], ["activity", "r", "必须看见搬运包裹"]], "queries": []}]}
    shot = editorial_contract.attach(project, [shot], editorial_contract.compile_responses(project, [shot], [reply]))[0]
    shot["overlay_label"] = "1760.8亿件" if change == "wrong-label" else "1750.8亿件"
    if change == "name-overlay":
        # Even a correct source-bound number overlay proves no named identity.
        shot["persons"] = [{"name": "张明", "aliases": []}]
    project["shots"] = [shot]
    observed = {"scene_description": "工作人员搬运纸箱", "frames": [
        {"frame_index": 0, "objects": "工作人员和纸箱", "action": "搬运包裹", "text": []}]}
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: observed}, 0))
    calls = []
    evidence = [{"frame_index": 0, "field": "action", "quote": "搬运包裹"}]
    def judge(system, content, **options):
        calls.append(1)
        payload = json.loads(content[0]["text"])
        assert set(payload["selected_visual_requirements"]) == {"source_alignment", "caption_context", "must_2"}
        assert payload["frame_observations"][0]["text"] == []
        local = payload["source_context"]["postproduction_text_checks"][0]
        assert local["key"] == "must_1" and local["evidence"][0]["source_text"] == "1750.8亿件"
        if offline_review_config == "strict":
            assert "onscreen_text服从postproduction_text_checks" in system
            assert "原片无需带后期文字，叠字不证明身份/动作/数字" in system
        else:
            assert "不得因原片没有后期文字而误拒" in system
            assert "后期文字也不能证明原片身份、数字或动作" in system
        checks = [{"key": key, "status": "supported", "reason": "实际搬运", "evidence": evidence}
                  for key in payload["selected_visual_requirements"] if change != "missing-action" or key != "must_2"]
        return {"reviews": [{"shot_id": shot["id"], "status": "supported", "reason": "内容与用途相符",
            "labels": [], "evidence": evidence, "primary_checks": checks}]}, {}
    monkeypatch.setattr(planner, "chat", judge)
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"},
        "project_context": project, "frame_paths": []}], lambda: None)[shot["id"]]
    assert result["status"] == expected and len(calls) == 1
    if change in {"none", "wrong-label"}:
        assert next(row for row in result["primary_checks"] if row["key"] == "must_1")["status"] == expected


def test_repeated_sales_preview_is_rejected_before_paid_text_audit(monkeypatch):
    project, shot = bound(script="展陈让更多人接触历史。")
    frames = [{"frame_index": i, "objects": "展品与画面中央大字", "action": "展示展品",
               "text": ["下单获取高清无水印版 NO:2040"]} for i in range(5)]
    observation = {"scene_description": "展览素材售卖预览", "frames": frames}
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check: ({shot["id"]: observation}, 1))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("Sale preview must not incur text audit"))
    result = planner.audit_selected_frames([{"shot": shot, "asset": {"kind": "video"},
        "project_context": project, "frame_paths": []}], lambda: None)[shot["id"]]
    assert result["status"] == "conflict" and result["request_count"] == 1
    assert len(result["evidence"]) == 5 and result["quality_check"]["status"] == "reject"


def test_sales_preview_does_not_discard_other_valid_review_or_hide_failed_request(monkeypatch):
    _, bad = bound(script="参观展览。")
    project, good = bound(script="工作人员查看包裹。")
    observations = {
        bad["id"]: {"scene_description": "售卖预览", "frames": [
            {"frame_index": i, "objects": "展品", "action": "静止展示", "text": ["下单获取无水印版"]}
            for i in range(2)]},
        good["id"]: {"scene_description": "查看包裹", "frames": [
            {"frame_index": 0, "objects": "工作人员和包裹", "action": "查看包裹", "text": []}]},
    }
    monkeypatch.setattr(neutral_observer, "observe", lambda records, check: (observations, 1))
    def unavailable(system, content, **kwargs):
        assert len(content) == 1 and json.loads(content[0]["text"])["shot_id"] == good["id"]
        raise ValueError("本次文本响应不完整")
    monkeypatch.setattr(planner, "chat", unavailable)
    result = planner.audit_selected_frames([{"shot": s, "asset": {"kind": "video"},
        "project_context": project, "frame_paths": []} for s in (bad, good)], lambda: None)
    assert result[bad["id"]]["status"] == "conflict" and result[good["id"]]["status"] == "unknown"
    assert all(r["request_count"] == 2 for r in result.values())


@pytest.mark.parametrize("edit", ["script", "requirements", "narration", "visual", "query", "persons"])
def test_api_edits_remove_old_source_binding_but_duration_does_not(tmp_path, monkeypatch, edit):
    monkeypatch.setattr(paths, "DATA", tmp_path / "isolated")
    with TestClient(create_app()) as client:
        template, shot = bound()
        project = client.post("/api/projects", json={"title": "契约编辑", "script": template["script"]}).json()
        store.mutate(project["id"], lambda p: p.update(shots=[deepcopy(shot)]))
        url = f"/api/projects/{project['id']}"
        duration = client.patch(url + f"/shots/{shot['id']}", json={"duration": 9.0})
        assert duration.status_code == 200
        assert editorial_contract.for_shot(duration.json()["shots"][0], duration.json())
        values = {"script": "新的原稿。", "requirements": "必须展示联系过程。", "narration": "新的口播。",
                  "visual": "明确要求展示拨号。", "query": "新查询", "persons": [{"name": "张三", "aliases": []}]}
        response = client.patch(url + ("" if edit in ("script", "requirements") else f"/shots/{shot['id']}"),
                                json={edit: values[edit]})
        assert response.status_code == 200, response.text
        assert "editorial_contract" not in response.json()["shots"][0]
