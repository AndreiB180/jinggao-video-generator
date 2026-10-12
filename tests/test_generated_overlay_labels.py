"""Generated labels never create a second, model-invented copy of source facts."""
from copy import deepcopy
import json

import pytest

from jinggao import editorial_contract, planner


ROUND15_SOURCE = "先看一个有明确年份的数字。国家邮政局公布，二〇二四年我国快递业务量完成一千七百五十点八亿件。"
ROUND15_BAD_LABEL = "国家邮政局2024年快递业务量1760.8亿件"


def plan_with_label(monkeypatch, tmp_path, *, script=ROUND15_SOURCE, requirements="", screen_source=None):
    project = {"script": script, "requirements": requirements, "style": "解说",
               "_plan_diagnostics_dir": str(tmp_path)}
    calls = []
    monkeypatch.setattr(planner.config, "read", lambda: {"search_provider": "mixed"})
    def chat(_system, content, **_kwargs):
        request = json.loads(content)
        calls.append(request)
        if request.get("stage") == editorial_contract.VERSION:
            return {"contracts": [{"ref": ref, "use": "illustration", "refs": [], "facts": [],
                "must": [["onscreen_text", screen_source, requirements if screen_source == "r" else script]]
                        if screen_source else [], "queries": ["快递 包裹"]} for ref in request["targets"]]}, {}
        return {"shots": [{"source_unit_id": unit["source_unit_id"], "overlay_label": ROUND15_BAD_LABEL}
                           for unit in request["source_units"]]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    shots, info = planner.plan(project, "model", False)
    assert len(calls) == 2  # Existing plan + source extraction, no new model stage.
    return project, shots, info


def test_round15_generated_label_cannot_turn_1750_point_8_into_1760_point_8(monkeypatch, tmp_path):
    project, shots, info = plan_with_label(monkeypatch, tmp_path)
    assert len(shots) == 1 and shots[0]["narration"] == ROUND15_SOURCE
    assert shots[0]["overlay_label"] == "资料画面"
    assert editorial_contract.for_shot(shots[0], project)["must"] == []
    assert info["generated_overlay_labels"] == [{"shot_id": shots[0]["id"], "source_unit_id": "u0001",
        "model_label": ROUND15_BAD_LABEL, "applied_label": "资料画面", "method": "local_purpose_only"}]


@pytest.mark.parametrize("source,literal", [("r", "1750.8亿件"), ("s1", "一千七百五十点八亿件")])
def test_explicit_screen_command_uses_source_literal_without_numeric_conversion(monkeypatch, tmp_path, source, literal):
    command = f"请在屏幕上显示“{literal}”。"
    project, shots, info = plan_with_label(monkeypatch, tmp_path,
        script=command if source == "s1" else ROUND15_SOURCE,
        requirements=command if source == "r" else "", screen_source=source)
    assert shots[0]["overlay_label"] == literal
    bound = editorial_contract.for_shot(shots[0], project)["onscreen_text"][0]
    assert bound["text"] == literal and bound["command_quote"] == command
    text = project["requirements"] if source == "r" else project["script"]
    assert all(text[s["start"]:s["end"]] == literal for s in bound["source_spans"])
    assert info["generated_overlay_labels"][0]["method"] == "source_command_literal"


@pytest.mark.parametrize("quote", ["统计报告显示1750.8亿件。", "屏幕显示“1750.8亿件”。",
                                  "当年记者要求屏幕显示“1750.8亿件”。"])
def test_ordinary_number_or_description_cannot_be_promoted_to_screen_command(quote):
    shot = planner.new_shot(quote)
    project = {"script": quote, "requirements": ""}
    response = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [], "facts": [],
        "must": [["onscreen_text", "s1", quote]], "queries": []}]}
    with pytest.raises(ValueError, match="明确屏幕文字命令"):
        editorial_contract.compile_responses(project, [shot], [response])


@pytest.mark.parametrize("protected", [{"visual_origin": "manual"}, {"locked": True},
                                       {"visual_origin": None}, {"selected_asset": "saved-video"}])
def test_manual_locked_legacy_and_selected_source_labels_are_preserved(protected):
    shot = planner.new_shot(ROUND15_SOURCE, overlay_label="2021年内蒙古双十一资料")
    shot.update(protected)
    original = deepcopy(shot)
    project = {"script": ROUND15_SOURCE, "requirements": ""}
    bundle = editorial_contract.compile_responses(project, [shot], [])
    attached = editorial_contract.attach(project, [shot], bundle)[0]
    assert attached == original and shot == original
    assert editorial_contract.initial_overlay_label(attached, project) == original["overlay_label"]
    # Construction still preserves old/manual labels; restriction is only in
    # returning a fresh, generated, valid source-contract plan.
    assert planner.new_shot("历史项目", overlay_label="北门现址资料")["overlay_label"] == "北门现址资料"
