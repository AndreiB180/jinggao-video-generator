from copy import deepcopy

import pytest

from jinggao import editorial_contract, review_evidence, review_view


def record():
    script = "李明在二〇二四年参加比赛。\n\n这场比赛不能借用2019年的其他赛事。"
    focus = "这场比赛不能借用2019年的其他赛事。"
    requirements = "必须展示计分牌，并在后期写明‘资料画面’。"
    start = script.index(focus)
    name_span = {"start": 0, "end": 2}
    contract = {"shot_id": "shot-a", "ref": "s2", "use": "event_record", "refs": ["s1"],
        "source_span": {"start": start, "end": len(script)},
        "facts": [{"kind": "person", "ref": "s1", "quote": "李明", "source": "script",
                   "source_spans": [name_span]}],
        "must": [{"kind": "document", "ref": "r", "quote": "必须展示计分牌", "source": "requirements",
                  "source_spans": [{"start": 0, "end": 7}]}],
        "state": "source_bound_only"}
    frames = [{"frame_index": 0, "objects": "计分牌与场上两名选手", "action": "站在球网两侧，未击球",
               "text": ["李明 VS 王强", "2019年另一比赛", "购买无水印版"]}]
    refs, _ = review_evidence.references("shot-a", frames)
    return {"shot_id": "shot-a", "narration": focus, "requirements": requirements,
        "requested_people": [{"name": "李明"}], "event": "原稿指定比赛", "event_scope": "exact",
        "narrative_context": [{"position": "before", "narration": script[:start]}],
        "source_context": {"full_script": script, "requirements": requirements,
            "contract": deepcopy(contract), "rules": editorial_contract.SELECTION_RULES,
            "source_alignment": review_view._ALIGNMENT_INSTRUCTION,
            "required_checks": ["source_alignment", "caption_context", "must_1"],
            "source_refs": [{"ref": "s2", "start": start, "end": len(script)},
                            {"ref": "s1", "start": 0, "end": start}],
            "local_context": {"source_span": {"start": start, "end": len(script)}, "narration": focus,
                "paragraphs": [{"source_span": {"start": start, "end": len(script)}, "quote": focus}]}},
        "source_contract": contract,
        "selected_visual_requirements": {"source_alignment": focus, "caption_context": "全部字幕逐项核对",
            "frame_scope_0": "本帧必须对应本镜", "caption_c1": frames[0]["text"][0],
            "caption_c2": frames[0]["text"][1], "caption_c3": frames[0]["text"][2], "must_1": "必须展示计分牌"},
        "evidence_binding": {"frame_sha256": ["a" * 64], "clip_start": 12.5, "duration": 8.0},
        "frame_observations": frames, "observation_references": refs,
        "scene_description": "两名选手未击球，画面另有2019年的年份。",
        "visual_quality": {"quality_status": "not_detected", "representation": "live_action", "frames": [
            {"frame_index": 0, "representation": "live_action", "type_evidence": "球场纹理、自然光影与真实摄影透视",
             "overlays": [{"kind": "sales_promotion", "text": "购买无水印版", "bbox": [0, 0, .1, .1],
                           "impact": "minor", "occludes": ""}]}]},
        "context_label_context": {"captions": [{"caption_id": f"c{i+1}", "text": text, "frame_indices": [0]}
                                                  for i, text in enumerate(frames[0]["text"])],
                                  "source_fields": {"title": "旧赛事报道", "published_at": "2019-01-01"}},
        "source_evidence": {"title": "旧赛事报道", "description": "其他赛事，不能当本场。", "url": "https://example.org"},
        "source_attribution": {"coverage": "unverified", "matched_names": [], "selection_blocked": True},
        "source_parts": [{"title": "首段", "source_at": 12.5, "duration": 8}], "overlay_label": "资料画面"}


def test_projection_preserves_all_user_words_pixels_captions_sources_and_identity():
    original = record()
    saved = deepcopy(original)
    view = review_view.build([original])[0]
    assert original == saved
    for key in ("narration", "requirements", "requested_people", "event", "event_scope", "narrative_context",
                "frame_observations", "observation_references", "scene_description",
                "context_label_context", "source_evidence", "source_attribution", "source_parts", "overlay_label"):
        assert view[key] == original[key]
    assert {key: condition["value"] for key, condition in view["selected_visual_requirements"].items()} == original["selected_visual_requirements"]
    for key in ("full_script", "requirements"):
        assert view["source_context"][key] == original["source_context"][key]
    assert not {"rules", "source_alignment", "required_checks", "contract"} & view["source_context"].keys()
    assert "evidence_binding" not in view
    assert "source_span" not in view["source_contract"]
    for key in ("facts", "must"):
        assert view["source_contract"][key] == [{k: v for k, v in citation.items() if k != "source_spans"}
                                                for citation in original["source_contract"][key]]
    assert view["visual_quality"] == original["visual_quality"]
    assert view["visual_quality"]["frames"][0]["overlays"][0]["text"] == "购买无水印版"
    view["frame_observations"][0]["text"].clear()
    assert original == saved


def test_antecedent_offsets_become_exact_full_source_words_not_a_guess():
    original = record()
    view = review_view.build([original])[0]
    for before, after in zip(original["source_context"]["source_refs"], view["source_context"]["source_refs"], strict=True):
        assert after == {"ref": before["ref"],
                         "quote": original["source_context"]["full_script"][before["start"]:before["end"]]}
    assert "二〇二四年" in view["source_context"]["source_refs"][1]["quote"]
    assert view["source_context"]["local_context"]["paragraphs"] == [{"quote": original["narration"]}]


def test_unknown_fields_are_not_pruned_by_recursively_named_keys():
    original = record()
    unknown = {"rules": "不要改变用户这一行", "source_spans": [{"unknown": "retain"}], "frame_sha256": "visible label"}
    original["future"] = deepcopy(unknown)
    original["evidence_binding"]["future"] = deepcopy(unknown)
    original["source_context"]["future"] = deepcopy(unknown)
    original["source_contract"]["future"] = deepcopy(unknown)
    original["source_contract"]["facts"][0]["future"] = deepcopy(unknown)
    original["visual_quality"]["frames"][0]["future"] = deepcopy(unknown)
    original["source_context"]["source_refs"][0]["future"] = deepcopy(unknown)
    view = review_view.build([original])[0]
    assert view["future"] == view["evidence_binding"]["future"] == view["source_context"]["future"] == unknown
    assert view["source_contract"]["future"] == view["source_contract"]["facts"][0]["future"] == unknown
    assert view["visual_quality"]["frames"][0]["future"] == unknown
    assert view["source_context"]["source_refs"][0]["future"] == unknown
    # The changed top-level contract must not silently replace the other copy.
    assert "contract" in view["source_context"]


@pytest.mark.parametrize("field", ["rules", "source_alignment", "required_checks"])
def test_changed_or_new_instructions_are_preserved(field):
    original = record()
    original["source_context"][field] = ["future_check"] if field == "required_checks" else "新增但未识别的约束"
    view = review_view.build([original])[0]
    assert view["source_context"][field] == original["source_context"][field]


def test_conflicting_contract_copies_stay_visible_without_changing_the_checklist():
    original = record()
    original["source_context"]["contract"]["must"][0]["quote"] = "另一条明确展示要求"
    view = review_view.build([original])[0]
    assert view["source_context"]["contract"]["must"][0]["quote"] == "另一条明确展示要求"
    assert view["source_contract"]["must"][0]["quote"] == "必须展示计分牌"
    assert {key: condition["value"] for key, condition in view["selected_visual_requirements"].items()} == original["selected_visual_requirements"]


@pytest.mark.parametrize("change", ["negative", "beyond", "bool", "wrong_quote", "unknown_span"])
def test_unvalidated_spans_remain_in_original_form(change):
    original = record()
    citation = original["source_contract"]["facts"][0]
    if change == "negative":
        citation["source_spans"][0]["start"] = -1
    elif change == "beyond":
        citation["source_spans"][0]["end"] = 10000
    elif change == "bool":
        citation["source_spans"][0]["start"] = False
    elif change == "wrong_quote":
        citation["quote"] = "王强"
    else:
        citation["source_spans"][0]["future"] = "retain"
    view = review_view.build([original])[0]
    assert view["source_contract"]["facts"][0] == citation


def test_existing_conflicting_source_ref_quote_is_not_overwritten():
    original = record()
    original["source_context"]["source_refs"][0]["quote"] = "别的原句"
    view = review_view.build([original])[0]
    assert view["source_context"]["source_refs"][0] == original["source_context"]["source_refs"][0]


@pytest.mark.parametrize("status", ["uncertain", "reject", "future_status"])
def test_uncompleted_or_adverse_quality_keeps_its_full_diagnostic(status):
    original = record()
    original["visual_quality"]["quality_status"] = status
    view = review_view.build([original])[0]
    assert view["visual_quality"] == original["visual_quality"]


def test_manual_legacy_fields_and_postproduction_commands_remain_intact():
    original = record()
    original["visual_origin"] = "manual"
    original["visual"] = "完整拍到本场计分牌，不许改成普通运动员"
    original["source_context"]["postproduction_text_checks"] = [{
        "key": "must_2", "required": "后期文字", "status": "conflict", "evidence": [
            {"field": "overlay_label", "quote": "错误说明", "source_spans": [{"start": 5, "end": 9}]}]}]
    original["source_context"]["required_checks"].append("must_2")
    original["source_contract"]["readonly_visual"] = {"visual": original["visual"], "locked": True}
    view = review_view.build([original])[0]
    assert view["visual"] == original["visual"] and view["visual_origin"] == "manual"
    assert view["source_contract"]["readonly_visual"] == original["source_contract"]["readonly_visual"]
    assert view["source_context"]["postproduction_text_checks"] == original["source_context"]["postproduction_text_checks"]


def test_views_require_original_records_and_do_not_replace_cache_binding():
    first, second = record(), record()
    second["evidence_binding"] = {"frame_sha256": ["b" * 64], "clip_start": 99, "duration": 8}
    a, b = review_view.build([first]), review_view.build([second])
    assert first != second and a == b
    with pytest.raises(ValueError):
        review_view.build(a)
    assert "evidence_binding" in first and "evidence_binding" in second


@pytest.mark.parametrize("records", [None, {}, [None], ["not a record"]])
def test_non_records_are_rejected_without_inventing_inputs(records):
    with pytest.raises(ValueError, match="完整记录"):
        review_view.build(records)
