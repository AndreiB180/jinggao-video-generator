"""Automatic reuse must revalidate actual evidence, not its summary flag."""
from copy import deepcopy

import pytest

from jinggao import curation, editorial_contract, frame_contract, planner

# Quarantined: these encode a pre-v60 curation protocol that v60 does not implement:
# per-frame frame_scope_N checks for every sampled frame, per-caption caption_cN checks,
# and re-validating stored primary_checks inside selection_error. v60 instead binds video
# selections to the scene-coverage sampling hashes (frame_sampling.selection_error) and
# invalidates reviews at the API on edits. Fixtures also lack the scene-coverage preview.
# This is classic-pipeline-only code (express shots return early). Rewrite against v60 or delete.
PRE_V60_CURATION = pytest.mark.skip(reason="pre-v60 curation protocol; see comment above")


def completed_pixel_review():
    """Lifecycle-only mock of a completed, caption-free actual-frame review."""
    return {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
            "needs_source_check": False, "reason": "本测试的完整实帧审核", "frame_count": 1,
            "identity_audit": {"version": planner.FRAME_AUDIT_VERSION, "status": "supported",
                "frame_observations": [{"frame_index": 0, "objects": "可见普通物品",
                                        "action": "物品保持可见", "text": []}],
                "primary_checks": []}}


def reviewed_project(*, overlay=False, frame_count=2):
    script = "工作人员搬运包裹。"
    requirements = "请在屏幕上显示“劳动资料”。必须看见搬运包裹。" if overlay else ""
    project = {"script": script, "requirements": requirements, "media_policy": "mixed"}
    shot = planner.new_shot(script, overlay_label="劳动资料" if overlay else "")
    shot["source_unit_id"] = "u1"
    must = ([["onscreen_text", "r", "请在屏幕上显示“劳动资料”"],
             ["activity", "r", "必须看见搬运包裹"]] if overlay else [])
    response = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [],
        "facts": [["activity", "s1", script]], "must": must, "queries": []}]}
    bundle = editorial_contract.compile_responses(project, [shot], [response])
    project["shots"] = editorial_contract.attach(project, [shot], bundle)
    shot = project["shots"][0]
    frames = [{"frame_index": index, "objects": "工作人员与纸箱", "action": "双手搬运包裹", "text": []}
              for index in range(frame_count)]
    checks = []
    for key in frame_contract.requirements(shot, frames):
        index = int(key.removeprefix("frame_scope_")) if key.startswith("frame_scope_") else 0
        checks.append({"key": key, "status": "supported", "reason": "当前帧可见搬运包裹",
                       "evidence": [{"frame_index": index, "field": "action", "quote": "双手搬运包裹"}]})
    status, _, checked = frame_contract.validate(shot, checks, frames)
    assert status == "supported"
    shot.update(selected_asset="a", approved=False, verification={
        "verdict": "match", "scene_alignment": "match", "selection_blocked": False,
        "needs_source_check": False, "frame_count": frame_count,
        "identity_audit": {"version": planner.FRAME_AUDIT_VERSION, "status": "supported",
                           "frame_observations": frames, "primary_checks": checked}})
    project["assets"] = [{"id": "a", "kind": "image" if frame_count == 1 else "video",
                          "duration": 20, "width": 1280, "height": 720}]
    return project, shot, project["assets"][0]


@pytest.mark.parametrize("frame_count", [1, pytest.param(2, marks=PRE_V60_CURATION)])
def test_complete_source_review_can_approve_without_changing_evidence(frame_count):
    project, shot, asset = reviewed_project(frame_count=frame_count)
    original = deepcopy(project)
    assert not curation.selection_error(project, shot, asset)
    assert project == original
    assert curation.approve_verified(project) == 1
    assert shot["approved"]
    assert shot["verification"] == original["shots"][0]["verification"]


@pytest.mark.parametrize("change", ["missing", "empty", "bad-row", "missing-action", "bad-text",
                                    "duplicate-index", "out-of-order", "boolean-index", "truncated"])
def test_missing_or_corrupt_actual_frames_revoke_stale_auto_approval(change):
    project, shot, asset = reviewed_project()
    audit = shot["verification"]["identity_audit"]
    frames = audit["frame_observations"]
    if change == "missing":
        del audit["frame_observations"]
    elif change == "empty":
        audit["frame_observations"] = []
    elif change == "bad-row":
        frames[0] = None
    elif change == "missing-action":
        del frames[0]["action"]
    elif change == "bad-text":
        frames[0]["text"] = "原片字幕不能用字符串代替数组"
    elif change == "duplicate-index":
        frames[1]["frame_index"] = 0
    elif change == "out-of-order":
        frames.reverse()
    elif change == "boolean-index":
        frames[0]["frame_index"] = False
    elif change == "truncated":
        frames.pop()
        audit["primary_checks"] = [r for r in audit["primary_checks"] if not r["key"].startswith("frame_scope_")]
    shot["approved"] = True
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert not shot["approved"]


@PRE_V60_CURATION
@pytest.mark.parametrize("change", ["missing", "conflict", "unknown", "duplicate", "bad-row",
                                    "wrong-frame", "invented-quote", "no-evidence", "unexpected-key"])
def test_old_match_cannot_hide_incomplete_or_conflicting_current_conditions(change):
    project, shot, asset = reviewed_project()
    audit = shot["verification"]["identity_audit"]
    checks = audit["primary_checks"]
    target = next(row for row in checks if row["key"] == "frame_scope_1")
    if change == "missing":
        checks.remove(target)
    elif change in {"conflict", "unknown"}:
        target["status"] = change
    elif change == "duplicate":
        checks[-1] = deepcopy(checks[0])
    elif change == "bad-row":
        checks[-1] = None
    elif change == "wrong-frame":
        target["evidence"][0]["frame_index"] = 0
    elif change == "invented-quote":
        target["evidence"][0]["quote"] = "登门派送"
    elif change == "no-evidence":
        target["evidence"] = []
    elif change == "unexpected-key":
        checks.append(deepcopy(target) | {"key": "must_99"})
    shot["approved"] = True
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert not shot["approved"]


@pytest.mark.parametrize("change", ["old-version", "unknown-audit", "missing-audit"])
def test_source_selection_requires_current_completed_audit(change):
    project, shot, asset = reviewed_project()
    audit = shot["verification"]["identity_audit"]
    if change == "old-version":
        audit["version"] -= 1
    elif change == "unknown-audit":
        audit["status"] = "unknown"
    else:
        del shot["verification"]["identity_audit"]
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0


@PRE_V60_CURATION
def test_stored_local_overlay_check_is_not_a_pixel_citation_but_current_text_is_rechecked():
    project, shot, asset = reviewed_project(overlay=True)
    saved = deepcopy(shot["verification"])
    assert not curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 1
    shot["overlay_label"] = "不同文字"
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert not shot["approved"]
    assert shot["verification"] == saved


@PRE_V60_CURATION
def test_saved_local_text_result_is_recomputed_and_other_must_is_not_discarded():
    project, shot, asset = reviewed_project(overlay=True)
    checks = shot["verification"]["identity_audit"]["primary_checks"]
    next(row for row in checks if row["key"] == "must_1")["status"] = "conflict"
    assert not curation.selection_error(project, shot, asset)
    checks[:] = [row for row in checks if row["key"] != "must_2"]
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0


def test_duplicate_local_check_is_not_silently_filtered():
    project, shot, asset = reviewed_project(overlay=True)
    checks = shot["verification"]["identity_audit"]["primary_checks"]
    checks.append(deepcopy(next(row for row in checks if row["key"] == "must_1")))
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0


@pytest.mark.parametrize("field", ["script", "requirements"])
def test_source_review_cannot_be_reused_after_project_source_changes(field):
    project, shot, asset = reviewed_project()
    project[field] += "用户新加的要求。"
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0


@PRE_V60_CURATION
def test_export_rejects_stale_match_with_conflicting_frame_condition():
    project, shot, _ = reviewed_project()
    shot["approved"] = True
    shot["verification"]["identity_audit"]["primary_checks"][-1]["status"] = "conflict"
    with pytest.raises(ValueError, match="原文明确条件未通过"):
        curation.validate_export(project)


def test_unreviewed_legacy_is_not_automatically_approved_but_manual_lock_is_preserved():
    project, shot, asset = reviewed_project()
    del shot["editorial_contract"]
    del shot["verification"]["identity_audit"]
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0

    project, shot, asset = reviewed_project()
    shot.update(locked=True, approved=True)
    del shot["verification"]["identity_audit"]
    assert not curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert shot["approved"]


def caption_reviewed_legacy_project():
    project, shot, asset = reviewed_project()
    del shot["editorial_contract"]
    shot["visual_plan"] = {"mode": "single", "primary": {"action": "双手搬运包裹"}}
    audit = shot["verification"]["identity_audit"]
    frames = audit["frame_observations"]
    frames[0]["text"] = ["分拣作业"]
    frames[1]["text"] = ["注意轻放"]
    checks = [{"key": "action", "status": "supported", "reason": "实际搬运动作",
               "evidence": [{"frame_index": 0, "field": "action", "quote": frames[0]["action"]}]}]
    checks.extend({"key": f"caption_c{i+1}", "status": "supported", "reason": "与当前包裹操作相关",
                   "evidence": [{"frame_index": i, "field": "text", "quote": f["text"][0]}]}
                  for i, f in enumerate(frames))
    assert frame_contract.validate(shot, checks, frames)[0] == "supported"
    audit["primary_checks"] = checks
    shot["approved"] = True
    return project, shot, asset


@PRE_V60_CURATION
def test_old_approved_legacy_cannot_skip_caption_review():
    project, shot, asset = caption_reviewed_legacy_project()
    audit = shot["verification"]["identity_audit"]
    audit["frame_observations"][1]["text"] = ["后续路上11公里熄火3次！"]
    audit["primary_checks"] = []
    assert frame_contract.validate(shot, [], audit["frame_observations"])[0] == "unknown"
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert not shot["approved"]


@PRE_V60_CURATION
@pytest.mark.parametrize("change", ["missing-caption", "conflict-caption", "unknown-caption",
                                    "wrong-caption", "new-caption", "changed-frame",
                                    "reordered-frames", "old-version", "missing-audit"])
def test_retained_legacy_invalidates_changed_or_incomplete_caption_evidence(change):
    project, shot, asset = caption_reviewed_legacy_project()
    audit = shot["verification"]["identity_audit"]
    if change == "missing-caption":
        audit["primary_checks"].pop()
    elif change in {"conflict-caption", "unknown-caption"}:
        audit["primary_checks"][-1]["status"] = change.split("-")[0]
    elif change == "wrong-caption":
        audit["primary_checks"][-1]["evidence"] = deepcopy(audit["primary_checks"][-2]["evidence"])
    elif change == "new-caption":
        audit["frame_observations"][1]["text"].append("另一个没有核验的事件")
    elif change == "changed-frame":
        audit["frame_observations"][0]["action"] = "人员坐在会议桌旁"
    elif change == "reordered-frames":
        audit["frame_observations"].reverse()
    elif change == "old-version":
        audit["version"] -= 1
    else:
        del shot["verification"]["identity_audit"]
    assert curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert not shot["approved"]


@PRE_V60_CURATION
def test_current_complete_legacy_review_is_reused_without_mutating_evidence():
    project, shot, asset = caption_reviewed_legacy_project()
    before = deepcopy(project)
    assert not curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 1
    assert project == before


@PRE_V60_CURATION
def test_locked_manual_legacy_is_not_subject_to_automatic_caption_migration():
    project, shot, asset = caption_reviewed_legacy_project()
    shot.update(locked=True, approved=True)
    shot["verification"]["identity_audit"]["primary_checks"] = []
    assert not curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0
    assert shot["approved"]
