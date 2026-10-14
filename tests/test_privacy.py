from copy import deepcopy

from PIL import Image
import pytest

from jinggao import curation, media, paths, planner, privacy, store


def test_phone_detection_does_not_confuse_short_service_codes_or_statistics():
    assert privacy.exposed_frames([{"frame_index": 2, "text": ["手机号 138 0013 8000"]}]) == [2]
    assert privacy.exposed_frames([{"frame_index": 0, "text": ["热线11183", "日均处理量150万件", "2024年业务量1750.8亿件"]}]) == []
    assert privacy.exposed_frames([{"frame_index": 0, "text": ["电话138****8000"]}]) == []


def test_phone_page_without_transcribed_number_still_requires_crop_scan():
    result = {"observations": {"frame_observations": [{"frame_index": 0, "objects": "手持手机显示物流页面",
        "action": "屏幕朝向镜头", "text": ["已签收", "2022-03-30 15:26:55"]}]}}
    assert privacy.needs_scan(result)
    assert "局部放大" in privacy.selection_error({"verification": result})
    result["privacy_scan"] = privacy.scan_stamp(result)
    assert not privacy.needs_scan(result)
    result["observations"]["frame_observations"][0]["text"].append("新的屏幕内容")
    assert privacy.needs_scan(result)


def test_wide_phone_mask_cannot_keep_a_prior_content_approval():
    shot = {"id": "shot", "selected_asset": "asset", "approved": True,
            "verification": {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
                             "needs_source_check": False}}
    asset = {"id": "asset", "kind": "video", "privacy_redaction": {
        "regions": [[.075, .14, .795, .54]]}}
    project = {"shots": [shot], "assets": [asset]}
    assert "影响主体" in curation.selection_error(project, shot, asset)
    assert curation.approve_verified(project) == 0 and not shot["approved"]


@pytest.mark.parametrize("response", [
    {"frames": []}, {"frames": [{"frame_index": 0, "boxes": []}]},
    {"frames": [{"frame_index": 0, "boxes": [[0, 0, 1, 1]]}]},
    {"frames": [{"frame_index": 0, "boxes": [[.6, .1, .2, .2]]}]},
])
def test_invalid_or_missing_phone_region_cannot_produce_a_safe_asset(response):
    with pytest.raises(ValueError):
        privacy.validate_boxes(response, 1, {0})


def test_native_redaction_changes_pixels_but_preserves_original_and_requires_new_review(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("隐私遮挡")
    target = paths.project_dir(project["id"])/"assets"/"source.jpg"
    target.parent.mkdir()
    Image.new("RGB", (800, 800), "white").save(target)
    asset = media.inspect_asset(project["id"], target, "页面资料")
    asset.update(source_url="https://v.qq.com/x/page/sample.html", source_clip_start=41.5)
    before = target.read_bytes()
    row = {"shot": {"id": "shot", "duration": 7, "clip_start": 0}, "asset": asset,
        "frame_paths": [target], "verification": {"observations": {"frame_observations": [
            {"frame_index": 0, "text": ["电话13800138000"]}]}}}
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: ({"frames": [
        {"frame_index": 0, "boxes": [[.2, .2, .8, .3]]}]}, {}))
    edited, calls = privacy.redact(project["id"], row, lambda: None)
    assert calls == 1 and edited["asset"]["id"] != asset["id"]
    assert edited["asset"]["source_clip_start"] == 41.5
    assert edited["asset"]["privacy_redaction"]["requires_fresh_review"]
    assert target.read_bytes() == before
    with Image.open(paths.file_path(project["id"], edited["asset"]["path"])) as picture:
        assert picture.size == (800, 800)
        assert max(picture.getpixel((400, 200))) < 24  # FFmpeg's limited-range black may decode as 16.
        assert min(picture.getpixel((50, 50))) > 230
    assert "13800138000" not in str(edited["asset"])


@pytest.mark.parametrize("current_audit", [False, True])
def test_old_approval_and_new_audit_cannot_ignore_readable_phone(current_audit):
    audit = {"status": "supported", "scene_description": "手机页面", "frame_observations": [
        {"frame_index": 0, "objects": "手机", "action": "通话中", "text": ["13800138000"]}]}
    verdict = {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
               "needs_source_check": False}
    checked = planner.apply_identity_audit(verdict, audit)
    assert checked["selection_blocked"] and checked["verdict"] == "uncertain"
    if current_audit:
        audit.update(version=planner.FRAME_AUDIT_VERSION, primary_checks=[{
            "key": "caption_c1", "status": "supported", "reason": "模拟内容审核已完成；隐私仍须独立检查",
            "evidence": [{"frame_index": 0, "field": "text", "quote": audit["frame_observations"][0]["text"][0]}]}])
    shot = {"id": "shot", "selected_asset": "asset", "duration": 8, "approved": True,
            "verification": verdict | {"identity_audit": deepcopy(audit)}}
    project = {"shots": [shot], "assets": [{"id": "asset", "kind": "image"}]}
    assert curation.approve_verified(project) == 0 and not shot["approved"]
    # v60 reports the privacy blocker first for both old and current audits.
    with pytest.raises(ValueError, match="手机号"):
        curation.validate_export(project)


def test_same_number_at_two_positions_keeps_both_masks_and_final_scan_can_reject(monkeypatch, tmp_path):
    target = tmp_path/"page.jpg"
    Image.new("RGB", (720, 1280), "white").save(target)
    frames = [{"frame_index": 0, "objects": "手机物流页面", "action": "屏幕展示联系人和物流记录", "text": []}]
    row = {"shot": {"id": "s", "duration": 4, "clip_start": 0},
           "asset": {"id": "masked", "kind": "image", "privacy_redaction": {"version": 3}},
           "frame_paths": [target], "verification": {"observations": {"frame_observations": frames}}}
    response = {"frames": [{"frame_index": 0, "boxes": [[.25, .25, .6, .28], [.25, .6, .6, .63]]}]}
    assert len(privacy.validate_boxes(response, 1, set())) == 2
    def chat(system, content, **options):
        assert "不同物理位置" in system and "不能按号码内容去重" in system
        assert len([r for r in content if r["type"] == "image_url"]) == 5
        return response, {}
    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(ValueError, match="仍检出联系方式"):
        privacy.redact("unused", row, lambda: None, allow_redaction=False)


def test_post_mask_scene_approval_cannot_skip_the_enlarged_privacy_rescan(monkeypatch):
    from jinggao import pooled_match
    row = {"shot": {"id": "s", "duration": 4, "clip_start": 0}, "asset": {"id": "source"},
           "preview": {"frames": []}, "verification": {"verdict": "match", "selection_blocked": False,
               "observations": {"frame_observations": [{"frame_index": 0, "objects": "手机屏幕",
                    "action": "显示物流页面", "text": []}]}}}
    calls = []
    monkeypatch.setattr(pooled_match, "_review_batch", lambda p, rows, check: (deepcopy(rows), [], {}))
    def redact(pid, current, check, *, request_started, allow_redaction=True):
        request_started()
        calls.append(allow_redaction)
        if not allow_redaction:
            raise ValueError("仍有第二处号码")
        return current | {"asset": {"id": "derivative-"+str(len(calls))}}, 1
    monkeypatch.setattr(privacy, "redact", redact)
    reviewed, errors, info = pooled_match.review_batch({"id": "p"}, [row], lambda: None)
    assert calls == [True, True, False] and info["model_requests"] == 3
    assert not errors and reviewed[0]["verification"]["selection_blocked"]
    assert reviewed[0]["verification"]["verdict"] == "uncertain"
