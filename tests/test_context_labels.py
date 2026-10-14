"""Archival disclosure may scope true sources; it cannot approve a scene."""
from copy import deepcopy
import json

import pytest

from jinggao import context_labels


def test_relative_processing_statistic_requires_report_date_even_when_ocr_lines_split():
    assert context_labels.relative_statistics([{"text": ["预计今年双十一分拣高峰期", "日均处理量150万件"]}])
    assert not context_labels.relative_statistics([{"text": ["资料画面", "分拣设备正在处理包裹"]}])


def source_context(tmp_path, *, historical=False):
    source = "https://www.chinanews.com.cn/shipin/cns-d/2024/09-04/news998993.shtml"
    title = "【聚焦中非合作】埃航电商库实现自动化分拣 提升非洲居民消费便利度"
    published = "2024-09-04 11:39:15"
    text = ["中新视频", "实现了跨境电商包裹的自动化分拣", "年处理货物能力15万吨"]
    action = "拍摄空置厂房内设备与平台"
    if historical:
        source = "https://www.chinanews.com/cj/shipin/cns-d/2021/11-11/news906908.shtml"
        title = "内蒙古快递初冬“大考”：冰雪和疫情下的“双十一”"
        published = "2021-11-11 19:08:42"
        text = ["中新视频", "所有货物进仓前都要经过消杀程序"]
        action = "男子在消杀设备旁操作，雾气持续喷出"
    shot = {"id": "shot", "narration": "说明设备与劳动。", "clip_start": 1.08,
            "duration": 10.0, "overlay_label": "资料画面", "visual_origin": "generated"}
    asset = {"id": "asset", "path": "assets/clip.mp4", "source_url": source, "source_clip_start": 12.04,
             "source_verified": True, "source_evidence": {"url": source, "title": title,
                 "description": "公开报道中的实际设施与操作。", "published_at": published,
                 "publisher": "中国新闻网", "readable": True, "evidence_origin": "publisher_page"}}
    frames = [{"frame_index": 0, "objects": "厂房里的设备与平台", "action": action, "text": text},
              {"frame_index": 1, "objects": "厂房里的设备与平台", "action": action, "text": text}]
    files = [tmp_path / "01.jpg", tmp_path / "02.jpg"]
    for i, path in enumerate(files):
        path.write_bytes(f"unchanged original frame bytes {i}".encode())
    context = context_labels.review_context(shot, asset, frames, frame_paths=files)
    return shot, asset, frames, files, context


def proposal(context, frames, *, historical=False):
    parts = ["内蒙古", "双十一"] if historical else ["埃航电商库"]
    return {"text": ("2021年报道·" if historical else "") + "·".join(parts) + "资料（来源中国新闻网）",
            "source_quotes": [{"field": "title", "quote": text} for text in parts],
            "report_period": "year" if historical else "none",
            "frame_evidence": [{"frame_index": 0, "field": "action", "quote": frames[0]["action"]},
                               {"frame_index": 0, "field": "text", "quote": frames[0]["text"][1]}],
            "caption_checks": [{"caption_id": c["caption_id"],
                                "status": "compatible" if c["text"] == "中新视频" else "scope_resolved",
                                "scope": "" if c["text"] == "中新视频" else "time" if historical else "project",
                                "reason": "来源标志" if c["text"] == "中新视频" else "准确限定来源报道的范围"}
                               for c in context["captions"]]}


def ordinary_caption_checks(frames):
    # v60 has no per-caption caption_cN obligations (context_labels.frame_captions does not exist);
    # these legacy shots carry no editorial contract or visual plan, so no primary checks are required.
    return []


@pytest.mark.parametrize("historical", [False, True])
def test_real_source_scope_is_extractive_and_never_asserts_capture_or_running(tmp_path, historical):
    shot, asset, frames, files, context = source_context(tmp_path, historical=historical)
    before = deepcopy((shot, asset, frames))
    suggestion = proposal(context, frames, historical=historical)
    label = context_labels.compile_label(context, suggestion, frames)
    assert label["text"] == suggestion["text"] and len(label["text"]) <= 30
    assert label["requires_content_review"] is True
    assert "status" not in label and "approved" not in label and "verdict" not in label
    assert (shot, asset, frames) == before
    assert context["captions"][1]["frame_indices"] == [0, 1]
    assert context["overlay_label"] == "资料画面"
    if historical:
        assert label["text"].startswith("2021年报道")
        assert "拍摄" not in label["text"] and "当下" not in label["text"]
    else:
        assert "运行" not in label["text"] and "国内" not in label["text"]
        # The supplied source calls it 埃航. An imagined expansion is not a quote.
        fabricated = deepcopy(suggestion)
        fabricated["source_quotes"][0]["quote"] = "埃塞俄比亚航空电商库"
        with pytest.raises(ValueError, match="逐字"):
            context_labels.compile_label(context, fabricated, frames)
    installed = shot | {"overlay_label": label["text"]}
    refreshed = context_labels.review_context(installed, asset, frames, frame_paths=files)
    assert context_labels.is_current(label, refreshed)
    with pytest.raises(ValueError, match="完整标签"):
        context_labels.compile_label(context, suggestion | {"text": "资料画面"}, frames)


def test_label_binding_expires_for_other_page_window_text_and_actual_pixels(tmp_path):
    shot, asset, frames, files, context = source_context(tmp_path)
    label = context_labels.compile_label(context, proposal(context, frames), frames)
    installed = shot | {"overlay_label": label["text"]}
    changes = [(installed | {"clip_start": 1.4}, asset, frames),
               (installed | {"overlay_label": "国内现场"}, asset, frames),
               (installed, asset | {"source_url": "https://www.chinanews.com.cn/other.shtml"}, frames),
               (installed, asset, [frames[0] | {"text": ["另一项目"]}, frames[1]])]
    for changed_shot, changed_asset, changed_frames in changes:
        updated = context_labels.review_context(changed_shot, changed_asset, changed_frames, frame_paths=files)
        assert not context_labels.is_current(label, updated)
    files[0].write_bytes(b"replaced original pixels")
    assert not context_labels.is_current(label, context_labels.review_context(installed, asset, frames,
                                                                            frame_paths=files))
    no_pixels = context_labels.review_context(shot, asset, frames)
    with pytest.raises(ValueError, match="实际帧"):
        context_labels.compile_label(no_pixels, proposal(no_pixels, frames), frames)
    unrelated_page = deepcopy(asset)
    unrelated_page["source_evidence"]["url"] = "https://www.chinanews.com.cn/another.shtml"
    unbound = context_labels.review_context(shot, unrelated_page, frames, frame_paths=files)
    assert not unbound["source_bound"] and unbound["source_fields"] == {}
    with pytest.raises(ValueError, match="同一原页"):
        context_labels.compile_label(unbound, proposal(context, frames), frames)


def test_palace_restoration_badge_cannot_hide_unrelated_alliance_speech(tmp_path):
    shot, asset, frames, files, _ = source_context(tmp_path)
    source = "https://www.chinanews.com.cn/sh/shipin/cns-d/2025/11-17/news1035293.shtml"
    asset.update(source_url=source, source_clip_start=31.12)
    asset["source_evidence"].update(url=source,
        title="“东北地区古建筑遗产保护与修缮研究联盟”在沈阳成立",
        description="由沈阳故宫博物院、沈阳建筑大学联合发起。", published_at="2025-11-17 20:20:48")
    shot.update(narration="游客看到的是屋顶、门窗和颜色，保护工作还要面对材料、结构、环境以及时间留下的变化。",
                clip_start=0, duration=8)
    frames[0].update(action="工人在木构件下持工具作业", text=["资料", "故宫古建园林", "东北地区的世界文化遗产"])
    frames[1].update(action="工人继续持工具作业", text=["资料", "(希望在这)一个平台之上", "能够发挥我们共同研究的作用"])
    context = context_labels.review_context(shot, asset, frames, frame_paths=files)
    candidate = {"text": "古建筑遗产保护资料（来源中国新闻网）",
                 "source_quotes": [{"field": "title", "quote": "古建筑遗产保护"}], "report_period": "none",
                 "frame_evidence": [{"frame_index": 0, "field": "action", "quote": frames[0]["action"]},
                                    {"frame_index": 0, "field": "text", "quote": "资料"}],
                 # Mirrors the failed real audit: only the convenient badge and uniform words.
                 "caption_checks": [{"caption_id": c["caption_id"], "status": "compatible", "scope": "",
                                     "reason": "资料或工装文字"} for c in context["captions"][:2]]}
    with pytest.raises(ValueError, match="全部原片字幕"):
        context_labels.compile_label(context, candidate, frames)
    candidate["caption_checks"] = [{"caption_id": c["caption_id"], "scope": "",
        "status": "unrelated_story" if "平台" in c["text"] or "共同研究" in c["text"] else "compatible",
        "reason": "联盟共同研究叙事不能靠建筑修缮标签消除"} for c in context["captions"]]
    with pytest.raises(ValueError, match="无关故事"):
        context_labels.compile_label(context, candidate, frames)
    # A source-page mention of Shenyang does not establish the pictured building.
    invented_place = deepcopy(candidate)
    invented_place["source_quotes"] = [{"field": "description", "quote": "北京故宫修缮"}]
    with pytest.raises(ValueError, match="逐字"):
        context_labels.compile_label(context, invented_place, frames)


def test_relative_statistics_date_comes_only_from_same_verified_source(tmp_path):
    shot, asset, frames, files, _ = source_context(tmp_path)
    frames[0]["text"] = ["当前快递业日均揽收和投递包裹量均超3亿件"]
    assert context_labels.relative_statistics(frames)
    context = context_labels.review_context(shot, asset, frames, frame_paths=files)
    label = context_labels.report_date_label(context, shot)
    assert label["text"] == "资料画面 · 2024年9月4日报道"
    assert label["requires_content_review"] and "approved" not in label
    assert context_labels.report_date_label(context | {"overlay_label": "2024年报道资料"}, shot) is None
    with pytest.raises(ValueError, match="已核实"):
        context_labels.report_date_label(context | {"source_bound": False}, shot)
    with pytest.raises(ValueError, match="原文叠字"):
        context_labels.report_date_label(context, shot | {"overlay_label_origin": "manual"})
    bad = deepcopy(context)
    bad["source_fields"]["published_at"] = "未知"
    with pytest.raises(ValueError, match="不能猜测"):
        context_labels.report_date_label(bad, shot)
    assert not context_labels.relative_statistics([frames[0] | {"text": ["目前正在处理快件"]}])


@pytest.mark.usefixtures("isolated_pixel_quality", "separate_pixel_observer")
@pytest.mark.parametrize("judgment", ["supported", "conflict"])
@pytest.mark.parametrize("manual", [False, True])
def test_replacing_source_clears_only_automatic_date_after_fresh_review(tmp_path, monkeypatch, judgment, manual):
    from jinggao import curation, neutral_observer, planner
    shot, asset, frames, files, _ = source_context(tmp_path)
    asset["kind"] = "video"
    old = "资料画面 · 2023年4月6日报道"
    shot.update(overlay_label=old, overlay_label_origin="manual" if manual else "source_context")
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: {
        "scene_description": "设备资料", "frames": frames}}, 0))
    def judge(system, content, **kwargs):
        data = json.loads(content[0]["text"])
        assert data["overlay_label"] == (old if manual else "资料画面")
        ref = next(r["ref"] for r in data["observation_references"] if r["field"] == "action")
        return {"reviews": [{"shot_id": shot["id"], "status": judgment, "reason": "逐帧核验",
            "labels": [], "evidence": [{"ref": ref}], "primary_checks": ordinary_caption_checks(frames)}]}, {}
    monkeypatch.setattr(planner, "chat", judge)
    audit = planner.audit_selected_frames([{"shot": shot, "asset": asset, "frame_paths": files}], lambda: None)[shot["id"]]
    verdict = planner.apply_identity_audit({"verdict": "match", "scene_alignment": "match",
        "selection_blocked": False, "needs_source_check": False}, audit)
    fields = curation.reviewed_label_fields(verdict)
    assert fields == ({"overlay_label": "资料画面", "overlay_label_origin": "generated"}
                      if judgment == "supported" and not manual else {})
    assert shot["overlay_label"] == old
    if not manual:
        assert "未绑定当前素材" in context_labels.selection_error(shot | {"verification": verdict}, asset)
    if fields:
        assert not context_labels.selection_error(shot | fields | {"verification": verdict}, asset)


@pytest.mark.usefixtures("isolated_pixel_quality", "separate_pixel_observer")
@pytest.mark.parametrize("judgment", ["supported", "conflict"])
def test_dated_label_is_audited_before_commit_and_does_not_override_conflict(tmp_path, monkeypatch, judgment):
    from jinggao import curation, neutral_observer, planner
    shot, asset, frames, files, _ = source_context(tmp_path)
    asset["kind"] = "video"
    frames[0]["text"] = ["当前快递业日均揽收和投递包裹量均超3亿件"]
    observed = {"scene_description": "分拣资料与业务量字幕", "frames": frames}
    monkeypatch.setattr(neutral_observer, "observe", lambda rows, check: ({shot["id"]: observed}, 0))
    def judge(system, content, **kwargs):
        data = json.loads(content[0]["text"])
        assert data["overlay_label"] == data["context_label_context"]["overlay_label"] == "资料画面 · 2024年9月4日报道"
        assert data["frame_observations"] == frames
        ref = next(r["ref"] for r in data["observation_references"] if r["field"] == "action")
        return {"reviews": [{"shot_id": shot["id"], "status": judgment, "reason": "独立判断场景与所有字幕",
            "labels": [], "evidence": [{"ref": ref}], "primary_checks": ordinary_caption_checks(frames)}]}, {}
    monkeypatch.setattr(planner, "chat", judge)
    audit = planner.audit_selected_frames([{"shot": shot, "asset": asset, "frame_paths": files}], lambda: None)[shot["id"]]
    assert audit["status"] == judgment
    verdict = planner.apply_identity_audit({"verdict": "match", "scene_alignment": "match",
        "selection_blocked": False, "needs_source_check": False}, audit)
    fields = curation.reviewed_label_fields(verdict)
    if judgment == "conflict":
        assert fields == {} and verdict["selection_blocked"]
        return
    installed = shot | fields | {"verification": verdict, "selected_asset": asset["id"]}
    assert not context_labels.selection_error(installed, asset)
    assert context_labels.selection_error(installed | {"clip_start": 2}, asset)
    assert context_labels.selection_error(installed | {"overlay_label": "资料画面"}, asset)
    stale_source = deepcopy(asset)
    stale_source["source_evidence"]["published_at"] = "2025-01-01"
    assert context_labels.selection_error(installed, stale_source)
    before = shot | {"verification": {"identity_audit": {"frame_observations": frames}}, "approved": True,
                     "selected_asset": asset["id"]}
    assert "报道时间" in context_labels.selection_error(before, asset)
    p = {"shots": [before], "assets": [asset]}
    curation.approve_verified(p)
    assert not before["approved"]
