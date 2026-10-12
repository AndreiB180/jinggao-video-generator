"""Incomplete model replies must not trigger another paid source search."""
from copy import deepcopy
import json

import pytest

from jinggao import (config, frame_fallback, library, library_match, media, montage_candidates,
                     paths, pixel_observer, pixel_packet, plan_reselection, planner,
                     pooled_match, project_pool, scene_recall, scene_windows, store, video_sources)
from test_pixel_observer import environment as _pixel_environment, joint_group

environment = _pixel_environment


@pytest.mark.parametrize("bad", ["missing_frame", "missing_group", "truncated"])
def test_audit_preserves_complete_sibling_and_marks_only_missing_observation(environment, monkeypatch, bad):
    cfg, records = environment
    cfg["pixel_observer_mode"] = "joint"
    records = records[:2]
    records[0]["frame_paths"] *= 2
    calls = []
    body = pixel_packet.pack({"groups": [joint_group("g0", 2), joint_group("g1")]})
    if bad == "missing_frame":
        body["groups"][1]["frames"] = []
    elif bad == "missing_group":
        body["groups"].pop()
    def request(*args):
        calls.append(1)
        return json.dumps(body), {"finish_reason": "length" if bad == "truncated" else "stop"}
    monkeypatch.setattr(pixel_observer, "_request", request)
    # A visibly obstructed complete sibling is a real content rejection; the
    # missing sibling is a transport/schema failure. Neither reaches text chat.
    if bad != "truncated":
        for frame in body["groups"][0]["frames"]:
            frame["overlays"][0].update(kind="sales_promotion", impact="material",
                                      occludes="工作人员和纸箱", text="购买无水印素材")
    monkeypatch.setattr(planner, "chat", lambda *a, **k: pytest.fail("Incomplete pixels cannot reach text review"))
    result = planner.audit_selected_frames(records, lambda: None)
    assert len(calls) == 1
    assert result["private-1"]["status"] == "unknown"
    assert result["private-1"]["failure_stage"] == "pixel_observation"
    assert result["private-1"]["frame_observations"] == []
    if bad == "truncated":
        assert result["private-0"]["failure_stage"] == "pixel_observation"
    else:
        assert result["private-0"]["status"] == "conflict"
        assert result["private-0"]["frame_observations"]
        assert "failure_stage" not in result["private-0"]


@pytest.mark.parametrize("failed_observation", [True, False])
@pytest.mark.parametrize("failed_stage", ["pixel_observation", "evidence_review"])
def test_normal_match_defers_model_failure_but_keeps_real_mismatch_rescue(tmp_path, monkeypatch, failed_observation, failed_stage):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("模型错误与素材不符分开处理", "工作人员扫描包裹。", media_policy="video_only")
    shot = planner.new_shot("工作人员扫描包裹。", query="扫描包裹", duration=8)
    asset = {"id": "local-asset", "path": "assets/local.mp4", "kind": "video", "title": "扫描包裹",
             "duration": 90, "width": 1280, "height": 720}
    store.mutate(project["id"], lambda p: p.update(shots=[shot], assets=[asset]))
    project = store.get_project(project["id"])
    group = {"label": "扫码", "indices": [0], "queries": ["扫描包裹"]}
    candidate = asset | {"id": "candidate", "asset_id": asset["id"], "provider": "local",
                         "source_url": "https://example.com/scanning", "download_status": "available"}
    searches, discoveries, reviews, rescue_targets, montage_targets = [], [], [], [], []
    monkeypatch.setattr(project_pool, "collect", lambda *a: ([], {}))
    monkeypatch.setattr(library_match, "match", lambda *a, **k: (set(), {
        "library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(montage_candidates, "match", lambda *a, **k: (set(), {"model_requests": 0, "failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([deepcopy(group)], {"cache_hit": True}))
    monkeypatch.setattr(scene_recall, "candidates", lambda *a: {})
    def search(query, **kwargs):
        searches.append(query)
        return [deepcopy(candidate)], []
    monkeypatch.setattr(video_sources, "search_videos", search)
    monkeypatch.setattr(planner, "rank_candidates", lambda p, s, rows, check: [{"candidate_id": c["id"]} for c in rows])
    def discover(p, groups, *args):
        discoveries.append(1)
        return {}, {}, {"model_requests": 0, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", discover)
    def rescue(groups, shots, remaining, found):
        rescue_targets.extend(s["id"] for s in remaining)
        return {}
    monkeypatch.setattr(pooled_match, "online_rescue_queries", rescue)
    monkeypatch.setattr(scene_windows, "align", lambda p, block, asset, used, check: (
        block["items"], {"adjusted": [], "error": ""}))
    monkeypatch.setattr(library, "remember_window", lambda *a, **k: None)
    monkeypatch.setattr(plan_reselection, "existing_windows", lambda *a: {})
    monkeypatch.setattr(plan_reselection, "remember", lambda *a: None)
    monkeypatch.setattr(frame_fallback, "match", lambda *a, **k: pytest.fail("Video-only job cannot buy images"))
    monkeypatch.setattr(video_sources, "download_video_candidate", lambda *a, **k: pytest.fail("Retain existing download"))
    monkeypatch.setattr(media, "inspect_asset", lambda *a: pytest.fail("Retain existing asset"))
    def review(p, items, check):
        reviews.extend(row["shot"]["id"] for row in items)
        verdict = {"verdict": "uncertain" if failed_observation else "mismatch", "scene_alignment": "uncertain",
                   "reason": "观察未完成" if failed_observation else "实际动作不符合口播",
                   "selection_blocked": True, "needs_source_check": True,
                   "identity_evidence": {"reason": "无人物身份要求"},
                   "identity_audit": {"failure_stage": failed_stage} if failed_observation else {}}
        return [row | {"verification": deepcopy(verdict), "preview": {"frames": []}} for row in items], [], {"model_requests": 1}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    def montage(p, remaining, *a, **k):
        montage_targets.extend(s["id"] for s in remaining)
        return set(), {"model_requests": 0, "failures": []}
    monkeypatch.setattr(montage_candidates, "match", montage)
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None, workers=1)
    final = store.get_project(project["id"])
    assert reviews == [shot["id"]]
    assert not final["shots"][0]["selected_asset"] and not final["shots"][0]["approved"]
    assert len(final["assets"]) == 1 and result["unmatched"] == [shot["id"]]
    assert result["selected"] == 0 and result["model_requests"] == 2  # One mocked rank, one review.
    if failed_observation:
        assert len(discoveries) == 1 and not rescue_targets and not montage_targets
        assert result["deferred_model_reviews"][0]["shot_id"] == shot["id"]
        assert "停止该镜的付费补搜" in final["shots"][0]["search_note"]
    else:
        assert len(discoveries) == 2 and rescue_targets == montage_targets == [shot["id"]]
        assert not result.get("deferred_model_reviews")
