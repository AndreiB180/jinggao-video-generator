import pytest

from jinggao import config, library, library_match, media, paths, planner, pooled_match, store


@pytest.mark.parametrize("first_matches", [False, True])
@pytest.mark.parametrize("allow_neighbour", [False, True])
def test_repair_releases_rejected_windows_but_preserves_selected_ones(
        tmp_path, monkeypatch, first_matches, allow_neighbour):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("共享片段占用回归", "扫码。运输。", media_policy="video_only")
    shots = [planner.new_shot("工作人员扫描包裹。", query="快递"),
             planner.new_shot("车辆运输包裹。", query="快递")]
    for shot in shots:
        shot["duration"] = 4
    store.mutate(project["id"], lambda p: p.update(shots=shots))
    project = store.get_project(project["id"])
    candidate = {"id": "source", "kind": "video", "title": "快递工作",
                 "source_url": "https://v.qq.com/x/page/shared.html", "provider": "tencent",
                 "duration": 90, "width": 1280, "height": 720, "download_status": "available"}
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {
        "library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{
        "label": "快递", "indices": [0, 1], "queries": ["快递"]}], {}))
    monkeypatch.setattr(pooled_match.video_sources, "search_videos", lambda *a, **kw: ([candidate], []))
    monkeypatch.setattr(planner, "rank_candidates", lambda *a: [{"candidate_id": "source"}])
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    calls, sampled = [], []
    def discovery(project, group_rows, *args):
        proposals = {}
        for members, candidates in group_rows:
            assert len(candidates) <= 3
            for candidate in candidates:
                points = candidate.get("sample_points", [])
                sampled.append(points)
                at = points[0] if points and allow_neighbour else 22
                proposals.update({s["id"]: [{"candidate_id": "source", "clip_start": at-2, "sample_at": at}]
                                  for s in members})
        return proposals, {}, {"model_requests": 0, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", discovery)
    def download(candidate, target, start, duration, check):
        calls.append((start, duration))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"mock-video")
        return {"source_clip_start": start, "download_mode": "fixture", "downloaded_bytes": 10}
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate", download)
    monkeypatch.setattr(media, "inspect_asset", lambda pid, target, title: {
        "id": store.uid(), "kind": "video", "title": title, "path": "assets/" + target.name,
        "duration": 4.12, "width": 1280, "height": 720})
    reviews = []
    def review(project, rows, check):
        result = []
        for row in rows:
            sid = row["shot"]["id"]
            reviews.append(sid)
            good = first_matches if sid == shots[0]["id"] else True
            result.append(row | {"preview": {}, "verification": {
                "verdict": "match" if good else "mismatch", "selection_blocked": not good,
                "needs_source_check": False, "reason": "本镜实际帧结果",
                "visual_quality": {"quality_status": "not_detected", "representation": "live_action",
                                   "frame_sha256": [f"{i:064x}" for i in range(5)]},
                "identity_evidence": {"reason": "通用画面"}}})
        return result, [], {"model": "fixture"}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(library, "remember_window", lambda *a, **kw: "clip")
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    final = store.get_project(project["id"])
    repaired_neighbour = first_matches and allow_neighbour
    assert result["selected"] == 1 + repaired_neighbour and len(calls) == 1 + repaired_neighbour
    assert bool(final["shots"][0]["selected_asset"]) is first_matches
    assert bool(final["shots"][1]["selected_asset"]) is (not first_matches or repaired_neighbour)
    assert reviews == [s["id"] for s in shots[:1 if first_matches and not allow_neighbour else 2]]
    assert not any(s["approved"] for s in final["shots"])
    assert bool(result.get("adjacent_sampling")) is first_matches
    if first_matches:
        assert any(sampled) and all(row["sampling_only"] for row in result["adjacent_sampling"])
    if repaired_neighbour:
        original, repaired = calls
        assert repaired[0] >= original[0] + 4
