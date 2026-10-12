"""Passed windows authorize bounded new reads, never new visual verdicts."""
from collections import defaultdict
from copy import deepcopy

from jinggao import config, library, library_match, media, paths, planner, pooled_match, source_expansion, store


def reviewed(shot_id="seed", asset_id="asset"):
    return {"id": shot_id, "selected_asset": asset_id, "duration": 8, "clip_start": 0,
            "match_preview": {"asset_id": asset_id}, "verification": {
                "verdict": "match", "selection_blocked": False, "needs_source_check": False, "frame_count": 5,
                "visual_quality": {"quality_status": "not_detected", "representation": "live_action",
                                   "frame_sha256": [f"{i:064x}" for i in range(5)]},
                "observations": {"observation_method": "blind_actual_frames_v1",
                                 "scene_description": "车间中工人操作设备并搬动产品。",
                                 "frame_observations": [{"frame_index": i} for i in range(5)]}}}


def asset(asset_id="asset", url="https://www.iqiyi.com/v_19rwi46v84.html"):
    return {"id": asset_id, "kind": "video", "source_url": url, "provider": "iqiyi", "title": "职业技能介绍",
            "source_clip_start": 28.48, "duration": 8.12, "width": 1280, "height": 720,
            "path": "assets/existing.mp4", "media_streams": [{"url": "must-not-copy-local-session-url"}]}


def test_source_seed_reopens_public_origin_not_cropped_local_asset():
    p = {"shots": [reviewed()], "assets": [asset()], "min_video_height": 720}
    before = deepcopy(p)
    seeds = source_expansion.seed_candidates(p)
    assert len(seeds) == 1 and seeds[0]["duration"] == 0 and not seeds[0]["metadata_checked"]
    assert seeds[0]["download_status"] == "unchecked" and seeds[0]["width"] == seeds[0]["height"] == 0
    assert not {"asset_id", "path", "media_streams", "verification", "approved"} & seeds[0].keys()
    assert seeds[0]["source_expansion"]["seed_windows"] == [
        {"shot_id": "seed", "asset_id": "asset", "start": 28.48, "end": 36.48}]
    assert "工人操作设备" in source_expansion.topic_record(seeds[0])["description"] and p == before


def test_only_current_complete_passes_become_seeds_and_quality_hook_can_reject():
    p = {"shots": [reviewed()], "assets": [asset()]}
    assert source_expansion.seed_candidates(p, seed_allowed=lambda shot, asset: False) == []
    for key, value in [("verdict", "uncertain"), ("selection_blocked", True),
                       ("needs_source_check", True), ("frame_count", 4)]:
        wrong = deepcopy(p)
        wrong["shots"][0]["verification"][key] = value
        assert source_expansion.seed_candidates(wrong) == []
    wrong = deepcopy(p)
    wrong["shots"][0]["match_preview"]["asset_id"] = "different-window"
    assert not source_expansion.seed_candidates(wrong)
    wrong = deepcopy(p)
    wrong["shots"][0]["verification"]["observations"]["observation_method"] = "title_only"
    assert not source_expansion.seed_candidates(wrong)


def test_quality_seed_hook_rejects_missing_uncertain_obstructed_and_animated_evidence():
    shot = reviewed()
    assert source_expansion.quality_seed_allowed(shot, asset())
    quality = shot["verification"]["visual_quality"]
    for change in [{"quality_status": "uncertain"}, {"quality_status": "reject"},
                   {"representation": "animation"}, {"representation": "mixed"},
                   {"representation": "uncertain"}, {"frame_sha256": quality["frame_sha256"][:4]}]:
        candidate = deepcopy(shot)
        candidate["verification"]["visual_quality"].update(change)
        assert not source_expansion.quality_seed_allowed(candidate, asset())
    del shot["verification"]["visual_quality"]
    assert not source_expansion.quality_seed_allowed(shot, asset())
    assert not source_expansion.seed_candidates({"shots": [shot], "assets": [asset()]},
                                               seed_allowed=source_expansion.quality_seed_allowed)


def test_seed_urls_deduplicate_and_source_count_stays_bounded():
    assets = [asset(str(i), f"https://www.iqiyi.com/v_public{i}.html") for i in range(7)]
    shots = [reviewed(str(i), str(i)) for i in range(7)]
    shots.append(reviewed("repeat", "0") | {"clip_start": 20})
    seeds = source_expansion.seed_candidates({"shots": shots, "assets": assets})
    assert len(seeds) == source_expansion.SOURCE_LIMIT == 4
    assert len(seeds[0]["source_expansion"]["seed_windows"]) == 2


def test_expansion_moves_beyond_nearest_two_probes_without_repeating_ranges():
    candidate = {"duration": 180}
    verified, used = [(60, 68)], [(60, 68)]
    before = deepcopy(used)
    first = source_expansion.neighbour_windows(candidate, verified, used, [8])
    second = source_expansion.neighbour_windows(candidate, verified, used, [8],
                                                probed=[(w["start"], w["end"]) for w in first])
    assert len(first) == len(second) == 4 and used == before
    all_ranges = sorted([*used, *((w["start"], w["end"]) for w in first + second)])
    assert all(a[1] <= b[0] for a, b in zip(all_ranges, all_ranges[1:], strict=False))
    assert all(0 <= w["start"] < w["at"] < w["end"] < 180 for w in first + second)
    assert not source_expansion.neighbour_windows(candidate, [], [], [8])
    assert not source_expansion.neighbour_windows(candidate, verified, [(0, 180)], [8])


def test_neighbour_slots_rotate_between_related_proven_sources():
    shots = [{"id": str(i), "duration": 8, "query": "车间 工人"} for i in range(3)]
    candidates = [{"id": str(i), "source_url": f"https://v.qq.com/{i}", "kind": "video", "title": "车间工人",
                   "width": 1280, "height": 720, "download_status": "available", "sample_points": [20]}
                  for i in range(2)]
    pending = {i: {"group": {"indices": [i], "queries": ["车间 工人"]}, "shots": [s],
                   "choices": {c["id"]: c for c in candidates}} for i, s in enumerate(shots)}
    chosen, budgeted = pooled_match.allocate_sampling_candidates({}, pending, shots)
    assert [chosen[i][0]["id"] for i in range(3)] == ["0", "1", "0"] and len(budgeted) == 2
    assert all(len(rows) == 1 for rows in chosen.values())


def test_local_mode_does_not_reopen_remote_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "local"})
    project = store.new_project("只用本地", "工人操作。工人搬运产品。", media_policy="video_only")
    new = planner.new_shot("工人搬运产品。", query="车间 工人")
    old = planner.new_shot("工人操作。") | reviewed()
    store.mutate(project["id"], lambda p: p.update(shots=[old, new], assets=[asset()]))
    project = store.get_project(project["id"])
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{
        "label": "车间劳动", "indices": [0], "queries": ["车间 工人"]}], {}))
    monkeypatch.setattr(pooled_match.sources, "local_candidates", lambda *a: [])
    def remote_forbidden(*a, **kw):
        raise AssertionError("本地模式不得重新打开远端来源")
    monkeypatch.setattr(source_expansion, "seed_candidates", remote_forbidden)
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", remote_forbidden)
    stats = pooled_match.match(project, [new], lambda: None, lambda *a: None)
    assert stats["source_expansion_seeds"] == [] and stats["downloads"] == 0


def test_library_source_absent_from_search_is_sampled_and_fully_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("来源扩展", "工人操作。工人在车间搬运产品。", media_policy="video_only")
    old = planner.new_shot("工人操作。") | reviewed()
    new = planner.new_shot("工人在车间搬运产品。", query="车间 工人", duration=8)
    new["duration"] = 8
    store.mutate(project["id"], lambda p: p.update(shots=[old, new], assets=[asset()]))
    project = store.get_project(project["id"])
    monkeypatch.setattr(library_match, "match", lambda *a: ({old["id"]}, {
        "library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{
        "label": "车间劳动", "indices": [0], "queries": ["车间 工人"]}], {}))
    monkeypatch.setattr(pooled_match.video_sources, "search_videos", lambda *a, **kw: ([], []))
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    inspections, samples, downloads, reviews = [], [], [], []
    def inspect(candidate, check):
        inspections.append(candidate)
        assert not candidate.get("asset_id") and candidate["duration"] == 0
        return candidate | {"download_status": "available", "metadata_checked": True,
                            "duration": 95, "width": 1280, "height": 720}
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", inspect)
    def discover(project, rows, *a, **kw):
        assert len(rows) == 1 and len(rows[0][1]) <= 3
        target, candidate = rows[0][0][0], rows[0][1][0]
        samples.extend(candidate["sample_points"])
        at = candidate["sample_points"][0]
        return {target["id"]: [{"candidate_id": candidate["id"], "clip_start": at-4, "sample_at": at}]}, {}, {
            "model_requests": 1, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", discover)
    def download(candidate, target, start, duration, check):
        downloads.append((start, duration))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"mock fresh source window")
        return {"source_clip_start": start, "download_mode": "fixture", "downloaded_bytes": 24}
    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate", download)
    monkeypatch.setattr(media, "inspect_asset", lambda pid, target, title: {
        "id": "new-asset", "path": str(target), "kind": "video", "duration": 8.12, "width": 1280, "height": 720})
    def review(project, rows, check):
        reviews.extend(row["shot"]["id"] for row in rows)
        return [row | {"preview": {}, "verification": reviewed()["verification"] | {
            "reason": "新窗口独立完整帧通过", "identity_evidence": {"reason": "通用场景"}}} for row in rows], [], {"model_requests": 1}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(library, "remember_window", lambda *a, **kw: "cached")
    stats = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    final = store.get_project(project["id"])
    assert len(inspections) == 1 and len(samples) == 4 and reviews == [new["id"]]
    assert stats["selected"] == 2 and len(downloads) == 1
    assert downloads[0][0] >= 36.48 and final["shots"][0]["selected_asset"] == "asset"
    assert final["shots"][1]["selected_asset"] == "new-asset" and not final["shots"][1]["approved"]
    assert stats["adjacent_sampling"][0]["phase"] == "initial"
    # Later successful reviews must not rewrite the earlier authorization log.
    assert {(round(a, 3), round(b, 3)) for a, b in stats["adjacent_sampling"][0]["verified_intervals"]} == {(28.48, 36.48)}
    assert stats["source_expansion_seeds"][0]["seed_windows"][0]["asset_id"] == "asset"
    # A reservation remains specific to the newly sampled vicinity.
    used = defaultdict(list, {asset()["source_url"]: [(28.48, 36.48)]})
    assert pooled_match.reserve_sampled({"id": "source", "source_url": asset()["source_url"], "duration": 95},
                                        {"sample_at": 32, "clip_start": 28}, 8, used) is None
