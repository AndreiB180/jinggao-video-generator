from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest

from jinggao import downloaded_pool, library, media, paths, planner, pooled_match, scene_windows, store


@pytest.fixture
def setup_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("Reuse downloaded footage", "骑车前行。", media_policy="video_only")
    shots = [planner.new_shot("骑车前行。", query="骑车", duration=6) for _ in range(3)]
    for shot in shots:
        shot["duration"] = 6
    store.mutate(project["id"], lambda p: p.update(shots=shots))

    def add(ident="a", source="https://www.bilibili.com/video/BV1WYMVz2EGD", base=100, duration=30):
        file = paths.file_path(project["id"], f"assets/{ident}.mp4")
        file.parent.mkdir(exist_ok=True)
        file.write_bytes((ident+"fixture bytes").encode())
        asset = {"id": ident, "kind": "video", "path": f"assets/{ident}.mp4", "source_url": source,
                 "source_clip_start": base, "candidate_id": "candidate-"+ident,
                 "download_mode": "fixture", "duration": duration, "width": 1280, "height": 720}
        store.mutate(project["id"], lambda p: p["assets"].append(asset))
        return asset, file

    monkeypatch.setattr(scene_windows, "_scenes", lambda p, a, begin, end, check: [(begin, end)])
    return project["id"], shots, add


def test_overlapping_downloads_share_source_reservations_and_sources_get_turns(setup_pool):
    pid, shots, add = setup_pool
    a, _ = add()
    for n in range(15):
        add(f"duplicate-{n}")
    b, _ = add("other", "https://www.bilibili.com/video/BV1AT4y1n7sx")
    project = store.get_project(pid)
    used = {a["source_url"]: [(108, 118)]}
    before = deepcopy(project)
    windows, stats = downloaded_pool.plan(project, shots, used, lambda: None)
    by_source = defaultdict(list)
    for w in windows:
        c = w["clip"]
        by_source[w["asset"]["source_url"]].append((c["start"], c["start"]+c["duration"]))
    assert b["source_url"] in by_source
    assert all(not (x < 118 and y > 108) for x, y in by_source[a["source_url"]])
    for spans in by_source.values():
        ordered = sorted(spans)
        assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:], strict=False))
    assert stats["model_requests"] == 0 and project == before
    assert not any(s.get("approved") or s.get("selected_asset") for s in project["shots"])


def test_no_model_requests_when_only_complete_target_window_was_already_tried(setup_pool, monkeypatch):
    pid, shots, add = setup_pool
    asset, _ = add(duration=8)
    tried = {s["id"]: [(asset["source_url"], 100, 108)] for s in shots}
    def forbidden(*a, **kw):
        pytest.fail("No paid observation is allowed for an unassignable window")
    monkeypatch.setattr(downloaded_pool.segment_discovery, "observe_samples", forbidden)
    ready, stats = downloaded_pool.propose(store.get_project(pid), shots, {}, lambda: None,
        lambda *a: None, tried=tried)
    assert ready == [] and stats["windows"] == stats["model_requests"] == 0


def test_short_scenes_and_per_shot_attempts_filter_before_sampling(setup_pool, monkeypatch):
    pid, shots, add = setup_pool
    asset, _ = add()
    monkeypatch.setattr(scene_windows, "_scenes", lambda *a: [(100, 105), (105.08, 120), (120.08, 123)])
    tried = {shots[0]["id"]: [(asset["source_url"], 105, 120)]}
    windows, _ = downloaded_pool.plan(store.get_project(pid), shots, {}, lambda: None, tried=tried)
    assert len(windows) == 1
    assert windows[0]["clip"]["start"] == pytest.approx(105.08)
    assert windows[0]["shot_ids"] == [s["id"] for s in shots[1:]]


def test_one_short_shot_cannot_buy_twelve_duplicate_opportunities(setup_pool, monkeypatch):
    pid, shots, add = setup_pool
    add(duration=90)
    monkeypatch.setattr(scene_windows, "_scenes", lambda *a: [(100+n*7, 107+n*7) for n in range(12)])
    windows, stats = downloaded_pool.plan(store.get_project(pid), shots[:1], {}, lambda: None)
    assert stats["windows"] == len(windows) == 2


def test_changed_source_or_bytes_cannot_reuse_stale_locations(setup_pool):
    pid, shots, add = setup_pool
    asset, file = add()
    project = store.get_project(pid)
    windows, _ = downloaded_pool.plan(project, shots, {}, lambda: None)
    clip = windows[0]["clip"]
    proposals = {shots[0]["id"]: [{"candidate_id": clip["id"], "source_url": asset["source_url"],
                                 "sample_at": 8, "clip_start": 5}]}
    file.write_bytes(b"different content")
    with pytest.raises(ValueError, match="字节"):
        downloaded_pool.allocate(project, shots, windows, proposals, {}, lambda: None)
    samples, errors = downloaded_pool.samples(windows, lambda: None)
    assert samples == [] and errors


def test_allocate_uses_original_source_time_and_normal_review_is_still_required(setup_pool):
    pid, shots, add = setup_pool
    asset, _ = add(base=100.013)
    project = store.get_project(pid)
    windows, _ = downloaded_pool.plan(project, shots, {}, lambda: None)
    clip = windows[0]["clip"]
    anchors = [n/25 for n in windows[0]["indices"]]
    proposal = {"candidate_id": clip["id"], "source_url": asset["source_url"], "sample_at": anchors[0], "clip_start": 5}
    proposals = {shots[0]["id"]: [proposal], shots[1]["id"]: [proposal | {"sample_at": anchors[1], "clip_start": 19}]}
    used = defaultdict(list)
    rows = downloaded_pool.allocate(project, shots, windows, proposals, used, lambda: None)
    assert [r["shot"]["clip_start"] for r in rows] == [5, 19]
    assert rows[0]["start"] == pytest.approx(105.013)
    assert rows[0]["discovery"]["sample_at"] == pytest.approx(100.013+anchors[0])
    assert used[asset["source_url"]][0][1] <= used[asset["source_url"]][1][0]
    assert not any(r.get("verification") or r["shot"].get("approved") or r["shot"].get("selected_asset") for r in rows)
    assert store.get_project(pid) == project
    assert downloaded_pool.allocate(project, shots, windows, proposals, {}, lambda: None,
                                    allowed=lambda s, c: False) == []


def test_real_local_decode_has_bound_pixels_and_no_new_download(setup_pool):
    pid, shots, add = setup_pool
    asset, file = add(duration=12)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=1280x720:rate=25:duration=12", "-c:v", "libx264", "-preset", "ultrafast",
        "-threads", "1", "-an", str(file)], timeout=30)
    windows, _ = downloaded_pool.plan(store.get_project(pid), shots, {}, lambda: None)
    samples, errors = downloaded_pool.samples(windows, lambda: None)
    assert not errors and len(samples) == 1
    assert [f["at"] for f in samples[0]["frames"]] == [3, 9]
    assert all(Image.open(f["path"]).size == (1280, 720) for f in samples[0]["frames"])
    assert windows[0]["clip"]["blob_id"] == library.digest_file(file)
    before = [library.digest_file(Path(f["path"])) for f in samples[0]["frames"]]
    repeated, errors = downloaded_pool.samples(windows, lambda: None)
    assert not errors and before == [library.digest_file(Path(f["path"])) for f in repeated[0]["frames"]]


def test_cancellation_is_never_converted_to_missing_pool(setup_pool):
    pid, shots, add = setup_pool
    add()
    def cancelled():
        raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        downloaded_pool.plan(store.get_project(pid), shots, {}, cancelled)


def test_observed_unseen_source_can_cross_groups_but_explicit_rejection_holds():
    candidate = {"id": "pov", "kind": "video", "source_url": "https://www.bilibili.com/video/BV1WYMVz2EGD"}
    group = {"source_allowed_ids": ["another"], "source_allowed_urls": [],
             "source_rejected_ids": [], "source_rejected_urls": []}
    assert not pooled_match.source_permitted(group, candidate)
    assert not pooled_match.source_explicitly_rejected(group, candidate)
    assert pooled_match.source_explicitly_rejected(group | {"source_rejected_ids": ["pov"]}, candidate)
    assert pooled_match.source_explicitly_rejected(group | {"source_rejected_urls": [candidate["source_url"]]}, candidate)


def test_normal_match_uses_local_pool_and_reviews_before_rescue_search(setup_pool, monkeypatch):
    from jinggao import config, library_match, source_metadata
    pid, shots, add = setup_pool
    asset, _ = add()
    config.update({"search_provider": "video"})
    store.mutate(pid, lambda p: p.update(shots=shots[:1]))
    project = store.get_project(pid)
    candidate = downloaded_pool._candidate(asset) | {"title": "骑车", "download_status": "available"}
    monkeypatch.setattr(library_match, "match", lambda *a, **kw: (set(), {"library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([{"label": "骑车", "indices": [0], "queries": ["骑车"]}], {}))
    events = []
    def search(query, **kw):
        events.append("search")
        assert "pool" not in events, "Completed local reuse must avoid remote rescue"
        return [candidate], []
    monkeypatch.setattr(pooled_match.video_sources, "search_videos", search)
    monkeypatch.setattr(planner, "rank_candidates", lambda *a, **kw: [{"candidate_id": candidate["id"]}])
    monkeypatch.setattr(source_metadata, "rank", lambda p, s, tasks, check: (
        [rows for _, rows in tasks], {"model_requests": 0, "failures": [], "completed_groups": list(range(len(tasks)))}))
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    def discovery(*a, **kw):
        events.append("discovery")
        return {}, {}, {"model_requests": 0, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(pooled_match, "discover_windows", discovery)
    # The candidate may be local, but ordinary first-pass allocation is disabled
    # here to exercise the second pass and its shared downloaded interval pool.
    monkeypatch.setattr(pooled_match, "reserve", lambda *a: None)
    def local(p, targets, used, check, progress, **kw):
        events.append("pool")
        windows, info = downloaded_pool.plan(p, targets, used, check, allowed=kw["allowed"], tried=kw["tried"])
        w = windows[0]["clip"]
        proposals = {s["id"]: [{"candidate_id": w["id"], "source_url": asset["source_url"],
                               "sample_at": windows[0]["indices"][0]/25-w["offset"], "clip_start": 5}] for s in targets}
        return downloaded_pool.allocate(p, targets, windows, proposals, used, check), info
    monkeypatch.setattr(downloaded_pool, "propose", local)
    monkeypatch.setattr(library, "remember_window", lambda *a, **kw: None)
    def review(p, rows, check):
        events.append("review")
        assert not any(s.get("selected_asset") for s in store.get_project(pid)["shots"])
        return [r | {"preview": {}, "verification": {"verdict": "match", "selection_blocked": False,
            "needs_source_check": False, "reason": "Fixture full review", "identity_evidence": {"reason": "Fixture"}}}
                for r in rows], [], {"model_requests": 0}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None, workers=1)
    assert events[-3:] == ["discovery", "pool", "review"]
    assert result["selected"] == result["downloaded_pool"]["selected"] == 1
    assert not store.get_project(pid)["shots"][0]["approved"]


def test_anchors_move_to_untried_gaps_before_any_paid_observation(setup_pool):
    pid, shots, add = setup_pool
    asset, _ = add(duration=20)
    shot = shots[0] | {"duration": 3}
    tried = {shot["id"]: [(asset["source_url"], 104, 116)]}
    project = store.get_project(pid)
    windows, _ = downloaded_pool.plan(project, [shot], {}, lambda: None, tried=tried)
    assert len(windows) == 1 and windows[0]["indices"] == [50, 450]
    clip = windows[0]["clip"]
    proposals = {shot["id"]: [{"candidate_id": clip["id"], "source_url": asset["source_url"],
                              "sample_at": 2, "clip_start": .5}]}
    rows = downloaded_pool.allocate(project, [shot], windows, proposals, {}, lambda: None, tried=tried)
    assert len(rows) == 1 and rows[0]["start"]+rows[0]["size"] <= 104
    # A center proposal may overlap a tried span while a nearby start retains
    # the same real anchor. Use the free interval, without shifting the anchor.
    placed = downloaded_pool._placement(shots[0], asset, 100, 120, 110, 105, [],
        {shots[0]["id"]: [(asset["source_url"], 100, 106)]})
    assert placed == 106


def test_real_privacy_derivative_drops_only_original_pool_binding(setup_pool, monkeypatch):
    from jinggao import privacy
    pid, shots, _ = setup_pool
    file = paths.file_path(pid, "assets/phone.jpg")
    file.parent.mkdir(exist_ok=True)
    Image.new("RGB", (800, 800), "white").save(file)
    asset = media.inspect_asset(pid, file, "page")
    asset.update(source_url="https://v.qq.com/x/page/sample.html", source_clip_start=0)
    row = {"shot": shots[0], "asset": asset, "candidate": {"id": "source"},
           "downloaded_pool": {"binding": "original bytes"}, "pending_receipt": "original receipt"}
    calls = []
    def review(project, rows, check):
        current = rows[0]
        if calls:
            assert "downloaded_pool" not in current and "pending_receipt" not in current
            assert current["asset"]["id"] != asset["id"]
            assert paths.file_path(pid, current["asset"]["path"]).read_bytes() != file.read_bytes()
        else:
            assert current["downloaded_pool"] == row["downloaded_pool"]
        calls.append(current["asset"]["id"])
        texts = ["电话13800138000"] if len(calls) == 1 else []
        return [current | {"frame_paths": [paths.file_path(pid, current["asset"]["path"])], "preview": {},
            "verification": {"verdict": "match", "selection_blocked": False,
                "observations": {"frame_observations": [{"frame_index": 0, "objects": "纸张",
                    "action": "静置", "text": texts}]}}}], [], {"model_requests": 0}
    monkeypatch.setattr(pooled_match, "_review_batch", review)
    scans = []
    def scan(*a, **kw):
        scans.append(1)
        return {"frames": [{"frame_index": 0, "boxes": [[.2, .2, .8, .3]] if len(scans) == 1 else []}]}, {}
    monkeypatch.setattr(planner, "chat", scan)
    rows, errors, _ = pooled_match._review_with_privacy(store.get_project(pid), [row], lambda: None)
    assert not errors and len(calls) == 2
    assert rows[0]["asset"].get("privacy_redaction") and not rows[0]["verification"]["selection_blocked"]
    assert not privacy.exposed_frames(privacy.observations(rows[0]["verification"]))
