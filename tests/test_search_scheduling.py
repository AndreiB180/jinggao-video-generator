import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from jinggao import config, library, library_match, media, paths, planner, pooled_match, store, video_sources


def test_first_query_is_shared_and_primary_words_precede_group_reserves():
    shots = [{"id": "a", "query": "甲动作"}, {"id": "b", "query": "乙动作"}, {"id": "c"}]
    groups = [{"indices": [0], "queries": ["共同场景", "甲备用"]},
              {"indices": [1], "queries": ["共同场景", "乙备用"]},
              {"indices": [2], "queries": ["已经完成", "不应再搜"]}]
    initial = pooled_match.initial_video_queries(groups)
    assert initial == ["共同场景", "已经完成"]
    found = {query: [] for query in initial}
    rescue = pooled_match.online_rescue_queries(groups, shots, shots[:2], found)
    assert list(rescue)[:2] == ["甲动作", "乙动作"]
    assert "不应再搜" not in rescue and set(found) == set(initial)


def test_image_fallback_respects_remaining_duration_and_prefers_planned_photos():
    shots = [{"id": str(i), "duration": 10, "preferred_media": "video"} for i in range(10)]
    shots[0]["selected_asset"] = "existing"
    shots[9]["preferred_media"] = "image"
    project = {"assets": [{"id": "existing", "kind": "image"}], "shots": shots,
               "media_policy": "video_preferred", "min_video_ratio": .8}
    groups = [{"indices": [i], "queries": [f"画面{i}"]} for i in range(10)]
    tasks, eligible = pooled_match.image_fallback_queries(project, groups, shots, {"0"})
    assert list(tasks) == ["画面9"] and eligible == {"9"}
    assert sum(s["duration"] for s in shots if s["id"] in eligible) == 10
    project["min_video_ratio"] = 1
    assert pooled_match.image_fallback_queries(project, groups, shots, {"0"}) == ({}, set())


def test_image_queries_share_compact_person_word_and_have_a_group_budget():
    shots = [{"id": str(i), "duration": 1} for i in range(10)]
    for shot in shots[:2]:
        shot["persons"] = [{"name": "马龙", "aliases": []}]
    groups = [{"indices": [i], "queries": [f"画面{i}"]} for i in range(10)]
    tasks, eligible = pooled_match.image_fallback_queries(
        {"assets": [], "shots": shots, "media_policy": "mixed"}, groups, shots, set())
    assert len(tasks) == pooled_match.IMAGE_QUERY_BUDGET
    assert len(tasks["马龙"]) == 2 and {"0", "1"} <= eligible


@pytest.mark.parametrize("video_ratio, expected_images", [(.8, 2), (1, 0)])
def test_match_has_one_video_query_per_group_total_budget_and_deferred_images(tmp_path, monkeypatch, video_ratio, expected_images):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "mixed"})
    project = store.new_project("检索预算", "多个独立场景。", media_policy="video_preferred", min_video_ratio=video_ratio)
    shots = [planner.new_shot("具体工作场景。", query=f"动作{i}") for i in range(12)]
    for shot in shots:
        shot.update(duration=5, preferred_media="video")
    groups = [{"label": f"场景{i}", "indices": [i], "queries": [f"首词{i}", f"备用{i}"]} for i in range(12)]
    store.mutate(project["id"], lambda p: p.update(shots=shots))
    project = store.get_project(project["id"])
    monkeypatch.setattr(library_match, "match", lambda *a: (set(), {
        "library_model_requests": 0, "library_failures": []}))
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: (groups, {}))
    calls = []
    def search(query, **kwargs):
        calls.append(("video", query, kwargs.get("use_index", True)))
        return [], []
    def images(query):
        calls.append(("image", query, False))
        return [], []
    def discovery(*args, **kwargs):
        calls.append(("discovery", "", False))
        return {}, {}, {"model_requests": 0, "sampled_bytes": 0, "failures": []}
    monkeypatch.setattr(video_sources, "search_videos", search)
    monkeypatch.setattr(pooled_match.sources, "search_images", images)
    monkeypatch.setattr(pooled_match, "discover_windows", discovery)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("empty searches must not call a model"))
    stats = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None, workers=3)
    videos = [c for c in calls if c[0] == "video"]
    assert len(videos) == pooled_match.VIDEO_QUERY_BUDGET
    assert {c[1] for c in videos if c[2]} == {f"首词{i}" for i in range(12)}
    assert {c[1] for c in videos if not c[2]} == {f"动作{i}" for i in range(4)}
    rescue_audits = [r for r in stats["search_tasks"] if r["phase"] == "rescue"]
    assert all(r["reason"] == "primary_gap" and r["gap_seconds"] == 5 for r in rescue_audits)
    assert {tuple(r["shot_indices"]) for r in rescue_audits} == {(i,) for i in range(4)}
    if expected_images:
        first_image = next(i for i, c in enumerate(calls) if c[0] == "image")
        assert sum(c[0] == "discovery" for c in calls[:first_image]) == 2
    assert sum(c[0] == "image" for c in calls) == expected_images
    assert stats["search_queries"] == len(stats["search_tasks"]) == 16+expected_images
    assert stats["downloads"] == stats["selected"] == 0
    assert stats["composition"]["target_video_ratio"] == video_ratio
    final = store.get_project(project["id"])
    assert final["min_video_height"] == 720
    if not expected_images:
        assert all("图片预算" in s["search_note"] for s in final["shots"])


def candidate(number):
    return {"id": str(number), "kind": "video", "provider": "cctv", "title": "公开工作场景",
            "source_url": f"https://tv.cctv.com/2026/10/09/VIDE{number}.shtml"}


def test_live_search_cannot_reset_recent_bad_availability_but_expiry_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    rows = [candidate(i) for i in range(5)]
    library.remember_sources(rows)
    now = time.time()
    states = [("preview_only", now), ("blocked", now), ("unsupported", now),
              ("blocked", now-video_sources.UNAVAILABLE_TTLS["blocked"]-1), ("available", now)]
    with library.connection() as con:
        for row, (state, checked) in zip(rows, states, strict=True):
            sid, _ = library.source_record(row)
            con.execute("UPDATE source_catalog SET state=?,checked=?,reason=? WHERE source_id=?",
                        (state, checked, "公开流为 480×270，短边低于 720p", sid))
    monkeypatch.setattr(video_sources, "cctv_search", lambda q: rows)
    diagnostics = {}
    result, errors = video_sources.search_videos("工作", use_index=False, channel_names={"央视"}, diagnostics=diagnostics)
    assert not errors and [r["id"] for r in result] == ["3", "4"]
    assert diagnostics["skipped_recent_unavailable"] == {"preview_only": 1, "blocked": 1, "unsupported": 1}
    with library.connection() as con:
        con.execute("UPDATE source_catalog SET checked=?", (now-86401,))
    result, _ = video_sources.search_videos("工作", use_index=False, channel_names={"央视"})
    assert [r["id"] for r in result] == [str(i) for i in range(5)]
    assert not any(r.get("approved") or r.get("verification") for r in result)


def test_previous_stricter_quality_rejection_does_not_hide_sufficient_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    rows = [candidate(i) for i in range(3)]
    library.remember_sources(rows)
    reasons = ["公开流为 1280×720，短边低于 1080p", "公开流短边低于 1080p", "未记录尺寸"]
    with library.connection() as con:
        for row, reason in zip(rows, reasons, strict=True):
            sid, _ = library.source_record(row)
            con.execute("UPDATE source_catalog SET state='preview_only',checked=?,reason=? WHERE source_id=?",
                        (time.time(), reason, sid))
    assert video_sources._recent_unavailable(rows, 720) == (rows, {})
    assert video_sources._recent_unavailable(rows, 1080) == ([rows[2]], {"preview_only": 2})


def test_provider_diagnostic_distinguishes_fresh_fetch_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    fetched = []
    def fetch():
        fetched.append(True)
        return [candidate(1)]
    monkeypatch.setattr(video_sources, "cctv_search", lambda q: video_sources._cached("test", q, fetch))
    first, second = {}, {}
    for diagnostic in (first, second):
        rows, failures = video_sources.search_videos("同词", use_index=False, channel_names={"央视"}, diagnostics=diagnostic)
        assert len(rows) == 1 and not failures
    assert len(fetched) == 1
    assert first["providers"][0]["caches"][0]["cache_hit"] is False
    assert second["providers"][0]["caches"][0]["cache_hit"] is True
    assert first["providers"][0]["queue_seconds"] >= 0
    assert first["providers"][0]["elapsed_seconds"] >= 0


def test_nested_searches_share_provider_slots_and_keep_stable_source_order(tmp_path, monkeypatch):
    from jinggao import platform_videos
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with library.connection():
        pass
    monkeypatch.setattr(video_sources, "SEARCH_PROVIDER_SLOTS", threading.BoundedSemaphore(2))
    lock, active, peak = threading.Lock(), 0, 0
    def search(query, number):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.01)
            return [candidate(number)]
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(platform_videos, "tencent_search", lambda q: search(q, 1))
    monkeypatch.setattr(video_sources, "cctv_search", lambda q: search(q, 2))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda q: video_sources.search_videos(q, use_index=False,
            channel_names={"腾讯视频", "央视"}), ["a", "b", "c"]))
    assert peak == 2 and active == 0
    assert all([r["id"] for r in rows] == ["1", "2"] and not failures for rows, failures in results)


def test_cancelled_queued_search_never_starts_provider_or_releases_another_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(video_sources, "SEARCH_PROVIDER_SLOTS", slots)
    monkeypatch.setattr(video_sources, "cctv_search", lambda q: pytest.fail("cancelled queued provider started"))
    calls = []
    def check():
        calls.append(True)
        if len(calls) > 1:
            raise media.Cancelled()
    try:
        with pytest.raises(media.Cancelled):
            video_sources.search_videos("取消", use_index=False, channel_names={"央视"}, check=check)
        assert not slots.acquire(blocking=False)
    finally:
        slots.release()


def test_provider_failure_diagnostic_does_not_cache_failure_or_leak_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(video_sources, "SEARCH_PROVIDER_SLOTS", slots)
    calls = []
    def fetch():
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("暂时无法读取")
        return [candidate(1)]
    monkeypatch.setattr(video_sources, "cctv_search", lambda q: video_sources._cached("test", q, fetch))
    diagnostic = {}
    rows, errors = video_sources.search_videos("重试", use_index=False, channel_names={"央视"}, diagnostics=diagnostic)
    assert not rows and errors and diagnostic["providers"][0]["status"] == "error"
    assert diagnostic["providers"][0]["caches"][0]["cache_hit"] is False
    rows, errors = video_sources.search_videos("重试", use_index=False, channel_names={"央视"})
    assert len(rows) == 1 and not errors and len(calls) == 2
    assert slots.acquire(blocking=False)
    slots.release()
