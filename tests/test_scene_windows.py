"""Real local cuts improve retrieval without approving or extending evidence."""

from collections import defaultdict
from copy import deepcopy

import pytest

from jinggao import media, paths, scene_windows, store


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory):
    file = tmp_path_factory.mktemp("scene-window") / "cut.mp4"
    media.run(
        [
            paths.binary("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=1280x720:rate=25:duration=8",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=1280x720:rate=25:duration=22",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "2",
            str(file),
        ],
        timeout=20,
    )
    return file


@pytest.fixture
def example(tmp_path, monkeypatch, cut_video):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("转场测试", "列车行驶")
    file = paths.file_path(project["id"], "assets/source.mp4")
    file.parent.mkdir()
    file.write_bytes(cut_video.read_bytes())
    source = "https://www.bilibili.com/video/BV1hXpu6MEhu"
    asset = {
        "id": "asset",
        "path": "assets/source.mp4",
        "source_clip_start": 100,
        "duration": 30,
        "kind": "video",
        "source_url": source,
        "title": "source",
    }
    candidate = {"id": "candidate", "source_url": source, "duration": 200}
    row = {
        "shot": {"id": "shot", "duration": 11.2},
        "candidate": candidate,
        "start": 106,
        "size": 11.32,
        "discovery": {"sample_at": 112, "source_url": source},
    }
    block = {"candidate": candidate, "start": 106, "end": 117.32, "items": [row]}
    used = defaultdict(list, {source: [(106, 117.32)]})
    return project, asset, block, used, file


def test_real_cut_moves_whole_window_into_anchor_scene_without_approving(example):
    project, asset, block, used, _ = example
    old = deepcopy(block)
    rows, info = scene_windows.align(project, block, asset, used, lambda: None)
    assert not info["error"] and info["scene_count"] == 2
    assert rows[0]["start"] == pytest.approx(108.08)
    assert rows[0]["start"] <= 112 < rows[0]["start"] + 11.2
    assert used[asset["source_url"]][0] == pytest.approx((108.08, 119.4))
    assert block == old and rows[0]["shot"]["duration"] == 11.2
    assert info["adjusted"][0]["content_approved"] is False
    assert not {"verification", "approved", "selected_asset"} & rows[0].keys()


def test_cached_cut_detection_replays_and_changed_bytes_force_decode(example, monkeypatch):
    project, asset, block, used, file = example
    calls, run = [], media.run

    def counted(*args, **kwargs):
        calls.append(1)
        return run(*args, **kwargs)

    monkeypatch.setattr(media, "run", counted)
    for _ in range(2):
        scene_windows.align(project, block, asset, deepcopy(used), lambda: None)
    assert len(calls) == 1
    file.write_bytes(file.read_bytes() + b"changed file bytes")
    scene_windows.align(project, block, asset, deepcopy(used), lambda: None)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "occupied,scenes",
    [
        ([(119, 125)], [(100, 107.92), (108.08, 130)]),
        ([], [(100, 107.92), (108.08, 116)]),
    ],
)
def test_short_or_occupied_scene_keeps_original_for_full_review(example, occupied, scenes):
    _, asset, block, used, _ = example
    used[asset["source_url"]].extend(occupied)
    before = deepcopy(used)
    rows, changes = scene_windows.adjust(block, asset, scenes, used)
    assert rows == block["items"] and not changes and used == before


def test_wrong_source_or_missing_reservation_cannot_shift_window(example):
    _, asset, block, used, _ = example
    block["items"][0]["discovery"]["source_url"] = "https://example.test/other"
    rows, changes = scene_windows.adjust(block, asset, [(108.08, 130)], used)
    assert rows == block["items"] and not changes
    block["items"][0]["discovery"]["source_url"] = asset["source_url"]
    used.clear()
    rows, changes = scene_windows.adjust(block, asset, [(108.08, 130)], used)
    assert rows == block["items"] and not changes


def test_download_neighbourhood_remains_bounded_and_ends_at_actual_source(example):
    _, _, block, _, _ = example
    assert scene_windows.download_bounds(block) == (98, 125.32)
    block["candidate"]["duration"] = 120
    assert scene_windows.download_bounds(block) == (98, 120)
    block["candidate"]["asset_id"] = "local-file"
    assert scene_windows.download_bounds(block) == (106, 117.32)


def test_decode_failure_falls_back_but_cancellation_propagates(example, monkeypatch):
    project, asset, block, used, _ = example

    def invalid(*args, **kwargs):
        raise ValueError("decoder failed")

    monkeypatch.setattr(media, "run", invalid)
    rows, info = scene_windows.align(project, block, asset, used, lambda: None)
    assert rows == block["items"] and info["error"] == "decoder failed"

    def cancel():
        raise media.Cancelled()

    with pytest.raises(media.Cancelled):
        scene_windows.align(project, block, asset, used, cancel)


def test_corrupt_partial_scene_cache_cannot_change_window(example, monkeypatch):
    project, asset, block, used, _ = example
    scene_windows.align(project, block, asset, deepcopy(used), lambda: None)
    cache = next((paths.DATA / "cache/scene-windows").glob("*.json"))
    cache.write_text('{"starts":[-12,50000]}')
    calls = []
    run = media.run

    def counted(*args, **kwargs):
        calls.append(1)
        return run(*args, **kwargs)

    monkeypatch.setattr(media, "run", counted)
    rows, info = scene_windows.align(project, block, asset, used, lambda: None)
    assert calls == [1] and not info["error"] and rows[0]["start"] == pytest.approx(108.08)


def test_normal_match_downloads_neighbourhood_then_reviews_shifted_actual_window(example, monkeypatch):
    from jinggao import config, library_match, montage_match, planner, pooled_match, source_metadata

    project, asset, block, _, file = example
    config.update({"search_provider": "video"})
    shot = planner.new_shot("列车在平原上行驶。", query="列车 行驶", duration=11.2)
    shot["duration"] = 11.2
    store.mutate(project["id"], lambda p: p.update(shots=[shot], media_policy="video_only"))
    project = store.get_project(project["id"])
    candidate = block["candidate"] | {
        "kind": "video",
        "title": "列车行驶实拍",
        "provider": "bilibili",
        "width": 1280,
        "height": 720,
        "download_status": "available",
    }
    monkeypatch.setattr(
        library_match,
        "match",
        lambda *a, **kw: (set(), {"library_model_requests": 0, "library_failures": []}),
    )
    monkeypatch.setattr(
        pooled_match,
        "retrieval_groups",
        lambda *a: ([{"label": "列车", "indices": [0], "queries": ["列车 行驶"]}], {}),
    )
    monkeypatch.setattr(pooled_match.video_sources, "search_videos", lambda *a, **kw: ([candidate], []))
    monkeypatch.setattr(planner, "rank_candidates", lambda *a, **kw: [{"candidate_id": candidate["id"]}])
    monkeypatch.setattr(
        source_metadata,
        "rank",
        lambda p, s, tasks, check: (
            [rows for _, rows in tasks],
            {"model_requests": 0, "failures": [], "completed_groups": list(range(len(tasks)))},
        ),
    )
    monkeypatch.setattr(pooled_match.video_sources, "inspect_video_candidate", lambda c, *a: c)
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    discoveries, downloads, reviews = [], [], []

    def discover(*args, **kwargs):
        discoveries.append(1)
        proposals = (
            [
                {
                    "candidate_id": candidate["id"],
                    "source_url": candidate["source_url"],
                    "clip_start": 106,
                    "sample_at": 112,
                }
            ]
            if len(discoveries) == 1
            else []
        )
        return (
            {shot["id"]: proposals},
            {shot["id"]: {candidate["id"]}},
            {"model_requests": 0, "sampled_bytes": 0, "failures": []},
        )

    monkeypatch.setattr(pooled_match, "discover_windows", discover)

    def download(c, target, start, duration, check):
        downloads.append((start, duration))
        # A real already-decoded response spans source [100,130). The caller
        # must use its returned source offset, not assume the requested start.
        target.write_bytes(file.read_bytes())
        return {
            "source_clip_start": 100,
            "download_mode": "offline_fixture",
            "downloaded_bytes": file.stat().st_size,
        }

    monkeypatch.setattr(pooled_match.video_sources, "download_video_candidate", download)

    def reject(p, rows, check):
        reviews.extend(rows)
        return (
            [
                r
                | {
                    "verification": {
                        "verdict": "mismatch",
                        "scene_alignment": "mismatch",
                        "selection_blocked": True,
                        "needs_source_check": False,
                        "reason": "Content still rejected",
                        "identity_evidence": {"reason": "Fixture"},
                    },
                    "preview": {},
                }
                for r in rows
            ],
            [],
            {"model_requests": 0},
        )

    monkeypatch.setattr(pooled_match, "review_batch", reject)
    monkeypatch.setattr(
        montage_match, "match", lambda *a, **kw: (set(), {"model_requests": 0, "failures": []})
    )
    result = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None, workers=1)
    assert len(downloads) == len(reviews) == 1
    assert downloads[0] == pytest.approx((98, 27.32))
    assert reviews[0]["shot"]["clip_start"] == pytest.approx(8.08)
    assert reviews[0]["shot"]["duration"] == 11.2
    assert result["scene_windows"]["adjusted"] and result["selected"] == 0
    saved = store.get_project(project["id"])["shots"][0]
    assert not saved.get("selected_asset") and not saved.get("approved")
