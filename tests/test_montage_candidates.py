"""Candidate assembly receives one real whole-window review before selection."""

from copy import deepcopy
import hashlib

import pytest

from jinggao import frame_sampling, library, library_match, media, montage_candidates
from jinggao import montage_match, paths, planner, pooled_match, store


@pytest.fixture(scope="module")
def movie(tmp_path_factory):
    file = tmp_path_factory.mktemp("montage-source") / "source.mp4"
    media.run(
        [
            paths.binary("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=25:duration=12",
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
def example(tmp_path, monkeypatch, movie):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("短素材整段复核", script="包裹的处理过程。", media_policy="video_only")
    target = paths.file_path(project["id"], "assets/source.mp4")
    target.parent.mkdir()
    target.write_bytes(movie.read_bytes())
    asset = media.inspect_asset(project["id"], target, "Test fixture; not content acceptance")
    asset.update(
        source_url="https://www.bilibili.com/video/BV1fixture",
        source_clip_start=80,
        candidate_id="fixture-source",
        download_mode="fixture",
    )
    shot = planner.new_shot(project["script"], query="包裹") | {"duration": 8.0}
    project = store.mutate(
        project["id"],
        lambda p: p.update(shots=[shot], assets=[asset], full_audio={"path": "unchanged.wav", "duration": 8}),
    )
    source = {"title": "包裹", "source_url": asset["source_url"]}
    clips = [
        {"id": str(i), "start": 80 + i * 4, "offset": i * 4, "duration": 4, "source": source}
        for i in range(3)
    ]
    monkeypatch.setattr(library, "search", lambda *a, **kw: deepcopy(clips))
    monkeypatch.setattr(library_match, "quick_candidates", lambda p, s, o, stats: (s, o))
    monkeypatch.setattr(library_match, "rank_windows", lambda *a, **kw: True)
    monkeypatch.setattr(library, "materialize", lambda *a: deepcopy(asset))
    return project, shot, asset, target


def reviewer(calls, *, approved=True, edit=False):
    def review(project, rows, check):
        result = []
        for row in rows:
            assert row["asset"].get("source_parts"), "No per-piece paid review"
            assert all(
                part["review_status"] == "unreviewed_candidate" and "verification" not in part
                for part in row["asset"]["source_parts"]
            )
            folder = paths.project_dir(project["id"]) / "review" / row["asset"]["id"]
            folder.mkdir(parents=True)
            files, preview = frame_sampling.sample(project["id"], row["shot"], row["asset"], folder, check)
            assert len(files) == 5
            calls.append(row)
            verdict = {
                "verdict": "match" if approved else "mismatch",
                "scene_alignment": "match" if approved else "mismatch",
                "selection_blocked": not approved,
                "needs_source_check": False,
                "reason": "Fixture whole-window result",
                "identity_evidence": {"reason": "Fixture"},
            }
            result.append(row | {"verification": verdict, "preview": preview})
            if edit:
                store.mutate(
                    project["id"], lambda p: p["shots"][0].update(narration="User changed narration")
                )
        return result, [], {"model_requests": 1}

    return review


@pytest.mark.parametrize("approved", [True, False])
def test_whole_actual_composite_is_reviewed_once_and_its_rejection_holds(example, approved):
    project, shot, _, source = example
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []
    passed, stats = montage_candidates.match(
        project,
        [shot],
        lambda: None,
        lambda *a: None,
        review_batch=reviewer(calls, approved=approved),
        workers=1,
    )
    assert len(calls) == 1 and stats["piece_review_requests"] == 0
    assert stats["model_requests"] == stats["whole_review_batches"] == 1
    assert passed == ({shot["id"]} if approved else set())
    after = store.get_project(project["id"])
    selected = after["shots"][0]
    assert bool(selected.get("selected_asset")) is approved and not selected["approved"]
    assert selected["narration"] == shot["narration"] and selected["duration"] == 8
    assert after["full_audio"] == project["full_audio"] and after["script"] == project["script"]
    composite = calls[0]["asset"]
    assert composite["duration"] == pytest.approx(8, abs=0.04)
    assert [part["source_at"] for part in composite["source_parts"]] == [80, 84]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_changed_narration_during_final_review_is_not_overwritten(example):
    project, shot, _, _ = example
    passed, _ = montage_candidates.match(
        project, [shot], lambda: None, lambda *a: None, review_batch=reviewer([], edit=True), workers=1
    )
    current = store.get_project(project["id"])["shots"][0]
    assert (
        not passed and not current.get("selected_asset") and current["narration"] == "User changed narration"
    )


def test_no_source_capacity_spends_no_ranking_or_review_request(example, monkeypatch):
    project, shot, _, _ = example
    monkeypatch.setattr(library, "search", lambda *a, **kw: [])
    monkeypatch.setattr(library_match, "rank_windows", lambda *a, **kw: pytest.fail("No useful candidate"))
    passed, stats = montage_candidates.match(
        project,
        [shot],
        lambda: None,
        lambda *a: None,
        review_batch=lambda *a: pytest.fail("No complete video to review"),
    )
    assert not passed and stats["model_requests"] == stats["candidates"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("locked", True),
        ("approved", True),
        ("location", "故宫"),
        ("persons", [{"name": "张三", "aliases": []}]),
        ("visual_plan", {"mode": "explicit_sequence"}),
    ],
)
def test_explicit_or_owned_shot_is_not_reinterpreted_as_generic_montage(example, field, value):
    project, shot, _, _ = example
    fixed = shot | {field: value}
    passed, stats = montage_candidates.match(
        project,
        [fixed],
        lambda: None,
        lambda *a: None,
        review_batch=lambda *a: pytest.fail("Ineligible shot"),
    )
    assert not passed and stats["model_requests"] == 0
    with pytest.raises(ValueError, match="不能使用"):
        montage_match.assemble_candidate(project, fixed, [], lambda: None)


def test_normal_pipeline_searches_before_one_complete_local_composition(example, monkeypatch):
    from jinggao import project_pool, scene_recall, video_sources
    project, shot, _, _ = example
    calls, searches = [], []
    monkeypatch.setattr(project_pool, "collect", lambda *a, **kw: ([], {}))
    monkeypatch.setattr(
        library_match,
        "match",
        lambda *a, **kw: (set(), {"library_model_requests": 0, "library_failures": []}),
    )
    review = reviewer(calls)
    def after_search(*args):
        assert searches, "Expensive composition must wait for actual source retrieval"
        return review(*args)
    monkeypatch.setattr(pooled_match, "review_batch", after_search)
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: (
        [{"label": "包裹处理", "indices": [0], "queries": ["包裹处理"]}], {"cache_hit": True}))
    def search(query, **kwargs):
        searches.append(query)
        return [], []
    monkeypatch.setattr(video_sources, "search_videos", search)
    monkeypatch.setattr(scene_recall, "candidates", lambda *a: {})
    monkeypatch.setattr(pooled_match, "online_rescue_queries", lambda *a: {})
    monkeypatch.setattr(montage_match, "match", lambda *a, **kw: pytest.fail("No five-round per-piece fallback"))
    result = pooled_match.match(project, [shot], lambda: None, lambda *a: None, workers=1)
    assert len(calls) == 1 and result["montage"]["selected"] == result["selected"] == 1
    assert searches and result["search_queries"] > 0 and result["downloads"] == 0
    assert "early_montage" not in result
