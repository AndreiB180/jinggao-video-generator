"""Project uploads reach normal online matching; model transport is offline."""
import json

import pytest

from jinggao import config, library, library_match, media, neutral_observer, paths
from jinggao import planner, pooled_match, project_pool, rank_cache, store


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    target = tmp_path_factory.mktemp("pool-video") / "real-decoded.mp4"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=2", "-t", "40", "-c:v", "libx264",
               "-preset", "ultrafast", "-crf", "35", str(target)], timeout=30)
    return target


@pytest.fixture
def environment(tmp_path, monkeypatch, video):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    cfg = config.read() | {"model_name": "qwen3.7-flash", "vision_model": "qwen3.7-flash",
                          "model_api_key": "offline-fixture", "search_provider": "video"}
    monkeypatch.setattr(config, "read", lambda: cfg)
    store.init()
    p = store.new_project("导入源视频测试", script="文案只用于匹配，不应进入画面索引请求")
    target = paths.file_path(p["id"], "assets/source.mp4")
    target.parent.mkdir()
    target.write_bytes(video.read_bytes())
    asset = {"id": store.uid(), "kind": "video", "path": "assets/source.mp4", "duration": 40,
             "width": 1280, "height": 720, "title": "不该进入盲观察的文件名"}
    shots = [{"id": store.uid(), "narration": "工人在仓库分拣包裹", "query": "仓库 包裹",
              "subject": "包裹", "visual": "仓库分拣包裹", "duration": 8,
              "persons": [], "event_scope": "background", "clip_start": 0,
              "approved": False, "selected_asset": None} for _ in range(4)]
    store.mutate(p["id"], lambda p: p.update(assets=[asset], shots=shots))
    p = store.get_project(p["id"])
    observations, rankings = [], []
    def observe(content, cfg, check):
        observations.append(content)
        assert "不该进入盲观察" not in str(content) and p["script"] not in str(content)
        manifest = json.loads(content[0]["text"])["observation_groups"]
        return json.dumps({"groups": [{"id": g["group_id"], "scene_description": "工人在仓库分拣包裹",
            "frames": [{"frame_index": i, "objects": "仓库、工人和包裹", "action": "分拣包裹", "text": []}
                       for i in g["frame_indices"]]} for g in manifest]}, ensure_ascii=False), {}
    def rank(system, content, **kwargs):
        data = json.loads(content)
        rankings.append(data)
        return {"rankings": [{"shot_id": s["shot_id"], "candidate_refs": [
            c["candidate_ref"] for c in s["candidates"][:6]]} for s in data["shots"]]}, {"model": cfg["model_name"]}
    monkeypatch.setattr(neutral_observer, "_request", observe)
    monkeypatch.setattr(planner, "chat", rank)
    return p, asset, target, observations, rankings, cfg


def collect(project):
    return project_pool.collect(project, lambda: None, lambda *a: None)


def test_real_frames_and_observations_replay_without_selecting_shots(environment):
    p, asset, target, requests, _, _ = environment
    clips, first = collect(p)
    again, second = collect(p)
    assert len(clips) == len(again) == 2
    assert [c["index_frame_times"] for c in clips] == [[5, 15], [25, 35]]
    assert first["model_requests"] == len(requests) == 1
    assert second["model_requests"] == 0 and second["frame_cache_hits"] == 1
    assert all(project_pool.resolve(c)["id"] == asset["id"] for c in clips)
    assert all(not s["selected_asset"] and not s["approved"] for s in store.get_project(p["id"])["shots"])
    assert not first["failures"]


def test_normal_online_match_uses_upload_before_remote_retrieval(environment, monkeypatch):
    p, asset, _, requests, rankings, _ = environment
    reviews = []
    def review(project, rows, check):
        reviews.extend(rows)
        return [r | {"verification": {"verdict": "match", "scene_alignment": "match",
            "selection_blocked": False, "needs_source_check": False, "reason": "fixture current review"},
            "preview": {}} for r in rows], [], {"model_requests": 1}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: pytest.fail("unnecessary online retrieval"))
    result = pooled_match.match(p, p["shots"], lambda: None, lambda *a: None, workers=2)
    saved = store.get_project(p["id"])
    assert result["selected"] == 4 and result["downloads"] == result["search_queries"] == 0
    assert len(requests) == len(rankings) == 1 and len(reviews) == 4
    assert {s["selected_asset"] for s in saved["shots"]} == {asset["id"]}
    intervals = sorted((s["clip_start"], s["clip_start"]+s["duration"]) for s in saved["shots"])
    assert intervals == [(1, 9), (11, 19), (21, 29), (31, 39)]
    assert all(not s["approved"] for s in saved["shots"])


def test_index_and_rank_never_override_current_content_rejection(environment, monkeypatch):
    p, _, _, _, _, _ = environment
    clips, _ = collect(p)
    called = []
    def reject(project, rows, check):
        called.extend(rows)
        return [r | {"verification": {"verdict": "mismatch", "scene_alignment": "mismatch",
            "selection_blocked": True, "needs_source_check": False, "reason": "wrong actual scene"},
            "preview": {}} for r in rows], [], {"model_requests": 1}
    monkeypatch.setattr(pooled_match, "review_batch", reject)
    passed, stats = library_match.match(p, p["shots"], lambda: None, lambda *a: None,
                                       project_clips=clips, include_library=False)
    assert called and not passed and stats["library_matched"] == 0
    assert all(not s["selected_asset"] for s in store.get_project(p["id"])["shots"])


def test_rank_cache_binds_upload_once_per_snapshot_and_rechecks_changes(environment, monkeypatch):
    p, _, target, _, _, _ = environment
    clips, _ = collect(p)
    candidates = {s["id"]: {str(i): c for i, c in enumerate(clips)} for s in p["shots"]}
    original, hashes = library.digest_file, []
    def digest(file):
        hashes.append(file)
        return original(file)
    monkeypatch.setattr(library, "digest_file", digest)
    cache = rank_cache.Ranking("policy", {}, candidates, {})
    assert hashes == [target]
    cache.save({"ok": True}, lambda r: True)
    assert cache.read(lambda r: True) == {"ok": True}
    target.write_bytes(target.read_bytes()+b"changed source")
    assert cache.read(lambda r: True) is None
    with pytest.raises(ValueError, match="字节"):
        project_pool.resolve(clips[0])


def test_source_changes_and_out_of_bounds_ranges_are_rejected(environment):
    p, asset, _, _, _, _ = environment
    clips, _ = collect(p)
    with pytest.raises(ValueError, match="时间"):
        project_pool.resolve(clips[0] | {"offset": 39})
    store.mutate(p["id"], lambda p: p["assets"][0].update(source_url="https://example.test/changed"))
    with pytest.raises(ValueError, match="来源"):
        project_pool.resolve(clips[0])


def test_downloaded_blocks_do_not_add_a_second_indexing_bill(environment):
    p, _, _, requests, _, _ = environment
    p["assets"][0]["download_mode"] = "hls_selected_segments"
    clips, stats = collect(p)
    assert clips == requests == [] and stats["model_requests"] == 0


def test_sampling_cap_reports_deferred_windows_and_cancellation(environment):
    p, asset, _, _, _, _ = environment
    p["assets"][0]["duration"] = 7200
    tasks, deferred = project_pool._tasks(p, lambda: None)
    assert len(tasks[0][3]) == 32 and deferred == 418
    assert all(start+duration <= 7200 for start, duration in tasks[0][3])
    def cancel():
        raise media.Cancelled()
    with pytest.raises(media.Cancelled):
        project_pool.collect(p, cancel, lambda *a: None)


def test_model_account_failure_stops_indexing(environment, monkeypatch):
    p, *_ = environment
    def unavailable(*args, **kwargs):
        raise planner.ModelUnavailable("quota fixture")
    monkeypatch.setattr(neutral_observer, "_request", unavailable)
    with pytest.raises(planner.ModelUnavailable, match="quota"):
        collect(p)


def test_incomplete_frame_rebuild_cannot_mix_old_and_new_frames(environment, monkeypatch):
    p, _, _, _, _, _ = environment
    collect(p)
    folder = next((paths.DATA / "cache" / "project-pool-frames").iterdir())
    (folder / "frames.json").write_text("corrupt")
    (folder / "0001.jpg").unlink()
    monkeypatch.setattr(media, "run", lambda *a, **kw: None)
    clips, stats = collect(p)
    assert not clips and stats["failures"] and stats["model_requests"] == 0
    assert not (folder / "frames.json").exists()


def test_retrieval_follows_later_observed_action_instead_of_block_start(environment):
    p, *_ = environment
    clips, _ = collect(p)
    first = clips[0]
    first["frame_observations"][0].update(objects="外部道路", action="车辆行驶")
    candidates = project_pool.search([first], "仓库 分拣 包裹", 8)
    assert len(candidates) == 1 and candidates[0]["offset"] == 11
    assert candidates[0]["start"] == 11 and candidates[0]["duration"] == 8


def test_long_source_uses_bounded_seeks_and_keeps_portrait_shape(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(paths, "DATA", tmp_path)
    file = tmp_path / "long.mp4"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
               "testsrc2=size=32x64:rate=1", "-t", "630", "-c:v", "libx264",
               "-preset", "ultrafast", str(file)], timeout=15)
    digest = library.digest_file(file)
    frames, cached = project_pool._frames(file, digest, [(608, 16)], lambda: None)
    assert sorted(frames) == [612*25, 620*25] and not cached
    assert all(Image.open(f).size == (640, 1280) for f in frames.values())
