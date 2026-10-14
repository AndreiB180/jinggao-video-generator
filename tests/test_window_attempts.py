from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import time

from PIL import Image
import pytest

from jinggao import frame_sampling, library, library_match, paths, pooled_match, store, window_attempts


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(window_attempts, "policy", lambda: {"protocol": "current"})
    store.init()
    project = store.new_project("窗口恢复")
    shot = {"id": store.uid(), "narration": "工人在分拣包裹。", "query": "分拣 包裹", "duration": 6,
            "clip_start": 0, "persons": [], "event": "", "event_scope": "background",
            "selected_asset": None, "approved": False,
            "visual_plan": {"mode": "single", "primary": {"subject": "工人", "action": "分拣包裹", "scene": "仓库"}}}
    def update(p):
        p.update(script=shot["narration"], requirements="不得用会议代替作业", shots=[shot])
    project = store.mutate(project["id"], update)
    root = paths.project_dir(project["id"])
    media = root / "assets" / "test.mp4"
    media.parent.mkdir(exist_ok=True)
    media.write_bytes(b"original media binding")
    asset = {"id": "asset", "kind": "video", "path": "assets/test.mp4", "duration": 9,
             "width": 1280, "height": 720, "title": "实际工作记录", "source_clip_start": 0,
             "source_url": "https://v.qq.com/x/page/work.html"}
    frames = []
    for i in range(5):
        file = root / f"frame-{i}.jpg"
        Image.new("RGB", (32, 32), (i*30, 40, 50)).save(file)
        frames.append(file)
    observations = [{"frame_index": i, "objects": "会议桌旁坐着发言者", "action": "发言者坐在桌前讲话",
                     "text": []} for i in range(5)]
    citation = {"frame_index": 0, "field": "action", "quote": observations[0]["action"]}
    checks = [{"key": key, "status": "conflict", "reason": "会议发言不是分拣包裹", "evidence": [citation]}
              for key in ("subject", "action", "scene")]
    audit = {"status": "conflict", "reason": "会议发言不是分拣包裹", "labels": [], "evidence": [citation],
             "scene_description": "坐在会议桌前发言", "frame_observations": observations,
             "primary_checks": checks,
             "visual_quality": {"frame_sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f in frames]}}
    result = {"verdict": "mismatch", "scene_alignment": "mismatch", "selection_blocked": True,
              "needs_source_check": True, "identity_audit": audit}
    # v60 binds every cached rejection to its scene-coverage sampling (frame_sampling.VERSION).
    start, duration, base, count = frame_sampling.window(shot, asset)
    indices = frame_sampling.indices_for([0], count)
    preview = {"frames": [{"kind": "video_frame", "at": round(base + start + n / frame_sampling.FPS, 3)}
                          for n in indices],
               "sampling": {"version": frame_sampling.VERSION, "policy_sha256": frame_sampling.policy_hash(),
                            "fps": frame_sampling.FPS, "frame_count": count,
                            "analysis_width": frame_sampling.ANALYSIS_WIDTH,
                            "scene_threshold": frame_sampling.SCENE_THRESHOLD, "scene_starts": [0],
                            "indices": indices, "clip_start": start, "duration": duration,
                            "asset_sha256": library.digest_file(media),
                            "frame_sha256": [library.digest_file(f) for f in frames]}}
    row = {"shot": shot, "asset": asset, "frame_paths": frames, "verification": result, "preview": preview}
    return project, shot, asset, row


def test_rejection_persists_and_asset_alias_does_not_reset_it(sample):
    project, shot, asset, row = sample
    cache = window_attempts.prepare(project, shot, asset)
    assert cache.save(row)
    alias = window_attempts.prepare(project, shot, asset | {"id": "new-id", "library_clip": "new-clip-id"})
    assert alias.key == cache.key and alias.read()["reason"] == row["verification"]["identity_audit"]["reason"]
    assert not {"approved", "verification"} & alias.read().keys()
    assert cache.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", ["script", "requirements", "narration", "persons", "event", "overlay_label",
                                  "clip_start", "duration", "fit", "source_evidence", "source_verified", "context_sources",
                                  "locked", "overlay_label_origin", "source", "media", "policy"])
def test_every_semantic_or_pixel_change_invalidates(sample, monkeypatch, change):
    project, shot, asset, row = sample
    cache = window_attempts.prepare(project, shot, asset)
    assert cache.save(row)
    p, s, a = deepcopy(project), deepcopy(shot), deepcopy(asset)
    if change in ("script", "requirements"):
        p[change] += "增加明确要求"
    elif change in ("clip_start", "duration"):
        s[change] += 1
    elif change == "fit":
        # Manual screen fitting is an input even where preferred_fit retains it.
        s.update(fit="contain", fit_origin="manual")
    elif change == "persons":
        s[change] = [{"name": "指定人物", "aliases": []}]
    elif change in ("narration", "event", "overlay_label"):
        s[change] = "另一个明确条件"
    elif change == "source_evidence":
        a[change] = {"description": "另一事件的发布者说明"}
    elif change == "source_verified":
        a[change] = True
    elif change == "context_sources":
        a[change] = [{"description": "补充来源明确当前年份"}]
    elif change == "locked":
        s[change] = True
    elif change == "overlay_label_origin":
        s[change] = "source_context"
    elif change == "source":
        a["source_url"] = "https://v.qq.com/x/page/another.html"
    elif change == "media":
        paths.file_path(project["id"], asset["path"]).write_bytes(b"changed media binding!")
    elif change == "policy":
        monkeypatch.setattr(window_attempts, "policy", lambda: {"protocol": "updated"})
    fresh = window_attempts.prepare(p, s, a)
    assert fresh is not None and fresh.read() is None


@pytest.mark.parametrize("failure", ["unknown", "error", "no_citation", "forged_quote", "wrong_frame",
                                    "missing_condition", "invalid_status", "altered_hash"])
def test_unknown_invalid_or_ungrounded_results_cannot_blacklist(sample, failure):
    project, shot, asset, row = sample
    row = deepcopy(row)
    audit = row["verification"]["identity_audit"]
    if failure == "unknown":
        audit["status"] = "unknown"
    elif failure == "error":
        row["verification"] = {"verdict": "uncertain", "reason": "429 timeout invalid JSON"}
    elif failure == "no_citation":
        audit["evidence"] = []
    elif failure == "forged_quote":
        audit["evidence"][0]["quote"] = "一段未观察到的动作"
    elif failure == "wrong_frame":
        audit["primary_checks"][0]["evidence"][0]["frame_index"] = 10
    elif failure == "missing_condition":
        audit["primary_checks"].pop()
    elif failure == "invalid_status":
        audit["primary_checks"][0]["status"] = []
    elif failure == "altered_hash":
        audit["visual_quality"]["frame_sha256"][0] = "0"*64
    cache = window_attempts.prepare(project, shot, asset)
    assert not cache.save(row) and cache.read() is None


@pytest.mark.parametrize("failure", ["expired", "corrupt", "frame_changed", "citation_changed", "media_changed"])
def test_stale_or_damaged_cache_is_a_miss(sample, failure):
    project, shot, asset, row = sample
    cache = window_attempts.prepare(project, shot, asset)
    assert cache.save(row)
    body = json.loads(cache.path.read_text())
    if failure == "expired":
        body["created_at"] -= window_attempts.TTL + 1
    elif failure == "corrupt":
        cache.path.write_text("{")
    elif failure == "frame_changed":
        row["frame_paths"][0].write_bytes(b"other pixels")
    elif failure == "citation_changed":
        body["audit"]["evidence"][0]["quote"] = "invented quote"
    elif failure == "media_changed":
        paths.file_path(project["id"], asset["path"]).write_bytes(b"other media")
    if failure in ("expired", "citation_changed"):
        cache.path.write_text(json.dumps(body))
    assert cache.read() is None


def setup_matching(sample, monkeypatch):
    project, shot, asset, row = sample
    clip = {"id": "clip", "start": 0, "offset": 0, "duration": 9, "score": 20, "visible_text": "", "source": {
        "title": asset["title"], "source_url": asset["source_url"]}}
    monkeypatch.setattr(library, "search", lambda *a, **kw: [clip])
    monkeypatch.setattr(library_match, "quick_candidates", lambda p, ss, opts, stats, **kw: (
        stats.update(library_quick_skips=[]) or ss, opts))
    monkeypatch.setattr(library_match, "rank_windows", lambda *a, **kw: True)
    monkeypatch.setattr(library, "materialize", lambda *a: asset)
    calls = []
    def review(p, rows, check):
        calls.append(len(rows))
        time.sleep(.05)
        return [r | {"verification": deepcopy(row["verification"]), "frame_paths": row["frame_paths"],
                     "preview": deepcopy(row["preview"])} for r in rows], [], {"model_requests": 3}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    return calls


def test_resume_skips_exact_rejected_window_without_model_calls(sample, monkeypatch):
    calls = setup_matching(sample, monkeypatch)
    project, shot, _, _ = sample
    first = library_match.match(project, [shot], lambda: None, lambda *a: None)
    second = library_match.match(project, [shot], lambda: None, lambda *a: None)
    assert calls == [1] and not first[0] and not second[0]
    assert first[1]["library_model_requests"] == 3
    assert second[1]["library_model_requests"] == 0 and second[1]["library_rejected_window_hits"] == 1
    assert not store.get_project(project["id"])["shots"][0]["approved"]


def test_concurrent_same_window_is_reviewed_once(sample, monkeypatch):
    calls = setup_matching(sample, monkeypatch)
    project, shot, _, _ = sample
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: library_match.match(project, [shot], lambda: None, lambda *a: None), range(4)))
    assert calls == [1]
    assert sum(r[1]["library_model_requests"] for r in results) == 3
    assert sum(r[1]["library_rejected_window_hits"] for r in results) == 3
