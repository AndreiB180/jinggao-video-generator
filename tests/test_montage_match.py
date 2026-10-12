from collections import defaultdict
from copy import deepcopy

import pytest

from jinggao import library, library_match, media, montage_match, paths, plan_reselection, pooled_match, store


def accepted():
    return {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
            "needs_source_check": False, "reason": "本次实帧审核通过"}


def test_real_parts_reserve_nonoverlapping_source_time_across_other_shots():
    clip = {"start": 100, "duration": 12, "source": {"source_url": "https://source.test/a"}}
    used = defaultdict(list, {clip["source"]["source_url"]: [(103, 109)]})
    assert montage_match.reserve(clip, 3, used)[1] == (100, 103)
    assert montage_match.reserve(clip, 3, used)[1] == (109, 112)
    assert montage_match.reserve(clip, 3, used) is None
    assert sum(montage_match.piece_lengths(8.64)) == pytest.approx(8.64)
    assert not montage_match.piece_lengths(13)
    asset = {"id": "combined", "kind": "video", "source_parts": [
        {"asset_id": "origin", "source_url": clip["source"]["source_url"], "source_at": 100, "duration": 3}]}
    project = {"assets": [asset], "shots": [{"id": "used", "selected_asset": "combined", "duration": 3}]}
    assert pooled_match.intervals(project)[clip["source"]["source_url"]] == [(100, 103)]
    original = {"id": "original", "source_url": clip["source"]["source_url"], "source_clip_start": 100}
    assert not plan_reselection.window_available(project, "new", original, 1, 2)
    assert plan_reselection.window_available(project, "new", original, 3, 2)


@pytest.mark.parametrize("final_match", [False, True])
def test_all_parts_and_the_whole_montage_need_separate_new_reviews(tmp_path, monkeypatch, final_match):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("短片段自动补足")
    shot = {"id": "gap", "duration": 8, "narration": "包裹处理过程", "query": "包裹", "persons": [],
            "selected_asset": None, "clip_start": 0, "approved": False}
    project = store.mutate(project["id"], lambda p: p.update(shots=[shot], full_audio={"path": "unchanged.wav"}))
    source = {"source_url": "https://source.test/a", "title": "包裹资料"}
    clips = [{"id": str(i), "start": i*5., "duration": 4., "offset": 0, "source": source} for i in range(4)]
    monkeypatch.setattr(library, "search", lambda *a, **kw: deepcopy(clips))
    monkeypatch.setattr(library_match, "quick_candidates", lambda p, s, o, stats: (s, o))
    monkeypatch.setattr(library_match, "rank_windows", lambda *a, **kw: True)
    monkeypatch.setattr(library, "materialize", lambda pid, c, check: {
        "id": c["id"], "kind": "video", "duration": 4, "source_url": source["source_url"]})
    piece_reviews, whole_reviews = [], []
    def review(p, rows, check):
        result = []
        for r in rows:
            whole = bool(r["asset"].get("source_parts"))
            (whole_reviews if whole else piece_reviews).append(r)
            verdict = accepted()
            if whole and not final_match:
                verdict.update(verdict="uncertain", selection_blocked=True)
            result.append(r | {"verification": verdict, "preview": {"frames": []}})
        return result, [], {"model_requests": 1}
    def combine(p, s, parts, check):
        assert len(parts) == 2 and sum(r["shot"]["duration"] for r in parts) == s["duration"]
        assert all(r["verification"] == accepted() for r in parts)
        asset = {"id": "assembled", "kind": "video", "source_parts": [{"source_url": source["source_url"]}]}
        store.mutate(p["id"], lambda current: current["assets"].append(asset), internal=True)
        return asset
    monkeypatch.setattr(montage_match, "assemble", combine)
    passed, stats = montage_match.match(project, [shot], lambda: None, lambda *a: None, review_batch=review)
    current = store.get_project(project["id"])
    assert len(piece_reviews) == 2 and len(whole_reviews) == 1
    assert passed == ({"gap"} if final_match else set())
    assert current["shots"][0]["selected_asset"] == ("assembled" if final_match else None)
    assert not current["shots"][0]["approved"]
    assert current["shots"][0]["narration"] == shot["narration"]
    assert current["full_audio"] == {"path": "unchanged.wav"}
    assert stats["selected"] == int(final_match)


def test_native_assembly_has_exact_real_duration_and_per_part_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("拼接编码", aspect="16:9", min_video_height=720)
    target = paths.project_dir(project["id"])/"assets"/"original.mp4"
    target.parent.mkdir(exist_ok=True)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=25:duration=4", "-c:v", "libx264", "-preset", "ultrafast", str(target)])
    asset = media.inspect_asset(project["id"], target, "编码测试图，不是实际内容验收")
    asset.update(source_url="https://source.test/a", source_clip_start=80)
    original = target.read_bytes()
    parts = [{"asset": asset, "shot": {"duration": 2, "clip_start": at}, "verification": accepted()} for at in (0, 2)]
    made = montage_match.assemble(project, {"duration": 4}, parts, lambda: None)
    assert made["duration"] == pytest.approx(4, abs=.04)
    assert (made["width"], made["height"]) == (1280, 720)
    assert not made["source_verified"] and not made["source_url"]
    assert [r["source_at"] for r in made["source_parts"]] == [80, 82]
    assert "approved" not in made and "verification" not in made
    assert target.read_bytes() == original
    rejected = deepcopy(parts)
    rejected[0]["verification"]["selection_blocked"] = True
    with pytest.raises(ValueError, match="逐段新审核"):
        montage_match.assemble(project, {"duration": 4}, rejected, lambda: None)
