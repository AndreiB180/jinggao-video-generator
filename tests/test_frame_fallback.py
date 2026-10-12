import pytest

from jinggao import curation, frame_fallback, library, library_match, media, paths, store


@pytest.fixture
def footage(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    store.init()
    original = store.new_project("原素材")
    target = paths.project_dir(original["id"]) / "assets" / "footage.mp4"
    target.parent.mkdir(exist_ok=True)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=1280x720:rate=5:duration=3", "-c:v", "libx264", "-preset", "ultrafast", str(target)])
    asset = media.inspect_asset(original["id"], target, "包裹检查资料")
    asset.update(source_url="https://v.qq.com/x/page/parcel.html", provider="tencent",
                 source_clip_start=100, source_verified=True, approved=True,
                 verification={"verdict": "match"})
    store.mutate(original["id"], lambda p: p["assets"].append(asset))
    clips = []
    for n in range(4):
        item = asset | {"source_url": f"https://v.qq.com/x/page/parcel{n}.html", "source_clip_start": 100+n*10}
        cid = library.remember_window(original["id"], item, .5, 2, description="工作人员检查包裹")
        with library.connection() as con:
            con.execute("INSERT INTO clip_observation_methods VALUES (?,?)", (cid, "blind_actual_frames_v1"))
        clips.append(library.get(cid))
    monkeypatch.setattr(library_match, "rank_windows", lambda *a, **kw: True)
    return clips


def test_frame_uses_native_pixels_and_exact_original_time_without_copying_approval(footage):
    project = store.new_project("图片目标")
    clip = footage[0]
    asset = frame_fallback.materialize_frame(project["id"], clip, .5, lambda: None)
    assert asset["kind"] == "image" and (asset["width"], asset["height"]) == (1280, 720)
    assert asset["source_clip_start"] == asset["derived_from_video"]["source_at"] == 101.5
    assert asset["source_url"] == clip["source"]["source_url"]
    assert asset["source_verified"] and asset["derived_from_video"]["image_only"]
    assert not any(key in asset for key in ("approved", "verification"))
    assert paths.file_path(project["id"], asset["path"]).is_file()


def pending_project(count=4):
    project = store.new_project("长片", media_policy="video_preferred", min_video_ratio=.8)
    selected = {"id": "existing", "duration": 60, "selected_asset": "video", "clip_start": 0,
                "verification": {"visual_quality": {"representation": "live_action"}}}
    shots = [{"id": f"gap-{i}", "duration": 10, "narration": "检查包裹", "query": "包裹",
              "selected_asset": None, "persons": [], "clip_start": 0} for i in range(count)]
    asset = {"id": "video", "kind": "video", "duration": 60, "width": 1280, "height": 720,
             "source_url": "https://v.qq.com/x/page/previous.html"}
    project = store.mutate(project["id"], lambda p: p.update(shots=[selected, *shots], assets=[asset]))
    return project, shots


def test_fresh_review_and_image_budget_apply_even_with_parallel_preparation(footage):
    project, shots = pending_project()
    reviewed = []
    def review(p, rows, check):
        for row in rows:
            assert row["asset"]["kind"] == "image" and row["shot"]["clip_start"] == 0
            reviewed.append(row["shot"]["id"])
        return [r | {"verification": {"verdict": "match", "scene_alignment": "match",
                 "selection_blocked": False, "needs_source_check": False}} for r in rows], [], {"model_requests": 1}
    def commit(row):
        def choose(p):
            shot = next(s for s in p["shots"] if s["id"] == row["shot"]["id"])
            shot.update(selected_asset=row["asset"]["id"], verification=row["verification"])
        store.mutate(project["id"], choose)
        return True
    passed, stats = frame_fallback.match(project, shots, lambda: None, review_batch=review, commit=commit)
    report = curation.composition(store.get_project(project["id"]))
    assert len(reviewed) == 4 and len(passed) == 2
    assert stats["images_selected"] == 2 and stats["image_seconds"] == 20
    assert report["image_seconds"] == 20 and report["video_seconds"] == 60
    assert report["missing_seconds"] == 20 and not report["meets_target"]


@pytest.mark.parametrize("result", [
    {"verdict": "uncertain", "scene_alignment": "uncertain", "selection_blocked": False, "needs_source_check": False},
    {"verdict": "match", "scene_alignment": "match", "selection_blocked": True, "needs_source_check": False},
    {"verdict": "match", "scene_alignment": "match", "selection_blocked": False, "needs_source_check": True},
])
def test_uncertain_or_blocked_image_never_commits(footage, result):
    project, shots = pending_project(1)
    def review(p, rows, check):
        return [r | {"verification": result} for r in rows], [], {}
    def commit(row):
        raise AssertionError("an incomplete review cannot select the image")
    passed, _ = frame_fallback.match(project, shots, lambda: None, review_batch=review, commit=commit)
    assert not passed


def test_all_video_policy_never_materializes_photos(monkeypatch):
    project = {"shots": [{"id": "gap", "duration": 10}], "assets": [], "media_policy": "video_only"}
    def forbidden(*args, **kwargs):
        raise AssertionError("all-video project may not enter photo fallback")
    monkeypatch.setattr(library, "search", forbidden)
    assert frame_fallback.match(project, project["shots"], lambda: None,
                               review_batch=forbidden, commit=forbidden)[0] == set()


def test_auto_image_commit_deduplicates_existing_assets_and_exact_source_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project, shots = pending_project(2)
    origin = {"source_url": "https://v.qq.com/x/page/parcel.html", "source_at": 15.8}
    asset = {"id": "image", "kind": "image", "path": "assets/photo.jpg", "derived_from_video": origin}
    duplicate = asset | {"id": "copy", "path": "assets/copy.jpg"}
    store.mutate(project["id"], lambda p: p["assets"].extend([asset, duplicate]))
    result = {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
              "needs_source_check": False, "reason": "逐帧复核通过"}
    def row(shot, image):
        return {"shot": shot, "asset": image, "verification": result, "preview": {"frames": []}}
    assert frame_fallback.commit_reviewed(project["id"], row(shots[0], asset))
    assert not frame_fallback.commit_reviewed(project["id"], row(shots[1], asset))
    assert not frame_fallback.commit_reviewed(project["id"], row(shots[1], duplicate))
    current = store.get_project(project["id"])
    assert curation.image_in_use(current, asset | {"id": "saved-image", "asset_id": "image"}, shots[1]["id"])
    assert not curation.image_in_use(current, asset, shots[0]["id"])
    assert current["shots"][-1]["selected_asset"] is None
