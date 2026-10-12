from jinggao import library, paths, planner, store


def test_independent_observation_replaces_script_aware_cache_description(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    project = store.new_project("中立观察缓存回归")
    target = paths.project_dir(project["id"]) / "assets" / "fixture.mp4"
    target.parent.mkdir(exist_ok=True)
    # Only index persistence is exercised here, not video decoding.
    target.write_bytes(b"index-persistence-fixture")
    asset = {"id": store.uid(), "kind": "video", "path": "assets/fixture.mp4", "duration": 6,
             "width": 1280, "height": 720, "title": "商品展示",
             "source_url": "https://www.iqiyi.com/v_neutraltest.html", "provider": "iqiyi"}
    first = {"verdict": "match", "reason": "错误地声称拆快递", "observations": {
        "scene_description": "拆开快递包裹", "visual_evidence": [
            {"frame_index": 0, "kind": "caption", "quote": "虚构快递字幕"}]}}
    clip_id = library.remember_window(project["id"], asset, 0, 3,
        description=first["observations"]["scene_description"], visible_text=library.observed_text(first))
    audit = {"status": "conflict", "reason": "实际展示化妆工具，未见拆包裹", "labels": [],
             "scene_description": "双手展示唇刷", "frame_observations": [
                 {"frame_index": 0, "objects": "双手、唇刷", "action": "展示", "text": []}]}
    review = planner.apply_identity_audit(first, audit)
    assert review["verdict"] == "mismatch"
    assert review["observations"]["scene_description"] == "双手展示唇刷"
    assert library.observed_text(review) == ""
    shot = {"id": store.uid(), "duration": 3, "clip_start": 0, "selected_asset": None,
            "match_result": review, "match_preview": {
                "asset_id": asset["id"], "clip_start": 0, "duration": 3}}
    store.mutate(project["id"], lambda p: (p["assets"].append(asset), p["shots"].append(shot)))
    assert library.rebuild()["processed"] == 1
    saved = library.get(clip_id)
    assert saved["description"] == "双手展示唇刷" and saved["visible_text"] == ""
    assert library.search("虚构快递字幕") == []
    assert library.search("唇刷")[0]["id"] == clip_id
    # A later full-library rebuild can encounter this same window in an old
    # project. Its biased description/OCR must not resurrect the corrected error.
    library.remember_window(project["id"], asset, 0, 3,
        description=first["observations"]["scene_description"], visible_text=library.observed_text(first))
    saved = library.get(clip_id)
    assert saved["description"] == "双手展示唇刷" and saved["visible_text"] == ""
    assert library.search("虚构快递字幕") == []
    assert first["observations"]["scene_description"] == "拆开快递包裹"


def test_blind_ocr_preserves_exact_words_and_does_not_inherit_verdict_labels():
    review = {"observations": {"observation_method": "blind_actual_frames_v1",
        "frame_observations": [
            {"frame_index": 0, "text": ["2023 年业务量", "快递"]},
            {"frame_index": 1, "text": ["快递", "顺丰"]},
            {"frame_index": True, "text": ["错误帧"]},
            {"frame_index": -1, "text": ["负数帧"]},
            {"frame_index": 3, "text": "错误类型"}],
        "visual_evidence": [{"frame_index": 0, "kind": "caption", "quote": "2024年业务量"}]}}
    assert library.observed_text(review) == "2023 年业务量 · 快递 · 顺丰"


def test_legacy_review_ocr_remains_readable():
    review = {"observations": {"visual_evidence": [
        {"frame_index": 0, "kind": "nameplate", "quote": "刘翔"}]}}
    assert library.observed_text(review) == "刘翔"
