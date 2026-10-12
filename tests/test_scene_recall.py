import hashlib
import json

from jinggao import paths, scene_recall, segment_discovery


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    frame = tmp_path / "cache/scene-samples/example/frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"recorded source pixels")
    shot = {"id": "shot", "narration": "工作人员走进楼道", "duration": 8}
    source = "https://www.bilibili.com/video/BV17jFqeoEBH"
    raw = {"status": "complete", "frame_observations": {"0": {"action": "上楼"}},
           "request": {"schema_version": segment_discovery.RANK_VERSION, "shots": [shot],
               "candidates": [{"source_url": source, "title": "工作实拍", "duration": 100}],
               "frames": [{"source_url": source, "frame_id": "f0", "at": 28, "path": str(frame),
                           "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(), "width": 1280, "height": 720}]},
           "proposals": {"shot": [{"source_url": source, "frame_id": "f0", "sample_at": 28,
                                    "clip_start": 24, "duration": 8}]}}
    target = tmp_path / "cache/scene-ranks/example.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(raw))
    return shot, source, frame


def test_resume_recovers_untried_real_position_without_copying_any_approval(tmp_path, monkeypatch):
    shot, source, _ = fixture(tmp_path, monkeypatch)
    rows = scene_recall.candidates({"assets": []}, [shot], lambda: None)["shot"]
    assert len(rows) == 1 and rows[0]["resume_points"] == [28]
    assert rows[0]["source_url"] == source and rows[0]["download_status"] == "unchecked"
    assert rows[0]["sampling_only"] and not {"approved", "verification", "verdict", "media_streams"} & rows[0].keys()
    # Existing full clips belong to the normal library pass, not another paid
    # source sampling/download cycle.
    asset = {"kind": "video", "source_url": source, "source_clip_start": 23.9, "duration": 8.2}
    assert scene_recall.candidates({"assets": [asset]}, [shot], lambda: None)["shot"] == []


def test_changed_script_or_pixels_cannot_reuse_saved_locator_output(tmp_path, monkeypatch):
    shot, _, frame = fixture(tmp_path, monkeypatch)
    assert scene_recall.candidates({"assets": []}, [shot | {"narration": "工作人员打电话"}], lambda: None)["shot"] == []
    assert scene_recall.candidates({"assets": []}, [shot | {"duration": 9}], lambda: None)["shot"] == []
    frame.write_bytes(b"changed pixels")
    assert scene_recall.candidates({"assets": []}, [shot], lambda: None)["shot"] == []
